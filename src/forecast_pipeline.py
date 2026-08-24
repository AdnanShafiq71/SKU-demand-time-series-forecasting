import pandas as pd
import numpy as np
import lightgbm as lgb

# --- Loading data (fast parquet copy, not the raw excel every time) ---
df = pd.read_parquet(r"data\processed\sku_demand_time_series_synthetic.parquet")

# Swapping real brand names for brand_1, brand_2 etc — don't want real names in a public repo
unique_brands = sorted(df["brand"].unique())
pad_width = len(str(len(unique_brands)))
brand_map = {brand: f"brand_{i+1:0{pad_width}d}" for i, brand in enumerate(unique_brands)}
df["brand"] = df["brand"].map(brand_map)

# --- clean + validate the data ---
print("Duplicate rows:", df.duplicated(subset=["sku_id", "warehouse_id", "date"]).sum())
print("Missing values per column:")
print(df.isna().sum())
print("Rows with negative units_sold:", (df["units_sold"] < 0).sum())

category_columns = ["sku_id", "category", "brand", "colour", "tip_size", "warehouse_id"]
for col in category_columns:
    df[col] = df[col].astype("category")

# checking for gaps in the weekly calendar per sku/warehouse
expected_weeks = df["date"].nunique()
actual_weeks_per_series = df.groupby(["sku_id", "warehouse_id"], observed=True)["date"].nunique()
series_with_gaps = actual_weeks_per_series[actual_weeks_per_series < expected_weeks]
print("Combos with missing weeks (before fix):", len(series_with_gaps))

# filling any gaps so every sku/warehouse has a full continuous run of weeks
full_date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="W-MON")
existing_combos = df[["sku_id", "warehouse_id"]].drop_duplicates()
scaffold = existing_combos.merge(pd.DataFrame({"date": full_date_range}), how="cross")
df = scaffold.merge(df, on=["sku_id", "warehouse_id", "date"], how="left")

df["units_sold"] = df["units_sold"].fillna(0)
df["promotion"] = df["promotion"].fillna(0)
df = df.sort_values(["sku_id", "warehouse_id", "date"])
attribute_cols = ["category", "brand", "colour", "tip_size", "price_try"]
df[attribute_cols] = df.groupby(["sku_id", "warehouse_id"], observed=True)[attribute_cols].transform(
    lambda s: s.ffill().bfill()
)

gap_check = df.groupby(["sku_id", "warehouse_id"], observed=True)["date"].nunique()
print("Combos with missing weeks (after fix):", (gap_check < len(full_date_range)).sum())

# --- Building features, for trend and seasonality ---
df["year"] = df["date"].dt.year
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month

# week 52 and week 1 are basically neighbours on a calendar, but as plain numbers
# the model sees them as far apart — this fixes that
df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

grp = df.groupby(["sku_id", "warehouse_id"], observed=True)["units_sold"]
df["lag_1"] = grp.shift(1)
df["lag_2"] = grp.shift(2)
df["lag_4"] = grp.shift(4)
df["lag_12"] = grp.shift(12)
df["lag_52"] = grp.shift(52)   # this time last year — the actual seasonality signal

df["roll_mean_4"] = grp.transform(lambda s: s.shift(1).rolling(4).mean())
df["roll_mean_12"] = grp.transform(lambda s: s.shift(1).rolling(12).mean())

# how jumpy has this sku/warehouse been lately — model couldn't see this before
df["roll_std_4"] = grp.transform(lambda s: s.shift(1).rolling(4).std())

# drop early rows that don't have a full year of lag history behind them yet
df = df.dropna(subset=["lag_52"]).reset_index(drop=True)
print("Rows after feature building:", df.shape[0])

# --- Train/test split — last 12 weeks held back to test on ---
cutoff_date = df["date"].max() - pd.Timedelta(weeks=12)
train = df[df["date"] <= cutoff_date]
test = df[df["date"] > cutoff_date]

feature_cols = [
    "sku_id", "category", "brand", "colour", "tip_size", "warehouse_id",
    "price_try", "promotion", "year", "week_sin", "week_cos", "month",
    "lag_1", "lag_2", "lag_4", "lag_12", "lag_52", "roll_mean_4", "roll_mean_12", "roll_std_4"
]
target_col = "units_sold"

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]
print("Training rows:", X_train.shape[0], "| Testing rows:", X_test.shape[0])

# --- Training the model ---
cat_features = ["sku_id", "category", "brand", "colour", "tip_size", "warehouse_id"]
train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
valid_set = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features, reference=train_set)

params = {
    "objective": "tweedie",       # fits skewed always-positive sales data better than plain regression
    "tweedie_variance_power": 1.2,
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 96,
    "min_data_in_leaf": 30,       # stops it obsessing over tiny things in the data
    "verbose": -1,
}

model = lgb.train(
    params,
    train_set,
    num_boost_round=3000,
    valid_sets=[valid_set],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200)
    ]
)

print("Best iteration:", model.best_iteration)
print("Best validation MAE:", model.best_score["valid_0"]["l1"])

# --- Scoring it properly + quick check against dumb guesses ---
preds = model.predict(X_test)
preds = np.clip(preds, 0, None)   # can't have negative demand

wmape = np.abs(y_test - preds).sum() / y_test.sum()
print(f"Overall WMAPE: {wmape:.2%}")

results = test.copy()
results["prediction"] = preds
category_wmape = results.groupby("category", observed=True).apply(
    lambda g: np.abs(g["units_sold"] - g["prediction"]).sum() / g["units_sold"].sum()
)
print("\nWMAPE by category:")
print(category_wmape.sort_values())

importance = pd.Series(model.feature_importance(), index=feature_cols).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)

# Quick check — if we can't beat "just repeat last week", nothing here is actually working
naive_last_week = test["lag_1"]
naive_last_year = test["lag_52"]
wmape_naive_last_week = np.abs(y_test - naive_last_week).sum() / y_test.sum()
wmape_naive_last_year = np.abs(y_test - naive_last_year).sum() / y_test.sum()

print(f"\nNaive (last week) WMAPE:  {wmape_naive_last_week:.2%}")
print(f"Naive (last year) WMAPE:  {wmape_naive_last_year:.2%}")
print(f"Our model WMAPE:          {wmape:.2%}")

# --- Recursive future forecast ---

horizon = 12   # how many weeks ahead we're forecasting
history = df.copy()   # working copy we'll keep growing with each new prediction
last_date = history["date"].max()
future_rows = []

for step in range(1, horizon + 1):
    next_date = last_date + pd.Timedelta(weeks=step)

    # start next week's rows from each series' most recent known row
    latest = (
        history.sort_values("date")
        .groupby(["sku_id", "warehouse_id"], observed=True)
        .tail(1)
        .copy()
    )
    latest["date"] = next_date
    latest["year"] = next_date.year
    latest["week_of_year"] = next_date.isocalendar().week
    latest["month"] = next_date.month
    latest["week_sin"] = np.sin(2 * np.pi * latest["week_of_year"] / 52)
    latest["week_cos"] = np.cos(2 * np.pi * latest["week_of_year"] / 52)
    # price_try and promotion just carry forward the last known value here —
    # swap this for a real future price/promo plan if you have one

    # look up lag values from history (now including any predictions I've already made)
    lookup = history.set_index(["sku_id", "warehouse_id", "date"])["units_sold"]

    def get_lag(row, weeks_back):
        key = (row["sku_id"], row["warehouse_id"], row["date"] - pd.Timedelta(weeks=weeks_back))
        return lookup.get(key, np.nan)

    for w in [1, 2, 4, 12, 52]:
        latest[f"lag_{w}"] = latest.apply(lambda r: get_lag(r, w), axis=1)

    # rolling stats from the most recent weeks in history
    recent = history[history["date"] > next_date - pd.Timedelta(weeks=13)]
    roll4 = recent.groupby(["sku_id", "warehouse_id"], observed=True)["units_sold"].apply(lambda s: s.tail(4).mean())
    roll12 = recent.groupby(["sku_id", "warehouse_id"], observed=True)["units_sold"].apply(lambda s: s.tail(12).mean())
    rollstd4 = recent.groupby(["sku_id", "warehouse_id"], observed=True)["units_sold"].apply(lambda s: s.tail(4).std())

    latest = latest.set_index(["sku_id", "warehouse_id"])
    latest["roll_mean_4"] = roll4
    latest["roll_mean_12"] = roll12
    latest["roll_std_4"] = rollstd4
    latest = latest.reset_index()

    # predict this week, clip negatives, then treat it as "real" for the next loop
    latest["units_sold"] = np.clip(model.predict(latest[feature_cols]), 0, None)

    future_rows.append(latest)
    history = pd.concat([history, latest], ignore_index=True)

future_df = pd.concat(future_rows, ignore_index=True)
print("Future forecast rows generated:", future_df.shape[0])
print(future_df[["date", "sku_id", "warehouse_id", "units_sold"]].head(15))