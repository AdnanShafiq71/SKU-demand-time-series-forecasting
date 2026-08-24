import pandas as pd

# --- load raw data ---

# --- from now on, load the fast Parquet copy instead ---
df = pd.read_parquet(r"data\processed\sku_demand_time_series_synthetic.parquet")

# --- anonymize brand names before anything else touches the data ---
unique_brands = sorted(df["brand"].unique())
pad_width = len(str(len(unique_brands)))
brand_map = {brand: f"brand_{i+1:0{pad_width}d}" for i, brand in enumerate(unique_brands)}
df["brand"] = df["brand"].map(brand_map)

print("Rows:", df.shape[0], "| Columns:", df.shape[1])
print("Unique brands:", len(unique_brands))
print(df.head(10))
print(df.dtypes)

# --- save a fast-loading copy for every future run ---
df.to_parquet(r"data\processed\sku_demand_time_series_synthetic.parquet", index=False)
print("Saved fast-loading copy to data/processed/")

# --- clean and validate data ---

# check for duplicate rows: each sku + warehouse + week should appear only once
duplicate_count = df.duplicated(subset=["sku_id", "warehouse_id", "date"]).sum()
print("Duplicate rows:", duplicate_count)

# check for missing values in any column
print("Missing values per column:")
print(df.isna().sum())

# check for impossible values — demand can't be negative
negative_demand = (df["units_sold"] < 0).sum()
print("Rows with negative units_sold:", negative_demand)

# convert text columns to "category" type — this uses less memory and
# helps our model later understand these are fixed groups, not free text
category_columns = ["sku_id", "category", "brand", "colour", "tip_size", "warehouse_id"]
for col in category_columns:
    df[col] = df[col].astype("category")

print("Data cleaning checks complete.")

# check every sku + warehouse has a full, continuous run of weekly dates
expected_weeks = df["date"].nunique()  # how many unique weeks exist overall
actual_weeks_per_series = df.groupby(["sku_id", "warehouse_id"], observed=True)["date"].nunique()

series_with_gaps = actual_weeks_per_series[actual_weeks_per_series < expected_weeks]
print("Expected weeks per series:", expected_weeks)
print("Number of sku/warehouse combinations with missing weeks:", len(series_with_gaps))

# --- fix missing weeks: give every EXISTING sku/warehouse pair a complete weekly calendar ---

full_date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="W-MON")

# only the sku/warehouse pairs that genuinely exist in your data — not every possible combination
existing_combos = df[["sku_id", "warehouse_id"]].drop_duplicates()

# cross each real combo with every week in the date range = the "complete calendar" we want
scaffold = existing_combos.merge(pd.DataFrame({"date": full_date_range}), how="cross")

# left-merge the real data onto that complete calendar — missing weeks become blank (NaN) rows
df = scaffold.merge(df, on=["sku_id", "warehouse_id", "date"], how="left")

# fill in the blanks sensibly
df["units_sold"] = df["units_sold"].fillna(0)     # no sale recorded that week = 0 units
df["promotion"] = df["promotion"].fillna(0)        # assume no promo if not recorded

# these describe the product/warehouse and don't change week to week — carry them across gaps
df = df.sort_values(["sku_id", "warehouse_id", "date"])
attribute_cols = ["category", "brand", "colour", "tip_size", "price_try"]
df[attribute_cols] = df.groupby(["sku_id", "warehouse_id"], observed=True)[attribute_cols].transform(
    lambda s: s.ffill().bfill()
)

print("Rows after filling gaps:", df.shape[0])
gap_check = df.groupby(["sku_id", "warehouse_id"], observed=True)["date"].nunique()
print("Combinations still missing weeks:", (gap_check < len(full_date_range)).sum())

# --- Build calendar and lag/rolling features ---

df["year"] = df["date"].dt.year
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month

# group by the unique series (one sku in one warehouse) before lagging —
# this stops SKU0001's history from leaking into SKU0002's lag features
grp = df.groupby(["sku_id", "warehouse_id"], observed=True)["units_sold"]

df["lag_1"] = grp.shift(1)     # demand 1 week ago
df["lag_2"] = grp.shift(2)     # demand 2 weeks ago
df["lag_4"] = grp.shift(4)     # demand 4 weeks ago
df["lag_12"] = grp.shift(12)   # demand 12 weeks ago (roughly a quarter)
df["lag_52"] = grp.shift(52)   # demand 52 weeks ago (this time last year — the yearly seasonality signal)

# rolling averages smooth out noise and show recent momentum
df["roll_mean_4"] = grp.transform(lambda s: s.shift(1).rolling(4).mean())
df["roll_mean_12"] = grp.transform(lambda s: s.shift(1).rolling(12).mean())

# the earliest rows of each series don't have a full year of lag history yet — drop them
rows_before = df.shape[0]
df = df.dropna(subset=["lag_52"]).reset_index(drop=True)
rows_after = df.shape[0]

print("Rows before dropping incomplete history:", rows_before)
print("Rows after dropping incomplete history:", rows_after)
print(df[["date", "sku_id", "warehouse_id", "units_sold", "lag_1", "lag_52", "roll_mean_4"]].head(10))

# --- Train/test split by date ---

cutoff_date = df["date"].max() - pd.Timedelta(weeks=12)   # hold back the last 12 weeks

train = df[df["date"] <= cutoff_date]
test = df[df["date"] > cutoff_date]

feature_cols = [
    "sku_id", "category", "brand", "colour", "tip_size", "warehouse_id",
    "price_try", "promotion", "year", "week_of_year", "month",
    "lag_1", "lag_2", "lag_4", "lag_12", "lag_52", "roll_mean_4", "roll_mean_12"
]
target_col = "units_sold"

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
print("Training date range:", train["date"].min(), "to", train["date"].max())
print("Testing date range:", test["date"].min(), "to", test["date"].max())

# --- Train the model ---

import lightgbm as lgb

# tell LightGBM which columns are categories (groups), not numbers to do math on
cat_features = ["sku_id", "category", "brand", "colour", "tip_size", "warehouse_id"]

train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
valid_set = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features, reference=train_set)

params = {
    "objective": "regression",   # we're predicting a number (units sold), not a category
    "metric": "mae",             # how we are measuring "being wrong" while training
    "learning_rate": 0.05,       # how big a step the model takes while learning each round
    "num_leaves": 64,            # how complex each individual tree in the model is allowed to be
    "verbose": -1,                # keep the console output clean
}

model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,        # train up to 1000 rounds...
    valid_sets=[valid_set],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),   # ...but stop early if it stops improving for 50 rounds straight
        lgb.log_evaluation(period=100)            # print progress every 100 rounds
    ]
)

print("Best iteration:", model.best_iteration)
print("Best validation MAE:", model.best_score["valid_0"]["l1"])

# --- Evaluate accuracy and check feature importance ---

import numpy as np

preds = model.predict(X_test)
preds = np.clip(preds, 0, None)   # demand can't be negative, so clip any stray negative predictions to 0

# WMAPE = weighted mean absolute percentage error
# unlike plain MAPE, it doesn't explode on near-zero-demand weeks, since it
# weighs errors by total volume rather than averaging raw percentages
wmape = np.abs(y_test - preds).sum() / y_test.sum()
print(f"Overall WMAPE: {wmape:.2%}")

# break it down by category, to see if some pen types forecast better than others
results = test.copy()
results["prediction"] = preds
category_wmape = results.groupby("category", observed=True).apply(
    lambda g: np.abs(g["units_sold"] - g["prediction"]).sum() / g["units_sold"].sum()
)
print("\nWMAPE by category:")
print(category_wmape.sort_values())

# which features actually drove the predictions?
importance = pd.Series(model.feature_importance(), index=feature_cols).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)

# --- Check: compare the model against simple naive forecasts ---
# if our trained model can't beat these simple guesses, something's wrong.
# if it does beat them, the remaining error is likely to be just real noise in the data.

naive_last_week = test["lag_1"]          # guess: "this week = last week"
naive_last_year = test["lag_52"]          # guess: "this week = same week last year"

wmape_naive_last_week = np.abs(y_test - naive_last_week).sum() / y_test.sum()
wmape_naive_last_year = np.abs(y_test - naive_last_year).sum() / y_test.sum()

print(f"Naive (last week) WMAPE:  {wmape_naive_last_week:.2%}")
print(f"Naive (last year) WMAPE:  {wmape_naive_last_year:.2%}")
print(f"Our model WMAPE:          {wmape:.2%}")