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