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