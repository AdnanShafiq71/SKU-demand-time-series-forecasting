import pandas as pd

# --- load raw data ---
df = pd.read_excel(r"data\raw\sku_demand_time_series_synthetic.xlsx")

# --- anonymize brand names before anything else touches the data ---
unique_brands = sorted(df["brand"].unique())
pad_width = len(str(len(unique_brands)))
brand_map = {brand: f"brand_{i+1:0{pad_width}d}" for i, brand in enumerate(unique_brands)}
df["brand"] = df["brand"].map(brand_map)

print("Rows:", df.shape[0], "| Columns:", df.shape[1])
print("Unique brands:", len(unique_brands))
print(df.head(10))
print(df.dtypes)