import pandas as pd

json_path = "./data/processed/instance_usage_small.json"
parquet_path = "./data/processed/instance_usage.parquet"

print("Reading instance_usage JSON...")
df = pd.read_json(json_path, lines=True)

df.to_parquet(parquet_path, engine="pyarrow")
print(f"Successfully saved {parquet_path}!\n")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())