import os
from google.cloud import bigquery
import pandas as pd

# Initialize BigQuery client
client = bigquery.Client(project="cluster-data-project-510504")

# Query the table you just created directly
query = """
SELECT * 
FROM `cluster-data-project-510504.clusterdata_local.scheduler_matched_100k`;
"""

print("Fetching 100,000 rows directly from pre-built BigQuery table...")
df = client.query(query).to_dataframe()

print(f"Downloaded {len(df):,} rows successfully.")

# Save to input parquet path
output_path = "./data/processed/scheduler_input100.parquet"
os.makedirs("./data/processed", exist_ok=True)
df.to_parquet(output_path, engine="pyarrow")

print(f"Saved directly to {output_path}!")