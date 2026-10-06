import pandas as pd
import json

# 1. Load Parquet dataset
df = pd.read_parquet("./data/processed/scheduler_input100.parquet")

# 2. Parse resource_request_json into explicit columns
def parse_req(json_str):
    try:
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
        return pd.Series([data.get('cpus', 0.0), data.get('memory', 0.0)])
    except Exception:
        return pd.Series([0.0, 0.0])

df[['req_cpus', 'req_mem']] = df['resource_request_json'].apply(parse_req)

# 3. Calculate Resource Slack (Unused allocated resources)
df['cpu_slack'] = df['req_cpus'] - df['avg_cpu_usage']
df['mem_slack'] = df['req_mem'] - df['avg_mem_usage']

print("--- Preprocessed Dataset Summary ---")
print(df[['priority', 'scheduling_class', 'req_cpus', 'avg_cpu_usage', 'cpu_slack', 'req_mem', 'avg_mem_usage', 'mem_slack']].describe())

# Save finalized dataset for simulator
df.to_parquet("./data/processed/final_scheduler_dataset100.parquet", engine="pyarrow")
print("\nSaved final dataset to ./data/processed/final_scheduler_dataset100.parquet")