import gzip
import json
import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def gzip_json_to_parquet(input_file, output_file):
    rows = []

    print(f"Reading {input_file}...")

    with gzip.open(input_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    print(f"Rows loaded: {len(rows):,}")

    df = pd.json_normalize(rows)

    print("Columns:")
    print(df.columns.tolist())

    print("Shape:", df.shape)

    df.to_parquet(
        output_file,
        engine="pyarrow",
        index=False
    )

    print(f"Saved: {output_file}")


gzip_json_to_parquet(
    RAW_DIR / "instance_events-000000000000.json.gz",
    OUT_DIR / "instance_events.parquet"
)


gzip_json_to_parquet(
    RAW_DIR / "instance_usage-000000000000.json.gz",
    OUT_DIR / "instance_usage.parquet"
)