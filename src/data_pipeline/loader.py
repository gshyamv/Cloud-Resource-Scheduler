import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Focus restricted to core capacity and scheduling tables[cite: 4]
        self.core_tables = [
            'machine_events', 
            'task_events', 
            'task_usage', 
            'task_constraints'
        ]

    def load_table(self, table_name: str, chunk_size: int = 100000) -> pd.DataFrame:
        """Loads dataset tables in chunks to efficiently handle large telemetry traces."""
        if table_name not in self.core_tables:
            raise ValueError(f"Table '{table_name}' is excluded from optimization scope.")
            
        file_path = os.path.join(self.data_dir, 'raw', f"{table_name}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raw data file not found: {file_path}")

        # Chunked data loading for memory safety[cite: 6]
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
            
        return pd.concat(chunks, ignore_index=True)