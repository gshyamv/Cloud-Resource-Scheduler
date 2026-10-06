import pandas as pd
import numpy as np
import os

class DataPreprocessor:
    def __init__(self):
        pass

    def integrate_tasks(self, task_events: pd.DataFrame, task_usage: pd.DataFrame) -> pd.DataFrame:
        """Integrates task events and usage metrics, handling missing values[cite: 6]."""
        # Task-level integration merging on job_id and task_idx[cite: 6]
        merged_tasks = pd.merge(
            task_events, 
            task_usage, 
            on=['job_id', 'task_idx'], 
            how='inner'
        )
        
        # Missing value handling for CPU/Memory requests[cite: 6]
        merged_tasks['req_cpu'] = merged_tasks['req_cpu'].fillna(merged_tasks['req_cpu'].median())
        merged_tasks['req_ram'] = merged_tasks['req_ram'].fillna(merged_tasks['req_ram'].median())
        
        return merged_tasks

    def apply_iqr_filtering(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Applies IQR-based outlier filtering to feature distributions[cite: 6]."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
    def generate_workload_profile(self, merged_tasks: pd.DataFrame, output_dir: str):
        """Processes features and exports the clean dataset to Parquet format[cite: 6]."""
        # Outlier filtering on resource requests[cite: 6]
        clean_df = self.apply_iqr_filtering(merged_tasks, 'req_cpu')
        clean_df = self.apply_iqr_filtering(clean_df, 'req_ram')
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'workload_profile.parquet')
        
        # Save processed dataset as Parquet for optimized downstream loading[cite: 6]
        clean_df.to_parquet(output_path, engine='pyarrow')
        return clean_df