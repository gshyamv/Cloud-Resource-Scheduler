import pandas as pd
from src.optimization.integer_model import IntegerProgrammingModel
from src.evaluation.metrics import Evaluator
from src.data_pipeline.batch_builder import build_task_batch, build_cluster

def main():
    print("Loading Real Cluster Data...")
    
    # 1. Load the 100k dataset
    df = pd.read_parquet("./data/processed/final_scheduler_dataset100.parquet")
    
    # 2. Use the robust batch builder (stratified sampling, gpu rules, durations)
    tasks_batch = build_task_batch(df, n_tasks=50, mode="stratified")
    cluster_nodes = build_cluster(tasks_batch, num_nodes=5, target_load=1.0)

    # --- Run Optimization ---
    print(f"Scheduling {len(tasks_batch)} tasks onto {len(cluster_nodes)} nodes...")
    solver = IntegerProgrammingModel(tasks_df=tasks_batch, nodes_df=cluster_nodes)
    solver.build_model()
    
    allocations, objective_value = solver.solve()
    
    print("\n--- Allocation Results (First 5) ---")
    print(allocations.head().to_string(index=False))
    
    # --- Evaluation ---s
    evaluator = Evaluator(allocations, tasks_batch, cluster_nodes)
    metrics = evaluator.calculate_resource_efficiency()
    quality = evaluator.calculate_scheduling_quality()
    
    print("\n--- Evaluation Metrics ---")
    print(f"CPU Utilization: {metrics['cpu_utilization']}%")
    print(f"Memory Utilization: {metrics['ram_utilization']}%")
    print(f"Makespan: {quality['makespan_hours']} hours")
    print(f"Avg Wait Time: {quality['avg_wait_time_mins']} minutes")

if __name__ == "__main__":
    main()