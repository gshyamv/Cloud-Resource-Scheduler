import pandas as pd
import numpy as np

from src.scheduling.timeline import simulate_schedule, schedule_metrics


class Evaluator:
    def __init__(self, assignments_df: pd.DataFrame, tasks_df: pd.DataFrame, nodes_df: pd.DataFrame):
        self.assignments = assignments_df
        self.tasks = tasks_df
        self.nodes = nodes_df
        self.timeline = pd.DataFrame()      # filled by calculate_scheduling_quality()
        self.tier_table = pd.DataFrame()

        # Merge task requirements with their assigned nodes
        if not self.assignments.empty:
            self.merged_data = pd.merge(self.assignments, self.tasks, left_on='task_id', right_index=True)
        else:
            self.merged_data = pd.DataFrame()

    def calculate_resource_efficiency(self) -> dict:
        """CPU / memory utilization and fragmentation of the tasks admitted at t=0."""
        if self.merged_data.empty:
            return {"cpu_utilization": 0, "ram_utilization": 0, "fragmentation": 100}

        # Aggregate allocated resources per node
        node_usage = self.merged_data.groupby('node_id')[['req_cpu', 'req_ram']].sum()

        total_cpu_capacity = self.nodes.loc[node_usage.index, 'cpus_norm'].sum()
        total_ram_capacity = self.nodes.loc[node_usage.index, 'mem_norm'].sum()

        used_cpu = node_usage['req_cpu'].sum()
        used_ram = node_usage['req_ram'].sum()

        cpu_utilization = (used_cpu / total_cpu_capacity) * 100 if total_cpu_capacity else 0
        ram_utilization = (used_ram / total_ram_capacity) * 100 if total_ram_capacity else 0

        # Fragmentation represents unutilized resource capacity across active nodes
        fragmentation = 100 - ((cpu_utilization + ram_utilization) / 2)

        return {
            "cpu_utilization": round(cpu_utilization, 2),
            "ram_utilization": round(ram_utilization, 2),
            "fragmentation": round(fragmentation, 2)
        }

    def calculate_scheduling_quality(self) -> dict:
        """
        Plays the schedule forward (see src/scheduling/timeline.py) and returns
        makespan, wait times and deadline misses. Also sets:
            self.timeline    one row per task (start/end/wait/status)
            self.tier_table  admission, wait and deadline misses per priority tier
        """
        self.timeline = simulate_schedule(self.tasks, self.nodes, self.assignments)
        summary, self.tier_table = schedule_metrics(self.timeline)
        return summary