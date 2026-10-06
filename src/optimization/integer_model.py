import pulp
import pandas as pd

class IntegerProgrammingModel:
    def __init__(self, tasks_df: pd.DataFrame, nodes_df: pd.DataFrame, w_c: float = 1.0, w_m: float = 1.0, w_f: float = 0.1):
        self.tasks = tasks_df
        self.nodes = nodes_df
        self.w_c = w_c
        self.w_m = w_m
        self.w_f = w_f  # Weight applied to the fragmentation penalty
        
        self.model = pulp.LpProblem("Cloud_GPU_Scheduling_MILP", pulp.LpMaximize)
        self.x = {}
        self.y = {}     # Node active indicator
        self.hit_time_limit = False

    def build_model(self):
        task_ids = self.tasks.index.tolist()
        node_ids = self.nodes.index.tolist()

        # Binary Decision Variables for Tasks (x) and Nodes (y)
        for i in task_ids:
            for j in node_ids:
                self.x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpBinary)
                
        for j in node_ids:
            self.y[j] = pulp.LpVariable(f"y_{j}", 0, 1, pulp.LpBinary)

        # Objective 1: Maximize Utilization (U)
        utilization = pulp.lpSum(
            self.x[(i, j)] * (self.w_c * self.tasks.loc[i, 'req_cpu'] + self.w_m * self.tasks.loc[i, 'req_ram'])
            for i in task_ids for j in node_ids
        )
        
        # Objective 2: Minimize Fragmentation (F)
        # F = Sum of Active Node Capacity - Sum of Allocated Resources
        total_active_capacity = pulp.lpSum(
            self.y[j] * (self.nodes.loc[j, 'cpus_norm'] + self.nodes.loc[j, 'mem_norm']) 
            for j in node_ids
        )
        total_allocated = pulp.lpSum(
            self.x[(i, j)] * (self.tasks.loc[i, 'req_cpu'] + self.tasks.loc[i, 'req_ram'])
            for i in task_ids for j in node_ids
        )
        fragmentation = total_active_capacity - total_allocated

        # Composite Objective: Maximize U - (w_f * F)
        # Subtracting fragmentation naturally minimizes it
        self.model += utilization - (self.w_f * fragmentation), "Maximize_Util_Minimize_Frag"

        # Constraint 1: Assignment
        for i in task_ids:
            self.model += pulp.lpSum(self.x[(i, j)] for j in node_ids) <= 1, f"Assign_Task_{i}"

        # Constraint 2: Node Activation Linking
        # If task i is assigned to node j (x=1), node j must be marked active (y=1)
        for i in task_ids:
            for j in node_ids:
                self.model += self.x[(i, j)] <= self.y[j], f"Link_x_{i}_y_{j}"

        # Constraint 3 & 4: Hard Capacity (CPU/RAM)
        for j in node_ids:
            self.model += pulp.lpSum(self.x[(i, j)] * self.tasks.loc[i, 'req_cpu'] for i in task_ids) <= self.nodes.loc[j, 'cpus_norm'], f"CPU_Node_{j}"
            self.model += pulp.lpSum(self.x[(i, j)] * self.tasks.loc[i, 'req_ram'] for i in task_ids) <= self.nodes.loc[j, 'mem_norm'], f"RAM_Node_{j}"

        # ADVANCED CONSTRAINT: Availability (Hardware affinity)
        if 'requires_gpu' in self.tasks.columns and 'has_gpu' in self.nodes.columns:
            for i in task_ids:
                if self.tasks.loc[i, 'requires_gpu']:
                    for j in node_ids:
                        if not self.nodes.loc[j, 'has_gpu']:
                            self.model += self.x[(i, j)] == 0, f"Hardware_Availability_{i}_{j}"

        # ADVANCED CONSTRAINT: Precedence (Data Co-location)
        if 'dependency' in self.tasks.columns:
            for i in task_ids:
                dep = self.tasks.loc[i, 'dependency']
                if pd.notna(dep) and dep in task_ids:
                    for j in node_ids:
                        self.model += self.x[(i, j)] <= self.x[(dep, j)], f"Precedence_{i}_depends_{dep}_Node_{j}"

    def solve(self, time_limit: int | None = 20, gap_rel: float = 0.01):
        self.model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap_rel))
        self.hit_time_limit = (time_limit is not None and self.model.sol_status == pulp.LpSolutionIntegerFeasible)

        assignments = []
        for (i, j), var in self.x.items():
            val = pulp.value(var)
            if val is not None and val > 0.5:
                assignments.append({'task_id': i, 'node_id': j})
                
        return pd.DataFrame(assignments), pulp.value(self.model.objective)