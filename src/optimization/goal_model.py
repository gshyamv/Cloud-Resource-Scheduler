"""
Goal Programming model for task-to-node allocation.

Place this file at:  src/optimization/goal_model.py

Hard constraints (never violated, same as the Integer model)
    * CPU and RAM capacity per node
    * each task runs on at most one node
    * GPU tasks only on GPU nodes
    * precedence: a task is only placed on the node that also hosts its
      dependency (data co-location)

Goals (soft, each measured by deviation variables)
    G1  Serve every task:   sum_j x[i,j] + e[i] = 1
        e[i] is the shortfall (task not admitted). It is penalised by the
        task's share of priority-weighted demand, so leaving a high-priority
        or large task unserved costs more than leaving a low-priority one.
    G2  Hit the utilisation target on every node, for CPU and for RAM:
        use[j,r] / capacity[j,r] - d_plus + d_minus = target
        d_plus  = above target (less headroom)
        d_minus = below target (idle capacity)

Objective (all terms are dimensionless, roughly 0..1)
    Z = w_service * sum_i W_i * e_i
      + w_over    * mean(d_plus)
      + w_idle    * mean(d_minus)

W_i = (priority tier weight) * (1 + duration/deadline, if available) * (task
size share), normalised so sum_i W_i = 1. Tight deadlines therefore raise a
task's weight, which is how deadlines enter the model.
"""
import numpy as np
import pandas as pd
import pulp

TIER_BINS = [-np.inf, 99, 115, 119, 359, np.inf]
# Free, Best-effort batch, Mid, Production, Monitoring
TIER_WEIGHTS = [1.0, 2.0, 3.0, 5.0, 6.0]


def priority_weights(tasks_df: pd.DataFrame, tier_weights=TIER_WEIGHTS) -> pd.Series:
    if "priority" not in tasks_df.columns:
        return pd.Series(1.0, index=tasks_df.index)
    tier = pd.cut(tasks_df["priority"], bins=TIER_BINS, labels=False)
    tier = tier.fillna(0).astype(int).to_numpy()
    return pd.Series(np.asarray(tier_weights, dtype=float)[tier], index=tasks_df.index)


class GoalProgrammingModel:
    def __init__(
        self,
        tasks_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        util_target: float = 0.90,
        w_service: float = 3.0,
        w_over: float = 1.0,
        w_idle: float = 1.0,
        use_deadline_urgency: bool = True,
    ):
        self.tasks = tasks_df
        self.nodes = nodes_df
        self.util_target = util_target
        self.w_service = w_service
        self.w_over = w_over
        self.w_idle = w_idle
        self.use_deadline_urgency = use_deadline_urgency

        self.model = pulp.LpProblem("Cloud_GPU_Scheduling_Goal_Prog", pulp.LpMinimize)
        self.x = {}
        self.e = {}                 # shortfall: task not admitted
        self.d_plus = {}            # (resource, node) -> above target
        self.d_minus = {}           # (resource, node) -> below target
        self.task_weight = pd.Series(dtype=float)
        self.hit_time_limit = False
        self.goal_summary = {}

    # ------------------------------------------------------------------ #
    def _task_weights(self) -> pd.Series:
        w = priority_weights(self.tasks)
        if (self.use_deadline_urgency
                and {"duration_s", "deadline_s"} <= set(self.tasks.columns)):
            ratio = (self.tasks["duration_s"] / self.tasks["deadline_s"]).clip(0, 1)
            w = w * (1 + ratio.fillna(0))
        cpu = self.tasks["req_cpu"].astype(float)
        ram = self.tasks["req_ram"].astype(float)
        size = 0.5 * (cpu / cpu.sum() + ram / ram.sum())
        tw = w * size
        return tw / tw.sum()

    def build_model(self):
        task_ids = self.tasks.index.tolist()
        node_ids = self.nodes.index.tolist()
        id_set = set(task_ids)
        cpu = self.tasks["req_cpu"].astype(float).to_dict()
        ram = self.tasks["req_ram"].astype(float).to_dict()
        cap = {"cpu": self.nodes["cpus_norm"].astype(float).to_dict(),
               "ram": self.nodes["mem_norm"].astype(float).to_dict()}
        req = {"cpu": cpu, "ram": ram}
        self.task_weight = self._task_weights()

        # Decision variables
        for i in task_ids:
            for j in node_ids:
                self.x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

        # Deviation variables
        for i in task_ids:
            self.e[i] = pulp.LpVariable(f"short_{i}", lowBound=0, upBound=1)
        for r in ("cpu", "ram"):
            for j in node_ids:
                self.d_plus[(r, j)] = pulp.LpVariable(f"dplus_{r}_{j}", lowBound=0)
                self.d_minus[(r, j)] = pulp.LpVariable(f"dminus_{r}_{j}", lowBound=0)

        # G1: serve every task (also enforces "at most one node")
        for i in task_ids:
            self.model += (
                pulp.lpSum(self.x[(i, j)] for j in node_ids) + self.e[i] == 1,
                f"Goal_Serve_{i}",
            )

        for j in node_ids:
            for r in ("cpu", "ram"):
                use = pulp.lpSum(self.x[(i, j)] * req[r][i] for i in task_ids)
                # Hard capacity
                self.model += use <= cap[r][j], f"Hard_{r}_Capacity_{j}"
                # G2: utilisation target, in fractions of node capacity
                self.model += (
                    use * (1.0 / cap[r][j])
                    - self.d_plus[(r, j)] + self.d_minus[(r, j)] == self.util_target,
                    f"Goal_{r}_Util_{j}",
                )

        # Hardware availability: GPU tasks only on GPU nodes
        if "requires_gpu" in self.tasks.columns and "has_gpu" in self.nodes.columns:
            for i in task_ids:
                if self.tasks.loc[i, "requires_gpu"]:
                    for j in node_ids:
                        if not self.nodes.loc[j, "has_gpu"]:
                            self.model += self.x[(i, j)] == 0, f"Hardware_Availability_{i}_{j}"

        # Precedence / data co-location
        if "dependency" in self.tasks.columns:
            for i in task_ids:
                dep = self.tasks.loc[i, "dependency"]
                if pd.notna(dep) and dep in id_set:
                    for j in node_ids:
                        self.model += (
                            self.x[(i, j)] <= self.x[(dep, j)],
                            f"Precedence_{i}_depends_{dep}_Node_{j}",
                        )

        # Weighted objective
        k = 2 * len(node_ids)
        self.model += (
            self.w_service * pulp.lpSum(float(self.task_weight[i]) * self.e[i] for i in task_ids)
            + (self.w_over / k) * pulp.lpSum(self.d_plus.values())
            + (self.w_idle / k) * pulp.lpSum(self.d_minus.values())
        ), "Minimize_Weighted_Deviations"

    # ------------------------------------------------------------------ #
    def solve(self, time_limit: int | None = 20, gap_rel: float = 0.01, gap_abs: float = 0.001):
        self.model.solve(pulp.PULP_CBC_CMD(
            msg=False, timeLimit=time_limit, gapRel=gap_rel, gapAbs=gap_abs))
        self.hit_time_limit = (time_limit is not None and self.model.sol_status == pulp.LpSolutionIntegerFeasible)

        assignments = []
        for (i, j), var in self.x.items():
            val = pulp.value(var)
            if val is not None and val > 0.5:
                assignments.append({"task_id": i, "node_id": j})

        self._summarise(assignments)
        return pd.DataFrame(assignments), pulp.value(self.model.objective)

    def _summarise(self, assignments):
        """How far each goal is from being met (for display in the dashboard)."""
        admitted = {a["task_id"] for a in assignments}
        unserved = sum(float(w) for i, w in self.task_weight.items() if i not in admitted)
        over = [pulp.value(v) or 0.0 for v in self.d_plus.values()]
        idle = [pulp.value(v) or 0.0 for v in self.d_minus.values()]
        self.goal_summary = {
            "unserved_weighted_share_pct": round(100 * unserved, 2),
            "avg_above_target_pct": round(100 * float(np.mean(over)), 2) if over else 0.0,
            "avg_below_target_pct": round(100 * float(np.mean(idle)), 2) if idle else 0.0,
            "tasks_admitted": len(admitted),
            "tasks_total": len(self.task_weight),
        }