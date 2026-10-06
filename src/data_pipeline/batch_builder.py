"""
Builds the scheduling problem (task batch + cluster) for the solvers.

Place this file at:  src/data_pipeline/batch_builder.py

Why this exists
---------------
* df.head(50) picks consecutive rows, which come from one or two collections
  and one or two priorities. stratified_sample() spreads the batch across
  collections and priority tiers instead.
* Request values are tiny (median CPU ~0.02, RAM ~0.001), so nodes with
  capacity 1.0 are almost never full and every task fits. build_cluster()
  sizes node capacity from the batch's own demand so that
  "total demand / total capacity" is an explicit knob (target_load).

Note: capacities here are synthetic. They are derived from the batch, not
taken from the trace's machine_events table. Say so in the report / viva.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIER_BINS = [-np.inf, 99, 115, 119, 359, np.inf]
TIER_LABELS = [
    "Free (0-99)",
    "Best-effort batch (100-115)",
    "Mid (116-119)",
    "Production (120-359)",
    "Monitoring (360+)",
]


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match the column names the solvers expect (req_cpu / req_ram)."""
    return df.rename(columns={"req_cpus": "req_cpu", "req_mem": "req_ram"})


def add_priority_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["priority_tier"] = pd.cut(
        df["priority"], bins=TIER_BINS, labels=TIER_LABELS
    ).astype(str)
    return df


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def stratified_sample(
    df: pd.DataFrame,
    n: int,
    strata_cols: tuple[str, ...] = ("collection_id", "priority_tier"),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n rows so that every stratum (collection x priority tier) is
    represented roughly in proportion to its size, with at least one row
    per stratum whenever n allows it.
    """
    n = min(n, len(df))
    cols = [c for c in strata_cols if c in df.columns]
    if not cols:
        return df.sample(n=n, random_state=seed)

    keys = df[cols].astype(str).agg("|".join, axis=1)
    sizes = keys.value_counts()
    rng = np.random.default_rng(seed)

    if n <= len(sizes):
        # More strata than rows requested: pick n different strata.
        chosen = rng.choice(sizes.index.to_numpy(), size=n, replace=False)
        alloc = pd.Series(1, index=chosen)
    else:
        ideal = sizes / sizes.sum() * n
        alloc = np.minimum(np.floor(ideal).astype(int).clip(lower=1), sizes)
        diff = n - int(alloc.sum())
        while diff != 0:
            if diff > 0:
                room = (sizes - alloc)[lambda s: s > 0].index
                alloc[(ideal - alloc)[room].idxmax()] += 1
                diff -= 1
            else:
                reducible = alloc[alloc > 1].index
                alloc[(alloc - ideal)[reducible].idxmax()] -= 1
                diff += 1

    parts = [
        df[keys == k].sample(n=int(a), random_state=seed)
        for k, a in alloc.items()
        if a > 0
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed)


# def build_task_batch(
#     raw: pd.DataFrame,
#     n_tasks: int = 60,
#     mode: str = "stratified",          # "stratified" or "head"
#     gpu_fraction: float = 0.2,
#     n_dependencies: int = 3,
#     seed: int = 42,
# ) -> pd.DataFrame:
#     """Return a task table indexed task_0..task_N with the solver's columns."""
#     df = add_priority_tier(standardize_columns(raw))

#     if mode == "stratified":
#         tasks = stratified_sample(df, n_tasks, seed=seed)
#     else:
#         tasks = df.head(n_tasks).copy()

#     tasks = tasks.reset_index(drop=True)
#     tasks.index = [f"task_{i}" for i in range(len(tasks))]

#     rng = np.random.default_rng(seed)
#     tasks["requires_gpu"] = rng.random(len(tasks)) < gpu_fraction

#     # Dependencies always point to an earlier task, so no cycles are possible.
#     tasks["dependency"] = None
#     if n_dependencies > 0 and len(tasks) > 1:
#         k = min(n_dependencies, len(tasks) - 1)
#         for i in rng.choice(np.arange(1, len(tasks)), size=k, replace=False):
#             tasks.loc[f"task_{i}", "dependency"] = f"task_{rng.integers(0, i)}"
#     return tasks

def build_task_batch(
    raw: pd.DataFrame,
    n_tasks: int = 60,
    mode: str = "stratified",
    gpu_fraction: float = 0.2,
    n_dependencies: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a task table indexed task_0..task_N with the solver's columns."""
    df = add_priority_tier(standardize_columns(raw))

    if mode == "stratified":
        tasks = stratified_sample(df, n_tasks, seed=seed)
    else:
        tasks = df.head(n_tasks).copy()

    tasks = tasks.reset_index(drop=True)
    tasks.index = [f"task_{i}" for i in range(len(tasks))]

    rng = np.random.default_rng(seed)
    tasks["requires_gpu"] = rng.random(len(tasks)) < gpu_fraction

    # 1. Add synthetic durations (e.g., between 1 minute and 2 hours)
    tasks["duration_s"] = rng.uniform(60, 7200, size=len(tasks))
    
    # 2. Add deadlines (duration + a random slack multiplier)
    # Tighter deadlines for higher priority tasks can also be enforced here
    slack = rng.uniform(1.1, 5.0, size=len(tasks))
    tasks["deadline_s"] = tasks["duration_s"] * slack

    # Dependencies always point to an earlier task, so no cycles are possible.
    tasks["dependency"] = None
    if n_dependencies > 0 and len(tasks) > 1:
        k = min(n_dependencies, len(tasks) - 1)
        for i in rng.choice(np.arange(1, len(tasks)), size=k, replace=False):
            tasks.loc[f"task_{i}", "dependency"] = f"task_{rng.integers(0, i)}"
            
    return tasks
# --------------------------------------------------------------------------- #
# Cluster
# --------------------------------------------------------------------------- #
def build_cluster(
    tasks: pd.DataFrame,
    num_nodes: int = 5,
    gpu_nodes: int = 2,
    target_load: float = 1.0,
    heterogeneity: float = 0.3,
) -> pd.DataFrame:
    """
    Size node capacity so that  total demand / total capacity == target_load
    for both CPU and RAM.

    target_load < 1  -> spare capacity, everything fits
    target_load = 1  -> tight, bin-packing losses force some rejections
    target_load > 1  -> overloaded, the solver must choose which tasks to run

    heterogeneity spreads capacity across nodes (0 = identical nodes). GPU
    nodes come first, so they are the larger machines.
    """
    gpu_nodes = max(0, min(gpu_nodes, num_nodes))
    heterogeneity = min(max(heterogeneity, 0.0), 0.9)

    mult = 1 + heterogeneity * np.linspace(1, -1, num_nodes)
    mult = mult / mult.mean()                      # total capacity unchanged

    base_cpu = tasks["req_cpu"].sum() / (target_load * num_nodes)
    base_ram = tasks["req_ram"].sum() / (target_load * num_nodes)

    return pd.DataFrame(
        {
            "cpus_norm": base_cpu * mult,
            "mem_norm": base_ram * mult,
            "has_gpu": [True] * gpu_nodes + [False] * (num_nodes - gpu_nodes),
        },
        index=[f"Node_{i}" for i in range(num_nodes)],
    )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def problem_report(tasks: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    """How tight is this problem? Used to warn when the comparison is trivial."""
    cpu_load = tasks["req_cpu"].sum() / nodes["cpus_norm"].sum() * 100
    ram_load = tasks["req_ram"].sum() / nodes["mem_norm"].sum() * 100

    gpu_tasks = tasks[tasks["requires_gpu"].astype(bool)] if "requires_gpu" in tasks else tasks.iloc[0:0]
    gpu_nodes = nodes[nodes["has_gpu"].astype(bool)] if "has_gpu" in nodes else nodes.iloc[0:0]

    if len(gpu_tasks) == 0:
        gpu_load = 0.0
    elif len(gpu_nodes) == 0:
        gpu_load = float("inf")
    else:
        gpu_load = gpu_tasks["req_cpu"].sum() / gpu_nodes["cpus_norm"].sum() * 100

    # Tasks larger than the biggest node they are allowed to run on
    def too_big(group: pd.DataFrame, pool: pd.DataFrame) -> int:
        if len(group) == 0:
            return 0
        if len(pool) == 0:
            return len(group)
        return int(
            ((group["req_cpu"] > pool["cpus_norm"].max())
             | (group["req_ram"] > pool["mem_norm"].max())).sum()
        )

    unplaceable = too_big(tasks[~tasks["requires_gpu"].astype(bool)] if "requires_gpu" in tasks else tasks, nodes)
    unplaceable += too_big(gpu_tasks, gpu_nodes)

    return {
        "n_tasks": len(tasks),
        "n_nodes": len(nodes),
        "cpu_load_pct": round(cpu_load, 1),
        "ram_load_pct": round(ram_load, 1),
        "overall_load_pct": round(max(cpu_load, ram_load), 1),
        "binding_resource": "CPU" if cpu_load >= ram_load else "RAM",
        "gpu_task_count": int(len(gpu_tasks)),
        "gpu_load_pct": round(gpu_load, 1) if np.isfinite(gpu_load) else float("inf"),
        "unplaceable_tasks": unplaceable,
    }
