"""
Real schedule timeline for a solver's assignment.

Place this file at:  src/scheduling/timeline.py

How time enters the project
---------------------------
The IP / GP models decide WHICH tasks are admitted at t=0 and on WHICH node
(a snapshot packing: everything admitted fits on its node at the same time).
This module then plays the schedule forward:

  * Admitted tasks start at t=0 on the node the solver chose. A task with a
    dependency starts when its dependency ends (same node, as the models
    enforce). Its capacity stays reserved from t=0, so the solver's packing
    guarantee holds.
  * Tasks the solver did NOT admit wait in a queue ordered by priority
    (highest first), then earliest deadline, then input order. Whenever a
    task ends and frees capacity, queued tasks start if they fit (small tasks
    may backfill ahead of larger ones that do not fit yet).
  * Deferred tasks respect GPU availability, and a deferred task with a
    dependency can only run on its dependency's node after it finishes.
  * All tasks are released together at t=0, so wait time = start time.

Because a solver that admits the right tasks first makes high-priority work
wait less, wait time, makespan and deadline misses now depend on the solver.

Durations and deadlines come from the task table (duration_s, deadline_s).
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from src.data_pipeline.batch_builder import TIER_LABELS

HIGH_PRIORITY_MIN = 120      # Production tier (120+) and above
_EPS = 1e-9


def _prepare(tasks: pd.DataFrame) -> pd.DataFrame:
    t = tasks.copy()
    if "duration_s" not in t:
        t["duration_s"] = 600.0
    if "deadline_s" not in t:
        t["deadline_s"] = np.inf
    if "priority" not in t:
        t["priority"] = 0
    if "requires_gpu" not in t:
        t["requires_gpu"] = False
    if "dependency" not in t:
        t["dependency"] = None
    return t


def simulate_schedule(
    tasks: pd.DataFrame,
    nodes: pd.DataFrame,
    assignments: pd.DataFrame,
    high_priority_min: int = HIGH_PRIORITY_MIN,
) -> pd.DataFrame:
    """
    Return one row per task with start/end/wait times and deadline outcome.
    status is "ran" or "never_ran" (e.g. it fits on no eligible node).
    """
    t = _prepare(tasks)
    ids = t.index.tolist()
    order = {tid: k for k, tid in enumerate(ids)}
    cpu = t["req_cpu"].astype(float).to_dict()
    ram = t["req_ram"].astype(float).to_dict()
    dur = t["duration_s"].astype(float).to_dict()
    ddl = t["deadline_s"].astype(float).to_dict()
    prio = t["priority"].to_dict()
    needs_gpu = t["requires_gpu"].astype(bool).to_dict()
    dep = {i: d for i, d in t["dependency"].items() if pd.notna(d) and d in cpu}

    node_ids = nodes.index.tolist()
    cap_c = nodes["cpus_norm"].astype(float).to_dict()
    cap_m = nodes["mem_norm"].astype(float).to_dict()
    has_gpu = (nodes["has_gpu"].astype(bool).to_dict() if "has_gpu" in nodes
               else {n: True for n in node_ids})

    admitted: dict = {}
    if assignments is not None and len(assignments):
        admitted = dict(zip(assignments["task_id"], assignments["node_id"]))

    used_c = {n: 0.0 for n in node_ids}
    used_m = {n: 0.0 for n in node_ids}
    for tid, n in admitted.items():                # reserve admitted capacity
        used_c[n] += cpu[tid]
        used_m[n] += ram[tid]

    rec: dict = {}                                 # tid -> (node, start, end)
    heap: list = []
    waiting_admitted = [i for i in ids if i in admitted]
    queue = sorted(
        (i for i in ids if i not in admitted),
        key=lambda i: (-prio[i], ddl[i], order[i]),
    )

    def launch(tid, node, now):
        end = now + dur[tid]
        rec[tid] = (node, now, end)
        heapq.heappush(heap, (end, order[tid], tid))

    def dep_ready(tid, now):
        d = dep.get(tid)
        return d is None or (d in rec and rec[d][2] <= now + _EPS)

    def try_start(now):
        for tid in list(waiting_admitted):
            if dep_ready(tid, now):
                launch(tid, admitted[tid], now)    # capacity already reserved
                waiting_admitted.remove(tid)
        for tid in list(queue):
            if not dep_ready(tid, now):
                continue
            d = dep.get(tid)
            candidates = [rec[d][0]] if d is not None else node_ids
            best = None
            for n in candidates:
                if needs_gpu[tid] and not has_gpu[n]:
                    continue
                left_c = cap_c[n] - used_c[n] - cpu[tid]
                left_m = cap_m[n] - used_m[n] - ram[tid]
                if left_c < -1e-12 or left_m < -1e-12:
                    continue
                # keep GPU nodes free for GPU tasks, then best fit
                key = (has_gpu[n] and not needs_gpu[tid],
                       left_c / cap_c[n] + left_m / cap_m[n])
                if best is None or key < best[0]:
                    best = (key, n)
            if best is not None:
                n = best[1]
                used_c[n] += cpu[tid]
                used_m[n] += ram[tid]
                launch(tid, n, now)
                queue.remove(tid)

    try_start(0.0)
    while heap:
        now = heap[0][0]
        while heap and heap[0][0] <= now + _EPS:
            _, _, tid = heapq.heappop(heap)
            n = rec[tid][0]
            used_c[n] -= cpu[tid]
            used_m[n] -= ram[tid]
        try_start(now)

    rows = []
    for tid in ids:
        if tid in rec:
            node, start, end = rec[tid]
            status = "ran"
        else:
            node, start, end, status = admitted.get(tid), np.nan, np.nan, "never_ran"
        rows.append({
            "task_id": tid, "node_id": node,
            "start_s": start, "end_s": end, "duration_s": dur[tid],
            "wait_s": start, "status": status,
            "admitted": tid in admitted,
            "priority": prio[tid],
            "high_priority": bool(prio[tid] >= high_priority_min),
            "deadline_s": ddl[tid],
            "met_deadline": bool(status == "ran" and end <= ddl[tid] + _EPS),
            "req_cpu": cpu[tid], "req_ram": ram[tid],
        })
    tl = pd.DataFrame(rows)
    if "priority_tier" in t.columns:
        tl["priority_tier"] = t["priority_tier"].reindex(tl["task_id"]).to_numpy()
    return tl


def schedule_metrics(tl: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Summary metrics and a per-priority-tier table for a simulated timeline."""
    nan = float("nan")
    n = len(tl)
    ran = tl[tl["status"] == "ran"]
    hp = tl[tl["high_priority"]]
    hp_ran = hp[hp["status"] == "ran"]

    def pct(part, whole):
        return round(100.0 * part / whole, 2) if whole else nan

    summary = {
        "makespan_hours": round(ran["end_s"].max() / 3600, 2) if len(ran) else 0.0,
        "avg_wait_time_mins": round(ran["wait_s"].mean() / 60, 2) if len(ran) else 0.0,
        "hp_avg_wait_mins": round(hp_ran["wait_s"].mean() / 60, 2) if len(hp_ran) else nan,
        "admitted_pct": pct(int(tl["admitted"].sum()), n),
        "hp_admitted_pct": pct(int(hp["admitted"].sum()), len(hp)),
        "deadline_miss_pct": pct(int((~tl["met_deadline"]).sum()), n),
        "hp_deadline_miss_pct": pct(int((~hp["met_deadline"]).sum()), len(hp)),
        "never_ran": int((tl["status"] != "ran").sum()),
        "high_priority_tasks": int(len(hp)),
    }

    tiers = pd.DataFrame()
    if "priority_tier" in tl.columns:
        g = tl.groupby("priority_tier")
        tiers = pd.DataFrame({
            "tasks": g.size(),
            "admitted_at_t0_%": g["admitted"].mean() * 100,
            "avg_wait_min": tl[tl["status"] == "ran"].groupby("priority_tier")["wait_s"].mean() / 60,
            "deadline_miss_%": (1 - g["met_deadline"].mean()) * 100,
        }).round(1)
        tiers = tiers.reindex([x for x in TIER_LABELS if x in tiers.index])
    return summary, tiers


def assign_lanes(tl_ran: pd.DataFrame) -> pd.DataFrame:
    """
    Give concurrent tasks on a node separate rows so Gantt bars do not overlap.
    Adds `lane` and `row` (e.g. "Node_0 #3") columns.
    """
    df = tl_ran.sort_values(["node_id", "start_s", "task_id"]).copy()
    lanes = []
    for _, grp in df.groupby("node_id", sort=False):
        lane_end: list[float] = []
        for _, r in grp.iterrows():
            for k, e in enumerate(lane_end):
                if e <= r["start_s"] + _EPS:
                    lane_end[k] = r["end_s"]
                    lanes.append(k)
                    break
            else:
                lane_end.append(r["end_s"])
                lanes.append(len(lane_end) - 1)
    df["lane"] = lanes
    df["row"] = df["node_id"].astype(str) + " #" + (df["lane"] + 1).astype(str)
    return df