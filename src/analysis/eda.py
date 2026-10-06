"""
EDA & workload insights for the Cloud GPU Resource Scheduler dashboard.

Place this file at:  src/analysis/eda.py

Hook it into app/dashboard.py:

    from src.analysis.eda import render_eda_tab

    with tab1:
        render_eda_tab()

Covers:
  1. describe() summary statistics (+ skew, missing %)
  2. Priority and CPU / RAM request histograms
  3. Correlation heatmap (Pearson or Spearman)
  4. Failure / eviction rates (overall, and by priority tier)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_PATH = (
    "./data/processed/final_scheduler_dataset100.parquet"
    if os.path.exists("./data/processed/final_scheduler_dataset100.parquet")
    else "./data/processed/final_scheduler_dataset.parquet"
)

# instance_events.parquet is written by scripts/prepare_clusterdata.py
EVENTS_PATHS = [
    "./data/instance_events.parquet",
    "./data/processed/instance_events.parquet",
]

# Google 2019 trace event-type enum (used when the column is numeric).
# If your export uses string labels (e.g. "EVICT"), those are used as-is.
EVENT_NAMES = {
    0: "SUBMIT", 1: "QUEUE", 2: "ENABLE", 3: "SCHEDULE", 4: "EVICT",
    5: "FAIL", 6: "FINISH", 7: "KILL", 8: "LOST",
    9: "UPDATE_PENDING", 10: "UPDATE_RUNNING",
}
TERMINAL_EVENTS = ["FINISH", "EVICT", "FAIL", "KILL", "LOST"]

# Google 2019 priority tiers
TIER_BINS = [-np.inf, 99, 115, 119, 359, np.inf]
TIER_LABELS = [
    "Free (0-99)",
    "Best-effort batch (100-115)",
    "Mid (116-119)",
    "Production (120-359)",
    "Monitoring (360+)",
]

CANDIDATE_CORR_COLS = [
    "priority", "scheduling_class", "req_cpu", "req_ram",
    "avg_cpu_usage", "avg_mem_usage", "cpu_slack", "mem_slack",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Use the same column names as the solvers (req_cpu / req_ram)."""
    df = df.rename(columns={"req_cpus": "req_cpu", "req_mem": "req_ram"})
    if "priority" in df.columns:
        df["priority_tier"] = pd.cut(
            df["priority"], bins=TIER_BINS, labels=TIER_LABELS
        ).astype(str)
    return df


@st.cache_data(show_spinner="Loading workload dataset...")
def load_workload(path: str = DATA_PATH) -> pd.DataFrame:
    return _standardize(pd.read_parquet(path))


@st.cache_data(show_spinner="Loading instance events...")
def load_events(path: str) -> pd.DataFrame | None:
    """Reads only the columns needed; returns None if no usable event column."""
    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    wanted = ["type", "event_type", "priority", "scheduling_class"]
    cols = [c for c in wanted if c in available]

    event_col = next((c for c in ("type", "event_type") if c in cols), None)
    if event_col is None:
        return None

    ev = pd.read_parquet(path, columns=cols)
    ev = ev.rename(columns={event_col: "event"})

    if pd.api.types.is_numeric_dtype(ev["event"]):
        ev["event"] = ev["event"].map(EVENT_NAMES).fillna("OTHER")
    else:
        ev["event"] = ev["event"].astype(str).str.upper()

    if "priority" in ev.columns:
        ev["priority_tier"] = pd.cut(
            ev["priority"], bins=TIER_BINS, labels=TIER_LABELS
        ).astype(str)
    return ev


def find_events() -> pd.DataFrame | None:
    for p in EVENTS_PATHS:
        if os.path.exists(p):
            return load_events(p)
    return None


# --------------------------------------------------------------------------- #
# Pure analysis helpers (no Streamlit calls, so they are easy to test)
# --------------------------------------------------------------------------- #
def summary_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    stats = df[cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats["skew"] = df[cols].skew()
    stats["missing_%"] = df[cols].isna().mean() * 100
    return stats.round(4)


def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str) -> pd.DataFrame:
    return df[cols].corr(method=method)


def terminal_event_rates(events: pd.DataFrame) -> pd.DataFrame:
    """Share of each terminal outcome (FINISH/EVICT/FAIL/KILL/LOST), in %."""
    counts = events["event"].value_counts().reindex(TERMINAL_EVENTS).fillna(0)
    total = counts.sum()
    out = pd.DataFrame({"count": counts.astype(int)})
    out["rate_%"] = (counts / total * 100).round(2) if total else 0.0
    return out.reset_index(names="outcome")


def terminal_rates_by_group(events: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-group outcome mix in % (rows sum to 100)."""
    sub = events[events["event"].isin(TERMINAL_EVENTS)]
    ct = pd.crosstab(sub[group_col], sub["event"]).reindex(
        columns=TERMINAL_EVENTS, fill_value=0
    )
    pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return pct.round(2)


# --------------------------------------------------------------------------- #
# Streamlit rendering
# --------------------------------------------------------------------------- #
def _section_overview(df: pd.DataFrame) -> list[str]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasks analysed", f"{len(df):,}")
    if "collection_id" in df.columns:
        c2.metric("Unique collections", f"{df['collection_id'].nunique():,}")
    c3.metric("Median CPU request", f"{df['req_cpu'].median():.4f}")
    c4.metric("Median RAM request", f"{df['req_ram'].median():.4f}")

    cols = [c for c in CANDIDATE_CORR_COLS if c in df.columns]
    st.subheader("Summary statistics")
    st.dataframe(summary_table(df, cols), use_container_width=True)

    notes = []
    if {"avg_cpu_usage", "req_cpu"} <= set(df.columns):
        over = (df["avg_cpu_usage"] < df["req_cpu"]).mean() * 100
        notes.append(f"{over:.1f}% of tasks use less CPU on average than they requested.")
    if {"avg_mem_usage", "req_ram"} <= set(df.columns):
        over = (df["avg_mem_usage"] < df["req_ram"]).mean() * 100
        notes.append(f"{over:.1f}% of tasks use less memory on average than they requested.")
    skewed = [c for c in ("req_cpu", "req_ram") if df[c].skew() > 2]
    if skewed:
        notes.append(
            f"{', '.join(skewed)} are heavily right-skewed. Use the log-scale option "
            "below and prefer Spearman correlation."
        )
    for n in notes:
        st.caption(n)
    return cols


def _section_distributions(df: pd.DataFrame) -> None:
    st.subheader("Distributions")
    c1, c2 = st.columns(2)
    nbins = c1.slider("Histogram bins", 10, 100, 40)
    log_y = c2.checkbox("Log-scale y-axis", value=True)

    left, right = st.columns(2)
    for col, title, target in (
        ("req_cpu", "CPU request (normalised)", left),
        ("req_ram", "RAM request (normalised)", right),
    ):
        fig = px.histogram(df, x=col, nbins=nbins, log_y=log_y, title=title)
        fig.update_layout(bargap=0.05, yaxis_title="Tasks")
        target.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    if "priority" in df.columns:
        prio = df["priority"].value_counts().sort_index().reset_index()
        prio.columns = ["priority", "tasks"]
        fig = px.bar(prio, x="priority", y="tasks", title="Tasks per priority value")
        fig.update_xaxes(type="category")
        left.plotly_chart(fig, use_container_width=True)

        tier = (
            df["priority_tier"].value_counts().reindex(TIER_LABELS).dropna().reset_index()
        )
        tier.columns = ["tier", "tasks"]
        right.plotly_chart(
            px.bar(tier, x="tier", y="tasks", title="Tasks per priority tier"),
            use_container_width=True,
        )

    if {"priority_tier", "req_cpu"} <= set(df.columns):
        fig = px.box(
            df, x="priority_tier", y="req_cpu", log_y=log_y,
            category_orders={"priority_tier": TIER_LABELS},
            title="CPU request by priority tier",
        )
        st.plotly_chart(fig, use_container_width=True)


def _section_correlation(df: pd.DataFrame, cols: list[str]) -> None:
    st.subheader("Feature relationships")
    method = st.radio(
        "Correlation method", ["spearman", "pearson"], horizontal=True,
        help="Spearman is rank-based and more reliable for skewed resource data.",
    )
    corr = correlation_matrix(df, cols, method)
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
        color_continuous_scale="RdBu_r", title=f"{method.title()} correlation",
    )
    st.plotly_chart(fig, use_container_width=True)


def _section_reliability() -> None:
    st.subheader("Reliability: failure and eviction rates")
    events = find_events()
    if events is None:
        st.info(
            "No instance event data found. Run `python scripts/prepare_clusterdata.py` "
            "to create `data/instance_events.parquet`, or update `EVENTS_PATHS` in "
            "`src/analysis/eda.py`. The file needs a `type` (or `event_type`) column."
        )
        return

    rates = terminal_event_rates(events)
    if rates["count"].sum() == 0:
        st.warning(
            "The event file has no terminal events (FINISH / EVICT / FAIL / KILL / LOST). "
            "Check that `EVENT_NAMES` matches your trace's event-type encoding."
        )
        return

    r = rates.set_index("outcome")["rate_%"]
    cols = st.columns(len(TERMINAL_EVENTS))
    for col, name in zip(cols, TERMINAL_EVENTS):
        col.metric(name.title(), f"{r[name]:.2f}%")
    st.caption(f"Based on {int(rates['count'].sum()):,} terminal events.")

    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(rates, x="outcome", y="rate_%", title="Terminal outcome mix (%)"),
        use_container_width=True,
    )

    if "priority_tier" in events.columns:
        by_tier = terminal_rates_by_group(events, "priority_tier")
        by_tier = by_tier.reindex([t for t in TIER_LABELS if t in by_tier.index])
        long = by_tier.reset_index().melt(
            id_vars="priority_tier", var_name="outcome", value_name="rate_%"
        )
        fig = px.bar(
            long, x="priority_tier", y="rate_%", color="outcome",
            category_orders={"priority_tier": TIER_LABELS, "outcome": TERMINAL_EVENTS},
            title="Outcome mix by priority tier (%)",
        )
        right.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_tier, use_container_width=True)


def render_eda_tab(data_path: str | None = None) -> None:
    """Call inside `with tab1:` in app/dashboard.py."""
    st.header("EDA & Workload Insights")

    path = data_path or DATA_PATH
    try:
        df = load_workload(path)
    except FileNotFoundError:
        st.error(
            f"Dataset not found at `{path}`. Run the dashboard from the project "
            "root, and make sure `miscs/build_scheduler.py` has been executed."
        )
        return

    if len(df) > 1000:
        n = st.slider(
            "Rows to analyse (random sample)", 1000, len(df),
            min(len(df), 100000), step=1000,
        )
        df = df.sample(n=n, random_state=42) if n < len(df) else df

    cols = _section_overview(df)
    st.divider()
    _section_distributions(df)
    st.divider()
    _section_correlation(df, cols)
    st.divider()
    _section_reliability()
