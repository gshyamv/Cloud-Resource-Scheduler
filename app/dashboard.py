import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

import streamlit as st
import pandas as pd
import plotly.express as px
from src.optimization.integer_model import IntegerProgrammingModel
from src.optimization.goal_model import GoalProgrammingModel
from src.evaluation.metrics import Evaluator
from src.analysis.eda import render_eda_tab
from src.data_pipeline.batch_builder import build_task_batch, build_cluster, problem_report
from src.scheduling.timeline import assign_lanes

DATA_PATH = os.path.join(ROOT, "data", "processed", "final_scheduler_dataset100.parquet")

st.set_page_config(page_title="Cloud GPU Resource Scheduler", layout="wide")
st.title("Cloud GPU Resource Scheduling System")

@st.cache_data
def load_raw():
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def make_problem(n_tasks, mode, num_nodes, gpu_nodes, target_load,
                 heterogeneity, gpu_fraction, n_deps, seed):
    tasks = build_task_batch(load_raw(), n_tasks=n_tasks, mode=mode,
                             gpu_fraction=gpu_fraction, n_dependencies=n_deps, seed=seed)
    nodes = build_cluster(tasks, num_nodes=num_nodes, gpu_nodes=gpu_nodes,
                          target_load=target_load, heterogeneity=heterogeneity)
    return tasks, nodes

# --- Sidebar ---
st.sidebar.header("Workload configuration")
sampling = st.sidebar.radio("Task sampling", ["Stratified (diverse)", "First N rows (original)"])
n_tasks = st.sidebar.slider("Number of tasks", 20, 200, 60, step=10)
num_nodes = st.sidebar.slider("Number of nodes", 2, 10, 5)
gpu_nodes = st.sidebar.slider("Nodes with GPU", 0, num_nodes, min(2, num_nodes))
load_pct = st.sidebar.slider("Demand as % of total capacity", 50, 200, 100, step=5)

st.sidebar.divider()
st.sidebar.header("Goal Programming Weights")
w_service = st.sidebar.slider("Task Admission Weight", 1.0, 10.0, 3.0, help="Importance of scheduling all tasks")
w_over = st.sidebar.slider("Over-utilization Penalty", 0.0, 5.0, 1.0, help="Penalty for exceeding target load")
w_idle = st.sidebar.slider("Under-utilization Penalty", 0.0, 5.0, 1.0, help="Penalty for idle capacity")

with st.sidebar.expander("Advanced"):
    heterogeneity = st.slider("Node size spread", 0.0, 0.9, 0.3, step=0.1)
    gpu_fraction = st.slider("Share of tasks needing a GPU", 0.0, 0.5, 0.2, step=0.05)
    n_deps = st.slider("Dependency pairs", 0, 10, 3)
    seed = st.number_input("Random seed", value=42, step=1)
    disable_time_limit = st.checkbox("Disable solver time limit", value=False)
    time_limit_val = st.slider("Solver time limit (seconds)", 5, 120, 20, disabled=disable_time_limit)
    time_limit = None if disable_time_limit else time_limit_val

try:
    real_tasks, real_nodes = make_problem(
        n_tasks, "stratified" if sampling.startswith("Strat") else "head",
        num_nodes, gpu_nodes, load_pct / 100, heterogeneity, gpu_fraction, n_deps, int(seed),
    )
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. Run `miscs/build_scheduler.py` first.")
    st.stop()

report = problem_report(real_tasks, real_nodes)

tab1, tab2, tab3, tab4 = st.tabs(
    ["EDA & Insights", "Run Optimization", "Evaluation & Timeline", "Robustness & Trade-off"]
)

# --- Tab 1: EDA ---
with tab1:
    render_eda_tab(DATA_PATH)
    with st.expander("Scheduled batch (the tasks sent to the solver)"):
        st.dataframe(real_tasks, use_container_width=True)
        st.dataframe(real_nodes, use_container_width=True)

# --- Tab 2: Optimization ---
with tab2:
    st.header("Optimization Engine")
    strategy = st.selectbox("Select Optimization Strategy",
                            ["Integer Programming (Maximize Util)", "Goal Programming (Target 90% Util)"])

    if st.button("Run Solver"):
        with st.spinner(f"Executing {strategy}..."):
            if "Integer" in strategy:
                solver = IntegerProgrammingModel(real_tasks, real_nodes)
            else:
                solver = GoalProgrammingModel(real_tasks, real_nodes,
                                              w_service=w_service, w_over=w_over, w_idle=w_idle)

            solver.build_model()
            allocations, obj_val = solver.solve(time_limit=time_limit)

            st.session_state['allocations'] = allocations
            st.session_state['tasks'] = real_tasks
            st.session_state['nodes'] = real_nodes
            st.session_state['run_info'] = f"{strategy} - Scheduled {len(allocations)} / {len(real_tasks)} tasks."
            st.success("Optimization complete.")

# --- Tab 3: Evaluation ---
with tab3:
    st.header("Comparative Decision & Schedule Timeline")
    if 'allocations' in st.session_state and not st.session_state['allocations'].empty:
        st.caption(f"Showing last run: {st.session_state['run_info']}")

        evaluator = Evaluator(st.session_state['allocations'], st.session_state['tasks'], st.session_state['nodes'])
        efficiency = evaluator.calculate_resource_efficiency()
        quality = evaluator.calculate_scheduling_quality()  # Populates evaluator.timeline

        # 1. Base Efficiency Metrics
        st.subheader("Resource Efficiency (Snapshot at t=0)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CPU Utilization", f"{efficiency['cpu_utilization']}%")
        c2.metric("Memory Utilization", f"{efficiency['ram_utilization']}%")
        c3.metric("Resource Fragmentation", f"{efficiency['fragmentation']}%")
        c4.metric("Tasks Admitted Initially", f"{quality['admitted_pct']}%")

        # 2. Timeline Quality Metrics
        st.subheader("Scheduling Quality (Simulated Timeline)")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Makespan (Hours)", f"{quality['makespan_hours']}h")
        t2.metric("Avg Wait Time (Mins)", f"{quality['avg_wait_time_mins']}m")
        t3.metric("High-Priority Wait (Mins)", f"{quality['hp_avg_wait_mins']}m")
        t4.metric("Deadline Miss Rate", f"{quality['deadline_miss_pct']}%")

        # 3. Gantt Chart
        st.subheader("Node Execution Schedule")
        tl_ran = evaluator.timeline[evaluator.timeline['status'] == 'ran'].copy()

        if not tl_ran.empty:
            # Give concurrent tasks on a node their own lane so bars don't overlap
            tl_ran = assign_lanes(tl_ran)          # adds "lane" and "row" columns

            # Convert float seconds to dummy datetimes for Plotly
            base_time = pd.Timestamp("2024-01-01 00:00:00")
            tl_ran["Start"] = base_time + pd.to_timedelta(tl_ran["start_s"], unit="s")
            tl_ran["End"] = base_time + pd.to_timedelta(tl_ran["end_s"], unit="s")

            # Order rows by node, then numeric lane (plain string sort puts "#10" before "#2")
            row_order = (tl_ran[["node_id", "lane", "row"]]
                         .drop_duplicates()
                         .sort_values(["node_id", "lane"])["row"].tolist())

            fig_gantt = px.timeline(
                tl_ran, x_start="Start", x_end="End", y="row", color="node_id",
                category_orders={"row": row_order},
                hover_data=["task_id", "priority", "duration_s", "wait_s"],
            )
            fig_gantt.update_yaxes(autorange="reversed", title="Node / lane")
            fig_gantt.update_layout(
                xaxis_title="Time (elapsed since t=0)",
                height=max(350, 28 * len(row_order) + 120),   # grow with number of lanes
            )
            st.plotly_chart(fig_gantt, use_container_width=True)
        else:
            st.warning("No tasks were successfully scheduled in the timeline.")
    else:
        st.info("Run optimization to view metrics.")

# --- Tab 4: Robustness & trade-off ---
with tab4:
    st.header("Trade-off Analysis (Integer vs Goal Programming)")

    simulate_failure = st.checkbox("Simulate a node failure (capacity drop)")
    failed_node = st.selectbox("Node to fail", real_nodes.index.tolist(), index=len(real_nodes) - 1, disabled=not simulate_failure)

    test_nodes = real_nodes.copy()
    if simulate_failure:
        test_nodes = test_nodes.drop(failed_node)
        st.warning(f"{failed_node} removed. Re-running comparison shows how solvers cope with less capacity.")

    if st.button("Run Side-by-Side Comparison"):
        with st.spinner("Evaluating both models..."):
            # Model A: IP
            solver_a = IntegerProgrammingModel(real_tasks, test_nodes)
            solver_a.build_model()
            alloc_a, _ = solver_a.solve(time_limit=time_limit)
            eval_a = Evaluator(alloc_a, real_tasks, test_nodes)
            eff_a = eval_a.calculate_resource_efficiency()
            qual_a = eval_a.calculate_scheduling_quality()

            # Model B: GP
            solver_b = GoalProgrammingModel(real_tasks, test_nodes, w_service=w_service, w_over=w_over, w_idle=w_idle)
            solver_b.build_model()
            alloc_b, _ = solver_b.solve(time_limit=time_limit)
            eval_b = Evaluator(alloc_b, real_tasks, test_nodes)
            eff_b = eval_b.calculate_resource_efficiency()
            qual_b = eval_b.calculate_scheduling_quality()

            comparison_df = pd.DataFrame({
                "Metric": [
                    "Tasks Admitted at t=0",
                    "CPU Utilization",
                    "Resource Fragmentation",
                    "Makespan (Hours)",
                    "Avg Wait Time (Mins)",
                    "High-Priority Wait (Mins)",
                    "Deadline Misses (%)"
                ],
                "Integer Programming (Strict Packing)": [
                    f"{qual_a['admitted_pct']}%",
                    f"{eff_a['cpu_utilization']}%",
                    f"{eff_a['fragmentation']}%",
                    f"{qual_a['makespan_hours']}h",
                    f"{qual_a['avg_wait_time_mins']}m",
                    f"{qual_a['hp_avg_wait_mins']}m",
                    f"{qual_a['deadline_miss_pct']}%"
                ],
                "Goal Programming (Soft Objectives)": [
                    f"{qual_b['admitted_pct']}%",
                    f"{eff_b['cpu_utilization']}%",
                    f"{eff_b['fragmentation']}%",
                    f"{qual_b['makespan_hours']}h",
                    f"{qual_b['avg_wait_time_mins']}m",
                    f"{qual_b['hp_avg_wait_mins']}m",
                    f"{qual_b['deadline_miss_pct']}%"
                ]
            })
            st.table(comparison_df)

            # Dynamic Insight Generation
            if float(qual_b['hp_avg_wait_mins']) < float(qual_a['hp_avg_wait_mins']):
                st.success("Insight: Goal Programming successfully prioritized critical workloads, resulting in lower wait times for High-Priority tasks.")
            elif float(eff_a['cpu_utilization']) > float(eff_b['cpu_utilization']):
                st.success("Insight: Integer Programming achieved tighter bin-packing, yielding higher total CPU utilization.")