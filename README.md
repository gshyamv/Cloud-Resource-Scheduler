# Cloud CPU/GPU Resource Scheduling System

An Operations Research-based scheduling system for allocating workloads across heterogeneous CPU/GPU cloud resources.

<br>

<p align="center">
  <img width="950" alt="Cloud CPU/GPU Resource Scheduling Dashboard" src="https://github.com/user-attachments/assets/416f88e8-650a-4a74-ab8a-ebc5135dadcc" />
</p>

<br>

The project uses workload characteristics derived from **Google Cluster Data** and applies two optimization approaches — **Integer Programming (IP)** and **Goal Programming (GP)** — to allocate tasks to available resources while considering CPU, memory, GPU availability, priorities, and dependencies.

---

## Features

* 📊 Exploratory Data Analysis of cluster workloads
* ⚙️ Integer Programming for task-to-node allocation
* 🎯 Goal Programming with soft optimization goals
* 🖥️ Interactive Streamlit dashboard
* 🧠 CPU/RAM-aware scheduling
* 🎮 GPU-aware task allocation
* 🔗 Task dependency handling
* ⏱️ Scheduling simulation with waiting time and deadlines
* 📈 IP vs GP performance comparison
* 💥 Resource-failure / robustness testing
* 📋 Gantt-style task execution timeline

---

## Project Architecture

```text
Google Cluster Data
        │
        ▼
Data Loading & Preprocessing
        │
        ▼
Scheduler Dataset
        │
        ├──────────────► EDA
        │
        ▼
Workload / Cluster Builder
        │
        ├──────────────► Integer Programming
        │
        └──────────────► Goal Programming
                         │
                         ▼
                  Task Allocation
                         │
                         ▼
                 Scheduling Simulation
                         │
                         ▼
              Evaluation & Visualization
```

---

## Dataset

The project is based on **Google Cluster Data**, using information from:

* `machine_events`
* `task_events`
* `task_usage`
* `task_constraints`

The preprocessing pipeline extracts workload characteristics such as:

* CPU requests
* Memory requests
* CPU/memory usage
* Priority
* Scheduling class
* Task and job information
* Failure/eviction information

Some scheduling attributes are generated for controlled experiments:

* Task duration
* Deadline
* Dependencies
* GPU requirement
* Heterogeneous node capacities

Therefore, this project should be understood as a **controlled scheduling simulation grounded in real-world workload characteristics**, rather than a reconstruction of Google's actual production scheduler.

---

## Project Structure

```text
cloud-gpu-scheduler/
│
├── app/
│   └── dashboard.py              # Streamlit dashboard
│
├── data/
│   ├── raw/                      # Raw cluster data
│   └── processed/                # Processed scheduler dataset
│
├── src/
│   ├── analysis/
│   │   └── eda.py                # Workload analysis
│   │
│   ├── data_pipeline/
│   │   ├── loader.py             # Dataset loading
│   │   ├── preprocessor.py       # Data preprocessing
│   │   └── batch_builder.py      # Scheduling workload generation
│   │
│   ├── evaluation/
│   │   └── metrics.py            # Scheduling metrics
│   │
│   ├── optimization/
│   │   ├── integer_model.py      # Integer Programming model
│   │   └── goal_model.py         # Goal Programming model
│   │
│   └── scheduling/
│       └── timeline.py           # Scheduling simulation
│
├── scripts/
│   └── prepare_clusterdata.py    # Data preparation
│
├── miscs/
│   ├── build_scheduler.py
│   └── seeparaquet.py
│
├── main.py
├── fetch100k.py
└── README.md
```

---

# Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd cloud-gpu-scheduler
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pulp pandas numpy streamlit plotly pyarrow
```

If the project contains a `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

---

# Data Preparation

The project expects the processed scheduler dataset inside:

```text
data/processed/
```

The main dashboard looks for:

```text
final_scheduler_dataset100.parquet
```

and falls back to:

```text
final_scheduler_dataset.parquet
```

The data preparation pipeline is handled primarily by:

```text
src/data_pipeline/loader.py
src/data_pipeline/preprocessor.py
src/data_pipeline/batch_builder.py
scripts/prepare_clusterdata.py
```

The general flow is:

```text
Raw Google Cluster Data
        ↓
Load required tables
        ↓
Preprocess / clean
        ↓
Combine task information
        ↓
Generate scheduling attributes
        ↓
Save Parquet dataset
```

---

# Running the Dashboard

From the project root:

```bash
python -m streamlit run app/dashboard.py
```

Streamlit will open the dashboard in your browser.

The dashboard allows you to configure:

* Number of tasks
* Number of nodes
* Number of GPU nodes
* Workload load
* Resource heterogeneity
* GPU workload fraction
* Task dependencies

You can then inspect the workload, run optimization models, and evaluate the resulting schedules.

---

# Dashboard Workflow

### 1. Configure Workload

Choose the scheduling scenario:

```text
Tasks
Nodes
GPU Nodes
Demand / Load
Heterogeneity
GPU Fraction
Dependencies
```

### 2. Explore the Data

The EDA section provides:

* Descriptive statistics
* Missing-value analysis
* Skewness
* Priority distribution
* CPU request distribution
* RAM request distribution
* Correlations
* Failure/eviction analysis

### 3. Run Optimization

The system can run:

* Integer Programming
* Goal Programming

The models generate task-to-node allocations subject to resource and scheduling constraints.

### 4. Evaluate

The evaluation section reports:

* CPU utilization
* RAM utilization
* Resource fragmentation
* Tasks admitted
* Makespan
* Average waiting time
* High-priority waiting time
* Deadline miss rate

### 5. Inspect the Timeline

The scheduling simulator converts the allocation into an execution timeline and displays the task schedule as a Gantt-style visualization.

### 6. Test Robustness

The system can reduce available node capacity to simulate resource failures and evaluate how scheduling performance changes.

---

# Optimization Models

## Integer Programming

The IP model uses binary assignment variables:

```text
x(i,j) = 1  → task i is assigned to node j
x(i,j) = 0  → otherwise
```

The model considers constraints such as:

* CPU capacity
* RAM capacity
* One-node-per-task assignment
* GPU availability
* Dependency/co-location requirements

The objective is to find a feasible resource allocation satisfying these constraints.

Implementation:

```text
src/optimization/integer_model.py
```

---

## Goal Programming

Goal Programming extends the scheduling problem by allowing multiple objectives and soft targets.

It considers goals such as:

* Serving tasks
* Achieving target resource utilization
* Penalizing deviations
* Giving higher importance to priority-sensitive workloads

Implementation:

```text
src/optimization/goal_model.py
```

---

# Scheduling Simulation

Optimization produces the resource allocation, while:

```text
src/scheduling/timeline.py
```

simulates how the allocated tasks execute.

The scheduler handles:

* Task dependencies
* Task priorities
* Deadlines
* Resource availability
* Waiting time
* Task completion
* Backfilling

Since tasks are initially released at time zero, a task's waiting time is represented by its start time.

---

# Evaluation

The system evaluates both resource efficiency and scheduling performance.

### Resource Metrics

| Metric          | Description                                               |
| --------------- | --------------------------------------------------------- |
| CPU Utilization | Percentage of available CPU capacity used                 |
| RAM Utilization | Percentage of available RAM capacity used                 |
| Fragmentation   | Unused capacity that cannot effectively accommodate tasks |
| Tasks Admitted  | Number of successfully allocated tasks                    |

### Scheduling Metrics

| Metric             | Description                                         |
| ------------------ | --------------------------------------------------- |
| Makespan           | Time required to complete the workload              |
| Average Wait       | Average task waiting time                           |
| High-Priority Wait | Waiting time experienced by high-priority tasks     |
| Deadline Miss Rate | Percentage of tasks finishing after their deadlines |

---

# Robustness Testing

The system can simulate reduced resource availability by lowering node capacity.

This allows comparison of scheduling performance under resource pressure:

```text
Normal Capacity
      ↓
Reduced Capacity
      ↓
Re-run Scheduling
      ↓
Compare:
  • Admission
  • Utilization
  • Fragmentation
  • Waiting Time
  • Makespan
  • Deadline Performance
```

---

# Technology Stack

* **Python**
* **Pandas / NumPy** — data processing
* **PyArrow / Parquet** — dataset storage
* **PuLP / CBC** — optimization
* **Streamlit** — dashboard
* **Plotly** — visualization
