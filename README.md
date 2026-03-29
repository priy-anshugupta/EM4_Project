# Big M Method — Linear Programming Solver & Visualizer
### Assignment 9 | EM-4 (BSC07) | INFT Engineering

---

## 📌 Project Overview

This project implements the **Big M Method**, a technique in **Linear Programming** used to find an initial Basic Feasible Solution (BFS) when the standard form of an LP problem contains **≥ (greater than or equal to)** or **= (equality)** constraints. By introducing artificial variables with a heavy penalty (M), the algorithm forces them out of the basis to find the optimal solution.

The project ships with **two interfaces**:
1. **Streamlit Web App** (`app.py`) — A modern, dark-themed web UI with interactive Plotly charts, sensitivity analysis, duality engine, and more.
2. **CLI Mode** (`main.py`) — A terminal-based solver with Rich-formatted tableaux and matplotlib graphs.

---

### ✨ Key Features

| Feature | Description |
|:---|:---|
| **Interactive Solver** | Input any LP problem (MAX / MIN) with mixed ≤, ≥, = constraints |
| **Step-by-Step Tableaux** | Color-coded simplex iterations with pivot element highlighting |
| **2D Feasible Region** | Interactive Plotly chart showing constraints, feasible region, corner points & optimal solution |
| **3D Objective Surface** | 3D surface plot of the objective function over the feasible region |
| **Simplex Path Animation** | Visualize the vertex-to-vertex path the simplex algorithm takes |
| **Sensitivity Analysis** | Shadow prices, binding constraints, allowable RHS & objective coefficient ranges |
| **Parametric What-If** | Vary a coefficient or RHS value and watch the optimal Z change live |
| **Duality Engine** | Auto-construct the Dual LP, verify Strong Duality Theorem & Complementary Slackness |
| **Educational Learn Tab** | Big M algorithm walkthrough, Big M vs Two-Phase comparison, key terms glossary |
| **Pre-Built Examples** | 4 one-click examples (MAX & MIN) with instant solve and visualization |
| **CLI Mode** | Terminal solver with Rich tables and matplotlib graphs (original `main.py`) |

---

## ⚙️ Requirements & Installation

**Python 3.7+** is required.

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `numpy`, `plotly`, `scipy`, `streamlit-lottie`, `requests`, `rich`, `matplotlib`

---

## 🚀 How to Run

### Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501) with 7 tabs:

| Tab | Purpose |
|:---|:---|
| 🏠 **Home** | Overview, feature cards, algorithm flowchart |
| 🧮 **Solver** | Input & solve any LP problem, view tableau iterations |
| 📊 **Visualization** | Feasible region, 3D surface, simplex path charts |
| 📈 **Sensitivity** | Shadow prices, allowable ranges, parametric what-if |
| 🔄 **Duality** | Primal vs Dual side-by-side, strong duality verification |
| 📚 **Learn** | Big M algorithm steps, comparison table, glossary |
| 🏆 **Examples** | 4 pre-built problems — solve with one click |

### CLI Mode (Original)

```bash
pip install numpy matplotlib rich
python main.py
```

---

## 📖 Documentation: Code, Methodology, & Example Usage

### 1️⃣ Code Explanation & Architecture

The project is structured into modular components, decoupling the mathematical engine from the user interfaces:

- **`solver.py` (Core Engine)**: Contains the `big_m_method()` function, which acts as the core Simplex solver. It handles converting inequalities to standard form, initializing the Big M penalty (`1e6`), performing pivot operations, executing the minimum ratio test, and returning a rich dictionary (iteration history, final tableau, shadow prices). It also hosts the `build_dual()` and `sensitivity_analysis()` functions.
- **`app.py` (Streamlit Web UI)**: Provides a modern web application with 7 interactive tabs. It maps user inputs directly into the `solver.py` engine and processes the output for visualization.
- **`utils.py` (Visualizations & Formatting)**: Stores modular helper functions for generating interactive Plotly charts (2D feasible region, 3D objective surfaces, simplex traversal paths) and parsing math formula strings into LaTeX notation.
- **`main.py` (CLI interface)**: A traditional command-line interface utilizing `rich` for styled terminal output. It takes user prompts, solves the LP via the core algorithm, builds matplotlib charts, and prints formatted step-by-step tableaux.

### 2️⃣ Methodology

The **Big M Method** algorithm is mathematically implemented as follows:
- **Step 1 — Standard Form Conversion**:
  - **≤ constraint** → Add a **slack variable** ($s \geq 0$)
  - **≥ constraint** → Subtract a **surplus variable** + Add an **artificial variable** ($a \geq 0$)
  - **= constraint** → Add an **artificial variable** only
- **Step 2 — Modified Objective Function**:
  - `Maximise: Z = c₁x₁ + c₂x₂ + ... - M·a₁ - M·a₂ - ...` *(Minimisation uses +M)*
- **Step 3 — Initial Tableau & Zero-out**: Artificial variables form the initial basis. The Z-row is updated to eliminate basic variable coefficients.
- **Step 4 — Simplex Iterations**: Repeat until optimal:
  1. **Entering variable**: Column with the most negative Z-row value.
  2. **Leaving variable**: Minimum positive ratio test ($\theta = \text{RHS} / \text{pivot column}$).
  3. **Row operations**: Normalise pivot row, eliminate pivot column in all other rows.
- **Step 5 — Optimality**: Reached when all Z-row coefficients $\geq 0$.

### 3️⃣ Example Usage

**Scenario:** We want to solve a Minimization LPP using the CLI.
**Problem:**
MinZ = 4x₁ + x₂
Subject to:
3x₁ + x₂ = 3
4x₁ + 3x₂ ≥ 6
x₁ + 2x₂ ≤ 4
x₁, x₂ ≥ 0

**CLI Walkthrough Example:**
1. Start the CLI using `python main.py`.
2. **Enter Problem Type:** Type `min` and hit enter.
3. **Number of Variables:** Enter `2`.
4. **Number of Constraints:** Enter `3`.
5. **Objective Coefficients:** Enter `4` for $x_1$ and `1` for $x_2$.
6. **Constraint 1:** Enter `3`, `1`, select `=`, RHS = `3`.
7. **Constraint 2:** Enter `4`, `3`, select `>=`, RHS = `6`.
8. **Constraint 3:** Enter `1`, `2`, select `<=`, RHS = `4`.

The solver will execute and output the step-by-step table highlighting the pivot cells until the final answer ($Z = 3.4$ at $x_1=0.6, x_2=1.2$) surfaces.

---

## 📁 File Structure
```
EM4_Project/
│
├── app.py              ← Streamlit web application (7-tab UI)
├── solver.py           ← Core Big M solver, sensitivity, duality & parametric engine
├── utils.py            ← Plotly visualizations, LaTeX formatting, chart helpers
├── main.py             ← Original CLI solver with Rich tables & matplotlib
├── requirements.txt    ← Python dependencies
├── README.md           ← Project documentation (this file)
└── *.png               ← Auto-generated graphs (saved during CLI runtime)
```

---

## 🔑 Key Concepts

| Term                | Meaning                                              |
|---------------------|------------------------------------------------------|
| Big M               | A very large positive number (penalty constant).     |
| Artificial Variable | Dummy variable to create an initial feasible basis.  |
| Slack Variable      | Added to ≤ constraint to convert to equality.        |
| Surplus Variable    | Subtracted from ≥ constraint to convert to equality. |
| Pivot               | Row operation to bring a variable into the basis.    |
| BFS                 | Basic Feasible Solution — a valid starting point.    |
| Shadow Price        | Rate of change of Z per unit change in a constraint's RHS. |
| Dual Problem        | A companion LP derived from the primal; provides bounds and insights. |
| Strong Duality      | Primal optimal Z* equals Dual optimal Z*.            |

---

## ⚠️ Limitations

- The solver uses `M = 1,000,000` — for extremely ill-conditioned problems, numerical rounding errors may occur.
- All RHS (Right Hand Side) values must be non-negative (the code attempts to auto-fix this by multiplying raw inputs by -1).
- 2D/3D visualizations are limited to **2-variable** problems.
- Parametric and sensitivity analysis use a perturbation-based approach (not exact inverse basis).

---

## 📚 References

1. Taha, H. A. — *Operations Research: An Introduction*, 10th Ed.
2. Hillier & Lieberman — *Introduction to Operations Research*
3. NPTEL Lectures — Linear Programming and Extensions

---

### 🎓 Prepared By: Group 2
**Topic:** Big-M Method (LPP)

| Name | Roll Number | Role |
| :--- | :--- | :--- |
| **Priyanshu Gupta** | 24101B0037 | Team Leader (TL) |
| **Aashutosh Mahajan** | 24101B0063 | Team Member |
| **Atharv Raut** | 24101B0052 | Team Member |
| **Aditya Patra** | 24101B0072 | Team Member |
| **Ronak Boddu** | 24101B0044 | Team Member |
| **Prem Ranmale** | 24101B0018 | Team Member |

*Assignment 9 | EM-4 (BSC07) | INFT Department*