# Big M Method — Linear Programming Solver & Visualizer
### Assignment 9 | EM-4 (BSC07) | INFT Engineering

---

## 📌 Project Overview

This project implements the **Big M Method**, a technique in **Linear Programming** used to find an initial Basic Feasible Solution (BFS) when the standard form of an LP problem contains **≥ (greater than or equal to)** or **= (equality)** constraints. By introducing artificial variables with a heavy penalty (M), the algorithm forces them out of the basis to find the optimal solution.

### ✨ Key Features
- **Interactive CLI Menu:** A fully colored, interactive terminal interface powered by `rich`.
- **Custom Problem Solver:** Enter your own custom constraints and objective functions interactively!
- **Beautiful Tableau Printing:** Step-by-step Simplex tableaus printed clearly with color-coded Z-rows and pivot markers.
- **2D Visualizations:** Automatically plots the feasible regions, constraint lines, and objective functions for 2-variable problems using `matplotlib`.
- **Poster-Ready Artifacts:** Automatically saves high-resolution `.png` graphs to your folder for use in your assignment presentations or posters!
- **Educational Mode:** Built-in explanation option to learn the steps of the Big M Method directly from the terminal.

---

## ⚙️ Requirements & Installation

To run this beautifully formatted script, you need to install standard mathematical and UI libraries.

```bash
pip install numpy matplotlib rich
```

*Python version: **3.7 or higher** required.*

---

## 🚀 How to Run

1. Open your terminal in the project directory.
2. Run the main python file:

```bash
python main.py
```

3. You will be greeted with the Main Menu. Choose an option:
    - `[1]` **Run pre-coded Examples:** Automatically solves 3 pre-configured examples, displays the Simplex iterations, and plots/saves the 2D graphs.
    - `[2]` **Interactive Mode:** Step-by-step prompts to input your exact variables, constraint values, and equations.
    - `[3]` **Learn:** Detailed explanation of the Big M algorithm directly in the terminal.
    - `[4]` **Exit**

---

## 🧠 Methodology

### Step 1 — Standard Form Conversion
- **≤ constraint** → Add a **slack variable** (s ≥ 0)
- **≥ constraint** → Subtract a **surplus variable** + Add an **artificial variable** (a ≥ 0)
- **= constraint** → Add an **artificial variable** only

### Step 2 — Modified Objective Function
```
Maximise: Z = c₁x₁ + c₂x₂ + ... - M·a₁ - M·a₂ - ...
```
*(For minimisation, the penalty is **+M**.)*

### Step 3 — Initial Tableau Setup
- Artificial variables form the **initial basis**.
- Z-row is updated to eliminate basic variable coefficients.

### Step 4 — Simplex Iterations
Repeat until optimal:
1. **Entering variable**: column with most negative Z-row value.
2. **Leaving variable**: minimum ratio test (θ = RHS / pivot column).
3. **Row operations**: normalise pivot row, eliminate pivot column in all other rows.

### Step 5 — Optimality & Feasibility Check
- **Optimal**: all Z-row coefficients ≥ 0.
- **Infeasible**: artificial variable remains in basis with value > 0.
- **Unbounded**: no valid pivot row exists (all ratios = ∞).

---

## 📁 File Structure
```
EM4_Project/
│
├── main.py             ← Main solver, UI, and graphing implementation
├── README.md           ← Project documentation (This file)
└── *.png               ← Automatically generated graphs (saved during runtime)
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

---

## ⚠️ Limitations

- The solver uses `M = 1,000,000` — for extremely ill-conditioned problems, numerical rounding errors may occur.
- All RHS (Right Hand Side) values must be non-negative (the code attempts to auto-fix this by multiplying raw inputs by -1).
- Visualizations (`matplotlib` graphs) are strictly limited to **2-variable** problems due to 2D spatial restrictions.

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
| **Aashutosh Mahajan** | 24101b0063 | Team Member |
| **Atharv Raut** | 24101B0052 | Team Member |
| **Aditya Patra** | 24101B0072 | Team Member |
| **Ronak Boddu** | 24101B0044 | Team Member |

*Assignment 9 | EM-4 (BSC07) | INFT Department*