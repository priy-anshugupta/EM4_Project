"""
=============================================================
  BIG M METHOD - Linear Programming Solver
  Implemented for Assignment-9 | EM-4 (BSC07)
=============================================================
  The Big M Method solves Linear Programming Problems (LPP)
  that contain >= or = constraints by introducing artificial
  variables with a very large penalty (M) in the objective.
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import re

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, FloatPrompt, IntPrompt
    from rich import print as rprint
except ImportError:
    print("This script requires the 'rich' library for fancy outputs.")
    print("Please install it by running: pip install rich")
    exit(1)

console = Console()

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
BIG_M = 1e6          # The "Big M" penalty constant
TOLERANCE = 1e-9     # Numerical zero threshold


# ─────────────────────────────────────────────────────────────
#  HELPER: Format values to show Big M explicitly
# ─────────────────────────────────────────────────────────────
def format_m(val, M=1e6, tol=1e-5):
    m_part = round(val / M)
    real_part = val - m_part * M
    
    if abs(real_part) < tol: real_str = ""
    else: real_str = f"{real_part:.2f}".rstrip('0').rstrip('.') if real_part % 1 != 0 else f"{int(real_part)}"
        
    if abs(m_part) < tol: m_str = ""
    else:
        if abs(m_part - 1) < tol: m_str = "M"
        elif abs(m_part + 1) < tol: m_str = "-M"
        else: m_str = f"{int(m_part)}M"

    if not real_str and not m_str: return "0"
    if not real_str: return m_str
    if not m_str: return real_str
    
    if m_part > 0: return f"{real_str} + {m_str}"
    else: return f"{real_str} - {m_str[1:]}"

# ─────────────────────────────────────────────────────────────
#  HELPER: Pretty-print the Simplex Tableau using Rich
# ─────────────────────────────────────────────────────────────
def print_tableau(tableau, col_labels, row_labels, iteration, ratios=None, pivot_row=None, pivot_col=None):
    table = Table(
        title=f"[bold cyan]ITERATION {iteration}[/bold cyan]", 
        show_header=True, 
        header_style="bold magenta",
        title_justify="center"
    )
    
    table.add_column("Basis", style="bold green", width=10)
    for c in col_labels:
        table.add_column(c, justify="right")
    table.add_column("RHS", justify="right", style="bold yellow")
    
    has_ratios = ratios is not None
    if has_ratios:
        table.add_column("Ratio (RHS/Pivot)", justify="right", style="bold red")

    for i, label in enumerate(row_labels):
        style = ""
        if i == pivot_row:
            style = "on dark_red"
            
        row = [label]
        for j in range(len(col_labels)):
            cell_val = format_m(tableau[i, j])
            if j == pivot_col and i == pivot_row:
                row.append(f"[bold bright_yellow]{cell_val}[/bold bright_yellow]")
            elif j == pivot_col:
                row.append(f"[green]{cell_val}[/green]")
            else:
                row.append(cell_val)
                
        row.append(format_m(tableau[i, -1]))
        
        if has_ratios:
            r = ratios[i]
            r_str = "-" if r == np.inf or r < 0 else f"{r:.4f}"
            row.append(r_str)
            
        table.add_row(*row, style=style)

    # Z-Row
    table.add_section()
    z_row = tableau[-1]
    z_str = ["Z"]
    for j in range(len(col_labels)):
        val = format_m(z_row[j])
        if j == pivot_col:
            z_str.append(f"[green]{val}[/green]")
        else:
            z_str.append(val)
    z_str.append(format_m(z_row[-1]))
    if has_ratios:
        z_str.append("")
        
    table.add_row(*z_str, style="bold blue")

    console.print(table)
    console.print("")

# ─────────────────────────────────────────────────────────────
#  VISUALIZATION: Plot Feasible Region & Optimal Solution
# ─────────────────────────────────────────────────────────────
def plot_2d_lp(c, A, b, constraint_types, result, title="Linear Programming: Big M Method", var_names=["x1", "x2"]):
    if len(c) != 2:
        return

    # Set up plot limits
    max_x = max(10, result["solution"][0] * 1.5 if result.get("status") == "optimal" else 10)
    max_y = max(10, result["solution"][1] * 1.5 if result.get("status") == "optimal" else 10)

    for i in range(len(b)):
        if A[i][0] > 0: max_x = max(max_x, b[i] / A[i][0] * 1.2)
        if A[i][1] > 0: max_y = max(max_y, b[i] / A[i][1] * 1.2)
        
    max_x = min(max_x, 100)
    max_y = min(max_y, 100)

    x = np.linspace(0, max_x, 400)
    y = np.linspace(0, max_y, 400)
    X, Y = np.meshgrid(x, y)
    
    feasible = np.ones_like(X, dtype=bool)

    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(len(b)):
        val = A[i][0]*X + A[i][1]*Y
        if constraint_types[i] == "<=":
            feasible &= (val <= b[i] + 1e-5)
        elif constraint_types[i] == ">=":
            feasible &= (val >= b[i] - 1e-5)
        elif constraint_types[i] == "=":
            feasible &= (np.abs(val - b[i]) <= max_x*0.01)

        if A[i][1] != 0:
            y_line = (b[i] - A[i][0] * x) / A[i][1]
            plt.plot(x, y_line, label=f'C{i+1}: {A[i][0]}x₁ + {A[i][1]}x₂ {constraint_types[i]} {b[i]}', color=colors[i % len(colors)], linewidth=2)
        else:
            x_line = b[i] / A[i][0]
            plt.axvline(x=x_line, label=f'C{i+1}: {A[i][0]}x₁ {constraint_types[i]} {b[i]}', color=colors[i % len(colors)], linewidth=2)

    plt.imshow(feasible, extent=(0, max_x, 0, max_y), origin='lower', alpha=0.3, cmap='Greys')

    if result["status"] == "optimal":
        opt_x, opt_y = result["solution"][0], result["solution"][1]
        
        Z = result["optimal_value"]
        if c[1] != 0:
            y_z = (Z - c[0] * x) / c[1]
            plt.plot(x, y_z, 'r--', label=f'Objective (Z={Z:.2f})', linewidth=2)
        else:
            plt.axvline(x=Z/c[0], color='r', linestyle='--', label=f'Objective (Z={Z:.2f})', linewidth=2)

        plt.scatter([opt_x], [opt_y], color='red', s=150, zorder=5, marker='*', edgecolor='k', label=f'Optimal: ({opt_x:.2f}, {opt_y:.2f})')

        plt.annotate(f'({opt_x:.2f}, {opt_y:.2f}) \n Z = {Z:.2f}', 
                     (opt_x, opt_y), textcoords="offset points", xytext=(15,10), 
                     ha='left', fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.xlabel(var_names[0], fontsize=12)
    plt.ylabel(var_names[1], fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save graph for poster
    safe_title = title.replace(' ', '_').replace(':', '')
    plt.savefig(f"{safe_title}.png", dpi=300, bbox_inches='tight')
    console.print(f"[bold green]✔[/bold green] Graph automatically saved as [bold cyan]'{safe_title}.png'[/bold cyan] for your poster!")

    plt.show(block=False)
    plt.pause(2.0)


# ─────────────────────────────────────────────────────────────
#  CORE: Big M Method Solver
# ─────────────────────────────────────────────────────────────
def big_m_method(c, A, b, constraint_types, problem_type="max", silent=False):
    num_vars = len(c)
    num_constraints = len(b)

    original_problem_type = problem_type
    C_orig = np.array(c, dtype=float)
    
    if problem_type == "min":
        c = [-ci for ci in c]

    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    for i in range(num_constraints):
        if b[i] < 0:
            b[i] *= -1
            A[i] *= -1
            if constraint_types[i] == "<=": constraint_types[i] = ">="
            elif constraint_types[i] == ">=": constraint_types[i] = "<="

    var_names   = [f"x{i+1}" for i in range(num_vars)]
    slack_names = []
    art_names   = []

    slack_count = sum(1 for ct in constraint_types if ct in ["<=", ">="])
    art_count   = sum(1 for ct in constraint_types if ct in ["=", ">="])

    total_vars  = num_vars + slack_count + art_count
    tableau_A   = np.zeros((num_constraints, total_vars))
    tableau_A[:, :num_vars] = A

    obj_full      = np.zeros(total_vars)
    obj_full[:num_vars] = c

    basis_indices = []
    s_col = num_vars
    a_col = num_vars + slack_count

    for i, ct in enumerate(constraint_types):
        if ct == "<=":
            tableau_A[i, s_col] = 1
            slack_names.append(f"s{s_col - num_vars + 1}")
            basis_indices.append(s_col)
            s_col += 1
        elif ct == ">=":
            tableau_A[i, s_col] = -1
            slack_names.append(f"s{s_col - num_vars + 1}")
            s_col += 1
            tableau_A[i, a_col] = 1
            art_names.append(f"a{a_col - num_vars - slack_count + 1}")
            obj_full[a_col] = -BIG_M
            basis_indices.append(a_col)
            a_col += 1
        elif ct == "=":
            tableau_A[i, a_col] = 1
            art_names.append(f"a{a_col - num_vars - slack_count + 1}")
            obj_full[a_col] = -BIG_M
            basis_indices.append(a_col)
            a_col += 1

    all_col_names = var_names + slack_names + art_names
    tableau = np.zeros((num_constraints + 1, total_vars + 1))
    tableau[:num_constraints, :total_vars] = tableau_A
    tableau[:num_constraints, -1]          = b
    tableau[-1, :total_vars] = -obj_full

    for i, bi in enumerate(basis_indices):
        if obj_full[bi] != 0:
            tableau[-1] += obj_full[bi] * tableau[i]

    row_labels = [all_col_names[bi] for bi in basis_indices]
    iteration  = 0
    iterations_log = []

    if not silent:
        console.rule("[bold cyan]BIG M METHOD — SIMPLEX TABLEAU ITERATIONS[/bold cyan]")

    while True:
            iterations_log.append(np.copy(tableau))
            z_row = tableau[-1, :-1]
            
            if np.all(z_row >= -TOLERANCE):
                if not silent:
                    print_tableau(tableau, all_col_names, row_labels, iteration)
                break

            pivot_col = int(np.argmin(z_row))
            
            ratios = []
            for i in range(num_constraints):
                if tableau[i, pivot_col] > TOLERANCE:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)

            pivot_row = None
            if not all(r == np.inf for r in ratios):
                valid_ratios = [r if r >= 0 else np.inf for r in ratios]
                pivot_row = int(np.argmin(valid_ratios))

            if not silent:
                print_tableau(tableau, all_col_names, row_labels, iteration, ratios, pivot_row, pivot_col)
                console.print(f"[bold green]→ Entering variable:[/bold green] {all_col_names[pivot_col]}")
                
            if pivot_row is None:
                if not silent:
                    console.print("\n[bold red]✗ Problem is UNBOUNDED.[/bold red]")
                return {"status": "unbounded", "optimal_value": None, "solution": None, "iterations": len(iterations_log)}

            if not silent:
                console.print(f"[bold red]→ Leaving variable:[/bold red]  {row_labels[pivot_row]}")
                console.print(f"[bold yellow]→ Pivot element:[/bold yellow]     {format_m(tableau[pivot_row, pivot_col])}\n")

            pivot_val = tableau[pivot_row, pivot_col]
            tableau[pivot_row] /= pivot_val
            for i in range(num_constraints + 1):
                if i != pivot_row:
                    tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

            row_labels[pivot_row] = all_col_names[pivot_col]
            basis_indices[pivot_row] = pivot_col
            iteration += 1

    for i, label in enumerate(row_labels):
        if label.startswith("a") and tableau[i, -1] > TOLERANCE:
            if not silent:
                console.print("\n[bold red]✗ Problem is INFEASIBLE (artificial variable in basis).[/bold red]")
            return {"status": "infeasible", "optimal_value": None, "solution": None, "iterations": iteration}

    solution = np.zeros(num_vars)
    for i, bi in enumerate(basis_indices):
        if bi < num_vars:
            solution[bi] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    if original_problem_type == "min":
        optimal_value = -optimal_value

    result = {
        "status"        : "optimal",
        "optimal_value" : optimal_value,
        "solution"      : solution,
        "iterations"    : iteration,
    }
    
    try:
        plot_2d_lp(C_orig if original_problem_type=="max" else -C_orig, A, b, constraint_types, result, title=f"Big M Method ({original_problem_type.upper()})")
    except Exception as e:
        if not silent:
            pass # Skipping plot error notice in prod mode

    return result

# ─────────────────────────────────────────────────────────────
#  RESULT PRINTER (Rich Panel)
# ─────────────────────────────────────────────────────────────
def print_result(result, var_names=None, problem_type="max"):
    if result["status"] != "optimal":
        console.print(f"\n[bold red]Status:[/bold red] {result['status'].upper()}")
        return

    sol = result["solution"]
    vars_str = ""
    for i, val in enumerate(sol):
        name = var_names[i] if var_names else f"x{i+1}"
        vars_str += f"[bold cyan]{name:>10}[/bold cyan] = [bold white]{val:.4f}[/bold white]\n"

    label = "Max Z" if problem_type == "max" else "Min Z"
    vars_str += f"\n[bold yellow]{label:>10}[/bold yellow] = [bold bright_green]{result['optimal_value']:.4f}[/bold bright_green]"

    panel = Panel.fit(
        vars_str.strip(), 
        title="[bold green]OPTIMAL SOLUTION[/bold green]", 
        border_style="green",
        padding=(1, 5)
    )
    console.print(panel)


# ─────────────────────────────────────────────────────────────
#  INPUT PARSERS
# ─────────────────────────────────────────────────────────────
def parse_equation(eq_str, num_vars):
    c = [0.0] * num_vars
    eq_str = eq_str.replace(" ", "").lower()
    if "=" in eq_str:
        eq_str = eq_str.split("=")[1]
        
    pattern = r'([+-]?\d*\.?\d*)x(\d+)'
    matches = re.findall(pattern, eq_str)
    for coef_str, var_idx in matches:
        idx = int(var_idx) - 1
        if 0 <= idx < num_vars:
            if coef_str == '+' or coef_str == '':
                val = 1.0
            elif coef_str == '-':
                val = -1.0
            else:
                val = float(coef_str)
            c[idx] += val
    return c

def parse_constraint(eq_str, num_vars):
    eq_str = eq_str.lower()
    if '<=' in eq_str: op, split_str = '<=', '<='
    elif '>=' in eq_str: op, split_str = '>=', '>='
    elif '=' in eq_str: op, split_str = '=', '='
    else: return None, None, None
    
    left, right = eq_str.split(split_str)
    coeffs = parse_equation(left, num_vars)
    try:
        rhs = float(right.strip())
    except:
        rhs = 0.0
    return coeffs, op, rhs

# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────
def interactive_mode():
    console.rule("[bold yellow]INTERACTIVE MODE[/bold yellow]")
    problem_type = Prompt.ask("Do you want to [bold green]Maximize[/bold green] or [bold red]Minimize[/bold red]?", choices=["max", "min"], default="max")
    
    num_vars = IntPrompt.ask("Enter the number of [cyan]decision variables[/cyan]")
    num_cons = IntPrompt.ask("Enter the number of [cyan]constraints[/cyan]")

    console.print("\n[bold]1. Objective Function ([/bold]" + problem_type.upper() + "[bold])[/bold]")
    console.print("Example format: [cyan]5x1 + 4x2[/cyan] or [cyan]Z = 5x1 - 3x2[/cyan]")
    obj_str = Prompt.ask("Enter Objective Function")
    c = parse_equation(obj_str, num_vars)

    console.print("\n[bold]2. Constraints[/bold]")
    console.print("Example formats: [cyan]2x1 + x2 <= 10[/cyan], [cyan]x1 + 3x2 >= 5[/cyan]")
    A = []
    b = []
    constraint_types = []
    for i in range(num_cons):
        while True:
            con_str = Prompt.ask(f"Enter Constraint {i+1}")
            coeffs, op, rhs = parse_constraint(con_str, num_vars)
            if op is not None:
                A.append(coeffs)
                constraint_types.append(op)
                b.append(rhs)
                break
            else:
                console.print("[red]Invalid format! Please include <=, >=, or = in the equation.[/red]")

    console.print("\n[bold cyan]Processing your problem...[/bold cyan]")
    
    var_names = [f"x{i+1}" for i in range(num_vars)]
    result = big_m_method(c, A, b, constraint_types, problem_type=problem_type)
    print_result(result, var_names, problem_type)

    if len(c) == 2:
        console.print("\n[dim]* Since this is a 2-variable problem, a graph has been plotted and saved automatically![/dim]")


# ─────────────────────────────────────────────────────────────
#  EXPLANATION: Big M Method
# ─────────────────────────────────────────────────────────────
def show_explanation():
    console.rule("[bold yellow]What is the Big M Method?[/bold yellow]")
    
    explanation_text = """
[bold cyan]1. Definition & Purpose[/bold cyan]
The [bold]Big M Method[/bold] is a modified version of the Simplex algorithm. Normal Simplex requires the origin (variables = 0) to be a feasible starting point. However, when an LP problem has [bold yellow]≥[/bold yellow] or [bold yellow]=[/bold yellow] constraints, the origin is usually not feasible.
To fix this, we introduce [bold magenta]"Artificial Variables."[/bold magenta]

[bold cyan]2. The Role of "Big M"[/bold cyan]
Artificial variables are purely mathematical placeholders—they have no real-world meaning. To ensure the algorithm removes them by the time we reach the final optimal solution, we penalize them heavily in the Objective Function using a massive constant called [bold red]M[/bold red] (e.g., M = 1,000,000).
• For [bold green]Maximization[/bold green], we assign [bold red]-M[/bold red]
• For [bold red]Minimization[/bold red], we assign [bold red]+M[/bold red]

[bold cyan]3. Step-by-Step Algorithm[/bold cyan]
  [bold]Step 1:[/bold] Convert all constraints to equalities:
          • For [bold yellow]≤[/bold yellow] constraints: Add a [italic green]Slack[/italic green] variable (+s).
          • For [bold yellow]≥[/bold yellow] constraints: Subtract a [italic red]Surplus[/italic red] variable (-s) AND add an [italic magenta]Artificial[/italic magenta] variable (+a).
          • For [bold yellow]=[/bold yellow] constraints: Add an [italic magenta]Artificial[/italic magenta] variable (+a).
  [bold]Step 2:[/bold] Modify the objective function Z by penalizing the artificial variables with Big M.
  [bold]Step 3:[/bold] Set up the initial Simplex Tableau. Since Z initially contains basic artificial variables, perform row operations to zero them out in the Z-row.
  [bold]Step 4:[/bold] Run the standard Simplex Algorithm:
          • Select entering variable (most negative Z-row value).
          • Select leaving variable (smallest positive ratio of RHS / Pivot column).
          • Pivot and repeat until all Z-row values are ≥ 0.
  [bold]Step 5:[/bold] Evaluate the final solution:
          • [bold green]Optimal:[/bold green] No artificial variables remain in the basis.
          • [bold red]Infeasible:[/bold red] An artificial variable remains in the basis with a positive value.
"""
    console.print(explanation_text)
    
# ─────────────────────────────────────────────────────────────
#  PRE-CODED EXAMPLES
# ─────────────────────────────────────────────────────────────
def show_examples():
    console.rule("[bold cyan]EXAMPLE 1: Maximise Z = 5x₁ + 4x₂[/bold cyan]")
    c, A, b, constraint_types = [5, 4], [[6, 4], [1, 2]], [24, 6], ["<=", ">="]
    result = big_m_method(c, A, b, constraint_types, problem_type="max")
    print_result(result, ["x1", "x2"], "max")

    console.rule("[bold cyan]EXAMPLE 2: Minimise Z = 2x₁ + 3x₂[/bold cyan]")
    c, A, b, constraint_types = [2, 3], [[1, 1], [1, 3]], [4, 6], ["=", ">="]
    result = big_m_method(c, A, b, constraint_types, problem_type="min")
    print_result(result, ["x1", "x2"], "min")


# ─────────────────────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    console.print(Panel(
        "[bold cyan]BIG M METHOD — Linear Programming Solver[/bold cyan]\n"
        "[white]Assignment 9 | EM-4 (BSC07) | INFT Engineering[/white]",
        style="on blue"
    ), justify="center")

    while True:
        console.print("\n[bold]Welcome! Please choose an option:[/bold]")
        console.print("  [1] Run pre-coded Examples (Shows Tableau + Graphs)")
        console.print("  [2] Enter Interactive Mode (Provide custom equations)")
        console.print("  [3] Learn about the Big M Method (Explanation & Steps)")
        console.print("  [4] Exit")

        choice = Prompt.ask("\nSelect your choice", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            show_examples()
            console.print("\n[bold green]✔ All examples completed successfully.[/bold green]")
            plt.show()
        elif choice == "2":
            interactive_mode()
            plt.show()
        elif choice == "3":
            show_explanation()
        else:
            console.print("[bold cyan]Goodbye![/bold cyan]")
            break
