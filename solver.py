"""
=============================================================
  BIG M METHOD — Core Solver Engine
  Refactored for Streamlit App | EM-4 (BSC07)
=============================================================
  Contains:
    - big_m_method()      — Core simplex solver with full iteration history
    - sensitivity_analysis() — Shadow prices, allowable ranges
    - build_dual()        — Primal-to-Dual LP construction
    - parametric_rhs()    — Parametric analysis on RHS values
=============================================================
"""

import numpy as np
from copy import deepcopy

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
BIG_M = 1e6
TOLERANCE = 1e-9


# ─────────────────────────────────────────────────────────────
#  CORE: Big M Method Solver
# ─────────────────────────────────────────────────────────────
def big_m_method(c, A, b, constraint_types, problem_type="max"):
    """
    Solve an LP using the Big M Method.

    Returns:
        dict with keys:
            status: 'optimal' | 'unbounded' | 'infeasible'
            optimal_value: float or None
            solution: np.array or None
            iterations: int
            iteration_log: list of dicts (full tableau history for display)
            all_col_names: list of column names
            final_basis: list of basis variable names
            final_tableau: np.array
            shadow_prices: np.array (dual values from final Z-row)
    """
    num_vars = len(c)
    num_constraints = len(b)
    constraint_types = list(constraint_types)  # make mutable copy

    original_problem_type = problem_type
    C_orig = np.array(c, dtype=float)

    if problem_type == "min":
        c = [-ci for ci in c]

    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Ensure RHS >= 0
    for i in range(num_constraints):
        if b[i] < 0:
            b[i] *= -1
            A[i] *= -1
            if constraint_types[i] == "<=":
                constraint_types[i] = ">="
            elif constraint_types[i] == ">=":
                constraint_types[i] = "<="

    var_names = [f"x{i+1}" for i in range(num_vars)]
    slack_names = []
    art_names = []

    slack_count = sum(1 for ct in constraint_types if ct in ["<=", ">="])
    art_count = sum(1 for ct in constraint_types if ct in ["=", ">="])

    total_vars = num_vars + slack_count + art_count
    tableau_A = np.zeros((num_constraints, total_vars))
    tableau_A[:, :num_vars] = A

    obj_full = np.zeros(total_vars)
    obj_full[:num_vars] = c

    basis_indices = []
    s_col = num_vars
    a_col = num_vars + slack_count
    art_col_indices = []

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
            art_col_indices.append(a_col)
            basis_indices.append(a_col)
            a_col += 1
        elif ct == "=":
            tableau_A[i, a_col] = 1
            art_names.append(f"a{a_col - num_vars - slack_count + 1}")
            obj_full[a_col] = -BIG_M
            art_col_indices.append(a_col)
            basis_indices.append(a_col)
            a_col += 1

    all_col_names = var_names + slack_names + art_names
    tableau = np.zeros((num_constraints + 1, total_vars + 1))
    tableau[:num_constraints, :total_vars] = tableau_A
    tableau[:num_constraints, -1] = b
    tableau[-1, :total_vars] = -obj_full

    # Eliminate basic artificial variable coefficients from Z-row
    for i, bi in enumerate(basis_indices):
        if obj_full[bi] != 0:
            tableau[-1] += obj_full[bi] * tableau[i]

    row_labels = [all_col_names[bi] for bi in basis_indices]
    iteration = 0
    iteration_log = []

    # Log initial tableau
    iteration_log.append({
        "iteration": iteration,
        "tableau": np.copy(tableau),
        "basis": list(row_labels),
        "entering": None,
        "leaving": None,
        "pivot_element": None,
        "pivot_row": None,
        "pivot_col": None,
    })

    while True:
        z_row = tableau[-1, :-1]
        if np.all(z_row >= -TOLERANCE):
            break

        pivot_col = int(np.argmin(z_row))
        entering_var = all_col_names[pivot_col]

        ratios = []
        for i in range(num_constraints):
            if tableau[i, pivot_col] > TOLERANCE:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(np.inf)

        if all(r == np.inf for r in ratios):
            return {
                "status": "unbounded",
                "optimal_value": None,
                "solution": None,
                "iterations": len(iteration_log),
                "iteration_log": iteration_log,
                "all_col_names": all_col_names,
                "final_basis": row_labels,
                "final_tableau": tableau,
                "shadow_prices": None,
            }

        pivot_row = int(np.argmin(ratios))
        leaving_var = row_labels[pivot_row]
        pivot_val = tableau[pivot_row, pivot_col]

        # Pivot operation
        tableau[pivot_row] /= pivot_val
        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        row_labels[pivot_row] = all_col_names[pivot_col]
        basis_indices[pivot_row] = pivot_col
        iteration += 1

        iteration_log.append({
            "iteration": iteration,
            "tableau": np.copy(tableau),
            "basis": list(row_labels),
            "entering": entering_var,
            "leaving": leaving_var,
            "pivot_element": pivot_val,
            "pivot_row": pivot_row,
            "pivot_col": pivot_col,
        })

    # Check for infeasibility
    for i, label in enumerate(row_labels):
        if label.startswith("a") and tableau[i, -1] > TOLERANCE:
            return {
                "status": "infeasible",
                "optimal_value": None,
                "solution": None,
                "iterations": iteration,
                "iteration_log": iteration_log,
                "all_col_names": all_col_names,
                "final_basis": row_labels,
                "final_tableau": tableau,
                "shadow_prices": None,
            }

    # Extract solution
    solution = np.zeros(num_vars)
    for i, bi in enumerate(basis_indices):
        if bi < num_vars:
            solution[bi] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    if original_problem_type == "min":
        optimal_value = -optimal_value

    # Extract shadow prices from Z-row (slack variable coefficients)
    shadow_prices = np.zeros(num_constraints)
    s_idx = num_vars
    for i, ct in enumerate(constraint_types):
        if ct in ["<=", ">="]:
            sp = tableau[-1, s_idx]
            if original_problem_type == "min":
                sp = -sp
            shadow_prices[i] = sp
            s_idx += 1
        else:
            # For equality constraints, shadow price from artificial variable column
            shadow_prices[i] = 0  # Approximated

    return {
        "status": "optimal",
        "optimal_value": optimal_value,
        "solution": solution,
        "iterations": iteration,
        "iteration_log": iteration_log,
        "all_col_names": all_col_names,
        "final_basis": row_labels,
        "final_tableau": tableau,
        "shadow_prices": shadow_prices,
        "constraint_types": constraint_types,
        "A": A,
        "b": b,
        "c": C_orig,
        "num_vars": num_vars,
        "num_constraints": num_constraints,
        "problem_type": original_problem_type,
    }


# ─────────────────────────────────────────────────────────────
#  SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────
def sensitivity_analysis(result):
    """
    Compute sensitivity analysis from an optimal Big M result.

    Returns:
        dict with:
            shadow_prices: np.array
            obj_coeff_ranges: list of (lower, upper) for each decision variable
            rhs_ranges: list of (lower, upper) for each constraint RHS
            binding_constraints: list of booleans
    """
    if result["status"] != "optimal":
        return None

    tableau = result["final_tableau"]
    A = result["A"]
    b = result["b"]
    c = result["c"]
    num_vars = result["num_vars"]
    num_constraints = result["num_constraints"]
    shadow_prices = result["shadow_prices"]

    # Determine binding constraints
    solution = result["solution"]
    binding = []
    for i in range(num_constraints):
        lhs_val = np.dot(A[i], solution)
        binding.append(abs(lhs_val - b[i]) < 1e-6)

    # Objective coefficient ranges (approximate via perturbation)
    obj_ranges = []
    for j in range(num_vars):
        # Try perturbation approach
        lower_delta = -1e6
        upper_delta = 1e6

        z_row_val = tableau[-1, j]
        # For non-basic variables, the range is bounded by z_row value
        is_basic = any(
            result["final_basis"][i] == f"x{j+1}"
            for i in range(num_constraints)
        )

        if not is_basic:
            # Non-basic: can increase obj coeff by z_row_val before it enters basis
            if result["problem_type"] == "max":
                upper_delta = z_row_val
                lower_delta = -np.inf
            else:
                lower_delta = -z_row_val
                upper_delta = np.inf
            obj_ranges.append((c[j] + lower_delta, c[j] + upper_delta))
        else:
            # Basic: use 100-rule approximation via parametric search
            low, high = _parametric_obj_range(c, A, b, result["constraint_types"],
                                              result["problem_type"], j, result["final_basis"])
            obj_ranges.append((low, high))

    # RHS ranges (how much each b_i can change while keeping same basis)
    rhs_ranges = []
    for i in range(num_constraints):
        low, high = _parametric_rhs_range(c, A, b, result["constraint_types"],
                                          result["problem_type"], i)
        rhs_ranges.append((low, high))

    return {
        "shadow_prices": shadow_prices,
        "obj_coeff_ranges": obj_ranges,
        "rhs_ranges": rhs_ranges,
        "binding_constraints": binding,
    }


def _parametric_obj_range(c, A, b, constraint_types, problem_type, var_idx, current_basis):
    """Find range of c[var_idx] that keeps the same basis optimal."""
    c = np.array(c, dtype=float)
    original_val = c[var_idx]
    low = original_val
    high = original_val

    # Search downward
    for delta in np.linspace(0, -50, 100):
        c_test = c.copy()
        c_test[var_idx] = original_val + delta
        try:
            res = big_m_method(c_test.tolist(), A.tolist(), b.tolist(),
                               list(constraint_types), problem_type)
            if res["status"] == "optimal" and list(res["final_basis"]) == list(current_basis):
                low = original_val + delta
            else:
                break
        except Exception:
            break

    # Search upward
    for delta in np.linspace(0, 50, 100):
        c_test = c.copy()
        c_test[var_idx] = original_val + delta
        try:
            res = big_m_method(c_test.tolist(), A.tolist(), b.tolist(),
                               list(constraint_types), problem_type)
            if res["status"] == "optimal" and list(res["final_basis"]) == list(current_basis):
                high = original_val + delta
            else:
                break
        except Exception:
            break

    return (low, high)


def _parametric_rhs_range(c, A, b, constraint_types, problem_type, constraint_idx):
    """Find range of b[constraint_idx] that keeps the same basis optimal."""
    b = np.array(b, dtype=float)
    original_val = b[constraint_idx]
    low = original_val
    high = original_val

    original_result = big_m_method(c.tolist() if isinstance(c, np.ndarray) else list(c),
                                   A.tolist() if isinstance(A, np.ndarray) else [list(r) for r in A],
                                   b.tolist(), list(constraint_types), problem_type)
    if original_result["status"] != "optimal":
        return (original_val, original_val)

    original_basis = list(original_result["final_basis"])

    # Search downward
    for delta in np.linspace(0, -50, 100):
        b_test = b.copy()
        b_test[constraint_idx] = original_val + delta
        if b_test[constraint_idx] < 0:
            break
        try:
            res = big_m_method(
                c.tolist() if isinstance(c, np.ndarray) else list(c),
                A.tolist() if isinstance(A, np.ndarray) else [list(r) for r in A],
                b_test.tolist(), list(constraint_types), problem_type
            )
            if res["status"] == "optimal" and list(res["final_basis"]) == original_basis:
                low = original_val + delta
            else:
                break
        except Exception:
            break

    # Search upward
    for delta in np.linspace(0, 50, 100):
        b_test = b.copy()
        b_test[constraint_idx] = original_val + delta
        try:
            res = big_m_method(
                c.tolist() if isinstance(c, np.ndarray) else list(c),
                A.tolist() if isinstance(A, np.ndarray) else [list(r) for r in A],
                b_test.tolist(), list(constraint_types), problem_type
            )
            if res["status"] == "optimal" and list(res["final_basis"]) == original_basis:
                high = original_val + delta
            else:
                break
        except Exception:
            break

    return (low, high)


# ─────────────────────────────────────────────────────────────
#  DUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────
def build_dual(c, A, b, constraint_types, problem_type="max"):
    """
    Construct the Dual LP from the Primal LP.

    Primal (standard MAX):
        max c^T x
        s.t. Ax <= b, x >= 0

    Dual:
        min b^T y
        s.t. A^T y >= c, y >= 0

    For mixed constraints, we handle sign conventions.

    Returns:
        dict with dual_c, dual_A, dual_b, dual_constraint_types, dual_problem_type,
        and formatted string representations.
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    num_vars = len(c)
    num_constraints = len(b)

    if problem_type == "max":
        dual_problem_type = "min"
        dual_c = b.tolist()
        dual_A = A.T.tolist()
        dual_b = c.tolist()
        dual_constraint_types = []

        for ct in constraint_types:
            if ct == "<=":
                dual_constraint_types.append(">=")
            elif ct == ">=":
                dual_constraint_types.append("<=")
            else:  # "="
                dual_constraint_types.append(">=")

        # Dual variable sign:
        # <= constraint -> y_i >= 0 (unrestricted handled as >=0 for simplicity)
        # >= constraint -> y_i <= 0
        # = constraint -> y_i unrestricted
        dual_var_signs = []
        for ct in constraint_types:
            if ct == "<=":
                dual_var_signs.append("≥ 0")
            elif ct == ">=":
                dual_var_signs.append("≤ 0")
            else:
                dual_var_signs.append("unrestricted")

    else:  # min
        dual_problem_type = "max"
        dual_c = b.tolist()
        dual_A = A.T.tolist()
        dual_b = c.tolist()
        dual_constraint_types = []

        for ct in constraint_types:
            if ct == ">=":
                dual_constraint_types.append("<=")
            elif ct == "<=":
                dual_constraint_types.append(">=")
            else:
                dual_constraint_types.append("<=")

        dual_var_signs = []
        for ct in constraint_types:
            if ct == ">=":
                dual_var_signs.append("≥ 0")
            elif ct == "<=":
                dual_var_signs.append("≤ 0")
            else:
                dual_var_signs.append("unrestricted")

    return {
        "dual_c": dual_c,
        "dual_A": dual_A,
        "dual_b": dual_b,
        "dual_constraint_types": dual_constraint_types,
        "dual_problem_type": dual_problem_type,
        "dual_var_signs": dual_var_signs,
        "num_dual_vars": num_constraints,
        "num_dual_constraints": num_vars,
    }


def verify_strong_duality(primal_result, dual_result):
    """Check if primal Z* == dual Z* (Strong Duality Theorem)."""
    if primal_result["status"] != "optimal" or dual_result["status"] != "optimal":
        return {
            "holds": False,
            "reason": "One or both problems are not optimal.",
            "primal_z": primal_result.get("optimal_value"),
            "dual_z": dual_result.get("optimal_value"),
        }

    primal_z = primal_result["optimal_value"]
    dual_z = dual_result["optimal_value"]
    holds = abs(primal_z - dual_z) < 1e-3

    return {
        "holds": holds,
        "primal_z": primal_z,
        "dual_z": dual_z,
        "difference": abs(primal_z - dual_z),
        "reason": "Strong Duality holds!" if holds else f"Gap = {abs(primal_z - dual_z):.6f}",
    }


def complementary_slackness(primal_result, dual_result):
    """
    Check complementary slackness conditions.
    For each primal constraint: either the constraint is tight OR the dual variable is 0.
    For each dual constraint: either the constraint is tight OR the primal variable is 0.
    """
    if primal_result["status"] != "optimal" or dual_result["status"] != "optimal":
        return []

    conditions = []
    primal_sol = primal_result["solution"]
    A = primal_result["A"]
    b = primal_result["b"]

    # Primal constraints
    for i in range(len(b)):
        lhs = np.dot(A[i], primal_sol)
        slack = b[i] - lhs
        shadow = primal_result["shadow_prices"][i] if primal_result["shadow_prices"] is not None else 0

        is_tight = abs(slack) < 1e-6
        dual_zero = abs(shadow) < 1e-6
        satisfied = is_tight or dual_zero

        conditions.append({
            "constraint": i + 1,
            "slack": slack,
            "dual_value": shadow,
            "is_tight": is_tight,
            "dual_is_zero": dual_zero,
            "satisfied": satisfied,
        })

    return conditions


# ─────────────────────────────────────────────────────────────
#  PARAMETRIC ANALYSIS
# ─────────────────────────────────────────────────────────────
def parametric_rhs_analysis(c, A, b, constraint_types, problem_type, constraint_idx, range_pct=0.5, steps=50):
    """
    Vary b[constraint_idx] over a range and track how optimal Z changes.

    Returns:
        list of (b_value, z_value) tuples
    """
    b = np.array(b, dtype=float)
    original_val = b[constraint_idx]
    low = max(0, original_val * (1 - range_pct))
    high = original_val * (1 + range_pct)

    results = []
    for val in np.linspace(low, high, steps):
        b_test = b.copy()
        b_test[constraint_idx] = val
        try:
            res = big_m_method(
                c.tolist() if isinstance(c, np.ndarray) else list(c),
                A.tolist() if isinstance(A, np.ndarray) else [list(r) for r in A],
                b_test.tolist(), list(constraint_types), problem_type
            )
            if res["status"] == "optimal":
                results.append((val, res["optimal_value"]))
        except Exception:
            pass

    return results


def parametric_obj_analysis(c, A, b, constraint_types, problem_type, var_idx, range_pct=0.5, steps=50):
    """
    Vary c[var_idx] over a range and track how optimal Z changes.

    Returns:
        list of (c_value, z_value) tuples
    """
    c = np.array(c, dtype=float)
    original_val = c[var_idx]
    margin = max(abs(original_val) * range_pct, 5)
    low = original_val - margin
    high = original_val + margin

    results = []
    for val in np.linspace(low, high, steps):
        c_test = c.copy()
        c_test[var_idx] = val
        try:
            res = big_m_method(
                c_test.tolist(),
                A.tolist() if isinstance(A, np.ndarray) else [list(r) for r in A],
                b.tolist() if isinstance(b, np.ndarray) else list(b),
                list(constraint_types), problem_type
            )
            if res["status"] == "optimal":
                results.append((val, res["optimal_value"]))
        except Exception:
            pass

    return results
