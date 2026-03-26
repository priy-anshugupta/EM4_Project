"""
=============================================================
  BIG M METHOD — Visualization & Formatting Utilities
  Plotly charts, LaTeX formatting, 3D surfaces
=============================================================
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────
#  LaTeX FORMATTING
# ─────────────────────────────────────────────────────────────
def format_objective_latex(c, problem_type="max"):
    """Format the objective function as a LaTeX string."""
    terms = []
    for i, coeff in enumerate(c):
        if coeff == 0:
            continue
        var = f"x_{{{i+1}}}"
        if i == 0:
            if coeff == 1:
                terms.append(var)
            elif coeff == -1:
                terms.append(f"-{var}")
            else:
                terms.append(f"{coeff:g}{var}")
        else:
            if coeff == 1:
                terms.append(f"+ {var}")
            elif coeff == -1:
                terms.append(f"- {var}")
            elif coeff > 0:
                terms.append(f"+ {coeff:g}{var}")
            else:
                terms.append(f"- {abs(coeff):g}{var}")

    obj_str = " ".join(terms) if terms else "0"
    label = "\\text{Maximize}" if problem_type == "max" else "\\text{Minimize}"
    return f"{label} \\quad Z = {obj_str}"


def format_constraint_latex(A_row, constraint_type, b_val, idx=None):
    """Format a single constraint as a LaTeX string."""
    terms = []
    for i, coeff in enumerate(A_row):
        if coeff == 0:
            continue
        var = f"x_{{{i+1}}}"
        if len(terms) == 0:
            if coeff == 1:
                terms.append(var)
            elif coeff == -1:
                terms.append(f"-{var}")
            else:
                terms.append(f"{coeff:g}{var}")
        else:
            if coeff == 1:
                terms.append(f"+ {var}")
            elif coeff == -1:
                terms.append(f"- {var}")
            elif coeff > 0:
                terms.append(f"+ {coeff:g}{var}")
            else:
                terms.append(f"- {abs(coeff):g}{var}")

    lhs = " ".join(terms) if terms else "0"
    ct_map = {"<=": "\\leq", ">=": "\\geq", "=": "="}
    ct = ct_map.get(constraint_type, constraint_type)
    return f"{lhs} {ct} {b_val:g}"


def format_full_problem_latex(c, A, b, constraint_types, problem_type="max"):
    """Format the complete LP problem as LaTeX."""
    lines = [format_objective_latex(c, problem_type)]
    lines.append("\\text{subject to:}")
    for i in range(len(b)):
        lines.append(format_constraint_latex(A[i], constraint_types[i], b[i], i))
    lines.append(f"x_{{1}}, x_{{2}}, \\ldots, x_{{{len(c)}}} \\geq 0")
    return lines


# ─────────────────────────────────────────────────────────────
#  PLOTLY: 2D Feasible Region
# ─────────────────────────────────────────────────────────────
def create_feasible_region_plot(c, A, b, constraint_types, result, problem_type="max"):
    """Create an interactive Plotly plot of the feasible region for 2-variable problems."""
    if len(c) != 2:
        return None

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    # Determine plot bounds
    max_x = 15
    max_y = 15
    for i in range(len(b)):
        if A[i][0] > 0:
            max_x = max(max_x, b[i] / A[i][0] * 1.5)
        if A[i][1] > 0:
            max_y = max(max_y, b[i] / A[i][1] * 1.5)

    if result and result["status"] == "optimal":
        max_x = max(max_x, result["solution"][0] * 2 + 2)
        max_y = max(max_y, result["solution"][1] * 2 + 2)

    max_x = min(max_x, 100)
    max_y = min(max_y, 100)

    fig = go.Figure()

    # Constraint lines
    neon_colors = [
        "#00d4ff", "#ff006e", "#39ff14", "#ffbe0b", "#8338ec",
        "#fb5607", "#3a86ff", "#ff595e",
    ]
    x_range = np.linspace(0, max_x, 500)

    for i in range(len(b)):
        color = neon_colors[i % len(neon_colors)]
        ct_sym = {"<=": "≤", ">=": "≥", "=": "="}.get(constraint_types[i], constraint_types[i])
        label = f"C{i+1}: {A[i][0]:g}x₁ + {A[i][1]:g}x₂ {ct_sym} {b[i]:g}"

        if A[i][1] != 0:
            y_line = (b[i] - A[i][0] * x_range) / A[i][1]
            mask = (y_line >= -1) & (y_line <= max_y + 5)
            fig.add_trace(go.Scatter(
                x=x_range[mask], y=y_line[mask],
                mode="lines",
                name=label,
                line=dict(color=color, width=3),
                hovertemplate=f"<b>{label}</b><br>x₁=%{{x:.2f}}<br>x₂=%{{y:.2f}}<extra></extra>",
            ))
        else:
            x_val = b[i] / A[i][0]
            fig.add_trace(go.Scatter(
                x=[x_val, x_val], y=[0, max_y],
                mode="lines",
                name=label,
                line=dict(color=color, width=3),
            ))

    # Feasible region shading (compute vertices via brute force)
    feasible_vertices = _compute_feasible_vertices(A, b, constraint_types, max_x, max_y)
    if len(feasible_vertices) >= 3:
        # Sort vertices by angle from centroid
        centroid = np.mean(feasible_vertices, axis=0)
        angles = np.arctan2(feasible_vertices[:, 1] - centroid[1],
                            feasible_vertices[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        sorted_verts = feasible_vertices[sorted_idx]

        fig.add_trace(go.Scatter(
            x=np.append(sorted_verts[:, 0], sorted_verts[0, 0]),
            y=np.append(sorted_verts[:, 1], sorted_verts[0, 1]),
            fill="toself",
            fillcolor="rgba(0, 212, 255, 0.12)",
            line=dict(color="rgba(0, 212, 255, 0.5)", width=2, dash="dot"),
            name="Feasible Region",
            hoverinfo="skip",
        ))

        # Corner points
        fig.add_trace(go.Scatter(
            x=sorted_verts[:, 0], y=sorted_verts[:, 1],
            mode="markers",
            marker=dict(size=8, color="#00d4ff", line=dict(width=1, color="white")),
            name="Corner Points",
            hovertemplate="<b>Corner Point</b><br>x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra></extra>",
        ))

    # Optimal point
    if result and result["status"] == "optimal":
        opt_x, opt_y = result["solution"][0], result["solution"][1]
        z_val = result["optimal_value"]

        fig.add_trace(go.Scatter(
            x=[opt_x], y=[opt_y],
            mode="markers+text",
            marker=dict(
                size=18, color="#ff006e",
                symbol="star",
                line=dict(width=2, color="white"),
            ),
            text=[f"Z*={z_val:.2f}"],
            textposition="top right",
            textfont=dict(size=14, color="#ff006e", family="monospace"),
            name=f"Optimal ({opt_x:.2f}, {opt_y:.2f})",
            hovertemplate=(
                f"<b>🌟 OPTIMAL SOLUTION</b><br>"
                f"x₁ = {opt_x:.4f}<br>"
                f"x₂ = {opt_y:.4f}<br>"
                f"Z* = {z_val:.4f}<extra></extra>"
            ),
        ))

        # Objective function line through optimal
        if c[1] != 0:
            y_obj = (z_val - c[0] * x_range) / c[1]
            mask = (y_obj >= -1) & (y_obj <= max_y + 5)
            fig.add_trace(go.Scatter(
                x=x_range[mask], y=y_obj[mask],
                mode="lines",
                line=dict(color="#ff006e", width=2, dash="dash"),
                name=f"Z = {z_val:.2f}",
                hoverinfo="skip",
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,12,41,0.0)",
        plot_bgcolor="rgba(15,12,41,0.3)",
        title=dict(
            text="<b>Feasible Region & Optimal Solution</b>",
            font=dict(size=20, color="#00d4ff"),
        ),
        xaxis=dict(
            title="x₁", range=[0, max_x],
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.2)",
        ),
        yaxis=dict(
            title="x₂", range=[0, max_y],
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.2)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(0, 212, 255, 0.3)",
            borderwidth=1,
            font=dict(color="white"),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        height=550,
    )

    return fig


def _compute_feasible_vertices(A, b, constraint_types, max_x, max_y):
    """Compute feasible region vertices by intersecting all constraint pairs + axes."""
    n = len(b)
    # Add axis constraints: x1 >= 0, x2 >= 0
    all_A = list(A) + [[-1, 0], [0, -1], [1, 0], [0, 1]]
    all_b = list(b) + [0, 0, max_x, max_y]
    all_ct = list(constraint_types) + ["<=", "<=", "<=", "<="]

    vertices = []
    total = len(all_b)

    for i in range(total):
        for j in range(i + 1, total):
            A_pair = np.array([all_A[i], all_A[j]], dtype=float)
            b_pair = np.array([all_b[i], all_b[j]], dtype=float)

            try:
                if abs(np.linalg.det(A_pair)) < 1e-10:
                    continue
                point = np.linalg.solve(A_pair, b_pair)
            except np.linalg.LinAlgError:
                continue

            if point[0] < -1e-6 or point[1] < -1e-6:
                continue
            if point[0] > max_x + 1e-6 or point[1] > max_y + 1e-6:
                continue

            # Check if point satisfies ALL constraints
            feasible = True
            for k in range(n):
                val = np.dot(A[k], point)
                if all_ct[k] == "<=" and val > b[k] + 1e-6:
                    feasible = False
                    break
                elif all_ct[k] == ">=" and val < b[k] - 1e-6:
                    feasible = False
                    break
                elif all_ct[k] == "=" and abs(val - b[k]) > 1e-4:
                    feasible = False
                    break

            if feasible:
                vertices.append(point)

    if len(vertices) == 0:
        return np.array([]).reshape(0, 2)

    # Remove duplicates
    vertices = np.array(vertices)
    unique = [vertices[0]]
    for v in vertices[1:]:
        if not any(np.allclose(v, u, atol=1e-4) for u in unique):
            unique.append(v)

    return np.array(unique)


# ─────────────────────────────────────────────────────────────
#  PLOTLY: 3D Objective Surface
# ─────────────────────────────────────────────────────────────
def create_3d_surface_plot(c, A, b, constraint_types, result, problem_type="max"):
    """Create a 3D surface plot of the objective function over the feasible region."""
    if len(c) != 2:
        return None

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    max_x = 15
    max_y = 15
    for i in range(len(b)):
        if A[i][0] > 0:
            max_x = max(max_x, b[i] / A[i][0] * 1.3)
        if A[i][1] > 0:
            max_y = max(max_y, b[i] / A[i][1] * 1.3)
    max_x = min(max_x, 80)
    max_y = min(max_y, 80)

    x = np.linspace(0, max_x, 100)
    y = np.linspace(0, max_y, 100)
    X, Y = np.meshgrid(x, y)
    Z = c[0] * X + c[1] * Y

    # Mask infeasible
    feasible = np.ones_like(X, dtype=bool)
    for i in range(len(b)):
        val = A[i][0] * X + A[i][1] * Y
        if constraint_types[i] == "<=":
            feasible &= (val <= b[i] + 1e-3)
        elif constraint_types[i] == ">=":
            feasible &= (val >= b[i] - 1e-3)
        elif constraint_types[i] == "=":
            feasible &= (np.abs(val - b[i]) <= max_x * 0.02)

    Z_masked = np.where(feasible, Z, np.nan)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_masked,
        colorscale=[
            [0.0, "#302b63"],
            [0.25, "#3a86ff"],
            [0.5, "#00d4ff"],
            [0.75, "#39ff14"],
            [1.0, "#ff006e"],
        ],
        opacity=0.85,
        name="Z Surface",
        hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Z", font=dict(color="white")),
            tickfont=dict(color="white"),
        ),
    ))

    # Optimal point marker
    if result and result["status"] == "optimal":
        opt_x, opt_y = result["solution"][0], result["solution"][1]
        z_opt = result["optimal_value"]
        fig.add_trace(go.Scatter3d(
            x=[opt_x], y=[opt_y], z=[z_opt],
            mode="markers+text",
            marker=dict(size=10, color="#ff006e", symbol="diamond"),
            text=[f"Z*={z_opt:.2f}"],
            textfont=dict(size=12, color="#ff006e"),
            textposition="top center",
            name="Optimal",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,12,41,0.0)",
        title=dict(
            text="<b>3D Objective Function Surface</b>",
            font=dict(size=20, color="#00d4ff"),
        ),
        scene=dict(
            xaxis=dict(title="x₁", backgroundcolor="rgba(15,12,41,0.3)",
                       gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="x₂", backgroundcolor="rgba(15,12,41,0.3)",
                       gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Z", backgroundcolor="rgba(15,12,41,0.3)",
                       gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(15,12,41,0.0)",
        ),
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


# ─────────────────────────────────────────────────────────────
#  PLOTLY: Simplex Path Animation
# ─────────────────────────────────────────────────────────────
def create_simplex_path_plot(c, A, b, constraint_types, result):
    """Show the path the simplex method takes from vertex to vertex."""
    if len(c) != 2 or result is None or result["status"] != "optimal":
        return None

    # Extract iteration solutions
    iteration_log = result["iteration_log"]
    num_vars = 2
    path_points = []

    for entry in iteration_log:
        tableau = entry["tableau"]
        basis = entry["basis"]
        sol = np.zeros(num_vars)
        for i, bv in enumerate(basis):
            if bv.startswith("x"):
                idx = int(bv[1:]) - 1
                if idx < num_vars:
                    sol[idx] = tableau[i, -1]
        path_points.append(sol.copy())

    if len(path_points) < 2:
        return None

    # Create base feasible region plot
    fig = create_feasible_region_plot(c, A, b, constraint_types, result)
    if fig is None:
        return None

    path_x = [p[0] for p in path_points]
    path_y = [p[1] for p in path_points]

    # Simplex path
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y,
        mode="lines+markers+text",
        line=dict(color="#ffbe0b", width=3, dash="solid"),
        marker=dict(
            size=[12] + [10] * (len(path_x) - 2) + [16],
            color=["#39ff14"] + ["#ffbe0b"] * (len(path_x) - 2) + ["#ff006e"],
            symbol=["circle"] + ["circle"] * (len(path_x) - 2) + ["star"],
            line=dict(width=2, color="white"),
        ),
        text=[f"Iter {i}" for i in range(len(path_x))],
        textposition="bottom right",
        textfont=dict(size=10, color="#ffbe0b"),
        name="Simplex Path",
        hovertemplate=(
            "<b>Iteration %{text}</b><br>"
            "x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra></extra>"
        ),
    ))

    # Arrows
    for i in range(len(path_x) - 1):
        fig.add_annotation(
            x=path_x[i + 1], y=path_y[i + 1],
            ax=path_x[i], ay=path_y[i],
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#ffbe0b",
        )

    fig.update_layout(
        title=dict(
            text="<b>Simplex Iteration Path (Vertex to Vertex)</b>",
            font=dict(size=20, color="#ffbe0b"),
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
#  PLOTLY: Parametric Analysis Chart
# ─────────────────────────────────────────────────────────────
def create_parametric_chart(data, xlabel, ylabel, title, color="#00d4ff"):
    """Create a parametric analysis line chart."""
    if not data:
        return None

    x_vals = [d[0] for d in data]
    y_vals = [d[1] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=6, color=color),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
        hovertemplate=f"<b>{xlabel}</b>: %{{x:.2f}}<br><b>{ylabel}</b>: %{{y:.2f}}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,12,41,0.0)",
        plot_bgcolor="rgba(15,12,41,0.3)",
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, color=color)),
        xaxis=dict(title=xlabel, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=ylabel, gridcolor="rgba(255,255,255,0.08)"),
        height=400,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    return fig
