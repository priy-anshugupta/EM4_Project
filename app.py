"""
=============================================================
  BIG M METHOD — Streamlit Web Application
  EM-4 (BSC07) | Group 2
=============================================================
"""

import streamlit as st
import numpy as np
import json
from solver import big_m_method, sensitivity_analysis, build_dual, verify_strong_duality
from solver import complementary_slackness, parametric_rhs_analysis, parametric_obj_analysis
from utils import (
    format_objective_latex, format_constraint_latex, format_full_problem_latex,
    create_feasible_region_plot, create_3d_surface_plot,
    create_simplex_path_plot, create_parametric_chart,
)

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Big M Method — LP Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  INJECTED CSS — Dark Glassmorphism Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,12,41,0.95), rgba(36,36,62,0.95)) !important;
    border-right: 1px solid rgba(0,212,255,0.15);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #00d4ff !important;
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 28px;
    margin: 12px 0;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.3);
    box-shadow: 0 8px 40px rgba(0,212,255,0.08);
    transform: translateY(-2px);
}

/* ── Neon Headers ── */
.neon-title {
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #00d4ff, #39ff14, #ff006e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 8px;
    letter-spacing: -1px;
    animation: glow 3s ease-in-out infinite;
}
@keyframes glow {
    0%, 100% { filter: drop-shadow(0 0 8px rgba(0,212,255,0.4)); }
    50% { filter: drop-shadow(0 0 20px rgba(57,255,20,0.4)); }
}
.neon-subtitle {
    font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.5);
    text-align: center;
    font-size: 1rem;
    margin-bottom: 30px;
    letter-spacing: 3px;
}

/* ── Stat Boxes ── */
.stat-box {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.stat-box:hover { border-color: rgba(0,212,255,0.5); transform: scale(1.02); }
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label { color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 4px; }

/* ── Optimal Banner ── */
.optimal-banner {
    background: linear-gradient(135deg, rgba(57,255,20,0.08), rgba(0,212,255,0.08));
    border: 1px solid rgba(57,255,20,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    animation: borderPulse 2s ease-in-out infinite;
}
@keyframes borderPulse {
    0%, 100% { border-color: rgba(57,255,20,0.3); }
    50% { border-color: rgba(57,255,20,0.7); }
}
.optimal-z {
    font-size: 3rem;
    font-weight: 900;
    color: #39ff14;
    font-family: 'JetBrains Mono', monospace;
    text-shadow: 0 0 30px rgba(57,255,20,0.4);
}
.optimal-label { color: rgba(255,255,255,0.7); font-size: 1rem; margin-bottom: 8px; }

/* ── Table Styles ── */
.tableau-container {
    overflow-x: auto;
    margin: 12px 0;
}
.tableau-container table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.tableau-container th {
    background: rgba(0,212,255,0.12);
    color: #00d4ff;
    padding: 12px 14px;
    font-weight: 600;
    border-bottom: 1px solid rgba(0,212,255,0.2);
}
.tableau-container td {
    padding: 10px 14px;
    color: #e0e0e0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.tableau-container tr:hover td { background: rgba(0,212,255,0.04); }
.tableau-container tr:last-child td {
    background: rgba(255,0,110,0.06);
    color: #ff006e;
    font-weight: 700;
    border-top: 2px solid rgba(255,0,110,0.3);
}
.pivot-cell {
    background: rgba(255,190,11,0.15) !important;
    color: #ffbe0b !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255,190,11,0.4) !important;
    border-radius: 6px;
}

/* ── Tags ── */
.tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.tag-enter { background: rgba(57,255,20,0.12); color: #39ff14; border: 1px solid rgba(57,255,20,0.3); }
.tag-leave { background: rgba(255,0,110,0.12); color: #ff006e; border: 1px solid rgba(255,0,110,0.3); }

/* ── Feature Cards ── */
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    transition: all 0.35s ease;
    height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.feature-card:hover {
    border-color: rgba(0,212,255,0.4);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,212,255,0.1);
}
.feature-icon { font-size: 2.5rem; margin-bottom: 12px; }
.feature-title { color: #00d4ff; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; }
.feature-desc { color: rgba(255,255,255,0.5); font-size: 0.82rem; }

/* ── Duality Side-by-Side ── */
.dual-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(131,56,236,0.25);
    border-radius: 14px;
    padding: 24px;
}
.dual-title {
    font-weight: 800;
    font-size: 1.2rem;
    margin-bottom: 12px;
}

/* ── Streamlit overrides ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.6);
    font-weight: 600;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.12) !important;
    color: #00d4ff !important;
    border-bottom: none !important;
}
.stNumberInput input, .stSelectbox select, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
}
.stButton>button {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(57,255,20,0.1)) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: #00d4ff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    border-color: rgba(0,212,255,0.6) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.2) !important;
    transform: translateY(-1px) !important;
}
h1, h2, h3, h4, h5, h6 { color: #e0e0e0 !important; }
p, li, span, label, div { color: #c0c0c0; }

/* ── Confetti ── */
@keyframes confettiFall {
    0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
    100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
}
.confetti-piece {
    position: fixed;
    width: 10px;
    height: 10px;
    top: -10px;
    z-index: 9999;
    animation: confettiFall 3s ease-in forwards;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧮 Big M Method")
    st.markdown("---")
    st.markdown("**EM-4 (BSC07) | Group 2**")
    st.markdown("""
    | Name | Roll |
    |:--|:--|
    | Priyanshu Gupta | 24101B0037 |
    | Aashutosh Mahajan | 24101b0063 |
    | Atharv Raut | 24101B0052 |
    | Aditya Patra | 24101B0072 |
    | Ronak Boddu | 24101B0044 |
    """)
    st.markdown("---")
    st.markdown("##### 🔧 Quick Settings")
    big_m_val = st.number_input("Big M Value", value=1000000.0, step=1000.0, format="%.0f")
    st.caption("Higher M → stronger penalty on artificial vars")


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs(["🏠 Home", "🧮 Solver", "📊 Visualization", "📈 Sensitivity",
                "🔄 Duality", "📚 Learn", "🏆 Examples"])


# ═══════════════════════════════════════════════════════════
#  TAB 0: HOME
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="neon-title">BIG M METHOD</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-subtitle">LINEAR PROGRAMMING SOLVER & VISUALIZER</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <p style="font-size:1.15rem; color:rgba(255,255,255,0.8); max-width:700px; margin:0 auto; line-height:1.8;">
            The <span style="color:#00d4ff; font-weight:700;">Big M Method</span> is a powerful technique
            in Linear Programming that handles <span style="color:#ff006e;">≥</span> and
            <span style="color:#ff006e;">=</span> constraints by introducing
            <span style="color:#ffbe0b;">artificial variables</span> with a massive penalty
            <span style="color:#39ff14; font-family:'JetBrains Mono';">M</span>,
            forcing the simplex algorithm to find the true optimal solution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("🧮", "Interactive Solver", "Input any LP problem with mixed constraints"),
        ("📊", "3D Visualization", "Feasible regions, surfaces & simplex paths"),
        ("📈", "Sensitivity Analysis", "Shadow prices & parametric what-if scenarios"),
        ("🔄", "Duality Engine", "Auto-construct dual & verify strong duality"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 📐 How It Works")
    st.markdown("""
    <div class="glass-card">
        <table style="width:100%; border-collapse:collapse;">
            <tr>
                <td style="padding:14px; border-right:1px solid rgba(255,255,255,0.06); width:25%; text-align:center;">
                    <div style="font-size:2rem;">1️⃣</div>
                    <div style="color:#00d4ff; font-weight:700; margin:8px 0;">Convert</div>
                    <div style="color:rgba(255,255,255,0.5); font-size:0.82rem;">Add slack, surplus & artificial variables</div>
                </td>
                <td style="padding:14px; border-right:1px solid rgba(255,255,255,0.06); width:25%; text-align:center;">
                    <div style="font-size:2rem;">2️⃣</div>
                    <div style="color:#ff006e; font-weight:700; margin:8px 0;">Penalize</div>
                    <div style="color:rgba(255,255,255,0.5); font-size:0.82rem;">Assign −M (max) or +M (min) to artificial vars</div>
                </td>
                <td style="padding:14px; border-right:1px solid rgba(255,255,255,0.06); width:25%; text-align:center;">
                    <div style="font-size:2rem;">3️⃣</div>
                    <div style="color:#ffbe0b; font-weight:700; margin:8px 0;">Iterate</div>
                    <div style="color:rgba(255,255,255,0.5); font-size:0.82rem;">Run simplex pivots until Z-row ≥ 0</div>
                </td>
                <td style="padding:14px; width:25%; text-align:center;">
                    <div style="font-size:2rem;">4️⃣</div>
                    <div style="color:#39ff14; font-weight:700; margin:8px 0;">Optimal!</div>
                    <div style="color:rgba(255,255,255,0.5); font-size:0.82rem;">Read solution from basis variables</div>
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  HELPER: Build tableau HTML
# ═══════════════════════════════════════════════════════════
def render_tableau_html(entry, all_col_names):
    """Render a single iteration's tableau as styled HTML."""
    tableau = entry["tableau"]
    basis = entry["basis"]
    iteration = entry["iteration"]
    pivot_r = entry.get("pivot_row")
    pivot_c = entry.get("pivot_col")
    entering = entry.get("entering")
    leaving = entry.get("leaving")
    pivot_elem = entry.get("pivot_element")

    num_rows = len(basis)
    html = f'<div class="tableau-container">'

    # Info tags
    if entering:
        html += f'<span class="tag tag-enter">▶ ENTER: {entering}</span> '
    if leaving:
        html += f'<span class="tag tag-leave">◀ LEAVE: {leaving}</span> '
    if pivot_elem is not None:
        html += f'<span class="tag" style="background:rgba(255,190,11,0.12);color:#ffbe0b;border:1px solid rgba(255,190,11,0.3);">⬥ PIVOT: {pivot_elem:.4f}</span>'

    html += '<table><thead><tr>'
    html += '<th>Basis</th><th>RHS</th>'
    for cn in all_col_names:
        html += f'<th>{cn}</th>'
    html += '</tr></thead><tbody>'

    for i in range(num_rows):
        html += '<tr>'
        html += f'<td style="color:#39ff14; font-weight:600;">{basis[i]}</td>'
        html += f'<td style="color:#ffbe0b; font-weight:600;">{tableau[i, -1]:.4f}</td>'
        for j in range(len(all_col_names)):
            cell_class = ""
            if pivot_r is not None and pivot_c is not None and i == pivot_r and j == pivot_c:
                cell_class = ' class="pivot-cell"'
            html += f'<td{cell_class}>{tableau[i, j]:.4f}</td>'
        html += '</tr>'

    # Z-row
    html += '<tr>'
    html += '<td style="font-weight:700;">Z</td>'
    html += f'<td style="font-weight:700;">{tableau[-1, -1]:.4f}</td>'
    for j in range(len(all_col_names)):
        html += f'<td>{tableau[-1, j]:.4f}</td>'
    html += '</tr>'

    html += '</tbody></table></div>'
    return html


# ═══════════════════════════════════════════════════════════
#  TAB 1: SOLVER
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🧮 Interactive LP Solver")
    st.markdown("Enter your Linear Programming problem below.")

    sol_col1, sol_col2 = st.columns([1, 1])

    with sol_col1:
        problem_type = st.selectbox("Problem Type", ["max", "min"],
                                    format_func=lambda x: "📈 Maximize" if x == "max" else "📉 Minimize")
        num_vars = st.number_input("Number of Decision Variables", min_value=1, max_value=10, value=2)
        num_cons = st.number_input("Number of Constraints", min_value=1, max_value=10, value=2)

    with sol_col2:
        st.markdown('<div class="glass-card" style="text-align:center;">'
                    '<div style="color:#00d4ff; font-weight:700; font-size:1.1rem; margin-bottom:10px;">📋 Problem Setup</div>'
                    f'<div style="color:rgba(255,255,255,0.6);">{problem_type.upper()} Z with {num_vars} vars, {num_cons} constraints</div>'
                    '</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Objective function input
    st.markdown("#### Objective Function Coefficients")
    obj_cols = st.columns(num_vars)
    c_vals = []
    for i, col in enumerate(obj_cols):
        with col:
            val = st.number_input(f"c(x{i+1})", value=0.0, key=f"obj_{i}", step=1.0)
            c_vals.append(val)

    # Display objective in LaTeX
    obj_latex = format_objective_latex(c_vals, problem_type)
    st.latex(obj_latex)

    st.markdown("---")

    # Constraints input
    st.markdown("#### Constraints")
    A_vals = []
    b_vals = []
    ct_vals = []

    for i in range(num_cons):
        st.markdown(f"**Constraint {i+1}**")
        cols = st.columns(num_vars + 2)
        row = []
        for j in range(num_vars):
            with cols[j]:
                val = st.number_input(f"a{i+1}{j+1}", value=0.0, key=f"a_{i}_{j}",
                                      label_visibility="collapsed")
                row.append(val)
        A_vals.append(row)
        with cols[num_vars]:
            ct = st.selectbox("Type", ["<=", ">=", "="], key=f"ct_{i}", label_visibility="collapsed")
            ct_vals.append(ct)
        with cols[num_vars + 1]:
            rhs = st.number_input("RHS", value=0.0, key=f"b_{i}", label_visibility="collapsed")
            b_vals.append(rhs)

        # Show constraint in LaTeX
        st.latex(format_constraint_latex(A_vals[i], ct_vals[i], b_vals[i]))

    st.markdown("---")

    # Solve button
    if st.button("🚀 Solve with Big M Method", width="stretch", type="primary"):
        with st.spinner("Running Big M Method..."):
            result = big_m_method(c_vals, A_vals, b_vals, ct_vals, problem_type)
            st.session_state["result"] = result
            st.session_state["c"] = c_vals
            st.session_state["A"] = A_vals
            st.session_state["b"] = b_vals
            st.session_state["ct"] = ct_vals
            st.session_state["ptype"] = problem_type

        if result["status"] == "optimal":
            # Confetti
            confetti_html = ""
            import random
            colors = ["#00d4ff", "#ff006e", "#39ff14", "#ffbe0b", "#8338ec"]
            for ci in range(30):
                left = random.randint(0, 100)
                delay = random.random() * 2
                color = colors[ci % len(colors)]
                confetti_html += (
                    f'<div class="confetti-piece" style="left:{left}%;'
                    f'background:{color};animation-delay:{delay:.1f}s;'
                    f'border-radius:{random.choice(["50%","0"])}"></div>'
                )
            st.markdown(confetti_html, unsafe_allow_html=True)

            # Optimal banner
            sol = result["solution"]
            sol_parts = " , ".join([f"x{i+1} = {sol[i]:.4f}" for i in range(len(sol))])
            label = "Maximum Z" if problem_type == "max" else "Minimum Z"

            st.markdown(f"""
            <div class="optimal-banner">
                <div class="optimal-label">✅ {label}</div>
                <div class="optimal-z">{result["optimal_value"]:.4f}</div>
                <div style="color:rgba(255,255,255,0.6); margin-top:10px; font-family:'JetBrains Mono';">
                    {sol_parts}
                </div>
                <div style="color:rgba(255,255,255,0.4); margin-top:6px; font-size:0.8rem;">
                    Solved in {result["iterations"]} iterations
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif result["status"] == "unbounded":
            st.error("❌ **UNBOUNDED** — The problem has no finite optimal solution.")
        else:
            st.error("❌ **INFEASIBLE** — No feasible solution exists (artificial variable remains in basis).")

        # Show iteration tableaux
        if result.get("iteration_log"):
            st.markdown("---")
            st.markdown("#### 📋 Simplex Tableau Iterations")
            all_col_names = result["all_col_names"]

            show_all = st.checkbox("Show all iterations expanded", value=False)

            for entry in result["iteration_log"]:
                it = entry["iteration"]
                label = f"Iteration {it}" + (" (Initial)" if it == 0 else "")
                if show_all:
                    st.markdown(f"**{label}**")
                    st.markdown(render_tableau_html(entry, all_col_names), unsafe_allow_html=True)
                else:
                    with st.expander(label, expanded=(it == len(result["iteration_log"]) - 1)):
                        st.markdown(render_tableau_html(entry, all_col_names), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  TAB 2: VISUALIZATION
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 📊 Visualization Studio")

    if "result" not in st.session_state:
        st.info("💡 Solve a problem in the **Solver** tab first to see visualizations here.")
    else:
        result = st.session_state["result"]
        c_v = st.session_state["c"]
        A_v = st.session_state["A"]
        b_v = st.session_state["b"]
        ct_v = st.session_state["ct"]
        pt_v = st.session_state["ptype"]

        if len(c_v) != 2:
            st.warning("⚠️ Visualizations are only available for **2-variable** problems.")
        else:
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🗺️ Feasible Region", "🏔️ 3D Surface", "🛤️ Simplex Path"])

            with viz_tab1:
                fig = create_feasible_region_plot(c_v, A_v, b_v, ct_v, result, pt_v)
                if fig:
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

            with viz_tab2:
                fig3d = create_3d_surface_plot(c_v, A_v, b_v, ct_v, result, pt_v)
                if fig3d:
                    st.plotly_chart(fig3d, width="stretch", config={"displayModeBar": True})

            with viz_tab3:
                fig_path = create_simplex_path_plot(c_v, A_v, b_v, ct_v, result)
                if fig_path:
                    st.plotly_chart(fig_path, width="stretch", config={"displayModeBar": True})
                else:
                    st.info("Simplex path requires a valid 2-variable optimal solution.")


# ═══════════════════════════════════════════════════════════
#  TAB 3: SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 📈 Sensitivity & Parametric Analysis")

    if "result" not in st.session_state or st.session_state["result"]["status"] != "optimal":
        st.info("💡 Solve an optimal problem in the **Solver** tab first.")
    else:
        result = st.session_state["result"]
        c_v = st.session_state["c"]
        A_v = st.session_state["A"]
        b_v = st.session_state["b"]
        ct_v = st.session_state["ct"]
        pt_v = st.session_state["ptype"]

        sa = sensitivity_analysis(result)

        if sa:
            # Shadow prices
            st.markdown("#### 🌓 Shadow Prices (Dual Values)")
            sp_cols = st.columns(len(sa["shadow_prices"]))
            for i, (col, sp) in enumerate(zip(sp_cols, sa["shadow_prices"])):
                with col:
                    binding_txt = "🔒 Binding" if sa["binding_constraints"][i] else "🔓 Non-binding"
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-value">{sp:.4f}</div>
                        <div class="stat-label">Constraint {i+1}</div>
                        <div style="color:{'#39ff14' if sa['binding_constraints'][i] else 'rgba(255,255,255,0.4)'}; font-size:0.75rem; margin-top:4px;">{binding_txt}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # RHS Ranges
            st.markdown("#### 📏 Allowable RHS Ranges")
            rhs_html = '<div class="tableau-container"><table><thead><tr><th>Constraint</th><th>Current RHS</th><th>Lower Bound</th><th>Upper Bound</th></tr></thead><tbody>'
            for i, (low, high) in enumerate(sa["rhs_ranges"]):
                rhs_html += f'<tr><td>C{i+1}</td><td>{b_v[i]:.2f}</td><td>{low:.2f}</td><td>{high:.2f}</td></tr>'
            rhs_html += '</tbody></table></div>'
            st.markdown(rhs_html, unsafe_allow_html=True)

            st.markdown("---")

            # Parametric analysis
            st.markdown("#### 🎛️ What-If Parametric Analysis")
            param_type = st.radio("Vary:", ["RHS of a constraint", "Objective coefficient"], horizontal=True)

            if param_type == "RHS of a constraint":
                con_idx = st.selectbox("Select constraint", range(len(b_v)),
                                       format_func=lambda i: f"Constraint {i+1} (RHS = {b_v[i]})")
                rng = st.slider("Analysis range (%)", 10, 200, 50)
                if st.button("Run Parametric Analysis", key="param_rhs"):
                    with st.spinner("Computing..."):
                        data = parametric_rhs_analysis(c_v, A_v, b_v, ct_v, pt_v, con_idx, rng / 100.0)
                        fig_p = create_parametric_chart(data, f"b{con_idx+1} (RHS)", "Optimal Z",
                                                        f"Z vs RHS of Constraint {con_idx+1}", "#00d4ff")
                        if fig_p:
                            st.plotly_chart(fig_p, width="stretch")
            else:
                var_idx = st.selectbox("Select variable", range(len(c_v)),
                                       format_func=lambda i: f"x{i+1} (c = {c_v[i]})")
                rng = st.slider("Analysis range (%)", 10, 200, 50, key="param_obj_range")
                if st.button("Run Parametric Analysis", key="param_obj"):
                    with st.spinner("Computing..."):
                        data = parametric_obj_analysis(c_v, A_v, b_v, ct_v, pt_v, var_idx, rng / 100.0)
                        fig_p = create_parametric_chart(data, f"c(x{var_idx+1})", "Optimal Z",
                                                        f"Z vs Coefficient of x{var_idx+1}", "#ff006e")
                        if fig_p:
                            st.plotly_chart(fig_p, width="stretch")


# ═══════════════════════════════════════════════════════════
#  TAB 4: DUALITY
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🔄 Duality Analysis")

    if "result" not in st.session_state or st.session_state["result"]["status"] != "optimal":
        st.info("💡 Solve an optimal problem in the **Solver** tab first.")
    else:
        result = st.session_state["result"]
        c_v = st.session_state["c"]
        A_v = st.session_state["A"]
        b_v = st.session_state["b"]
        ct_v = st.session_state["ct"]
        pt_v = st.session_state["ptype"]

        dual = build_dual(c_v, A_v, b_v, ct_v, pt_v)

        # Side-by-side primal vs dual
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown('<div class="dual-panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="dual-title" style="color:#00d4ff;">PRIMAL ({pt_v.upper()})</div>', unsafe_allow_html=True)
            primal_lines = format_full_problem_latex(c_v, A_v, b_v, ct_v, pt_v)
            for line in primal_lines:
                st.latex(line)
            st.markdown('</div>', unsafe_allow_html=True)

        with d_col2:
            st.markdown('<div class="dual-panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="dual-title" style="color:#ff006e;">DUAL ({dual["dual_problem_type"].upper()})</div>', unsafe_allow_html=True)
            dual_lines = format_full_problem_latex(dual["dual_c"], dual["dual_A"], dual["dual_b"],
                                                    dual["dual_constraint_types"], dual["dual_problem_type"])
            for line in dual_lines:
                st.latex(line)
            # Dual variable signs
            sign_str = ", ".join([f"y{i+1} {dual['dual_var_signs'][i]}" for i in range(dual['num_dual_vars'])])
            st.markdown(f"**Variable signs:** {sign_str}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Solve dual and verify strong duality
        if st.button("🔬 Verify Strong Duality Theorem", width="stretch"):
            with st.spinner("Solving dual problem..."):
                dual_result = big_m_method(dual["dual_c"], dual["dual_A"], dual["dual_b"],
                                           dual["dual_constraint_types"], dual["dual_problem_type"])
                duality_check = verify_strong_duality(result, dual_result)

            if duality_check["holds"]:
                st.markdown(f"""
                <div class="optimal-banner">
                    <div class="optimal-label">✅ Strong Duality Theorem VERIFIED</div>
                    <div style="display:flex; justify-content:center; gap:40px; margin-top:15px;">
                        <div>
                            <div style="color:rgba(255,255,255,0.5); font-size:0.8rem;">Primal Z*</div>
                            <div style="color:#00d4ff; font-size:1.5rem; font-weight:800; font-family:'JetBrains Mono';">{duality_check['primal_z']:.4f}</div>
                        </div>
                        <div style="color:#39ff14; font-size:1.5rem; align-self:center;">=</div>
                        <div>
                            <div style="color:rgba(255,255,255,0.5); font-size:0.8rem;">Dual Z*</div>
                            <div style="color:#ff006e; font-size:1.5rem; font-weight:800; font-family:'JetBrains Mono';">{duality_check['dual_z']:.4f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Strong Duality does not hold. {duality_check['reason']}")

            # Complementary slackness
            cs = complementary_slackness(result, dual_result)
            if cs:
                st.markdown("#### ⚖️ Complementary Slackness Conditions")
                cs_html = '<div class="tableau-container"><table><thead><tr><th>Constraint</th><th>Slack</th><th>Dual Value</th><th>Tight?</th><th>Satisfied?</th></tr></thead><tbody>'
                for cond in cs:
                    ok = "✅" if cond["satisfied"] else "❌"
                    tight = "Yes" if cond["is_tight"] else "No"
                    cs_html += f'<tr><td>C{cond["constraint"]}</td><td>{cond["slack"]:.4f}</td><td>{cond["dual_value"]:.4f}</td><td>{tight}</td><td>{ok}</td></tr>'
                cs_html += '</tbody></table></div>'
                st.markdown(cs_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  TAB 5: LEARN
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 📚 Learn the Big M Method")

    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#00d4ff;">What is the Big M Method?</h4>
        <p>The <b>Big M Method</b> is a modification of the Simplex Algorithm designed to find an
        initial Basic Feasible Solution (BFS) when the LP problem contains <b>≥</b> or <b>=</b>
        constraints — situations where the origin isn't feasible.</p>
        <p>It works by introducing <span style="color:#ffbe0b;"><b>artificial variables</b></span>
        and penalizing them with a very large number <span style="color:#ff006e; font-family:'JetBrains Mono';"><b>M</b></span>
        in the objective function, ensuring they leave the basis on the way to optimality.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔑 Step-by-Step Algorithm")

    steps = [
        ("1️⃣ Standard Form Conversion",
         "• **≤ constraint**: Add slack variable (+s)\n"
         "• **≥ constraint**: Subtract surplus (−s) AND add artificial (+a)\n"
         "• **= constraint**: Add artificial (+a) only"),
        ("2️⃣ Modify Objective Function",
         "Penalize artificial variables:\n"
         "• **Maximization**: Z = ... **−M·aᵢ** for each artificial\n"
         "• **Minimization**: Z = ... **+M·aᵢ** for each artificial"),
        ("3️⃣ Set Up Initial Tableau",
         "Artificial variables form the initial basis. Perform row operations "
         "to eliminate basic variable coefficients from the Z-row."),
        ("4️⃣ Iterate (Simplex Pivots)",
         "• **Entering variable**: Most negative Z-row coefficient\n"
         "• **Leaving variable**: Minimum ratio test (θ = RHS ÷ pivot column)\n"
         "• **Pivot**: Normalize pivot row, eliminate pivot column elsewhere\n"
         "• **Repeat** until all Z-row values ≥ 0"),
        ("5️⃣ Check Optimality",
         "• **Optimal**: All Z-row ≥ 0, no artificials in basis\n"
         "• **Infeasible**: Artificial variable remains with value > 0\n"
         "• **Unbounded**: No valid pivot row (all ratios = ∞)"),
    ]

    for title, content in steps:
        with st.expander(title, expanded=False):
            st.markdown(content)

    st.markdown("---")
    st.markdown("#### ⚔️ Big M Method vs Two-Phase Method")

    st.markdown("""
    <div class="tableau-container">
    <table>
        <thead><tr><th>Feature</th><th style="color:#00d4ff;">Big M Method</th><th style="color:#ff006e;">Two-Phase Method</th></tr></thead>
        <tbody>
            <tr><td>Approach</td><td>Uses penalty constant M</td><td>Two separate optimization phases</td></tr>
            <tr><td>Numerical Stability</td><td>Can suffer from large M values</td><td>More numerically stable</td></tr>
            <tr><td>Simplicity</td><td>Simpler to implement</td><td>More complex (two phases)</td></tr>
            <tr><td>Infeasibility Detection</td><td>After full solve</td><td>After Phase I</td></tr>
            <tr><td>Computational Cost</td><td>Single pass but larger numbers</td><td>Two passes but cleaner</td></tr>
            <tr><td>Best For</td><td>Small-medium problems, teaching</td><td>Large-scale or ill-conditioned</td></tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📖 Key Terms")
    terms = {
        "Big M": "A very large positive number used as a penalty constant.",
        "Artificial Variable": "Dummy variable introduced to create an initial feasible basis.",
        "Slack Variable": "Added to ≤ constraints to convert them to equalities.",
        "Surplus Variable": "Subtracted from ≥ constraints to convert them to equalities.",
        "Pivot": "Row operation that brings a variable into the basis.",
        "BFS": "Basic Feasible Solution — a corner-point solution.",
        "Shadow Price": "Rate of change of Z per unit change in a constraint's RHS.",
        "Dual Problem": "A companion LP derived from the primal; provides bounds and insights.",
    }
    for term, defn in terms.items():
        st.markdown(f"**{term}** — {defn}")


# ═══════════════════════════════════════════════════════════
#  TAB 6: EXAMPLES
# ═══════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 🏆 Pre-Built Examples")
    st.markdown("Click any example to load and solve it instantly.")

    examples = [
        {
            "name": "📈 Maximize Z = 5x₁ + 4x₂",
            "desc": "Mixed ≤ and ≥ constraints",
            "c": [5, 4], "A": [[6, 4], [1, 2]], "b": [24, 6],
            "ct": ["<=", ">="], "ptype": "max",
        },
        {
            "name": "📉 Minimize Z = 2x₁ + 3x₂",
            "desc": "Equality and ≥ constraints",
            "c": [2, 3], "A": [[1, 1], [1, 3]], "b": [4, 6],
            "ct": ["=", ">="], "ptype": "min",
        },
        {
            "name": "📈 Maximize Z = 3x₁ + 5x₂",
            "desc": "All ≤ constraints (standard form)",
            "c": [3, 5], "A": [[1, 0], [0, 2], [3, 2]], "b": [4, 12, 18],
            "ct": ["<=", "<=", "<="], "ptype": "max",
        },
        {
            "name": "📉 Minimize Z = 4x₁ + x₂",
            "desc": "Three mixed constraints",
            "c": [4, 1], "A": [[3, 1], [4, 3], [1, 2]], "b": [3, 6, 4],
            "ct": [">=", ">=", "<="], "ptype": "min",
        },
    ]

    for idx, ex in enumerate(examples):
        with st.expander(f"{ex['name']} — {ex['desc']}", expanded=False):
            # Show problem
            obj_latex = format_objective_latex(ex["c"], ex["ptype"])
            st.latex(obj_latex)
            st.markdown("**Subject to:**")
            for i in range(len(ex["b"])):
                st.latex(format_constraint_latex(ex["A"][i], ex["ct"][i], ex["b"][i]))

            if st.button(f"🚀 Solve Example {idx+1}", key=f"ex_{idx}"):
                with st.spinner("Solving..."):
                    result = big_m_method(ex["c"], ex["A"], ex["b"], ex["ct"], ex["ptype"])
                    st.session_state["result"] = result
                    st.session_state["c"] = ex["c"]
                    st.session_state["A"] = ex["A"]
                    st.session_state["b"] = ex["b"]
                    st.session_state["ct"] = ex["ct"]
                    st.session_state["ptype"] = ex["ptype"]

                if result["status"] == "optimal":
                    sol = result["solution"]
                    sol_str = ", ".join([f"x{i+1} = {sol[i]:.4f}" for i in range(len(sol))])
                    st.markdown(f"""
                    <div class="optimal-banner">
                        <div class="optimal-label">✅ {"Maximum" if ex["ptype"]=="max" else "Minimum"} Z</div>
                        <div class="optimal-z">{result["optimal_value"]:.4f}</div>
                        <div style="color:rgba(255,255,255,0.6); margin-top:8px; font-family:'JetBrains Mono';">{sol_str}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Quick viz for 2-var problems
                    if len(ex["c"]) == 2:
                        fig = create_feasible_region_plot(ex["c"], ex["A"], ex["b"], ex["ct"], result, ex["ptype"])
                        if fig:
                            st.plotly_chart(fig, width="stretch")

                    st.success("✅ Solution stored — check **Visualization**, **Sensitivity**, and **Duality** tabs for more analysis!")
                else:
                    st.error(f"❌ {result['status'].upper()}")
