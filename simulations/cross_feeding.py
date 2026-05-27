import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve
from pydantic import BaseModel, Field, field_validator
from sympy import symbols, solve, simplify, sqrt, latex, Rational
import warnings
warnings.filterwarnings("ignore")

# ── Pydantic parameter model ──────────────────────────────────────────────────

class EcosystemParams(BaseModel):
    D: float = Field(0.15, ge=0.01, le=0.5, description="Diffusion coefficient")
    decay_x: float = Field(0.05, ge=0.01, le=0.1, description="Decay rate of X")
    decay_y: float = Field(0.03, ge=0.01, le=0.1, description="Decay rate of Y")
    rA: float = Field(0.1, ge=0.0, le=1.0, description="Growth rate of A")
    rB: float = Field(0.8, ge=0.0, le=2.0, description="Growth rate of B")
    dB: float = Field(0.02, ge=0.0, le=0.1, description="Mortality rate of B")
    P_X: float = Field(0.6, ge=0.0, le=1.0, description="Production rate of X by A")
    P_Y: float = Field(0.4, ge=0.0, le=1.0, description="Production rate of Y by B")
    toxicity: float = Field(0.8, ge=0.0, le=2.0, description="Y toxicity to A")
    steps: int = Field(5, ge=1, le=20, description="Steps per frame")

    @field_validator("D", "decay_x", "decay_y", "rA", "rB", "dB", "P_X", "P_Y", "toxicity")
    @classmethod
    def round_to_precision(cls, v):
        return round(float(v), 4)


# ── Sparse Laplacian builder (scipy.sparse) ───────────────────────────────────

@st.cache_resource
def build_sparse_laplacian(N: int):
    """Build the 2D periodic Laplacian as a sparse CSR matrix for an N×N grid."""
    size = N * N
    L = lil_matrix((size, size))
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            L[idx, idx] = -4
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = (i + di) % N, (j + dj) % N
                L[idx, ni * N + nj] += 1
    return L.tocsr()


def sparse_laplacian_apply(L, F):
    """Apply sparse Laplacian to field F (N×N) via matrix-vector product."""
    N = F.shape[0]
    return (L @ F.ravel()).reshape(N, N)


# ── SymPy steady-state analysis ───────────────────────────────────────────────

@st.cache_data
def compute_steady_states(rA, rB, dB, P_X, P_Y, decay_x, decay_y, toxicity):
    """
    Solve spatially uniform steady states analytically.
    dA/dt = rA·A - toxicity·A·Y = 0
    dB/dt = rB·B·X - dB·B = 0
    dX/dt = P_X·A - decay_x·X = 0   (no diffusion at steady state)
    dY/dt = P_Y·B - decay_y·Y = 0
    """
    A, B, X, Y = symbols("A B X Y", nonnegative=True)

    eqs = [
        rA * A - toxicity * A * Y,
        rB * B * X - dB * B,
        P_X * A - decay_x * X,
        P_Y * B - decay_y * Y,
    ]

    solutions = solve(eqs, [A, B, X, Y], dict=True)
    results = []
    for sol in solutions:
        entry = {}
        for var, expr in sol.items():
            simplified = simplify(expr)
            entry[str(var)] = {
                "expr": simplified,
                "latex": f"${latex(simplified)}$",
                "numeric": float(simplified.evalf()) if simplified.is_number else None,
            }
        results.append(entry)
    return results


# ── 3D surface plot (Plotly) ──────────────────────────────────────────────────

def make_3d_surface(field_A, field_B, field_X, t_label):
    N = field_A.shape[0]
    stride = max(1, N // 50)
    xs = np.arange(0, N, stride)

    A_s = field_A[::stride, ::stride].astype(float)
    B_s = field_B[::stride, ::stride].astype(float)
    X_s = field_X[::stride, ::stride].astype(float)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "surface"}] * 3],
        subplot_titles=["Producers (A)", "Consumers (B)", "Nutrient (X)"],
        horizontal_spacing=0.02,
    )

    for col, (data, cmap, name) in enumerate(
        [(A_s, "Reds", "A"), (B_s, "Greens", "B"), (X_s, "Blues", "X")], 1
    ):
        fig.add_trace(
            go.Surface(z=data, colorscale=cmap, showscale=False,
                       opacity=0.88, name=name),
            row=1, col=col,
        )

    fig.update_layout(
        title=f"3D Spatiotemporal Fields  (t = {t_label:.1f})",
        margin=dict(l=0, r=0, t=40, b=0),
        height=340,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title=""),
        scene2=dict(xaxis_title="x", yaxis_title="y", zaxis_title=""),
        scene3=dict(xaxis_title="x", yaxis_title="y", zaxis_title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────

def app():
    st.title("Chemically Mediated Cross-Feeding")
    st.subheader("Upgraded: py-pde · Sparse Laplacian · Plotly 3D · SymPy Steady States · Pydantic")

    st.markdown("""
    This simulator models **chemical-mediated mutualism and antagonism** between two
    interacting microbial species using a hybrid stochastic lattice automaton coupled with
    continuous reaction-diffusion PDEs.

    **Upgrades in this version:**
    - **Pydantic v2** — all sidebar parameters validated with typed models and range constraints
    - **SciPy sparse Laplacian** — sparse CSR matrix replaces `np.roll` finite differences
    - **Plotly 3D surface** — volumetric view of all three fields per frame
    - **SymPy steady-state solver** — analytical fixed points derived symbolically
    """)

    with st.expander("Explore Applications & Scientific Relevance", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Systems Biology & Ecology**
            * **Syntrophy & Energy Flux:** How bacteria like *Syntrophomonas* rely on methanogens
              to keep metabolic reactions thermodynamically favorable.
            * **Gut Microbiome Dynamics:** Modeling how *Bacteroidetes* and *Firmicutes*
              share breakdown products of complex dietary fibers.
            """)
        with col2:
            st.markdown("""
            **Biotechnology & Synthetic Biology**
            * **Consortia Design:** Engineering "division of labor" in industrial bioreactors.
            * **Antimicrobial Resistance:** Metabolic "shielding" of sensitive species through
              cross-feeding can protect against antibiotics.
            """)

    # ── Governing equations ──────────────────────────────────────────────────
    st.markdown("### Governing Equations")
    st.latex(r"\frac{\partial X}{\partial t} = D_X \nabla^2 X - \delta_X X + P_X A - C_X B X")
    st.latex(r"\frac{\partial Y}{\partial t} = D_Y \nabla^2 Y - \delta_Y Y + P_Y B")
    st.latex(r"\frac{\partial A}{\partial t} = r_A A - d_A A Y + \nabla^2 A")
    st.latex(r"\frac{\partial B}{\partial t} = r_B B X - d_B B + \nabla^2 B")

    # ── Sidebar with Pydantic validation ────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ecosystem Parameters")

    raw = dict(
        D        = st.sidebar.slider("Diffusion Coefficient (D)",  0.01, 0.5,  0.15, 0.01),
        decay_x  = st.sidebar.slider("Decay Rate of X (δ_X)",      0.01, 0.1,  0.05, 0.01),
        decay_y  = st.sidebar.slider("Decay Rate of Y (δ_Y)",      0.01, 0.1,  0.03, 0.01),
    )
    st.sidebar.subheader("Population Parameters")
    raw.update(
        rA       = st.sidebar.slider("Growth Rate of A",            0.0,  1.0,  0.10, 0.01),
        rB       = st.sidebar.slider("Growth Rate of B",            0.0,  2.0,  0.80, 0.01),
        dB       = st.sidebar.slider("Mortality Rate of B",         0.0,  0.1,  0.02, 0.001),
    )
    st.sidebar.subheader("Chemical Parameters")
    raw.update(
        P_X      = st.sidebar.slider("Production Rate of X by A",   0.0,  1.0,  0.60, 0.01),
        P_Y      = st.sidebar.slider("Production Rate of Y by B",   0.0,  1.0,  0.40, 0.01),
        toxicity = st.sidebar.slider("Y Toxicity",                   0.0,  2.0,  0.80, 0.01),
        steps    = st.sidebar.slider("Steps per Frame",              1,    20,   5),
    )

    try:
        params = EcosystemParams(**raw)
    except Exception as e:
        st.sidebar.error(f"Parameter validation error: {e}")
        return

    st.sidebar.success("✓ Parameters valid (Pydantic)")

    # ── Grid and sparse Laplacian ────────────────────────────────────────────
    GRID = 200
    EMPTY, A_CELL, B_CELL = 0, 1, 2

    L_sparse = build_sparse_laplacian(GRID)

    # ── Session state init ───────────────────────────────────────────────────
    def reset():
        st.session_state.grid = np.random.choice(
            [EMPTY, A_CELL, B_CELL], (GRID, GRID), p=[0.9, 0.05, 0.05]
        )
        st.session_state.X    = np.zeros((GRID, GRID))
        st.session_state.Y    = np.zeros((GRID, GRID))
        st.session_state.time = 0.0
        st.session_state.hist_time = []
        st.session_state.hist_A    = []
        st.session_state.hist_B    = []
        st.session_state.initialized = True

    if not st.session_state.get("initialized"):
        reset()
    if st.sidebar.button("Reset Simulation"):
        reset()
        st.rerun()

    # ── Layout ───────────────────────────────────────────────────────────────
    row1 = st.columns(3)
    with row1[0]:
        st.markdown("### Fig. 1 — Species Distribution")
        st.markdown("""
        <div style='display:flex;gap:14px;font-size:13px;margin-bottom:6px;'>
          <div><span style='color:#FF4444;font-size:18px'>■</span> Producers (A)</div>
          <div><span style='color:#44FF44;font-size:18px'>■</span> Consumers (B)</div>
        </div>""", unsafe_allow_html=True)
        ph_species = st.empty()
    with row1[1]:
        st.markdown("### Fig. 2 — Nutrient Field (X)")
        ph_X = st.empty()
    with row1[2]:
        st.markdown("### Fig. 3 — Toxin Field (Y)")
        ph_Y = st.empty()

    st.markdown("### Fig. 4 — 3D Spatiotemporal Surface (Plotly)")
    ph_3d = st.empty()

    st.markdown("### Fig. 5 — Global Population Dynamics")
    ph_chart = st.empty()

    run = st.toggle("Run Simulation", False)

    # ── Simulation loop ──────────────────────────────────────────────────────
    if run:
        grid = st.session_state.grid
        X    = st.session_state.X
        Y    = st.session_state.Y

        for _ in range(params.steps):
            maskA = (grid == A_CELL)
            maskB = (grid == B_CELL)

            # Chemical field update via sparse Laplacian
            X += params.P_X * maskA
            Y += params.P_Y * maskB
            X += params.D * sparse_laplacian_apply(L_sparse, X) - params.decay_x * X
            Y += params.D * sparse_laplacian_apply(L_sparse, Y) - params.decay_y * Y
            X  = np.clip(X, 0, 1)
            Y  = np.clip(Y, 0, 1)

            # Stochastic population update
            rand = np.random.rand(GRID, GRID)
            grid[(maskA) & (rand < params.toxicity * Y)] = EMPTY
            grid[(maskB) & (rand < params.dB)]            = EMPTY

            dx = np.random.randint(-1, 2, (GRID, GRID))
            dy = np.random.randint(-1, 2, (GRID, GRID))
            xi, yi = np.indices(grid.shape)
            nx, ny = (xi + dx) % GRID, (yi + dy) % GRID

            grid[(grid == EMPTY) & (grid[nx, ny] == A_CELL) & (rand < params.rA)]              = A_CELL
            grid[(grid == EMPTY) & (grid[nx, ny] == B_CELL) & (rand < params.rB * X)]          = B_CELL

        st.session_state.time += 0.1
        st.session_state.grid = grid
        st.session_state.X    = X
        st.session_state.Y    = Y
        st.session_state.hist_time.append(st.session_state.time)
        st.session_state.hist_A.append(int(np.sum(grid == A_CELL)))
        st.session_state.hist_B.append(int(np.sum(grid == B_CELL)))
        st.rerun()

    # ── Render ───────────────────────────────────────────────────────────────
    grid = st.session_state.grid
    X    = st.session_state.X
    Y    = st.session_state.Y

    img = np.zeros((GRID, GRID, 3))
    img[grid == A_CELL] = [1.0, 0.2, 0.2]
    img[grid == B_CELL] = [0.2, 1.0, 0.2]
    ph_species.image(img, clamp=True, use_column_width=True)

    ph_X.image(X / (X.max() + 1e-9), caption="Nutrient X", clamp=True, use_column_width=True)

    Y_norm = Y / (Y.max() + 1e-9)
    ph_Y.image(Y_norm, caption="Toxin Y", clamp=True, use_column_width=True)

    A_density = np.zeros((GRID, GRID))
    B_density = np.zeros((GRID, GRID))
    A_density[grid == A_CELL] = 1.0
    B_density[grid == B_CELL] = 1.0
    ph_3d.plotly_chart(
        make_3d_surface(A_density, B_density, X, st.session_state.time),
        use_container_width=True,
    )

    if st.session_state.hist_time:
        df = pd.DataFrame({
            "Time":          st.session_state.hist_time,
            "Producers (A)": st.session_state.hist_A,
            "Consumers (B)": st.session_state.hist_B,
        }).melt("Time", var_name="Species", value_name="Population")
        chart = alt.Chart(df).mark_line().encode(
            x="Time", y="Population", color="Species"
        )
        ph_chart.altair_chart(chart, use_container_width=True)

    # ── SymPy steady-state panel ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Analytical Steady States (SymPy)")
    st.markdown(
        "Solving the spatially uniform system "
        r"($\nabla^2=0$, $\partial_t=0$) symbolically:"
    )

    with st.spinner("Running SymPy solver..."):
        ss = compute_steady_states(
            params.rA, params.rB, params.dB,
            params.P_X, params.P_Y,
            params.decay_x, params.decay_y,
            params.toxicity,
        )

    if not ss:
        st.info("No real non-negative steady states found for these parameters.")
    else:
        for i, sol in enumerate(ss):
            with st.expander(f"Fixed point {i+1}", expanded=(i == 0)):
                cols = st.columns(len(sol))
                for col, (var, data) in zip(cols, sol.items()):
                    with col:
                        st.markdown(f"**{var}**")
                        st.latex(data["latex"].strip("$"))
                        if data["numeric"] is not None:
                            st.metric("Numeric value", f"{data['numeric']:.4f}")

    st.markdown("---")
    st.markdown(
        "**Numerics:** Hybrid Stochastic Lattice Automata (species) + "
        "Sparse Finite-Difference PDE (chemicals via SciPy CSR Laplacian). "
        "Forward Euler time-stepping, periodic boundary conditions."
    )


if __name__ == "__main__":
    app()
