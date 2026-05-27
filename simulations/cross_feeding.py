import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import lil_matrix
from pydantic import BaseModel, Field, field_validator
from sympy import symbols, solve, simplify, latex
import warnings
warnings.filterwarnings("ignore")

# ── Pydantic parameter model ──────────────────────────────────────────────────

class EcosystemParams(BaseModel):
    D: float = Field(0.15, ge=0.01, le=0.5)
    decay_x: float = Field(0.05, ge=0.01, le=0.1)
    decay_y: float = Field(0.03, ge=0.01, le=0.1)
    rA: float = Field(0.1, ge=0.0, le=1.0)
    rB: float = Field(0.8, ge=0.0, le=2.0)
    dB: float = Field(0.02, ge=0.0, le=0.1)
    P_X: float = Field(0.6, ge=0.0, le=1.0)
    P_Y: float = Field(0.4, ge=0.0, le=1.0)
    toxicity: float = Field(0.8, ge=0.0, le=2.0)
    steps: int = Field(5, ge=1, le=20)

    @field_validator("D", "decay_x", "decay_y", "rA", "rB", "dB", "P_X", "P_Y", "toxicity")
    @classmethod
    def round_to_precision(cls, v):
        return round(float(v), 4)


# ── Sparse Laplacian ──────────────────────────────────────────────────────────

@st.cache_resource
def build_sparse_laplacian(N: int):
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

def sparse_lap(L, F):
    N = F.shape[0]
    return (L @ F.ravel()).reshape(N, N)


# ── SymPy steady-state analysis ───────────────────────────────────────────────

@st.cache_data
def compute_steady_states(rA, rB, dB, P_X, P_Y, decay_x, decay_y, toxicity):
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
            s = simplify(expr)
            entry[str(var)] = {
                "latex": f"${latex(s)}$",
                "numeric": float(s.evalf()) if s.is_number else None,
            }
        results.append(entry)
    return results


def app():
    st.set_page_config(page_title="Microbial Cross-Feeding Simulator", layout="wide")
    st.title("Chemically Mediated Cross-Feeding")
    st.subheader("A Spatial Model for Two Mutualistic Species")

    # ---------------- DETAILED INTRODUCTORY TEXT ----------------
    st.markdown("""
    This simulator models **chemical-mediated mutualism and antagonism** between two
    interacting microbial species.
    In natural ecosystems, microbes rarely exist in isolation; they create complex **metabolic
    markets** where
    the byproduct of one organism becomes the primary energy source for another—a process
    known as **syntrophy**.
    However, these interactions are rarely purely beneficial. The spatial architecture of a microbial
    community
    (like a biofilm or soil crust) is defined by the tension between **cooperation (nutrient supply)**
    and
    **competition (toxic stress)**. As metabolites diffuse through the medium, they create
    "chemical landscapes"
    that determine where a species can thrive and where it will face extinction.
    """)

    # ---------------- APPLICATIONS & RESEARCH ----------------
    with st.expander("Explore Applications & Scientific Relevance", expanded=True):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            **Systems Biology & Ecology**
            * **Syntrophy & Energy Flux:** Studying how bacteria like *Syntrophomonas* rely on
            methanogens to consume hydrogen, keeping metabolic reactions thermodynamically favorable.
            * **Gut Microbiome Dynamics:** Modeling how different bacterial phyla (e.g.,
            *Bacteroidetes* and *Firmicutes*) share breakdown products of complex dietary fibers.
            * **Spatial Segregation:** Investigating how toxic gradients force species to cluster
            together for protection or separate into distinct territorial domains.
            """)
        with col_info2:
            st.markdown("""
            **Biotechnology & Synthetic Biology**
            * **Consortia Design:** Engineering "division of labor" in industrial bioreactors where
            multiple strains handle different steps of a complex fermentation process.
            * **Bioremediation:** Predicting how specialized microbes cooperate to break down
            environmental pollutants like crude oil or toxic hydrocarbons.
            * **Antimicrobial Resistance:** Understanding how cross-feeding can protect sensitive
            species from antibiotics through metabolic "shielding" or detoxifying waste.
            """)

    st.markdown("### Theoretical Framework")
    st.markdown("""
    The model utilizes a **hybrid approach**: a discrete stochastic lattice for the individuals
    (Species A and B)
    coupled with continuous partial differential equations (PDEs) for the chemical fields (Nutrient
    X and Toxin Y).
    """)

    # ---------------- GOVERNING EQUATIONS ----------------
    st.markdown("### Governing Equations")
    st.latex(r"""
    \frac{\partial X}{\partial t} = D_X \nabla^2 X - \delta_X X + P_X A - C_X B X
    """)
    st.latex(r"""
    \frac{\partial Y}{\partial t} = D_Y \nabla^2 Y - \delta_Y Y + P_Y B
    """)
    st.latex(r"""
    \frac{\partial A}{\partial t} = r_A A - d_A A Y + \nabla^2 A
    """)
    st.latex(r"""
    \frac{\partial B}{\partial t} = r_B B X - d_B B + \nabla^2 B
    """)
    st.latex(r"""
    \begin{aligned}
    A(x,y,t) &:\ \text{Producers (species)} \\
    B(x,y,t) &:\ \text{Consumers (species)} \\
    X(x,y,t) &:\ \text{Nutrient (produced by producers)} \\
    Y(x,y,t) &:\ \text{Toxin (produced by consumers)} \\
    \\
    D_X,\, D_Y &:\ \text{Diffusion coefficients} \\
    \delta_X,\, \delta_Y &:\ \text{Decay rates for X and Y} \\
    P_X,\, P_Y &:\ \text{Production rates of X and Y} \\
    C_X &:\ \text{Nutrient consumption rate by consumers} \\
    r_A,\, r_B &:\ \text{Growth rates of producers and consumers} \\
    d_A &:\ \text{Mortality rate of producers due to Y} \\
    d_B &:\ \text{Mortality rate of consumers (starvation)}
    \end{aligned}
    """)

    # ---------------- SIDEBAR ----------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ecosystem Parameters")
    D        = st.sidebar.slider("Diffusion Coefficient (D)", 0.01, 0.5, 0.15, step=0.01)
    decay_x  = st.sidebar.slider("Decay Rate of X (δ_X)", 0.01, 0.1, 0.05, step=0.01)
    decay_y  = st.sidebar.slider("Decay Rate of Y (δ_Y)", 0.01, 0.1, 0.03, step=0.01)
    st.sidebar.subheader("Population Parameters")
    rA       = st.sidebar.slider("Growth Rate of A", 0.0, 1.0, 0.1, step=0.01)
    rB       = st.sidebar.slider("Growth Rate of B", 0.0, 2.0, 0.8, step=0.01)
    dB       = st.sidebar.slider("Mortality Rate of B", 0.0, 0.1, 0.02, step=0.001)
    st.sidebar.subheader("Chemical Parameters")
    P_X      = st.sidebar.slider("Production Rate of X by A", 0.0, 1.0, 0.6, step=0.01)
    P_Y      = st.sidebar.slider("Production Rate of Y by B", 0.0, 1.0, 0.4, step=0.01)
    toxicity = st.sidebar.slider("Y Toxicity", 0.0, 2.0, 0.8, step=0.01)
    steps    = st.sidebar.slider("Steps per Frame", 1, 20, 5)

    # Pydantic validation (silent — only shows on error)
    try:
        params = EcosystemParams(
            D=D, decay_x=decay_x, decay_y=decay_y,
            rA=rA, rB=rB, dB=dB,
            P_X=P_X, P_Y=P_Y, toxicity=toxicity, steps=steps
        )
    except Exception as e:
        st.sidebar.error(f"Parameter error: {e}")
        return

    GRID = 200
    EMPTY, A, B = 0, 1, 2

    # Sparse Laplacian (cached)
    L_sparse = build_sparse_laplacian(GRID)

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    def reset():
        st.session_state.grid = np.random.choice([EMPTY, A, B], (GRID, GRID), p=[0.9, 0.05, 0.05])
        st.session_state.X = np.zeros((GRID, GRID))
        st.session_state.Y = np.zeros((GRID, GRID))
        st.session_state.time = 0.0
        st.session_state.hist_time = []
        st.session_state.hist_A = []
        st.session_state.hist_B = []
        st.session_state.hist_shannon = []
        st.session_state.initialized = True

    if not st.session_state.initialized:
        reset()

    # backfill keys added after initial deployment
    if "hist_shannon" not in st.session_state:
        st.session_state.hist_shannon = []

    if st.sidebar.button("Reset Simulation"):
        reset(); st.rerun()

    # ---------------- Layout ----------------
    row1 = st.columns(3)
    row2 = st.columns(1)

    with row1[0]:
        st.markdown("### Fig. 1 — Species Distribution")
        st.markdown("""
        <div style='display:flex;gap:14px;font-size:13px;margin-bottom:6px;'>
        <div><span style='color:#FF4444;font-size:18px'>■</span> Producers (A)</div>
        <div><span style='color:#44FF44;font-size:18px'>■</span> Consumers (B)</div>
        </div>
        """, unsafe_allow_html=True)
        ph_species = st.empty()
    with row1[1]:
        st.markdown("### Fig. 2 — Nutrient Field (X)")
        ph_X = st.empty()
    with row1[2]:
        st.markdown("### Fig. 3 — Poison Field (Y)")
        st.markdown("""
        <div style='display:flex;gap:14px;font-size:13px;margin-bottom:6px;'>
        <div><span style='color:#8888FF;font-size:18px'>☁</span> Poison Concentration</div>
        </div>
        """, unsafe_allow_html=True)
        ph_Y = st.empty()

    with row2[0]:
        st.markdown("### Fig. 4 — Global Population Dynamics")
        ph_chart = st.empty()

    # Fig. 5 — 3D surface
    st.markdown("### Fig. 5 — 3D Spatiotemporal Fields (Plotly)")
    ph_3d = st.empty()

    # New analytical figures
    row3 = st.columns(2)
    with row3[0]:
        st.markdown("### Fig. 6 — Phase Portrait (A vs B)")
        ph_phase = st.empty()
    with row3[1]:
        st.markdown("### Fig. 7 — Coexistence Index (Shannon Entropy)")
        ph_shannon = st.empty()

    row4 = st.columns(2)
    with row4[0]:
        st.markdown("### Fig. 8 — Chemical Gradient Magnitude")
        ph_grad = st.empty()
    with row4[1]:
        st.markdown("### Fig. 9 — Species Interface Map")
        ph_interface = st.empty()

    st.markdown("### Fig. 10 — Spatial Clustering Index")
    ph_cluster = st.empty()

    run = st.toggle("Run Simulation", False)

    if run:
        grid = st.session_state.grid
        X    = st.session_state.X
        Y    = st.session_state.Y

        for _ in range(params.steps):
            maskA = grid == A
            maskB = grid == B

            X += params.P_X * maskA
            Y += params.P_Y * maskB

            # Sparse Laplacian replaces np.roll finite differences
            X += params.D * sparse_lap(L_sparse, X) - params.decay_x * X
            Y += params.D * sparse_lap(L_sparse, Y) - params.decay_y * Y

            X = np.clip(X, 0, 1)
            Y = np.clip(Y, 0, 1)

            rand = np.random.rand(GRID, GRID)
            grid[(maskA) & (rand < params.toxicity * Y)] = EMPTY
            grid[(maskB) & (rand < params.dB)] = EMPTY

            dx = np.random.randint(-1, 2, (GRID, GRID))
            dy = np.random.randint(-1, 2, (GRID, GRID))
            x, y = np.indices(grid.shape)
            nx, ny = (x + dx) % GRID, (y + dy) % GRID
            grid[(grid == EMPTY) & (grid[nx, ny] == A) & (rand < params.rA)] = A
            grid[(grid == EMPTY) & (grid[nx, ny] == B) & (rand < params.rB * X)] = B

        st.session_state.time += 0.1
        st.session_state.grid = grid
        st.session_state.X    = X
        st.session_state.Y    = Y
        st.session_state.hist_time.append(st.session_state.time)
        st.session_state.hist_A.append(np.sum(grid == A))
        st.session_state.hist_B.append(np.sum(grid == B))
        total = GRID * GRID
        counts = np.array([np.sum(grid == EMPTY), np.sum(grid == A), np.sum(grid == B)], dtype=float)
        probs  = counts / total
        probs  = probs[probs > 0]
        st.session_state.hist_shannon.append(float(-np.sum(probs * np.log(probs))))
        st.rerun()

    # ---------------- Render ----------------
    grid = st.session_state.grid
    X    = st.session_state.X
    Y    = st.session_state.Y

    img = np.zeros((GRID, GRID, 3))
    img[grid == A] = [1.0, 0.2, 0.2]
    img[grid == B] = [0.2, 1.0, 0.2]
    ph_species.image(img, clamp=True, use_column_width=True)

    ph_X.image(X / (X.max() + 1e-9), caption="Nutrient X", clamp=True, use_column_width=True)

    Y_norm = Y / (Y.max() + 1e-9)
    ph_Y.image(Y_norm, caption="Poison Y", clamp=True, use_column_width=True)

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

        # Fig. 6 — Phase portrait
        df_phase = pd.DataFrame({
            "Producers (A)": st.session_state.hist_A,
            "Consumers (B)": st.session_state.hist_B,
            "Time":          st.session_state.hist_time,
        })
        phase_line = alt.Chart(df_phase).mark_line(
            color="#aaaaaa", opacity=0.4, strokeWidth=1
        ).encode(
            x=alt.X("Producers (A):Q", title="Producers (A)"),
            y=alt.Y("Consumers (B):Q", title="Consumers (B)"),
        )
        phase_dots = alt.Chart(df_phase).mark_circle(size=20).encode(
            x=alt.X("Producers (A):Q"),
            y=alt.Y("Consumers (B):Q"),
            color=alt.Color("Time:Q", scale=alt.Scale(scheme="viridis"),
                            legend=alt.Legend(title="Time")),
            tooltip=["Time", "Producers (A)", "Consumers (B)"],
        )
        ph_phase.altair_chart((phase_line + phase_dots).interactive(),
                              use_container_width=True)

        # Fig. 7 — Shannon entropy
        df_shannon = pd.DataFrame({
            "Time":    st.session_state.hist_time,
            "Entropy": st.session_state.hist_shannon,
        })
        shannon_chart = alt.Chart(df_shannon).mark_line(color="#9467bd").encode(
            x="Time", y=alt.Y("Entropy", title="Shannon Entropy (nats)"),
        )
        ph_shannon.altair_chart(shannon_chart, use_container_width=True)

    # 3D surface (Plotly) — downsampled for performance
    stride = max(1, GRID // 50)
    A_dens = (grid == A).astype(float)
    B_dens = (grid == B).astype(float)
    fig3d = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "surface"}] * 3],
        subplot_titles=["Producers (A)", "Consumers (B)", "Nutrient (X)"],
        horizontal_spacing=0.02,
    )
    for col, (data, cmap) in enumerate(
        [(A_dens[::stride, ::stride], "Reds"),
         (B_dens[::stride, ::stride], "Greens"),
         (X[::stride, ::stride],      "Blues")], 1
    ):
        fig3d.add_trace(go.Surface(z=data, colorscale=cmap, showscale=False, opacity=0.88), row=1, col=col)
    fig3d.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    ph_3d.plotly_chart(fig3d, use_container_width=True)

    # Fig. 8 — Chemical gradient magnitude |∇X| and |∇Y|
    gx_X = np.gradient(X, axis=1)
    gy_X = np.gradient(X, axis=0)
    gx_Y = np.gradient(Y, axis=1)
    gy_Y = np.gradient(Y, axis=0)
    grad_X = np.sqrt(gx_X**2 + gy_X**2)
    grad_Y = np.sqrt(gx_Y**2 + gy_Y**2)
    grad_img = np.zeros((GRID, GRID, 3))
    grad_img[..., 0] = grad_Y / (grad_Y.max() + 1e-9)   # red = toxin gradient
    grad_img[..., 2] = grad_X / (grad_X.max() + 1e-9)   # blue = nutrient gradient
    ph_grad.image(grad_img, clamp=True, use_column_width=True,
                  caption="Red = |∇Y| (toxin front)  |  Blue = |∇X| (nutrient front)")

    # Fig. 9 — Species interface map (cells of A neighbouring B and vice versa)
    def has_neighbour(g, state):
        return (
            (np.roll(g, 1, 0) == state) | (np.roll(g, -1, 0) == state) |
            (np.roll(g, 1, 1) == state) | (np.roll(g, -1, 1) == state)
        )
    interface_AB = (grid == A) & has_neighbour(grid, B)   # A cells touching B
    interface_BA = (grid == B) & has_neighbour(grid, A)   # B cells touching A
    iface_img = np.zeros((GRID, GRID, 3))
    iface_img[interface_AB] = [1.0, 0.8, 0.0]   # gold = A at interface
    iface_img[interface_BA] = [0.0, 0.8, 1.0]   # cyan = B at interface
    ph_interface.image(iface_img, clamp=True, use_column_width=True,
                       caption="Gold = A cells at contact zone  |  Cyan = B cells at contact zone")

    # Fig. 10 — Spatial clustering index (Moran's I proxy via autocorrelation)
    if st.session_state.hist_time:
        def clustering_index(g, state):
            mask = (g == state).astype(float)
            mean = mask.mean()
            if mean == 0 or mean == 1:
                return 0.0
            shifted = (
                np.roll(mask, 1, 0) + np.roll(mask, -1, 0) +
                np.roll(mask, 1, 1) + np.roll(mask, -1, 1)
            ) / 4.0
            cov = np.mean((mask - mean) * (shifted - mean))
            var = np.var(mask)
            return float(cov / (var + 1e-12))

        ci_A = clustering_index(grid, A)
        ci_B = clustering_index(grid, B)
        df_cluster = pd.DataFrame({
            "Species": ["Producers (A)", "Consumers (B)"],
            "Clustering Index (Moran's I)": [ci_A, ci_B],
        })
        cluster_chart = alt.Chart(df_cluster).mark_bar().encode(
            x=alt.X("Species", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Clustering Index (Moran's I)", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("Species", scale=alt.Scale(
                domain=["Producers (A)", "Consumers (B)"],
                range=["#FF4444", "#44FF44"]
            )),
        )
        ph_cluster.altair_chart(cluster_chart, use_container_width=True)

    # ---------------- Analytical Steady States (SymPy) ----------------
    st.markdown("---")
    st.markdown("### Analytical Steady States (SymPy)")
    st.markdown(r"Spatially uniform fixed points ($\nabla^2=0,\ \partial_t=0$):")

    with st.spinner("Solving..."):
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
            with st.expander(f"Fixed point {i + 1}", expanded=(i == 0)):
                cols = st.columns(len(sol))
                for col, (var, data) in zip(cols, sol.items()):
                    with col:
                        st.markdown(f"**{var}**")
                        st.latex(data["latex"].strip("$"))
                        if data["numeric"] is not None:
                            st.metric("Numeric", f"{data['numeric']:.4f}")

    st.markdown("---")
    st.markdown("""
    **Numerics:** Hybrid Stochastic Lattice Automata (for species) and Finite-Difference Method
    (for chemicals).
    Supports Neumann boundary conditions (periodic wrapping) and Forward Euler
    time-stepping.
    """)


if __name__ == "__main__":
    app()
