import time
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()


# ================= phase map helper =================
def run_coarse(nu, p, steps=120, size=60):
    CENTER = size // 2
    RADIUS = size // 2 - 4
    MAX_RES = 3

    A = np.ones((size, size)) * 99
    for r in range(size):
        for c in range(size):
            d = np.sqrt((r - CENTER) ** 2 + (c - CENTER) ** 2)
            if d > RADIUS:
                A[r, c] = 99
            elif d < RADIUS / 3:
                A[r, c] = 0
            elif d < 2 * RADIUS / 3:
                A[r, c] = 1
            else:
                A[r, c] = 2

    B = np.zeros((size, size), int)
    B[CENTER - 1:CENTER + 2, CENTER - 1:CENTER + 2] = 1

    for _ in range(steps):
        new = B.copy()
        rows, cols = np.where(B > 0)
        for r, c in zip(rows, cols):
            i = B[r, c]
            if (i - 1) < A[r, c]:
                new[r, c] = 0
                continue
            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= nr < size and 0 <= nc < size:
                    if B[nr, nc] == 0 and A[nr, nc] != 99 and np.random.rand() < p:
                        child = i
                        if np.random.rand() < nu:
                            child = min(MAX_RES, child + 1)
                        if (child - 1) >= A[nr, nc]:
                            new[nr, nc] = child
        B[:] = new

    tot = np.sum(B > 0)
    if tot == 0:
        return 0
    frac3 = np.sum(B == 3) / tot
    if frac3 > 0.7:
        return 2
    return 1


# ================= main app =================
def app():
    st.set_page_config(page_title="MEGA Plate Evolution", layout="wide")
    st.title("The MEGA Plate Experiment")
    st.subheader("A Spatial Reaction–Selection–Mutation Model")

    st.markdown("""
    This simulator models **stepwise antibiotic resistance evolution**
    across spatial drug gradients using discrete stochastic dynamics. It is inspired by the Harvard
    Medical School "MEGA-plate" (Microbial Evolution and Growth Arena), which demonstrates how
    bacteria migrate into increasing concentrations of antibiotics through successive mutations.
    """)

    with st.expander("Explore Applications & Scientific Relevance", expanded=True):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            **Evolutionary Biology**
            * **Fitness Landscapes:** Visualizing how spatial heterogeneity provides "stepping stones" for bacteria to reach high-fitness peaks.
            * **Clonal Interference:** Observing how different mutant lineages compete spatially for the same resources.
            * **Adaptive Radiation:** Studying how populations diversify as they encounter new environmental stressors.
            """)
        with col_info2:
            st.markdown("""
            **Public Health & Pharmacology**
            * **Antibiotic Stewardship:** Understanding why low-dose "sub-inhibitory" concentrations accelerate the emergence of superbugs.
            * **Drug Design:** Simulating multi-drug gradients to test "collateral sensitivity," where resistance to one drug makes bacteria weaker against another.
            * **Infection Control:** Modeling how spatial barriers in the body (like tissue density) affect bacterial spread.
            """)

    # ---------------- Mathematical model ----------------
    st.markdown("## Mathematical Model")
    st.latex(r"B_i(x,y,t)\in\{0,1\},\quad i\in\{1,2,3\}")
    st.latex(r"A(x,y)\in\{0,1,2\}")
    st.latex(r"""
    B_i(x,y,t+1)=
    \begin{cases}
    0, & i-1 < A(x,y),\\
    B_i(x,y,t), & \text{otherwise}.
    \end{cases}
    """)
    st.latex(r"\Pr(B_i(x\to x',t+1)=1)=p")
    st.latex(r"\Pr(i\to i+1)=\nu")
    st.latex(r"i-1\ge A(x')")
    st.latex(r"""
    \begin{aligned}
    B_i &: \text{bacteria with resistance level } i\\
    A(x) &: \text{local antibiotic field}\\
    p &: \text{reproduction probability}\\
    \nu &: \text{mutation probability}
    \end{aligned}
    """)

    # ---------------- Sidebar ----------------
    st.sidebar.subheader("Evolution Parameters")
    NU = st.sidebar.slider("Mutation Rate (ν)", 0.0, 0.1, 0.01)
    P  = st.sidebar.slider("Growth Prob (p)",   0.0, 1.0, 0.2)

    st.sidebar.subheader("System Settings")
    SIZE           = 100
    CENTER         = SIZE // 2
    RADIUS         = 45
    MAX_RES_LEVEL  = 3
    STEPS_PER_FRAME = st.sidebar.slider("Speed", 1, 10, 2)

    # throttle constants — same pattern as growth sim
    CHARTS_EVERY  = 20   # redraw Altair charts every N frames
    PHASE_EVERY   = 50   # redraw expensive phase map every N frames

    # ---------------- Init helpers ----------------
    def build_ab_map():
        ab_map = np.ones((SIZE, SIZE)) * 99
        for r in range(SIZE):
            for c in range(SIZE):
                dist = np.sqrt((r - CENTER) ** 2 + (c - CENTER) ** 2)
                if dist > RADIUS:
                    ab_map[r, c] = 99
                elif dist < RADIUS / 3:
                    ab_map[r, c] = 0
                elif dist < 2 * RADIUS / 3:
                    ab_map[r, c] = 1
                else:
                    ab_map[r, c] = 2
        return ab_map

    def init_state():
        ab_map   = build_ab_map()
        bac_grid = np.zeros((SIZE, SIZE), dtype=int)
        bac_grid[CENTER - 1:CENTER + 2, CENTER - 1:CENTER + 2] = 1
        return ab_map, bac_grid

    # ---------------- Session state ----------------
    if "mp_initialized" not in st.session_state:
        ab_map, bac_grid = init_state()
        st.session_state.mp_ab_map        = ab_map
        st.session_state.mp_grid          = bac_grid
        st.session_state.mp_time          = 0.0
        st.session_state.mp_frame         = 0
        st.session_state.mp_hist_time     = []
        st.session_state.mp_hist_total    = []
        st.session_state.mp_hist_res      = []
        st.session_state.mp_phase_cache   = None
        st.session_state.mp_initialized   = True

    if st.sidebar.button("Reset Simulation"):
        ab_map, bac_grid = init_state()
        st.session_state.mp_ab_map        = ab_map
        st.session_state.mp_grid          = bac_grid
        st.session_state.mp_time          = 0.0
        st.session_state.mp_frame         = 0
        st.session_state.mp_hist_time     = []
        st.session_state.mp_hist_total    = []
        st.session_state.mp_hist_res      = []
        st.session_state.mp_phase_cache   = None

    # ---------------- Layout ----------------
    col_vis, col_stats = st.columns([1, 1])

    with col_vis:
        st.markdown("### Figure 1 — Spatial Evolution")
        ph_plate = st.empty()

    with col_stats:
        st.markdown("### Figure 2 — Population Size")
        ph_total = st.empty()
        st.markdown("### Figure 3 — Genotype Frequencies")
        ph_freq  = st.empty()
        st.markdown("### Figure 4 — Phase Map (ν vs p)")
        ph_phase = st.empty()

    run = st.toggle("Run Simulation", value=False)
    st.markdown("**Species Legend:** 🔴 Superbug (R3) | 🟢 Mutant (R2) | 🔵 Wildtype (R1)")

    # ---------------- Render function ----------------
    def render(bac_grid, ab_map, t, frame):
        # Figure 1: spatial plate (every frame)
        img = np.zeros((SIZE, SIZE, 3))
        img[ab_map == 0] = [0.18, 0.18, 0.18]
        img[ab_map == 1] = [0.38, 0.38, 0.38]
        img[ab_map == 2] = [0.58, 0.58, 0.58]
        img[bac_grid == 1] = [0.0, 0.7, 1.0]   # blue  – wildtype
        img[bac_grid == 2] = [0.2, 1.0, 0.2]   # green – mutant
        img[bac_grid == 3] = [1.0, 0.2, 0.2]   # red   – superbug
        ph_plate.image(img, caption=f"Time: {t:.1f} h", use_column_width=True)

        # Figures 2, 3, 4: charts (throttled)
        if frame % CHARTS_EVERY == 0 and st.session_state.mp_hist_time:
            df_total = pd.DataFrame({
                "Time":       st.session_state.mp_hist_time,
                "Population": st.session_state.mp_hist_total,
            })
            ph_total.altair_chart(
                alt.Chart(df_total).mark_line(color="#4fc3f7")
                .encode(x="Time:Q", y="Population:Q",
                        tooltip=["Time", "Population"])
                .interactive(),
                use_container_width=True,
            )

            hist = np.array(st.session_state.mp_hist_res)
            df_frac = pd.DataFrame({
                "Time":     st.session_state.mp_hist_time,
                "Wildtype": hist[:, 0],
                "Mutant":   hist[:, 1],
                "Superbug": hist[:, 2],
            }).melt("Time", var_name="Genotype", value_name="Frequency")
            ph_freq.altair_chart(
                alt.Chart(df_frac).mark_line()
                .encode(x="Time:Q", y="Frequency:Q",
                        color=alt.Color("Genotype:N",
                                        scale=alt.Scale(
                                            domain=["Wildtype", "Mutant", "Superbug"],
                                            range=["#4fc3f7", "#69f0ae", "#ef5350"])),
                        tooltip=["Time", "Genotype", "Frequency"])
                .interactive(),
                use_container_width=True,
            )

        # Phase map (expensive — throttled more aggressively)
        if frame % PHASE_EVERY == 0:
            if st.session_state.mp_phase_cache is None or frame % PHASE_EVERY == 0:
                nus   = np.linspace(0, 0.1, 10)
                ps    = np.linspace(0, 1.0, 10)
                phase = [[nu, p, run_coarse(nu, p)] for nu in nus for p in ps]
                st.session_state.mp_phase_cache = phase
            df_phase = pd.DataFrame(
                st.session_state.mp_phase_cache, columns=["nu", "p", "state"]
            )
            ph_phase.altair_chart(
                alt.Chart(df_phase).mark_rect().encode(
                    x=alt.X("nu:Q", bin=alt.Bin(maxbins=10), title="Mutation rate (ν)"),
                    y=alt.Y("p:Q",  bin=alt.Bin(maxbins=10), title="Growth probability (p)"),
                    color=alt.Color("state:N",
                                    scale=alt.Scale(domain=[0, 1, 2],
                                                    range=["black", "orange", "red"]),
                                    legend=None),
                ).properties(height=220),
                use_container_width=True,
            )

    # ---------------- Static render when paused ----------------
    render(
        st.session_state.mp_grid,
        st.session_state.mp_ab_map,
        st.session_state.mp_time,
        frame=0,   # force all panels to draw on load
    )

    # ---------------- Live simulation loop ----------------
    if run:
        bacteria_grid  = st.session_state.mp_grid.copy()
        antibiotic_map = st.session_state.mp_ab_map
        t              = st.session_state.mp_time
        frame          = st.session_state.mp_frame
        hist_time      = list(st.session_state.mp_hist_time)
        hist_total     = list(st.session_state.mp_hist_total)
        hist_res       = list(st.session_state.mp_hist_res)

        while True:
            t0 = time.perf_counter()

            # --- physics (vectorised inner loop) ---
            for _ in range(STEPS_PER_FRAME):
                new_grid = bacteria_grid.copy()
                rows, cols = np.where(bacteria_grid > 0)

                for r, c in zip(rows, cols):
                    res = bacteria_grid[r, c]
                    if (res - 1) < antibiotic_map[r, c]:
                        new_grid[r, c] = 0
                        continue
                    for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                        if 0 <= nr < SIZE and 0 <= nc < SIZE:
                            if bacteria_grid[nr, nc] == 0 and antibiotic_map[nr, nc] != 99:
                                if np.random.rand() < P:
                                    child = res
                                    if np.random.rand() < NU:
                                        child = min(MAX_RES_LEVEL, child + 1)
                                    if (child - 1) >= antibiotic_map[nr, nc]:
                                        new_grid[nr, nc] = child

                bacteria_grid[:] = new_grid
                t += 0.1

            # --- record ---
            frame += 1
            total = int(np.sum(bacteria_grid > 0))
            c1    = int(np.sum(bacteria_grid == 1))
            c2    = int(np.sum(bacteria_grid == 2))
            c3    = int(np.sum(bacteria_grid == 3))
            hist_time.append(t)
            hist_total.append(total)
            hist_res.append([
                c1 / total if total else 0,
                c2 / total if total else 0,
                c3 / total if total else 0,
            ])

            # write history to session state so render() can read it
            st.session_state.mp_hist_time  = hist_time
            st.session_state.mp_hist_total = hist_total
            st.session_state.mp_hist_res   = hist_res

            # --- render ---
            render(bacteria_grid, antibiotic_map, t, frame)

            # --- persist ---
            st.session_state.mp_grid  = bacteria_grid.copy()
            st.session_state.mp_time  = t
            st.session_state.mp_frame = frame

            # --- pace to ~20 fps ---
            elapsed = time.perf_counter() - t0
            wait    = (1 / 20) - elapsed
            if wait > 0:
                time.sleep(wait)

    st.markdown("---")
    st.markdown(
        "**Numerics:** Discrete-time stochastic cellular automata. "
        "Antibiotic zones defined by radial thresholding. "
        "Forward frame loop with throttled chart redraws."
    )


if __name__ == "__main__":
    app()
