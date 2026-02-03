import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# ---------------- PHASE MAP UTILITY ----------------
def run_coarse(mu, p, steps=120, size=60):
    CENTER = size//2
    RADIUS = size//2 - 4
    MAX_RES = 3

    A = np.ones((size,size))*99
    for r in range(size):
        for c in range(size):
            d = np.sqrt((r-CENTER)**2+(c-CENTER)**2)
            if d>RADIUS: A[r,c]=99
            elif d<RADIUS/3: A[r,c]=0
            elif d<2*RADIUS/3: A[r,c]=1
            else: A[r,c]=2

    B = np.zeros((size,size),int)
    B[CENTER-1:CENTER+2, CENTER-1:CENTER+2] = 1

    for _ in range(steps):
        new = B.copy()
        rows, cols = np.where(B>0)
        for r,c in zip(rows,cols):
            i = B[r,c]
            if (i-1)<A[r,c]:
                new[r,c]=0; continue
            for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0<=nr<size and 0<=nc<size:
                    if B[nr,nc]==0 and A[nr,nc]!=99 and np.random.rand()<p:
                        child=i
                        if np.random.rand()<mu:
                            child=min(MAX_RES,child+1)
                        if (child-1)>=A[nr,nc]:
                            new[nr,nc]=child
        B[:]=new

    tot=np.sum(B>0)
    if tot==0: return 0
    frac3=np.sum(B==3)/tot
    if frac3>0.7: return 2
    return 1

# ---------------- APP ----------------
def app():
    st.title("The MEGA Plate Experiment")
    st.subheader("Spatial Reaction–Selection–Mutation Model")

    st.markdown("""
    Discrete stochastic model of **stepwise resistance evolution**
    across spatial antibiotic gradients.
    """)

    # ---------------- THEORY ----------------
    st.markdown("### Governing Model")
    st.latex(r"B_i(x,y,t)\in\{0,1\},\ i\in\{1,2,3\}")
    st.latex(r"A(x,y)\in\{0,1,2\}")
    st.latex(r"""
    B_i(x,y,t+1)=
    \begin{cases}
    0, & i-1<A(x,y)\\
    B_i(x,y,t), & \text{otherwise}
    \end{cases}
    """)
    st.latex(r"\Pr(B_i(x\to x',t+1)=1)=p")
    st.latex(r"\Pr(i\to i+1)=\mu")
    st.latex(r"i-1\ge A(x')")

    # ---------------- SIDEBAR ----------------
    st.sidebar.subheader("Evolution Parameters")
    MUTATION_RATE = st.sidebar.slider("Mutation Rate (μ)", 0.0, 0.1, 0.01, format="%.3f")
    REGROW_PROB = st.sidebar.slider("Growth Prob (p)", 0.0, 1.0, 0.2)

    st.sidebar.subheader("System Settings")
    SIZE = 100
    CENTER = SIZE // 2
    RADIUS = 45
    MAX_RES_LEVEL = 3
    STEPS_PER_FRAME = st.sidebar.slider("Speed", 1, 10, 2)

    if "mp_grid" not in st.session_state:
        st.session_state.mp_initialized = False

    def reset_simulation():
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

        bac_grid = np.zeros((SIZE, SIZE), dtype=int)
        bac_grid[CENTER-1:CENTER+2, CENTER-1:CENTER+2] = 1

        st.session_state.mp_ab_map = ab_map
        st.session_state.mp_grid = bac_grid
        st.session_state.mp_time = 0.0
        st.session_state.mp_hist_time = []
        st.session_state.mp_hist_total = []
        st.session_state.mp_hist_res = []
        st.session_state.mp_initialized = True

    if not st.session_state.mp_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    col_vis, col_stats = st.columns([1, 1])

    with col_vis:
        st.markdown("### Figure 1 — Spatial Evolution")
        plate_placeholder = st.empty()

    with col_stats:
        st.markdown("### Figure 2 — Population Dynamics")
        chart_total = st.empty()
        chart_frac = st.empty()
        st.markdown("#### μ–p Phase Diagram")
        phase_placeholder = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    # ---------------- SIMULATION ----------------
    if run_sim:
        bacteria_grid = st.session_state.mp_grid
        antibiotic_map = st.session_state.mp_ab_map

        for _ in range(STEPS_PER_FRAME):
            new_grid = bacteria_grid.copy()
            rows, cols = np.where(bacteria_grid > 0)
            idx = np.arange(len(rows))
            np.random.shuffle(idx)

            for i in idx:
                r, c = rows[i], cols[i]
                res_level = bacteria_grid[r, c]

                if (res_level - 1) < antibiotic_map[r, c]:
                    new_grid[r, c] = 0
                    continue

                neighbors = [(r - 1, c), (r + 1, c),
                             (r, c - 1), (r, c + 1)]
                np.random.shuffle(neighbors)

                for nr, nc in neighbors:
                    if 0 <= nr < SIZE and 0 <= nc < SIZE:
                        if bacteria_grid[nr, nc] == 0 and antibiotic_map[nr, nc] != 99:
                            if np.random.random() < REGROW_PROB:
                                child = res_level
                                if np.random.random() < MUTATION_RATE:
                                    child = min(MAX_RES_LEVEL, child + 1)
                                if (child - 1) >= antibiotic_map[nr, nc]:
                                    new_grid[nr, nc] = child

            bacteria_grid[:] = new_grid
            st.session_state.mp_time += 0.1

            total = np.sum(bacteria_grid > 0)
            c1 = np.sum(bacteria_grid == 1)
            c2 = np.sum(bacteria_grid == 2)
            c3 = np.sum(bacteria_grid == 3)

            st.session_state.mp_hist_time.append(st.session_state.mp_time)
            st.session_state.mp_hist_total.append(total)

            if total > 0:
                st.session_state.mp_hist_res.append([c1 / total, c2 / total, c3 / total])
            else:
                st.session_state.mp_hist_res.append([0, 0, 0])

        st.session_state.mp_grid = bacteria_grid
        st.rerun()

    # ---------------- VISUAL ----------------
    bac_grid = st.session_state.mp_grid
    ab_map = st.session_state.mp_ab_map

    img = np.zeros((SIZE, SIZE, 3))
    img[ab_map == 0] = [0.2, 0.2, 0.2]
    img[ab_map == 1] = [0.4, 0.4, 0.4]
    img[ab_map == 2] = [0.6, 0.6, 0.6]

    img[bac_grid == 1] = [0, 1, 1]
    img[bac_grid == 2] = [0, 1, 0]
    img[bac_grid == 3] = [1, 0, 0]

    plate_placeholder.image(img, caption=f"Time: {st.session_state.mp_time:.1f} h", use_column_width=True)

    # ---------------- PLOTS ----------------
    if st.session_state.mp_hist_time:
        df_total = pd.DataFrame({"Time": st.session_state.mp_hist_time,
                                 "Population": st.session_state.mp_hist_total})
        chart_total.altair_chart(
            alt.Chart(df_total).mark_line().encode(x="Time", y="Population"),
            use_container_width=True
        )

        hist_res = np.array(st.session_state.mp_hist_res)
        df_frac = pd.DataFrame({
            "Time": st.session_state.mp_hist_time,
            "Wildtype": hist_res[:, 0],
            "Mutant": hist_res[:, 1],
            "Superbug": hist_res[:, 2]
        }).melt("Time", var_name="Genotype", value_name="Frequency")

        chart_frac.altair_chart(
            alt.Chart(df_frac).mark_line().encode(x="Time", y="Frequency", color="Genotype"),
            use_container_width=True
        )

    # ---------------- PHASE MAP ----------------
    mus = np.linspace(0, 0.1, 10)
    ps = np.linspace(0, 1, 10)
    phase = []
    for mu in mus:
        for p in ps:
            phase.append([mu, p, run_coarse(mu, p)])

    df_phase = pd.DataFrame(phase, columns=["mu", "p", "state"])
    phase_chart = alt.Chart(df_phase).mark_rect().encode(
        x="mu:O", y="p:O",
        color=alt.Color("state:Q",
                        scale=alt.Scale(domain=[0,1,2],
                                        range=["black","orange","red"]),
                        legend=None)
    ).properties(height=250)

    phase_placeholder.altair_chart(phase_chart, use_container_width=True)

if __name__ == "__main__":
    app()
