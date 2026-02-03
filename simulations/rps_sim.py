import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
    st.title("Cyclic Dominance (Rock–Paper–Scissors)")
    st.subheader("A Spatial Non-Transitive Competition Model")

    st.markdown("""
    This simulator models **cyclic ecological dominance**
    observed in colicin-producing *E. coli* strains.
    Local interactions generate spiral waves and long-term coexistence.
    """)

    # ---------------- MATHEMATICAL MODEL ----------------
    st.markdown("## Mathematical Model")

    st.latex(r"S(x,y,t) \in \{0,R,B,G\}")

    st.latex(r"R \succ G,\quad G \succ B,\quad B \succ R")

    st.latex(r"""
    \Pr\big[S(x,y,t+\Delta t)=S(x',y',t)\big]
    = \sigma \;\; \text{if } S(x,y,t)=\emptyset
    """)

    st.latex(r"""
    \Pr\big[S(x,y,t+\Delta t)=S(x',y',t)\big]
    = \beta \;\; \text{if } S(x',y',t) \succ S(x,y,t)
    """)

    st.latex(r"""
    S(x,y,t+\Delta t)=
    \begin{cases}
    S(x',y',t), & \text{reproduction or dominance},\\
    S(x,y,t), & \text{otherwise}.
    \end{cases}
    """)

    st.latex(r"""
    \begin{aligned}
    S(x,y,t) &: \text{species at lattice site } (x,y)\\
    R,G,B &: \text{toxic, sensitive, resistive strains}\\
    \sigma &: \text{reproduction probability}\\
    \beta &: \text{interaction (kill) probability}\\
    (x',y') &: \text{random neighboring site}
    \end{aligned}
    """)

    # ---------------- SIDEBAR ----------------
    st.sidebar.subheader("Ecosystem Parameters")
    spread_rate = st.sidebar.slider("Reproduction Rate (σ)", 0.0, 1.0, 0.25)
    eat_prob = st.sidebar.slider("Interaction Probability (β)", 0.0, 1.0, 0.45)

    st.sidebar.subheader("Initial Density")
    init_red = st.sidebar.slider("Init Toxic (R)", 0.0, 0.3, 0.02)
    init_green = st.sidebar.slider("Init Sensitive (G)", 0.0, 0.3, 0.02)
    init_blue = st.sidebar.slider("Init Resistive (B)", 0.0, 0.3, 0.02)

    GRID = 200
    steps_per_frame = st.sidebar.slider("Steps per Frame", 1, 20, 5)

    EMPTY, RED, BLUE, GREEN = 0, 1, 2, 3

    if "rps_grid" not in st.session_state:
        st.session_state.rps_initialized = False

    def reset_simulation():
        yy, xx = np.indices((GRID, GRID))
        center = GRID // 2
        radius = GRID // 2 - 2
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2

        grid = np.zeros((GRID, GRID), dtype=int)
        rand = np.random.rand(GRID, GRID)

        grid[(rand < init_red) & mask] = RED
        grid[(rand >= init_red) & (rand < init_red + init_blue) & mask] = BLUE
        grid[(rand >= init_red + init_blue) &
             (rand < init_red + init_blue + init_green) & mask] = GREEN

        st.session_state.rps_grid = grid
        st.session_state.rps_mask = mask
        st.session_state.rps_time = 0
        st.session_state.rps_hist_time = []
        st.session_state.rps_hist_red = []
        st.session_state.rps_hist_blue = []
        st.session_state.rps_hist_green = []
        st.session_state.rps_initialized = True

    if not st.session_state.rps_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    col_vis, col_stats = st.columns([1, 1])

    with col_vis:
        st.markdown("### Figure 1 — Spatial Population Field")
        dish_placeholder = st.empty()

    with col_stats:
        st.markdown("### Figure 2 — Global Population Dynamics")
        chart_counts = st.empty()
        chart_fracs = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    st.markdown("**RGB legend:** 🔴 Toxic | 🟢 Sensitive | 🔵 Resistive")

    # ---------------- SIMULATION ----------------
    if run_sim:
        grid = st.session_state.rps_grid
        mask = st.session_state.rps_mask

        for _ in range(steps_per_frame):
            dx = np.random.randint(-1, 2, size=(GRID, GRID))
            dy = np.random.randint(-1, 2, size=(GRID, GRID))

            x_indices, y_indices = np.indices((GRID, GRID))
            nx = (x_indices + dx) % GRID
            ny = (y_indices + dy) % GRID

            valid = mask & mask[nx, ny]
            rand_spread = np.random.rand(GRID, GRID)
            rand_eat = np.random.rand(GRID, GRID)

            neighbor = grid[nx, ny]
            self_state = grid

            repro = valid & (self_state == EMPTY) & (neighbor != EMPTY) & (rand_spread < spread_rate)
            grid[repro] = neighbor[repro]

            eat_mask = valid & (rand_eat < eat_prob)

            toxic_kill = eat_mask & (neighbor == RED) & (self_state == GREEN)
            sensitive_win = eat_mask & (neighbor == GREEN) & (self_state == BLUE)
            resistive_win = eat_mask & (neighbor == BLUE) & (self_state == RED)

            replaced = toxic_kill | sensitive_win | resistive_win
            grid[replaced] = neighbor[replaced]

            grid[~mask] = EMPTY

        st.session_state.rps_time += steps_per_frame

        st.session_state.rps_hist_time.append(st.session_state.rps_time)
        st.session_state.rps_hist_red.append(np.sum(grid == RED))
        st.session_state.rps_hist_blue.append(np.sum(grid == BLUE))
        st.session_state.rps_hist_green.append(np.sum(grid == GREEN))

        st.session_state.rps_grid = grid
        st.rerun()

    # ---------------- VISUAL ----------------
    grid = st.session_state.rps_grid
    mask = st.session_state.rps_mask

    img = np.zeros((GRID, GRID, 3))
    img[grid == RED] = [1.0, 0.2, 0.2]
    img[grid == BLUE] = [0.2, 0.4, 1.0]
    img[grid == GREEN] = [0.2, 1.0, 0.2]
    img[~mask] = 0

    dish_placeholder.image(img, caption=f"Time step: {st.session_state.rps_time}", use_column_width=True)

    # ---------------- PLOTS ----------------
    if st.session_state.rps_hist_time:
        df = pd.DataFrame({
            'Time': st.session_state.rps_hist_time,
            'Toxic (R)': st.session_state.rps_hist_red,
            'Sensitive (G)': st.session_state.rps_hist_green,
            'Resistive (B)': st.session_state.rps_hist_blue
        })

        df_melt = df.melt('Time', var_name='Strain', value_name='Count')
        chart_c = alt.Chart(df_melt).mark_line().encode(
            x='Time', y='Count', color='Strain'
        ).properties(height=200)
        chart_counts.altair_chart(chart_c, use_container_width=True)

        df['Total'] = df[['Toxic (R)', 'Sensitive (G)', 'Resistive (B)']].sum(axis=1).replace(0, 1)
        df_frac = pd.DataFrame({
            'Time': df['Time'],
            'Toxic (R)': df['Toxic (R)'] / df['Total'],
            'Sensitive (G)': df['Sensitive (G)'] / df['Total'],
            'Resistive (B)': df['Resistive (B)'] / df['Total']
        })

        df_frac_melt = df_frac.melt('Time', var_name='Strain', value_name='Fraction')
        chart_f = alt.Chart(df_frac_melt).mark_line().encode(
            x='Time', y='Fraction', color='Strain'
        ).properties(height=200)
        chart_fracs.altair_chart(chart_f, use_container_width=True)
        st.markdown("---")
        st.markdown(
        "**Numerics:** stochastic lattice updates, nearest-neighbor sampling, Bernoulli reproduction, cyclic dominance rules, circular domain mask."
        )



if __name__ == "__main__":
    app()

