import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
    st.title("Chemically Mediated Cross-feeding")
    st.subheader("A Spatial Model for Two Mutualistic Species")

    st.markdown("""
    This simulator models **chemical-mediated mutualism** between two species:
    Species **A** produces a nutrient **X**, while species **B** consumes **X** and produces
    a toxin **Y**. The simulation demonstrates spatial interactions, metabolic flows, and 
    oscillating dynamics between populations of both species.
    """)

    # ---------------- MATHEMATICAL MODEL ----------------
    st.markdown("## Mathematical Model")

    st.latex(r"""
    \begin{aligned}
    A(x, y, t): \text{population density of species } A \text{ (producers)} \\
    B(x, y, t): \text{population density of species } B \text{ (consumers)} \\
    X(x, y, t): \text{spatial concentration of resource } X \text{ (nutrient)} \\
    Y(x, y, t): \text{spatial concentration of resource } Y \text{ (toxin)}
    \end{aligned}
    """)

    st.latex(r"""
    \textbf{Dynamics:}
    """)

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

    st.markdown("""
    * **$D_X, D_Y$**: Diffusion coefficients for X and Y.  
    * **$P_X, P_Y$**: Secretion rates for nutrient (X) and toxin (Y).  
    * **$C_X$**: Consumption rate of X by B.  
    * **$r_A, r_B$**: Growth rates of A and B respectively.  
    * **$\delta_X, \delta_Y$**: Decay rates of X and Y respectively.  
    * **$d_A, d_B$**: Mortality rates of A and B respectively.  
    """)

    # ---------------- SIDEBAR ----------------
    st.sidebar.subheader("Ecosystem Parameters")
    D_diffusion = st.sidebar.slider("Diffusion Coefficient (D)", 0.01, 0.5, 0.15, step=0.01)
    decay_rate_x = st.sidebar.slider("Decay Rate of X (δ_X)", 0.01, 0.1, 0.05, step=0.01)
    decay_rate_y = st.sidebar.slider("Decay Rate of Y (δ_Y)", 0.01, 0.1, 0.03, step=0.01)

    st.sidebar.subheader("Species Parameters")
    growth_rate_a = st.sidebar.slider("Growth Rate of A (r_A)", 0.0, 1.0, 0.1, step=0.01)
    growth_rate_b = st.sidebar.slider("Growth Rate of B (r_B)", 0.0, 2.0, 0.8, step=0.01)
    mortality_a = st.sidebar.slider("Mortality of A by Y (d_A)", 0.0, 2.0, 0.8, step=0.01)
    mortality_b = st.sidebar.slider("Starvation Rate of B (d_B)", 0.0, 0.1, 0.02, step=0.01)

    st.sidebar.subheader("Field Strengths")
    prod_x = st.sidebar.slider("Nutrient X Production (P_X)", 0.0, 1.0, 0.4, step=0.01)
    prod_y = st.sidebar.slider("Poison Y Production (P_Y)", 0.0, 1.0, 0.4, step=0.01)
    toxicity = st.sidebar.slider("Poison Toxicity", 0.0, 2.0, 0.8, step=0.01)

    GRID = 300
    steps_per_frame = st.sidebar.slider("Steps per Iteration", 1, 20, 5)

    EMPTY, SPECIES_A, SPECIES_B = 0, 1, 2

    if "cf_grid" not in st.session_state:
        st.session_state.cf_initialized = False

    def reset_simulation():
        grid = np.zeros((GRID, GRID), dtype=int)
        r = np.random.rand(GRID, GRID)
        mask = r < 0.2
        grid[mask & (r < 0.1)] = SPECIES_A
        grid[mask & (r >= 0.1)] = SPECIES_B

        field_x = np.zeros((GRID, GRID))
        field_y = np.zeros((GRID, GRID))

        st.session_state.cf_grid = grid
        st.session_state.cf_field_x = field_x
        st.session_state.cf_field_y = field_y
        st.session_state.cf_time = 0
        st.session_state.cf_hist_a, st.session_state.cf_hist_b, st.session_state.cf_hist_time = [], [], []
        st.session_state.cf_initialized = True

    if not st.session_state.cf_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    col_vis, col_stats = st.columns([1, 1])

    with col_vis:
        st.markdown("### Figure 1 — Spatial Simulation of A and B")
        dish_placeholder = st.empty()

    with col_stats:
        st.markdown("### Figure 2 — Global Populations")
        chart_counts = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    st.markdown("**Legend:** 🔴 Producers (A) | 🟢 Consumers (B)")
    # ---------------- SIMULATION ----------------
    if run_sim:
        grid = st.session_state.cf_grid
        X = st.session_state.cf_field_x
        Y = st.session_state.cf_field_y

        def laplacian(field):
            return (
                np.roll(field, 1, axis=0) +
                np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) +
                np.roll(field, -1, axis=1) -
                4 * field
            )

        for _ in range(steps_per_frame):
            mask_a = (grid == SPECIES_A)
            mask_b = (grid == SPECIES_B)

            # Secretion
            X += prod_x * mask_a
            Y += prod_y * mask_b

            # Diffusion/Decay (PDE integration)
            X += D_diffusion * laplacian(X)
            Y += D_diffusion * laplacian(Y)
            X -= decay_rate_x * X
            Y -= decay_rate_y * Y

            # Clip fields to avoid negative values
            X = np.clip(X, 0, 1)
            Y = np.clip(Y, 0, 1)

            # Random indexes and actions
            dx = np.random.randint(-1, 2, size=(GRID, GRID))
            dy = np.random.randint(-1, 2, size=(GRID, GRID))
            x_indices, y_indices = np.indices(grid.shape)
            nx = (x_indices + dx) % GRID
            ny = (y_indices + dy) % GRID

            neighbor = grid[nx, ny]

            # Reproduction and replacement
            repro_a = (grid == EMPTY) & (neighbor == SPECIES_A) & (np.random.rand(GRID, GRID) < growth_rate_a)
            repro_b = (grid == EMPTY) & (neighbor == SPECIES_B) & (
                        np.random.rand(GRID, GRID) < growth_rate_b * X)
            grid[repro_a] = SPECIES_A
            grid[repro_b] = SPECIES_B

            # Mortality from poison
            death_a = (grid == SPECIES_A) & (np.random.rand(GRID, GRID) < toxicity * Y)
            death_b = (grid == SPECIES_B) & (np.random.rand(GRID, GRID) < mortality_b)
            grid[death_a | death_b] = EMPTY

        st.session_state.cf_hist_a.append(np.sum(grid == SPECIES_A))
        st.session_state.cf_hist_b.append(np.sum(grid == SPECIES_B))
        st.session_state.cf_hist_time.append(st.session_state.cf_time)
        st.session_state.cf_grid = grid
        st.session_state.cf_field_x = X
        st.session_state.cf_field_y = Y
        st.session_state.cf_time += steps_per_frame

        st.rerun()

    grid = st.session_state.cf_grid

    # Spatial visualization
    img = np.zeros((GRID, GRID, 3))
    img[grid == SPECIES_A] = [1, 0.2, 0.2]
    img[grid == SPECIES_B] = [0.0, 1, 0.0]
    dish_placeholder.image(img, caption=f"Time step: {st.session_state.cf_time}", use_column_width=True)

    # Population dynamics plot
    if st.session_state.cf_hist_time:
        df = pd.DataFrame({
            'Time': st.session_state.cf_hist_time,
            'Producers (A)': st.session_state.cf_hist_a,
            'Consumers (B)': st.session_state.cf_hist_b
        })
        df_melt = df.melt('Time', var_name='Species', value_name='Count')
        chart_c = alt.Chart(df_melt).mark_line().encode(
            x='Time', y='Count', color='Species'
        ).properties(height=250)
        chart_counts.altair_chart(chart_c, use_container_width=True)


if __name__ == "__main__":
    app()
