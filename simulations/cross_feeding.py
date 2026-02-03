import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
    st.title("Chemically Mediated Cross-Feeding")
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
    S(x,y,t) &: \text{species at lattice site } (x,y)\\
    A(x, y, t) &: \text{population density of `producers`}\\
    B(x, y, t) &: \text{population density of `consumers`}\\
    X(x, y, t) &: \text{spatial concentration of resource } X \text{ (nutrient)} \\
    Y(x, y, t): \text{spatial concentration of resource } Y \text{ (toxin)}
    \end{aligned}
    """)

    st.markdown("The chemical and population dynamics are modeled with the following partial differential equations:")

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
    * **$C_X$**: Consumption rate of X by consumers (B).
    * **$r_A, r_B$**: Growth rates of A (producers) and B (consumers).
    * **$\delta_X, \delta_Y$**: Decay rates of X and Y respectively.
    * **$d_A, d_B$**: Mortality rates of A and B respectively.
    """)

    # ---------------- SIDEBAR ----------------
    st.sidebar.subheader("Ecosystem Parameters")
    D_diffusion = st.sidebar.slider("Diffusion Coefficient (D)", 0.01, 0.5, 0.1, step=0.01)
    decay_rate_x = st.sidebar.slider("Decay Rate of X (δ_X)", 0.01, 0.1, 0.02, step=0.01)
    decay_rate_y = st.sidebar.slider("Decay Rate of Y (δ_Y)", 0.01, 0.1, 0.01, step=0.01)

    st.sidebar.subheader("Population Parameters")
    growth_a = st.sidebar.slider("Growth Rate of A (r_A)", 0.0, 1.0, 0.1, step=0.01)
    growth_b = st.sidebar.slider("Growth Rate of B (r_B)", 0.0, 1.0, 0.5, step=0.01)
    death_a = st.sidebar.slider("Mortality of A due to Y (d_A)", 0.0, 1.0, 0.2, step=0.01)
    death_b = st.sidebar.slider("Starvation Rate of B (d_B)", 0.0, 0.05, 0.01, step=0.001)

    st.sidebar.subheader("Chemical Parameters")
    prod_x = st.sidebar.slider("X Production by A (P_X)", 0.0, 1.0, 0.5, step=0.01)
    prod_y = st.sidebar.slider("Y Production by B (P_Y)", 0.0, 1.0, 0.5, step=0.01)
    toxicity = st.sidebar.slider("Y Toxicity", 0.0, 1.0, 0.5, step=0.01)

    steps_per_frame = st.sidebar.slider("Steps per Iteration", 1, 50, 10)

    GRID = 200
    EMPTY, SPECIES_A, SPECIES_B = 0, 1, 2

    if "grid" not in st.session_state:
        st.session_state.initialized = False

    def reset_simulation():
        grid = np.random.choice([EMPTY, SPECIES_A, SPECIES_B], (GRID, GRID), p=[0.9, 0.05, 0.05])
        X = np.zeros((GRID, GRID))
        Y = np.zeros((GRID, GRID))
        st.session_state.grid = grid
        st.session_state.X = X
        st.session_state.Y = Y
        st.session_state.time = 0
        st.session_state.hist_time = []
        st.session_state.hist_species_a = []
        st.session_state.hist_species_b = []
        st.session_state.initialized = True

    if not st.session_state.initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    col_main, col_chart = st.columns([1.5, 1])

    with col_main:
        st.markdown("### Figure 1 — Spatial Dynamics")
        dish_placeholder = st.empty()

    with col_chart:
        st.markdown("### Figure 2 — Global Populations")
        chart_placeholder = st.empty()

    run_simulation = st.toggle("Run Simulation", value=False)

    st.markdown("**Legend:** 🔴 Producers (A) | 🟢 Consumers (B)")

    # ---------------- SIMULATION ----------------
    if run_simulation:
        grid = st.session_state.grid
        X = st.session_state.X
        Y = st.session_state.Y

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

            # Secretion, Diffusion, and Decay
            X += prod_x * mask_a
            Y += prod_y * mask_b
            X += D_diffusion * laplacian(X) - decay_rate_x * X
            Y += D_diffusion * laplacian(Y) - decay_rate_y * Y

            # Clipping
            X = np.clip(X, 0, 1)
            Y = np.clip(Y, 0, 1)

            # Birth and Death
            rand_arr = np.random.random(grid.shape)
            grid[mask_a & (rand_arr < toxicity * Y)] = EMPTY  # A killed by Y
            grid[mask_b & (rand_arr < death_b)] = EMPTY       # B dies from starvation

            # Propagation of A and B
            dx = np.random.randint(-1, 2, size=(GRID, GRID))
            dy = np.random.randint(-1, 2, size=(GRID, GRID))
            x_idx, y_idx = np.indices(grid.shape)
            nx, ny = (x_idx + dx) % GRID, (y_idx + dy) % GRID

            reproduction_a = (grid == EMPTY) & (grid[nx, ny] == SPECIES_A) & (rand_arr < growth_a)
            reproduction_b = (grid == EMPTY) & (grid[nx, ny] == SPECIES_B) & (rand_arr < growth_b * X)
            grid[reproduction_a] = SPECIES_A
            grid[reproduction_b] = SPECIES_B

        st.session_state.time += steps_per_frame
        st.session_state.hist_time.append(st.session_state.time)
        st.session_state.hist_species_a.append(np.sum(grid == SPECIES_A))
        st.session_state.hist_species_b.append(np.sum(grid == SPECIES_B))
        st.session_state.grid = grid
        st.session_state.X = X
        st.session_state.Y = Y
        st.rerun()

    # ---------------- VISUAL ----------------
    grid = st.session_state.grid
    img = np.zeros((GRID, GRID, 3))
    img[grid == SPECIES_A] = [1.0, 0.2, 0.2]
    img[grid == SPECIES_B] = [0.2, 1.0, 0.2]
    dish_placeholder.image(img, caption=f"Time Step: {st.session_state.time}", use_column_width=True)

    # ---------------- PLOTS ----------------
    if st.session_state.hist_time:
        df = pd.DataFrame({
            'Time': st.session_state.hist_time,
            'Producers (A)': st.session_state.hist_species_a,
            'Consumers (B)': st.session_state.hist_species_b
        })

        df_melted = df.melt('Time', var_name='Species', value_name='Population')

        chart = alt.Chart(df_melted).mark_line().encode(
            x='Time', y='Population', color='Species',
        ).properties(height=250)
        chart_placeholder.altair_chart(chart, use_container_width=True)

    # ---------------- NUMERICS ----------------
    st.markdown("## Numerics")
    st.markdown("""
    Discrete-time spatial dynamics, regular grid lattice, uniform random sampling, periodic boundary conditions, nearest-neighbor interactions, Laplacian diffusion and decay updates, stochastic reproduction, consumption, and mortality events.
    """)


if __name__ == "__main__":
    app()
