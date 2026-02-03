import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
    st.title("Chemically Mediated Cross-Feeding")
    st.subheader("A Spatial Model for Two Mutualistic Species")

    st.markdown("""
    This simulator models **chemical-mediated mutualism** between two interacting species:  
    - Species **A** produces a nutrient **X** consumed by species **B**.  
    - Species **B** produces a toxin **Y** harmful to species **A**.  

    The spatial interaction of these two species and their metabolic byproducts (**X** and **Y**) creates distinct population dynamics and chemical distributions.
    """)

    # ---------------- MATHEMATICAL MODEL ----------------
    st.markdown("## Mathematical Model")

    st.latex(r"A(x, y, t) \in \{0, A\}, \quad B(x, y, t) \in \{0, B\}")
    st.latex(r"X(x, y, t), \quad Y(x, y, t)")

    st.latex(r"""
    X(x, y, t+1) =
    \begin{cases}
    X(x, y, t) + P_X A(x, y) + D_X \nabla^2 X(x, y) - \delta_X X(x, y), & \text{if } B(x, y)=0,\\
    X(x, y, t) - C_B B(x, y) X(x, y), & \text{if } B(x, y)>0.
    \end{cases}
    """)

    st.latex(r"""
    \Pr[A(x \to x', t+1) = A] = r_A \quad \text{{(birth of Species A)}}
    """)

    st.latex(r"""
    Y(x, y, t+1) =
    \begin{cases}
    Y(x, y, t) + P_Y B(x, y) + D_Y \nabla^2 Y(x, y) - \delta_Y Y(x, y), & \text{otherwise.}\\
    \end{cases}
    """)

    st.latex(r"""
    \Pr[B(x \to x', t+1) = B] = r_B X \quad \text{{(birth of Species B based on X availability)}}
    """)

    st.latex(r"""
    \begin{aligned}
    A &: \text{density of producers}\\
    B &: \text{density of consumers}\\
    X &: \text{nutrient concentration, produced by } A(x, y)\\
    Y &: \text{toxin concentration, produced by } B(x, y)\\
    P_X, P_Y &: \text{production rates of X and Y respectively}\\
    D_X, D_Y &: \text{diffusion coefficients for X and Y}\\
    \delta_X, \delta_Y &: \text{decay rates for X and Y}\\
    r_A &: \text{growth rate of producers (A)}\\
    r_B &: \text{growth rate of consumers (B)}\\
    C_B &: \text{X consumption efficiency of consumers}\\
    \end{aligned}
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

    # ---------------- Layout ----------------
    col_vis, col_stats = st.columns([1, 1])

    with col_vis:
        st.markdown("### Figure 1 — Spatial Dynamics")
        dish_placeholder = st.empty()

    with col_stats:
        st.markdown("### Figure 2 — Global Population Dynamics")
        chart_placeholder = st.empty()

    run_simulation = st.toggle("Run Simulation", value=False)

    st.markdown("**Legend:** 🔴 Producers (A) | 🟢 Consumers (B)")

    # ---------------- Simulation ----------------
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

    # ---------------- Visual ----------------
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

if __name__ == "__main__":
    app()
