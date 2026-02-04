import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
    st.set_page_config(layout="wide")
    st.title("Chemically Mediated Cross-Feeding")
    st.subheader("A Spatial Model for Two Mutualistic Species")

    st.markdown("""
    This simulator models **chemical-mediated mutualism** between two interacting species:  
    - Species **A** produces a nutrient (**X**) that is consumed by species **B**.  
    - Species **B** produces a toxin (**Y**) that negatively affects species **A**.  

    The spatial interaction between the two species and their metabolic byproducts (**X** and **Y**) creates distinct population dynamics and chemical distributions.
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
    st.sidebar.subheader("Ecosystem Parameters")
    D = st.sidebar.slider("Diffusion Coefficient (D)", 0.01, 0.5, 0.15, step=0.01)
    decay_x = st.sidebar.slider("Decay Rate of X (δ_X)", 0.01, 0.1, 0.05, step=0.01)
    decay_y = st.sidebar.slider("Decay Rate of Y (δ_Y)", 0.01, 0.1, 0.03, step=0.01)

    st.sidebar.subheader("Population Parameters")
    rA = st.sidebar.slider("Growth Rate of A", 0.0, 1.0, 0.1, step=0.01)
    rB = st.sidebar.slider("Growth Rate of B", 0.0, 2.0, 0.8, step=0.01)
    dA = st.sidebar.slider("Mortality of A due to Y", 0.0, 1.0, 0.5, step=0.01)
    dB = st.sidebar.slider("Mortality Rate of B", 0.0, 0.1, 0.02, step=0.001)

    st.sidebar.subheader("Chemical Parameters")
    P_X = st.sidebar.slider("Production Rate of X by A", 0.0, 1.0, 0.6, step=0.01)
    P_Y = st.sidebar.slider("Production Rate of Y by B", 0.0, 1.0, 0.4, step=0.01)
    toxicity = st.sidebar.slider("Y Toxicity", 0.0, 2.0, 0.8, step=0.01)

    steps = st.sidebar.slider("Steps per Frame", 1, 20, 5)

    GRID = 200
    EMPTY, A, B = 0, 1, 2

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
        st.session_state.initialized = True

    if not st.session_state.initialized:
        reset()

    if st.sidebar.button("Reset Simulation"):
        reset(); st.rerun()

    col1, col2, col3 = st.columns(3)
    ph_species = col1.empty()
    ph_X = col2.empty()
    ph_Y = col3.empty()

    chart_ph = st.empty()

    run = st.toggle("Run Simulation", False)

    def laplacian(F):
        return (
            np.roll(F, 1, 0) + np.roll(F, -1, 0) +
            np.roll(F, 1, 1) + np.roll(F, -1, 1) -
            4 * F
        )

    if run:
        grid = st.session_state.grid
        X = st.session_state.X
        Y = st.session_state.Y

        for _ in range(steps):
            maskA = grid == A
            maskB = grid == B

            X += P_X * maskA
            Y += P_Y * maskB

            X += D * laplacian(X) - decay_x * X
            Y += D * laplacian(Y) - decay_y * Y

            X = np.clip(X, 0, 1)
            Y = np.clip(Y, 0, 1)

            rand = np.random.rand(GRID, GRID)
            grid[(maskA) & (rand < toxicity * Y)] = EMPTY
            grid[(maskB) & (rand < dB)] = EMPTY

            dx = np.random.randint(-1, 2, (GRID, GRID))
            dy = np.random.randint(-1, 2, (GRID, GRID))
            x, y = np.indices(grid.shape)
            nx, ny = (x + dx) % GRID, (y + dy) % GRID

            grid[(grid == EMPTY) & (grid[nx, ny] == A) & (rand < rA)] = A
            grid[(grid == EMPTY) & (grid[nx, ny] == B) & (rand < rB * X)] = B

        st.session_state.time += 0.1
        st.session_state.grid = grid
        st.session_state.X = X
        st.session_state.Y = Y

        st.session_state.hist_time.append(st.session_state.time)
        st.session_state.hist_A.append(np.sum(grid == A))
        st.session_state.hist_B.append(np.sum(grid == B))

        st.rerun()

    grid = st.session_state.grid
    X = st.session_state.X
    Y = st.session_state.Y

    img = np.zeros((GRID, GRID, 3))
    img[grid == A] = [1, 0.2, 0.2]
    img[grid == B] = [0.2, 1, 0.2]

    ph_species.image(img, caption="Species Distribution", use_column_width=True)
    ph_X.image(X, caption="Nutrient Field X", clamp=True, use_column_width=True)
    ph_Y.image(Y, caption="Poison Field Y", clamp=True, use_column_width=True)

    if st.session_state.hist_time:
        df = pd.DataFrame({
            "Time": st.session_state.hist_time,
            "Producers (A)": st.session_state.hist_A,
            "Consumers (B)": st.session_state.hist_B
        }).melt("Time", var_name="Species", value_name="Population")

        chart = alt.Chart(df).mark_line().encode(
            x="Time", y="Population", color="Species"
        ).properties(title="Global Population Dynamics")

        chart_ph.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    app()