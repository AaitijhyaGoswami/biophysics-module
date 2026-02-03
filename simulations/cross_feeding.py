import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
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
    D_diffusion = st.sidebar.slider("Diffusion Coefficient (D)", 0.01, 0.5, 0.15, step=0.01)
    decay_rate_x = st.sidebar.slider("Decay Rate of X (δ_X)", 0.01, 0.1, 0.05, step=0.01)
    decay_rate_y = st.sidebar.slider("Decay Rate of Y (δ_Y)", 0.01, 0.1, 0.03, step=0.01)

    st.sidebar.subheader("Population Parameters")
    growth_a = st.sidebar.slider("Growth Rate of A (r_A)", 0.0, 1.0, 0.1, step=0.01)
    growth_b = st.sidebar.slider("Growth Rate of B (r_B)", 0.0, 2.0, 0.8, step=0.01)
    death_a = st.sidebar.slider("Mortality of A due to Y (d_A)", 0.0, 1.0, 0.5, step=0.01)
    death_b = st.sidebar.slider("Mortality Rate of B (d_B)", 0.0, 0.1, 0.02, step=0.001, format="%.3f")

    st.sidebar.subheader("Chemical Parameters")
    prod_x = st.sidebar.slider("Production Rate of X by A (P_X)", 0.0, 1.0, 0.6, step=0.01)
    prod_y = st.sidebar.slider("Production Rate of Y by B (P_Y)", 0.0, 1.0, 0.4, step=0.01)
    toxicity = st.sidebar.slider("Y Toxicity", 0.0, 2.0, 0.8, step=0.01)

    steps_per_frame = st.sidebar.slider("Steps per Frame", 1, 20, 5)

    GRID = 200
    EMPTY, SPECIES_A, SPECIES_B = 0, 1, 2

    if "grid" not in st.session_state:
        st.session_state.initialized = False

    def reset_simulation():
        grid = np.random.choice([EMPTY, SPECIES_A, SPECIES_B], (GRID, GRID), p=[0.9, 0.05, 0.05])
        field_x = np.zeros((GRID, GRID))
        field_y = np.zeros((GRID, GRID))
        st.session_state.grid = grid
        st.session_state.field_x = field_x
        st.session_state.field_y = field_y
        st.session_state.time = 0.0
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
        st.markdown("### Figure 1 — Spatial Population Dynamics")
        plate_placeholder = st.empty()

    with col_stats:
        st.markdown("### Figure 2 — Global Population Dynamics")
        chart_placeholder = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    st.markdown("**Legend:** 🔴 Producers (A) | 🟢 Consumers (B)")

    # ---------------- Simulation ----------------
    if run_sim:
        grid = st.session_state.grid
        X = st.session_state.field_x
        Y = st.session_state.field_y

        def laplacian(field):
            return (
                np.roll(field, 1, axis=0) +
                np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) +
                np.roll(field, -1, axis=1) -
                4 * field
            )

        for _ in range(steps_per_frame):
            mask_a = grid == SPECIES_A
            mask_b = grid == SPECIES_B

            # Secretion, Diffusion, Decay processes
            X += prod_x * mask_a
            Y += prod_y **
