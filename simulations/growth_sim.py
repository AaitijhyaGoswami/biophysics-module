import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import time

def app():
    st.set_page_config(page_title="In Silico Morphogenesis", layout="wide")
    st.title("🧫 Stochastic Bacterial Colony Morphogenesis")
    st.subheader("Coupled Reaction–Diffusion Systems & Non-Linear Growth Dynamics")

    # ---------------- INTRODUCTORY TEXT ----------------
    st.markdown("""
    This simulator provides an *in silico* environment to observe **emergent multicellular organization**. 
    The model explores how microscopic stochasticity at the cellular level translates into macroscopic 
    phenotypes—specifically the dendritic, fractal-like branching patterns observed in motile bacterial 
    species under nutrient stress.

    The morphology is governed by the competition between the **expansion rate** of the colony front 
    and the **diffusion rate** of limiting substrates. When nutrient levels are low, the circular 
    symmetry of a colony breaks down due to **Mullins-Sekerka instability**, leading to the 
    formation of discrete "fingers" or branches that seek higher nutrient gradients.
    """)

    

    # ---------------- RELEVANT LITERATURE ----------------
    with st.expander("📚 Key Scientific Literature & Theoretical Foundations", expanded=False):
        st.markdown("""
        The algorithms implemented here are grounded in the following seminal works:
        
        * **Ben-Jacob, E., et al. (1994).** *Generic modelling of self-organization during bacterial colony growth.* Nature.
        * **Mimura, M., et al. (2000).** *Reaction-diffusion models for the formation of bacterial patterns.* Mathematical Biosciences.
        * **Golding, I., et al. (1998).** *Studies of bacterial branching growth using reaction-diffusion models.* Physica A.
        * **Farrell, F. D., et al. (2013).** *Mechanical interactions in bacterial colonies and spatial expansion.* J. R. Soc. Interface.
        """)

    # ---------------- THEORY ----------------
    st.markdown("### Governing PDE Framework")
    
    st.latex(r"\frac{\partial B}{\partial t}=D_B\nabla^2B + r B(1-B)F \,\Phi(x,y,t)")
    st.latex(r"\frac{\partial F}{\partial t}=D_F\nabla^2F - \lambda B F")

    st.markdown("**The Stochastic Modulation Field ($\Phi$):**")
    st.latex(r"\Phi(x,y,t) = \eta + (1-\eta)\big(\bar{B} + \xi(x,y,t) + \kappa T(x,y,t)\big)")

    # ---------------- SIDEBAR ----------------
    st.sidebar.header("Kinetic Parameters")
    food_diff = st.sidebar.slider("Substrate Diffusion ($D_F$)", 0.0, 0.02, 0.008, format="%.4f")
    bact_diff = st.sidebar.slider("Biomass Diffusion ($D_B$)", 0.0, 0.05, 0.02, format="%.4f")
    growth_rate = st.sidebar.slider("Max Growth Rate ($r$)", 0.0, 0.1, 0.05, format="%.4f")
    self_growth = st.sidebar.slider("Basal Growth ($\eta$)", 0.0, 0.05, 0.012, format="%.4f")
    consumption_rate = st.sidebar.slider("Consumption ($\lambda$)", 0.0, 0.02, 0.006, format="%.4f")
    noise_strength = st.sidebar.slider("Stochasticity ($\xi$)", 0.0, 1.0, 0.65)
    tip_factor = st.sidebar.slider("Tip Amplification ($\kappa$)", 0.5, 2.0, 1.0)

    st.sidebar.header("Boundary & Initial Conditions")
    grid = 250 # Adjusted for performance
    num_seeds = st.sidebar.slider("Initial Inoculation Sites", 1, 12, 6)
    steps_per_frame = st.sidebar.slider("Temporal Resolution", 1, 100, 40)

    # ---------------- UTILS ----------------
    def laplacian(arr):
        # Using 5-point stencil for diffusion
        return (np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) +
                np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) - 4 * arr)

    # ---------------- INITIALIZATION ----------------
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    def reset():
        y, x = np.ogrid[-grid/2:grid/2, -grid/2:grid/2]
        mask = x**2 + y**2 <= (grid/2 - 5)**2
        st.session_state.bacteria = np.zeros((grid, grid))
        st.session_state.food = np.zeros((grid, grid))
        st.session_state.food[mask] = 1.0
        st.session_state.seed_ids = np.zeros((grid, grid), dtype=int)
        st.session_state.mask = mask
        st.session_state.time = 0
        st.session_state.pop_history = []
        st.session_state.nut_history = []
        st.session_state.time_axis = []
        st.session_state.colony_history = {i: [] for i in range(1, 13)}

        np.random.seed(int(time.time()))
        for sid in range(1, num_seeds + 1):
            r, c = np.random.randint(20, grid-20, 2)
            st.session_state.bacteria[r, c] = 0.05
            st.session_state.seed_ids[r, c] = sid
        st.session_state.initialized = True

    if not st.session_state.initialized: reset()
    if st.sidebar.button("Reset Experiment"): reset(); st.rerun()

    # ---------------- LAYOUT ----------------
    col1, col2 = st.columns(2)
    with col1: ph_colony = st.empty()
    with col2: ph_3d = st.empty()
    
    col3, col4 = st.columns(2)
    with col3: ph_global = st.empty()
    with col4: ph_local = st.empty()

    run = st.toggle("Initiate Growth Simulation", value=False)

    # ---------------- MAIN LOOP ----------------
    while run:
        B = st.session_state.bacteria
        F = st.session_state.food
        S = st.session_state.seed_ids
        M = st.session_state.mask

        for _ in range(steps_per_frame):
            # Diffusion
            F += food_diff * laplacian(F)
            B += bact_diff * laplacian(B)
            
            # Nutrition Consumption
            F -= consumption_rate * B * F
            F = np.clip(F, 0, 1)

            # Growth Term with Tip Amplification
            nbr = (np.roll(B,1,0)+np.roll(B,-1,0)+np.roll(B,1,1)+np.roll(B,-1,1))/4
            tip_drive = nbr * (1 - B) * tip_factor
            noise = np.random.random(B.shape)
            Phi = self_growth + (1-self_growth) * np.clip(nbr - noise_strength*(noise-0.5) + tip_drive, 0, 1)
            
            B += growth_rate * B * (1-B) * Phi * F
            B[~M] = 0
            B = np.clip(B, 0, 1)

            # Lineage Tracking (Dilation)
            growing_edge = (B > 0.01) & (S == 0)
            if np.any(growing_edge):
                for sid in range(1, num_seeds + 1):
                    # Neighbors of current lineage
                    lineage_mask = (S == sid)
                    dilation = (np.roll(lineage_mask,1,0)|np.roll(lineage_mask,-1,0)|
                                np.roll(lineage_mask,1,1)|np.roll(lineage_mask,-1,1))
                    S[(dilation) & (S == 0) & (B > 0)] = sid

        # Stats Update
        st.session_state.time += steps_per_frame
        st.session_state.time_axis.append(st.session_state.time)
        st.session_state.pop_history.append(np.sum(B))
        st.session_state.nut_history.append(np.sum(F))
        for sid in range(1, num_seeds+1):
            st.session_state.colony_history[sid].append(np.sum(B[S==sid]))

        # --- RE-RENDERING VISUALS ---
        # Colony Image Construction
        colors = np.array([[0,0,0],[1,0.2,0.2],[0.2,1,0.2],[0.2,0.2,1],[1,1,0.2],[1,0.2,1],[0.2,1,1],[0.7,0.4,0],[0.5,0,0.5],[0,0.5,0.5],[1,0.5,0],[0.5,1,0],[1,0,0.5]])
        img = np.zeros((grid, grid, 3))
        for sid in range(1, num_seeds+1):
            for c in range(3):
                img[..., c] += (S == sid) * B * colors[sid, c]
        img = np.clip(img, 0, 1)
        ph_colony.image(img, caption=f"Morphology at T={st.session_state.time}", use_container_width=True)

        # 3D Plot
        z_data = gaussian_filter(B, 1.0)
        fig3d = go.Figure(data=[go.Surface(z=z_data, colorscale="Viridis")])
        fig3d.update_layout(title="Biomass Topography", scene=dict(zaxis=dict(range=[0,0.5])), margin=dict(l=0,r=0,b=0,t=30))
        ph_3d.plotly_chart(fig3d, use_container_width=True)

        # Charts
        df_glob = pd.DataFrame({"T": st.session_state.time_axis, "Biomass": st.session_state.pop_history, "Nutrients": st.session_state.nut_history}).melt("T")
        ph_global.altair_chart(alt.Chart(df_glob).mark_line().encode(x="T", y="value", color="variable"), use_container_width=True)
        
        # Lineage Chart
        colony_data = {"T": st.session_state.time_axis}
        for sid in range(1, num_seeds+1): colony_data[f"L{sid}"] = st.session_state.colony_history[sid]
        df_loc = pd.DataFrame(colony_data).melt("T")
        ph_local.altair_chart(alt.Chart(df_loc).mark_line().encode(x="T", y="value", color="variable"), use_container_width=True)

        time.sleep(0.01)

if __name__ == "__main__":
    app()
