import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

def app():
    st.set_page_config(page_title="Stochastic Bacterial Simulator", layout="wide")
    st.title("Stochastic Bacterial Colony Growth")
    st.subheader("Reaction–Diffusion + Stochastic Tip-Driven Branching")

    # ---------------- INTRODUCTORY TEXT ----------------
    st.markdown("""
    This advanced simulator models the emergence of complex, fractal-like structures in **nutrient-limited bacterial colonies**. 
    Unlike simple growth models, this system simulates the interplay between metabolic consumption, spatial diffusion, 
    and the stochastic (random) nature of biological branching. 
    
    As the bacteria consume local nutrients, they create a depletion zone, forcing the colony to "reach" outward. 
    The resulting **morphogenesis**—the birth of shape—is driven by small fluctuations at the colony's edge, 
    amplified by the physics of the environment.
    """)

    with st.expander("Explore Applications & Scientific Relevance", expanded=True):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            **Biological Research**
            * **Pattern Formation:** Studying how *Paenibacillus dendritiformis* or *Bacillus subtilis* create intricate dendritic patterns.
            * **Quorum Sensing:** Modeling how local density affects growth signals and survival strategies.
            * **Metabolic Competition:** Visualizing how different lineages (colored seeds) compete for a finite nutrient pool.
            """)
        with col_info2:
            st.markdown("""
            **Computational & Clinical Use**
            * **Biofilm Engineering:** Predicting the structural integrity and growth limits of bacterial films on medical devices.
            * **Antimicrobial Testing:** Simulating how diffusion barriers impact the efficacy of treatments in dense colonies.
            * **Mathematical Ecology:** Applying Reaction-Diffusion equations to understand invasive species spread in heterogeneous landscapes.
            """)

    with st.expander("📚 Key Reference Papers & Further Reading", expanded=False):
        col_ref1, col_ref2 = st.columns(2)
        with col_ref1:
            st.markdown("""
            **Reaction–Diffusion & Pattern Formation**
            * Turing, A. M. (1952). [The Chemical Basis of Morphogenesis.](https://doi.org/10.1098/rstb.1952.0012)
              *Philosophical Transactions of the Royal Society B.* — The foundational paper establishing how diffusion-driven instability generates spatial patterns.
            * Kondo, S. & Miura, T. (2010). [Reaction-Diffusion Model as a Framework for Understanding Biological Pattern Formation.](https://doi.org/10.1126/science.1179047)
              *Science, 329(5999).* — A modern review connecting Turing's theory to real biological systems.
            * Cross, M. C. & Hohenberg, P. C. (1993). [Pattern Formation Outside of Equilibrium.](https://doi.org/10.1103/RevModPhys.65.851)
              *Reviews of Modern Physics, 65(3).* — Comprehensive treatment of spatiotemporal pattern formation in non-equilibrium systems.

            **Bacterial Colony Morphogenesis**
            * Ben-Jacob, E. et al. (1994). [Generic Modelling of Cooperative Growth Patterns in Bacterial Colonies.](https://doi.org/10.1038/368046a0)
              *Nature, 368.* — Pioneering work showing how cooperative strategies produce complex branching colony shapes.
            * Matsushita, M. & Fujikawa, H. (1990). [Diffusion-Limited Growth in Bacterial Colony Formation.](https://doi.org/10.1016/0378-4371(90)90081-A)
              *Physica A, 168(1).* — Describes DLA-like fractal growth in *Bacillus subtilis* colonies on nutrient agar.
            """)
        with col_ref2:
            st.markdown("""
            **Stochastic Modeling & Branching**
            * Goldenfeld, N. & Kadanoff, L. P. (1999). [Simple Lessons from Complexity.](https://doi.org/10.1126/science.284.5411.87)
              *Science, 284(5411).* — Explores how stochastic fluctuations at small scales give rise to large-scale emergent structure.
            * Mimura, M. et al. (2000). [Reaction–Diffusion Modelling of Bacterial Colony Patterns.](https://doi.org/10.1016/S0378-4371(99)00549-5)
              *Physica A, 282(1–2).* — Mathematical framework for tip-driven branching and nutrient-limited colony expansion.
            * Wakita, J. et al. (1994). [Experimental Investigation on the Validity of Population Dynamics Approach to Bacterial Colony Formation.](https://doi.org/10.1143/JPSJ.63.1205)
              *Journal of the Physical Society of Japan, 63(3).* — Experimental validation of continuum PDE models against real colony growth data.

            **Biofilms & Clinical Relevance**
            * Costerton, J. W. et al. (1999). [Bacterial Biofilms: A Common Cause of Persistent Infections.](https://doi.org/10.1126/science.284.5418.1318)
              *Science, 284(5418).* — Seminal paper on the role of biofilm structure in chronic infection and antibiotic resistance.
            * Stewart, P. S. & Costerton, J. W. (2001). [Antibiotic Resistance of Bacteria in Biofilms.](https://doi.org/10.1016/S0140-6736(01)05321-1)
              *The Lancet, 358(9276).* — Explains how diffusion barriers within dense colonies limit antimicrobial penetration.
            """)

    # ---------------- THEORY ----------------
    st.markdown("### Governing Equations")
    st.latex(r"\frac{\partial B}{\partial t}=D_B\nabla^2B + r B(1-B)F \,\Phi(x,y,t)")
    st.latex(r"\frac{\partial F}{\partial t}=D_F\nabla^2F - \lambda B F")
    st.latex(r"\Phi(x,y,t) = \eta + (1-\eta)\big(\bar{B}(x,y,t) + \xi(x,y,t) + \kappa T(x,y,t)\big)")
    st.latex(r"""
    \begin{aligned}
    B(x,y,t) &:\ \text{Bacterial biomass density} \\
    F(x,y,t) &:\ \text{Nutrient concentration} \\
    \bar{B} &:\ \text{Local neighbor-averaged biomass} \\
    T &:\ \text{Tip indicator field (branch fronts)} \\
    \xi &:\ \text{Stochastic noise field} \\
    D_B, D_F &:\ \text{Diffusion coefficients} \\
    r &:\ \text{Growth rate} \\
    \lambda &:\ \text{Consumption rate} \\
    \kappa &:\ \text{Tip amplification strength} \\
    \eta &:\ \text{Self-growth baseline}
    \end{aligned}
    """)

    # ---------------- SIDEBAR ----------------
    st.sidebar.subheader("Physics Parameters")
    food_diff        = st.sidebar.slider("Food Diffusion",        0.0, 0.02, 0.008, format="%.4f")
    bact_diff        = st.sidebar.slider("Bacteria Diffusion",    0.0, 0.05, 0.02,  format="%.4f")
    growth_rate      = st.sidebar.slider("Growth Rate",           0.0, 0.1,  0.05,  format="%.4f")
    self_growth      = st.sidebar.slider("Self Growth (η)",       0.0, 0.05, 0.012, format="%.4f")
    consumption_rate = st.sidebar.slider("Consumption Rate (λ)",  0.0, 0.02, 0.006, format="%.4f")
    noise_strength   = st.sidebar.slider("Stochastic Noise (ξ)",  0.0, 1.0,  0.65)
    tip_factor       = st.sidebar.slider("Tip Growth Factor (κ)", 0.5, 2.0,  1.0)

    st.sidebar.subheader("System Settings")
    grid             = 300
    num_seeds        = st.sidebar.slider("Number of Colonies", 1, 12, 12)
    seed_intensity   = 0.03
    steps_per_frame  = st.sidebar.slider("Simulation Speed", 1, 100, 40)

    # ---------------- UTILS ----------------
    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1, 1:-1] = (
            arr[:-2, 1:-1] + arr[2:, 1:-1] +
            arr[1:-1, :-2] + arr[1:-1, 2:] -
            4 * arr[1:-1, 1:-1]
        )
        return lap

    # ---------------- INIT ----------------
    if "bg_initialized" not in st.session_state:
        st.session_state.bg_initialized = False

    def reset():
        y, x = np.ogrid[-grid/2:grid/2, -grid/2:grid/2]
        mask = x**2 + y**2 <= (grid/2 - 2)**2

        bacteria = np.zeros((grid, grid))
        food = np.zeros((grid, grid))
        food[mask] = 1.0
        seed_ids = np.zeros_like(bacteria, int)

        np.random.seed(42)
        for sid in range(1, num_seeds + 1):
            while True:
                r, c = np.random.randint(10, grid-10, 2)
                if mask[r, c] and bacteria[r, c] == 0:
                    bacteria[r, c] = seed_intensity
                    seed_ids[r, c] = sid
                    break

        st.session_state.bg_bacteria = bacteria
        st.session_state.bg_food = food
        st.session_state.bg_seed_ids = seed_ids
        st.session_state.bg_mask = mask
        st.session_state.bg_time = 0
        st.session_state.bg_hist_time = []
        st.session_state.bg_pop_history = []
        st.session_state.bg_nut_history = []
        st.session_state.bg_colony_history = {i: [] for i in range(1, 13)}
        st.session_state.bg_initialized = True

    if not st.session_state.bg_initialized:
        reset()

    if st.sidebar.button("Reset Simulation"):
        reset()
        st.rerun()

    # ---------------- LAYOUT ----------------
    row1 = st.columns(2)
    row2 = st.columns(2)

    with row1[0]:
        st.markdown("### Figure 1 — 2D Colony Morphology")
        ph_colony = st.empty()

    with row1[1]:
        st.markdown("### Figure 2 — 3D Biomass Surface")
        ph_3d = st.empty()

    with row2[0]:
        st.markdown("### Figure 3 — Nutrient Field")
        ph_nutrient = st.empty()

    with row2[1]:
        st.markdown("### Figure 4 — Biomass Density")
        ph_biomass = st.empty()

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("### Figure 5 — Global Dynamics")
        ph_global = st.empty()
    with col_g2:
        st.markdown("### Figure 6 — Growth per Colony")
        ph_local = st.empty()

    run = st.toggle("Run Simulation", value=False)

    # ---------------- SIMULATION ----------------
    if run:
        bacteria = st.session_state.bg_bacteria
        food = st.session_state.bg_food
        seed_ids = st.session_state.bg_seed_ids
        mask = st.session_state.bg_mask

        for _ in range(steps_per_frame):
            food += food_diff * laplacian(food)
            bacteria += bact_diff * laplacian(bacteria)

            food = np.clip(food, 0, 1)
            bacteria = np.clip(bacteria, 0, 1)
            bacteria[~mask] = 0

            food -= consumption_rate * bacteria
            food = np.clip(food, 0, 1)

            nbr = (np.roll(bacteria,1,0)+np.roll(bacteria,-1,0)+
                   np.roll(bacteria,1,1)+np.roll(bacteria,-1,1))/4

            tip_drive = nbr * (1 - bacteria) * tip_factor
            noise = np.random.random(bacteria.shape)
            noisy = np.clip(nbr - noise_strength*(noise-0.5) + tip_drive, 0, 1)

            local_drive = self_growth + (1-self_growth)*noisy
            growth = growth_rate * bacteria * (1-bacteria) * local_drive * food
            bacteria += growth
            bacteria = np.clip(bacteria, 0, 1)
            bacteria[~mask] = 0

            for sid in range(1, num_seeds+1):
                nbr_mask = (np.roll(seed_ids==sid,1,0)|
                            np.roll(seed_ids==sid,-1,0)|
                            np.roll(seed_ids==sid,1,1)|
                            np.roll(seed_ids==sid,-1,1))
                seed_ids[(nbr_mask)&(seed_ids==0)&(bacteria>0)] = sid

        st.session_state.bg_time += steps_per_frame
        t = st.session_state.bg_time
        st.session_state.bg_hist_time.append(t)
        st.session_state.bg_pop_history.append(np.sum(bacteria))
        st.session_state.bg_nut_history.append(np.sum(food))

        for sid in range(1, num_seeds+1):
            st.session_state.bg_colony_history[sid].append(
                np.sum(bacteria[seed_ids==sid])
            )

        st.session_state.bg_bacteria = bacteria
        st.session_state.bg_food = food
        st.session_state.bg_seed_ids = seed_ids
        st.rerun()

    # ---------------- VISUALS ----------------
    bacteria = st.session_state.bg_bacteria
    food = st.session_state.bg_food
    seed_ids = st.session_state.bg_seed_ids
    mask = st.session_state.bg_mask

    base_colors = np.array([
        [0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],
        [0,1,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1,.5,0],[.5,1,0],[1,0,.5]
    ])

    medium = np.zeros((grid,grid,3))
    for sid in range(1, num_seeds+1):
        sid_mask = seed_ids==sid
        for c in range(3):
            medium[...,c] += sid_mask*bacteria*base_colors[sid,c]

    nbr_field = (np.roll(bacteria,1,0)+np.roll(bacteria,-1,0)+
                 np.roll(bacteria,1,1)+np.roll(bacteria,-1,1))/4
    tips = (bacteria>0)&(nbr_field<0.3)
    halo = gaussian_filter(tips.astype(float),1.2)
    if halo.max()>0: halo/=halo.max()
    medium += halo[...,None]*0.6
    medium = np.clip(medium,0,1)
    medium[~mask]=0

    nutr_img = np.zeros((grid,grid,3))
    nutr_img[...,1]=food
    nutr_img[~mask]=0

    bio_img = np.zeros((grid,grid,3))
    bio_img[...,0]=bacteria
    bio_img[...,2]=bacteria*0.5
    bio_img[~mask]=0

    z = gaussian_filter(bacteria,1.5)
    z = z / (z.max() + 1e-9) * 0.15

    fig3d = go.Figure(data=[go.Surface(z=z, colorscale="Inferno")])
    fig3d.update_layout(title=f"3D Biomass Surface (t={st.session_state.bg_time})",
                        margin=dict(l=0,r=0,b=0,t=30))

    ph_colony.image(medium, use_column_width=True)
    ph_3d.plotly_chart(fig3d, use_container_width=True)
    ph_nutrient.image(nutr_img, use_column_width=True)
    ph_biomass.image(bio_img, use_column_width=True)

    # ---------------- PLOTS ----------------
    if st.session_state.bg_hist_time:
        df_global = pd.DataFrame({
            "Time (mins)": st.session_state.bg_hist_time,
            "Total Biomass": st.session_state.bg_pop_history,
            "Total Nutrient": st.session_state.bg_nut_history
        })
        df_melt = df_global.melt("Time (mins)", var_name="Metric", value_name="Value")

        chart_global = alt.Chart(df_melt).mark_line().encode(
            x="Time (mins)", y="Value", color="Metric",
            tooltip=["Time (mins)", "Metric", "Value"]
        ).interactive()

        ph_global.altair_chart(chart_global, use_container_width=True)

        data = {"Time (mins)": st.session_state.bg_hist_time}
        for sid in range(1, num_seeds+1):
            data[f"Colony {sid}"] = st.session_state.bg_colony_history[sid]

        df_col = pd.DataFrame(data)
        df_col_melt = df_col.melt("Time (mins)", var_name="Colony", value_name="Biomass")

        chart_local = alt.Chart(df_col_melt).mark_line().encode(
            x="Time (mins)", y="Biomass", color="Colony",
            tooltip=["Time (mins)", "Colony", "Biomass"]
        ).interactive()

        ph_local.altair_chart(chart_local, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Numerics:** Forward Euler diffusion, stochastic branching, 
    tip amplification, lineage tracking, 3D biomass projection.
    """)

if __name__ == "__main__":
    app()