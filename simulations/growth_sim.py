import time
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

    st.markdown("## Local Run Video")
    st.video("https://youtu.be/f6vUCx-SFOI")

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
    grid            = 300
    num_seeds       = st.sidebar.slider("Number of Colonies", 1, 12, 12)
    seed_intensity  = 0.03
    steps_per_frame = st.sidebar.slider("Simulation Speed", 1, 100, 40)

    # hardcoded throttle constants — tuned for smoothness
    SURF_STRIDE    = 4   # 300→75px for 3D (16× fewer points)
    SURF_EVERY     = 10  # redraw 3D surface every 10 frames
    CHARTS_EVERY   = 25  # redraw Altair charts every 25 frames

    # ---------------- UTILS ----------------
    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1, 1:-1] = (
            arr[:-2, 1:-1] + arr[2:, 1:-1] +
            arr[1:-1, :-2] + arr[1:-1, 2:] -
            4 * arr[1:-1, 1:-1]
        )
        return lap

    def init_state():
        y, x = np.ogrid[-grid/2:grid/2, -grid/2:grid/2]
        mask = x**2 + y**2 <= (grid/2 - 2)**2
        bacteria = np.zeros((grid, grid))
        food = np.zeros((grid, grid))
        food[mask] = 1.0
        seed_ids = np.zeros_like(bacteria, int)
        np.random.seed(42)
        for sid in range(1, num_seeds + 1):
            while True:
                r, c = np.random.randint(10, grid - 10, 2)
                if mask[r, c] and bacteria[r, c] == 0:
                    bacteria[r, c] = seed_intensity
                    seed_ids[r, c] = sid
                    break
        return bacteria, food, seed_ids, mask

    # ---------------- SESSION STATE ----------------
    if "bg_initialized" not in st.session_state:
        b, f, s, m = init_state()
        st.session_state.bg_bacteria       = b
        st.session_state.bg_food           = f
        st.session_state.bg_seed_ids       = s
        st.session_state.bg_mask           = m
        st.session_state.bg_time           = 0
        st.session_state.bg_frame          = 0
        st.session_state.bg_hist_time      = []
        st.session_state.bg_pop_history    = []
        st.session_state.bg_nut_history    = []
        st.session_state.bg_colony_history = {i: [] for i in range(1, 13)}
        st.session_state.bg_initialized    = True

    if st.sidebar.button("Reset Simulation"):
        b, f, s, m = init_state()
        st.session_state.bg_bacteria       = b
        st.session_state.bg_food           = f
        st.session_state.bg_seed_ids       = s
        st.session_state.bg_mask           = m
        st.session_state.bg_time           = 0
        st.session_state.bg_frame          = 0
        st.session_state.bg_hist_time      = []
        st.session_state.bg_pop_history    = []
        st.session_state.bg_nut_history    = []
        st.session_state.bg_colony_history = {i: [] for i in range(1, 13)}

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

    st.markdown("---")
    st.markdown("""
    **Numerics:** Forward Euler diffusion, stochastic branching,
    tip amplification, lineage tracking, 3D biomass projection.
    """)

    # ---------------- COLOUR PALETTE ----------------
    base_colors = np.array([
        [0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],
        [0,1,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1,.5,0],[.5,1,0],[1,0,.5]
    ])

    # ---------------- RENDER ----------------
    def render(bacteria, food, seed_ids, mask, t, frame):
        # --- Figure 1: colony (every frame) ---
        medium = np.zeros((grid, grid, 3))
        for sid in range(1, num_seeds + 1):
            sid_mask = seed_ids == sid
            for c in range(3):
                medium[..., c] += sid_mask * bacteria * base_colors[sid, c]
        nbr_field = (np.roll(bacteria,1,0)+np.roll(bacteria,-1,0)+
                     np.roll(bacteria,1,1)+np.roll(bacteria,-1,1)) / 4
        tips = (bacteria > 0) & (nbr_field < 0.3)
        halo = gaussian_filter(tips.astype(float), 1.2)
        if halo.max() > 0: halo /= halo.max()
        medium = np.clip(medium + halo[..., None] * 0.6, 0, 1)
        medium[~mask] = 0
        ph_colony.image(medium, use_column_width=True)

        # --- Figure 3: nutrient (every frame) ---
        nutr_img = np.zeros((grid, grid, 3))
        nutr_img[..., 1] = food
        nutr_img[~mask] = 0
        ph_nutrient.image(nutr_img, use_column_width=True)

        # --- Figure 4: biomass (every frame) ---
        bio_img = np.zeros((grid, grid, 3))
        bio_img[..., 0] = bacteria
        bio_img[..., 2] = bacteria * 0.5
        bio_img[~mask] = 0
        ph_biomass.image(bio_img, use_column_width=True)

        # --- Figure 2: 3D surface (throttled + downsampled) ---
        if frame % SURF_EVERY == 0:
            b_small = gaussian_filter(bacteria, 1.5)[::SURF_STRIDE, ::SURF_STRIDE]
            z = b_small / (b_small.max() + 1e-9) * 0.15
            fig3d = go.Figure(data=[go.Surface(z=z, colorscale="Inferno", showscale=False)])
            fig3d.update_layout(
                title=f"3D Biomass Surface (t={t})",
                margin=dict(l=0, r=0, b=0, t=30),
                uirevision="static",
            )
            ph_3d.plotly_chart(fig3d, use_container_width=True)

        # --- Figures 5 & 6: charts (throttled) ---
        if frame % CHARTS_EVERY == 0 and st.session_state.bg_hist_time:
            df_global = pd.DataFrame({
                "Time (mins)":    st.session_state.bg_hist_time,
                "Total Biomass":  st.session_state.bg_pop_history,
                "Total Nutrient": st.session_state.bg_nut_history,
            })
            ph_global.altair_chart(
                alt.Chart(df_global.melt("Time (mins)", var_name="Metric", value_name="Value"))
                .mark_line().encode(x="Time (mins)", y="Value", color="Metric",
                                    tooltip=["Time (mins)", "Metric", "Value"])
                .interactive(), use_container_width=True
            )
            data = {"Time (mins)": st.session_state.bg_hist_time}
            for sid in range(1, num_seeds + 1):
                data[f"Colony {sid}"] = st.session_state.bg_colony_history[sid]
            ph_local.altair_chart(
                alt.Chart(pd.DataFrame(data).melt("Time (mins)", var_name="Colony", value_name="Biomass"))
                .mark_line().encode(x="Time (mins)", y="Biomass", color="Colony",
                                    tooltip=["Time (mins)", "Colony", "Biomass"])
                .interactive(), use_container_width=True
            )

    # ---------------- STATIC RENDER WHEN PAUSED ----------------
    render(
        st.session_state.bg_bacteria,
        st.session_state.bg_food,
        st.session_state.bg_seed_ids,
        st.session_state.bg_mask,
        st.session_state.bg_time,
        frame=0,  # force all panels to draw on load
    )

    # ---------------- LIVE SIMULATION LOOP ----------------
    if run:
        bacteria       = st.session_state.bg_bacteria.copy()
        food           = st.session_state.bg_food.copy()
        seed_ids       = st.session_state.bg_seed_ids.copy()
        mask           = st.session_state.bg_mask
        t              = st.session_state.bg_time
        frame          = st.session_state.bg_frame
        hist_time      = list(st.session_state.bg_hist_time)
        pop_history    = list(st.session_state.bg_pop_history)
        nut_history    = list(st.session_state.bg_nut_history)
        colony_history = {i: list(st.session_state.bg_colony_history[i]) for i in range(1, 13)}

        while True:
            t0 = time.perf_counter()

            # --- physics ---
            for _ in range(steps_per_frame):
                food     += food_diff  * laplacian(food)
                bacteria += bact_diff  * laplacian(bacteria)
                food      = np.clip(food,     0, 1)
                bacteria  = np.clip(bacteria, 0, 1)
                bacteria[~mask] = 0

                food -= consumption_rate * bacteria
                food  = np.clip(food, 0, 1)

                nbr = (np.roll(bacteria, 1,0)+np.roll(bacteria,-1,0)+
                       np.roll(bacteria, 1,1)+np.roll(bacteria,-1,1)) / 4

                tip_drive   = nbr * (1 - bacteria) * tip_factor
                noise       = np.random.random(bacteria.shape)
                noisy       = np.clip(nbr - noise_strength*(noise-0.5) + tip_drive, 0, 1)
                local_drive = self_growth + (1 - self_growth) * noisy
                growth      = growth_rate * bacteria * (1-bacteria) * local_drive * food
                bacteria    = np.clip(bacteria + growth, 0, 1)
                bacteria[~mask] = 0

                for sid in range(1, num_seeds + 1):
                    nbr_mask = (np.roll(seed_ids==sid, 1,0)|np.roll(seed_ids==sid,-1,0)|
                                np.roll(seed_ids==sid, 1,1)|np.roll(seed_ids==sid,-1,1))
                    seed_ids[(nbr_mask)&(seed_ids==0)&(bacteria>0)] = sid

            # --- record ---
            t     += steps_per_frame
            frame += 1
            hist_time.append(t)
            pop_history.append(float(np.sum(bacteria)))
            nut_history.append(float(np.sum(food)))
            for sid in range(1, num_seeds + 1):
                colony_history[sid].append(float(np.sum(bacteria[seed_ids==sid])))

            # write history into session state so render() can read it
            st.session_state.bg_hist_time      = hist_time
            st.session_state.bg_pop_history    = pop_history
            st.session_state.bg_nut_history    = nut_history
            st.session_state.bg_colony_history = colony_history

            # --- render ---
            render(bacteria, food, seed_ids, mask, t, frame)

            # --- persist ---
            st.session_state.bg_bacteria  = bacteria.copy()
            st.session_state.bg_food      = food.copy()
            st.session_state.bg_seed_ids  = seed_ids.copy()
            st.session_state.bg_time      = t
            st.session_state.bg_frame     = frame

            # --- pace to ~20 fps ---
            elapsed = time.perf_counter() - t0
            wait    = (1/20) - elapsed
            if wait > 0:
                time.sleep(wait)

if __name__ == "__main__":
    app()
