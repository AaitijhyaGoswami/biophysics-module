import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

def app():
    st.title("Stochastic Bacterial Colony Growth (2D + 3D)")

    st.markdown("""
    **Model:** Reaction–Diffusion + Stochastic Tip Growth  
    **Features:** Branching, nutrient depletion, lineage tracking, 3D biomass surface  
    """)

    # ---------------- SIDEBAR ----------------
    st.sidebar.subheader("Physics Parameters")
    food_diff = st.sidebar.slider("Food Diffusion", 0.0, 0.02, 0.008, format="%.4f")
    bact_diff = st.sidebar.slider("Bacteria Diffusion", 0.0, 0.05, 0.02, format="%.4f")
    growth_rate = st.sidebar.slider("Growth Rate", 0.0, 0.1, 0.05, format="%.4f")
    self_growth = st.sidebar.slider("Self Growth", 0.0, 0.05, 0.012, format="%.4f")
    consumption_rate = st.sidebar.slider("Consumption Rate", 0.0, 0.02, 0.006, format="%.4f")
    noise_strength = st.sidebar.slider("Stochastic Noise", 0.0, 1.0, 0.65)
    tip_factor = st.sidebar.slider("Tip Growth Factor", 0.5, 2.0, 1.0)

    st.sidebar.subheader("System Settings")
    grid = 300
    num_seeds = st.sidebar.slider("Number of Colonies", 1, 12, 12)
    seed_intensity = 0.03
    steps_per_frame = st.sidebar.slider("Simulation Speed", 1, 100, 40)

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

    # ---------------- 2×2 GRID ----------------
    row1 = st.columns(2)
    row2 = st.columns(2)

    ph_colony = row1[0].empty()
    ph_3d = row1[1].empty()
    ph_nutrient = row2[0].empty()
    ph_biomass = row2[1].empty()

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)
    ph_global = col_g1.empty()
    ph_local = col_g2.empty()

    run = st.toggle("Run Simulation", value=False)

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
    z = z / (z.max() + 1e-9) * 0.25  # flatten vertical scale

    fig3d = go.Figure(data=[go.Surface(z=z, colorscale="Inferno")])
    fig3d.update_layout(title=f"3D Biomass Surface (t={st.session_state.bg_time})",
                        margin=dict(l=0,r=0,b=0,t=30))

    ph_colony.image(medium, caption="2D Colony Morphology", use_column_width=True)
    ph_3d.plotly_chart(fig3d, use_container_width=True)
    ph_nutrient.image(nutr_img, caption="Nutrient Field", use_column_width=True)
    ph_biomass.image(bio_img, caption="Biomass Density", use_column_width=True)

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
        ).properties(title="Global Dynamics").interactive()

        ph_global.altair_chart(chart_global, use_container_width=True)

        data = {"Time (mins)": st.session_state.bg_hist_time}
        for sid in range(1, num_seeds+1):
            data[f"Colony {sid}"] = st.session_state.bg_colony_history[sid]

        df_col = pd.DataFrame(data)
        df_col_melt = df_col.melt("Time (mins)", var_name="Colony", value_name="Biomass")

        chart_local = alt.Chart(df_col_melt).mark_line().encode(
            x="Time (mins)", y="Biomass", color="Colony",
            tooltip=["Time (mins)", "Colony", "Biomass"]
        ).properties(title="Growth per Colony").interactive()

        ph_local.altair_chart(chart_local, use_container_width=True)

if __name__ == "__main__":
    app()
