import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

def app():
    st.set_page_config(layout="wide")
    st.title("Stochastic Bacterial Colony Growth (Smooth 2D + 3D)")

    st.markdown("""
    **Model:** Reaction–Diffusion with stochastic, surface-tension regulated growth  
    **Visuals:** 2D lineage map, 3D biomass surface, nutrient field, density map  
    """)

    # -------- Sidebar --------
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

    # -------- Utils --------
    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1,1:-1] = (
            arr[:-2,1:-1] + arr[2:,1:-1] +
            arr[1:-1,:-2] + arr[1:-1,2:] -
            4*arr[1:-1,1:-1]
        )
        return lap

    # -------- Init --------
    if "bg_initialized" not in st.session_state:
        st.session_state.bg_initialized = False

    def reset():
        y,x = np.ogrid[-grid/2:grid/2, -grid/2:grid/2]
        mask = x**2 + y**2 <= (grid/2-2)**2

        bacteria = np.zeros((grid,grid))
        food = np.zeros((grid,grid))
        food[mask] = 1
        seed_ids = np.zeros_like(bacteria, int)

        for sid in range(1, num_seeds+1):
            while True:
                r,c = np.random.randint(10,grid-10,2)
                if mask[r,c] and bacteria[r,c]==0:
                    bacteria[r,c] = seed_intensity
                    seed_ids[r,c] = sid
                    break

        st.session_state.update({
            "bg_bacteria": bacteria,
            "bg_food": food,
            "bg_seed_ids": seed_ids,
            "bg_mask": mask,
            "bg_time": 0,
            "bg_hist_time": [],
            "bg_pop_history": [],
            "bg_nut_history": [],
            "bg_colony_history": {i:[] for i in range(1,13)},
            "bg_initialized": True
        })

    if not st.session_state.bg_initialized:
        reset()
    if st.sidebar.button("Reset Simulation"):
        reset(); st.rerun()

    # -------- Layout --------
    row1 = st.columns(2)
    row2 = st.columns(2)
    ph_colony = row1[0].empty()
    ph_3d = row1[1].empty()
    ph_nutrient = row2[0].empty()
    ph_biomass = row2[1].empty()

    st.markdown("---")
    g1,g2 = st.columns(2)
    ph_global = g1.empty()
    ph_local = g2.empty()

    run = st.toggle("Run Simulation")

    if run:
        b = st.session_state.bg_bacteria
        f = st.session_state.bg_food
        s = st.session_state.bg_seed_ids
        m = st.session_state.bg_mask

        for _ in range(steps_per_frame):
            f += food_diff * laplacian(f)
            b += bact_diff * laplacian(b)
            b = gaussian_filter(b, 0.6)  # surface tension

            f -= consumption_rate * b
            f = np.clip(f,0,1)
            b = np.clip(b,0,1)
            b[~m] = 0

            nbr = gaussian_filter(b, 1.2)
            tip = nbr*(1-b)*tip_factor
            noise = np.random.rand(*b.shape)
            drive = np.clip(nbr - noise_strength*(noise-0.5) + tip, 0, 1)

            growth = growth_rate*b*(1-b)*(self_growth+(1-self_growth)*drive)*f
            b += growth
            b = gaussian_filter(b, 0.6)

            for sid in range(1,num_seeds+1):
                nbr_mask = (np.roll(s==sid,1,0)|np.roll(s==sid,-1,0)|
                            np.roll(s==sid,1,1)|np.roll(s==sid,-1,1))
                s[(nbr_mask)&(s==0)&(b>0)] = sid

        st.session_state.bg_time += steps_per_frame
        t = st.session_state.bg_time
        st.session_state.bg_hist_time.append(t)
        st.session_state.bg_pop_history.append(b.sum())
        st.session_state.bg_nut_history.append(f.sum())
        for sid in range(1,num_seeds+1):
            st.session_state.bg_colony_history[sid].append(b[s==sid].sum())

        st.session_state.bg_bacteria = b
        st.session_state.bg_food = f
        st.session_state.bg_seed_ids = s
        st.rerun()

    # -------- Render (FIXED) --------
    b = st.session_state.bg_bacteria
    f = st.session_state.bg_food
    s = st.session_state.bg_seed_ids
    m = st.session_state.bg_mask

    b_norm = b.copy()
    if b_norm.max() > 0:
        b_norm /= b_norm.max()

    medium = np.zeros((grid, grid, 3))
    medium[..., 0] = b_norm
    medium[..., 2] = b_norm * 0.7
    medium[~m] = 0

    f_norm = f.copy()
    if f_norm.max() > 0:
        f_norm /= f_norm.max()

    nutr = np.zeros((grid, grid, 3))
    nutr[..., 1] = f_norm
    nutr[~m] = 0

    z = gaussian_filter(b,1.5)
    fig3d = go.Figure(data=[go.Surface(z=z, colorscale="Inferno")])
    fig3d.update_layout(title=f"3D Biomass (t={st.session_state.bg_time})",
                        margin=dict(l=0,r=0,b=0,t=30))

    ph_colony.image(medium, caption="Colony Morphology",
                    clamp=True, use_column_width=True)
    ph_3d.plotly_chart(fig3d, use_container_width=True)
    ph_nutrient.image(nutr, caption="Nutrient Field",
                      clamp=True, use_column_width=True)
    ph_biomass.image(medium, caption="Biomass Density",
                     clamp=True, use_column_width=True)

    if st.session_state.bg_hist_time:
        df = pd.DataFrame({
            "Time": st.session_state.bg_hist_time,
            "Total Biomass": st.session_state.bg_pop_history,
            "Total Nutrient": st.session_state.bg_nut_history
        }).melt("Time")

        ph_global.altair_chart(
            alt.Chart(df).mark_line().encode(x="Time", y="value", color="variable"),
            use_container_width=True
        )

        data = {"Time": st.session_state.bg_hist_time}
        for sid in range(1,num_seeds+1):
            data[f"C{sid}"] = st.session_state.bg_colony_history[sid]

        df2 = pd.DataFrame(data).melt("Time")
        ph_local.altair_chart(
            alt.Chart(df2).mark_line().encode(x="Time", y="value", color="variable"),
            use_container_width=True
        )

if __name__ == "__main__":
    app()
