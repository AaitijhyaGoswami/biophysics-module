import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

def app():
    st.title("Stochastic Bacterial Colony Growth")
    st.markdown("""
    **Simulation Details:**
    * **Model:** Reaction-Diffusion system with stochastic noise.
    * **Dynamics:** Nutrient consumption, metabolic diffusion, and exclusion principles.
    * **Visualization:** Real-time 2D morphology, nutrient fields, and **3D Biomass Terrain**.
    """)

    # --- Sidebar Parameters ---
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

    # --- Helper Functions ---
    def laplacian(arr):
        """Discrete 5-point stencil Laplacian for diffusion."""
        lap = np.zeros_like(arr)
        lap[1:-1, 1:-1] = (
            arr[:-2, 1:-1] + arr[2:, 1:-1] +
            arr[1:-1, :-2] + arr[1:-1, 2:] -
            4 * arr[1:-1, 1:-1]
        )
        return lap

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    # --- Initialization ---
    if "bg_bacteria" not in st.session_state:
        st.session_state.bg_initialized = False

    def reset():
        """Initialize circular domain, uniform nutrient, and random colony seeds."""
        y, x = np.ogrid[-grid / 2:grid / 2, -grid / 2:grid / 2]
        mask = x**2 + y**2 <= (grid / 2 - 2)**2

        bacteria = np.zeros((grid, grid), float)
        food = np.zeros((grid, grid), float)
        food[mask] = 1.0
        seed_ids = np.zeros_like(bacteria, int)

        np.random.seed(42)
        for sid in range(1, num_seeds + 1):
            attempts = 0
            while True:
                r = np.random.randint(10, grid - 10)
                c = np.random.randint(10, grid - 10)
                attempts += 1

                if mask[r, c] and bacteria[r, c] == 0:
                    bacteria[r, c] = seed_intensity
                    seed_ids[r, c] = sid
                    break

                if attempts > 5000:
                    ys, xs = np.where(mask & (bacteria == 0))
                    if len(ys) == 0:
                        break
                    idx = np.random.randint(len(ys))
                    r, c = ys[idx], xs[idx]
                    bacteria[r, c] = seed_intensity
                    seed_ids[r, c] = sid
                    break

        # State Variables
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

    # --- Layout: 2x2 Grid for 4 Visuals ---
    st.markdown("### Real-Time Visualization")
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # Create placeholders in a 2x2 grid
    with row1_col1:
        st.write("**1. Colony Morphology**")
        ph_colony = st.empty()
    with row1_col2:
        st.write("**2. 3D Biomass Terrain**")
        ph_3d = st.empty()
    with row2_col1:
        st.write("**3. Nutrient Field**")
        ph_nutrient = st.empty()
    with row2_col2:
        st.write("**4. Biomass Density (2D)**")
        ph_biomass = st.empty()

    st.markdown("---")

    # --- Analytics Plots ---
    col_g1, col_g2 = st.columns(2)
    ph_global = col_g1.empty()
    ph_local = col_g2.empty()

    run = st.toggle("Run Simulation", value=False)

    # --- Simulation Loop ---
    if run:
        bacteria = st.session_state.bg_bacteria
        food = st.session_state.bg_food
        seed_ids = st.session_state.bg_seed_ids
        mask = st.session_state.bg_mask

        for _ in range(steps_per_frame):
            # Diffusion
            food += food_diff * laplacian(food)
            bacteria += bact_diff * laplacian(bacteria)

            food = np.clip(food, 0.0, 1.0)
            bacteria = np.clip(bacteria, 0.0, 1.0)
            bacteria[~mask] = 0.0

            # Consumption
            food -= consumption_rate * bacteria
            food = np.clip(food, 0.0, 1.0)

            # Neighbor Sensing & Tip Growth
            nbr = (
                np.roll(bacteria, 1, 0) + np.roll(bacteria, -1, 0) +
                np.roll(bacteria, 1, 1) + np.roll(bacteria, -1, 1)
            ) / 4.0

            tip_drive = nbr * (1 - bacteria) * tip_factor
            noise = np.random.random(bacteria.shape)
            noisy_factor = np.clip(nbr - noise_strength * (noise - 0.5) + tip_drive, 0, 1)

            # Logistic Growth
            local_drive = self_growth + (1 - self_growth) * noisy_factor
            growth = growth_rate * bacteria * (1 - bacteria) * local_drive * food

            bacteria += growth
            bacteria = np.clip(bacteria, 0.0, 1.0)
            bacteria[~mask] = 0.0

            # Seed ID Propagation
            for sid in range(1, num_seeds + 1):
                nbr_mask = (
                    np.roll(seed_ids == sid, 1, 0) |
                    np.roll(seed_ids == sid, -1, 0) |
                    np.roll(seed_ids == sid, 1, 1) |
                    np.roll(seed_ids == sid, -1, 1)
                )
                seed_ids[(nbr_mask & (seed_ids == 0) & (bacteria > 0))] = sid

        # Update Metrics
        st.session_state.bg_time += steps_per_frame
        st.session_state.bg_hist_time.append(st.session_state.bg_time)
        st.session_state.bg_pop_history.append(np.sum(bacteria))
        st.session_state.bg_nut_history.append(np.sum(food))

        for sid in range(1, num_seeds + 1):
            st.session_state.bg_colony_history[sid].append(np.sum(bacteria[seed_ids == sid]))

        st.rerun()

    # --- Rendering Logic (Runs every rerun) ---
    bacteria = st.session_state.bg_bacteria
    food = st.session_state.bg_food
    seed_ids = st.session_state.bg_seed_ids
    mask = st.session_state.bg_mask

    # 1. Colony Morphology (RGB)
    base_colors = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
        [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0], [0.5, 0, 0.5],
        [0, 0.5, 0.5], [1, 0.5, 0], [0.5, 1, 0], [1, 0, 0.5]
    ])
    
    medium = np.zeros((grid, grid, 3), float)
    for sid in range(1, num_seeds + 1):
        sid_mask = (seed_ids == sid)
        for c in range(3):
            medium[..., c] += sid_mask * bacteria * base_colors[sid, c]

    # Add Halo Effect
    nbr_field = (np.roll(bacteria, 1, 0) + np.roll(bacteria, -1, 0) +
                 np.roll(bacteria, 1, 1) + np.roll(bacteria, -1, 1)) / 4.0
    tips = (bacteria > 0) & (nbr_field < 0.3)
    halo = gaussian_filter(tips.astype(float), sigma=1.2)
    if halo.max() > 0: halo /= halo.max()
    medium += halo[..., None] * 0.6
    medium = np.clip(medium, 0, 1)
    medium[~mask] = 0.0

    ph_colony.image(medium, clamp=True, use_column_width=True)

    # 2. 3D Terrain Render (Plotly)
    # Downsample by factor of 3 to maintain FPS (100x100 grid instead of 300x300)
    stride = 3
    z_data = bacteria[::stride, ::stride]
    
    fig_3d = go.Figure(data=[go.Surface(
        z=z_data,
        colorscale='Viridis',
        cmin=0, cmax=1.0,
        showscale=False # Hide colorbar to save space
    )])
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(visible=False), # Clean look
            yaxis=dict(visible=False),
            zaxis=dict(range=[0, 1], title="Dens"),
            aspectratio=dict(x=1, y=1, z=0.4),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=300 # Match height with other images
    )
    ph_3d.plotly_chart(fig_3d, use_container_width=True)

    # 3. Nutrient Field
    nutr_img = np.zeros((grid, grid, 3))
    nutr_img[..., 1] = food
    nutr_img[~mask] = 0.0
    ph_nutrient.image(nutr_img, clamp=True, use_column_width=True)

    # 4. Biomass Density (2D)
    bio_img = np.zeros((grid, grid, 3))
    bio_img[..., 0] = bacteria
    bio_img[..., 2] = bacteria * 0.5
    bio_img[~mask] = 0.0
    ph_biomass.image(bio_img, clamp=True, use_column_width=True)

    # --- Analytics Charts ---
    if st.session_state.bg_pop_history:
        # Global Chart
        df_global = pd.DataFrame({
            "Time": st.session_state.bg_hist_time,
            "Biomass": st.session_state.bg_pop_history,
            "Nutrient": st.session_state.bg_nut_history
        }).melt("Time", var_name="Metric", value_name="Value")

        chart_global = alt.Chart(df_global).mark_line().encode(
            x="Time", y="Value", color="Metric"
        ).properties(title="Global Dynamics", height=250)
        ph_global.altair_chart(chart_global, use_container_width=True)

        # Local Chart
        data = {"Time": st.session_state.bg_hist_time}
        for sid in range(1, num_seeds + 1):
            data[f"C{sid}"] = st.session_state.bg_colony_history[sid][:len(st.session_state.bg_hist_time)]
        
        df_local = pd.DataFrame(data).melt("Time", var_name="Colony", value_name="Biomass")
        domain = [f"C{sid}" for sid in range(1, num_seeds + 1)]
        colors = [rgb_to_hex(base_colors[sid]) for sid in range(1, num_seeds + 1)]

        chart_local = alt.Chart(df_local).mark_line().encode(
            x="Time", y="Biomass",
            color=alt.Color("Colony", scale=alt.Scale(domain=domain, range=colors))
        ).properties(title="Colony Growth", height=250)
        ph_local.altair_chart(chart_local, use_container_width=True)

if __name__ == "__main__":
    app()
