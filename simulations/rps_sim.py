import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
    st.title("Cyclic Dominance (Rock-Paper-Scissors)")

    st.markdown("""
    **Simulation Details:**
    * **Model:** Spatial non-transitive competition (Colicin-type).
    * **Dynamics:**  
        * <span style='color:#FF3333'>Toxic</span> kills 
          <span style='color:#33FF33'>Sensitive</span>.  
        * <span style='color:#33FF33'>Sensitive</span> outgrows 
          <span style='color:#3366FF'>Resistive</span>.  
        * <span style='color:#3366FF'>Resistive</span> outcompetes 
          <span style='color:#FF3333'>Toxic</span>.  
    * **Outcome:** Spiral-wave coexistence or extinction depending on mobility.
    """, unsafe_allow_html=True)

    # Core parameters controlling reproduction and dominance
    st.sidebar.subheader("Ecosystem Parameters")
    spread_rate = st.sidebar.slider("Reproduction Rate", 0.0, 1.0, 0.25)
    eat_prob = st.sidebar.slider("Interaction Probability", 0.0, 1.0, 0.45)

    st.sidebar.subheader("Initial Density")
    init_red = st.sidebar.slider("Init Toxic (Red)", 0.0, 0.3, 0.02)
    init_green = st.sidebar.slider("Init Sensitive (Green)", 0.0, 0.3, 0.02)
    init_blue = st.sidebar.slider("Init Resistive (Blue)", 0.0, 0.3, 0.02)

    GRID = 200
    steps_per_frame = st.sidebar.slider("Steps per Frame", 1, 20, 5)

    EMPTY = 0
    RED = 1
    BLUE = 2
    GREEN = 3

    # Session initialization
    if "rps_grid" not in st.session_state:
        st.session_state.rps_initialized = False

    # Initialize circular domain with random seeded species
    def reset_simulation():
        yy, xx = np.indices((GRID, GRID))
        center = GRID // 2
        radius = GRID // 2 - 2
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2

        grid = np.zeros((GRID, GRID), dtype=int)
        rand = np.random.rand(GRID, GRID)

        mask_red = (rand < init_red) & mask
        mask_blue = (rand >= init_red) & (rand < init_red + init_blue) & mask
        mask_green = (rand >= init_red + init_blue) & (
            rand < init_red + init_blue + init_green
        ) & mask

        grid[mask_red] = RED
        grid[mask_blue] = BLUE
        grid[mask_green] = GREEN

        st.session_state.rps_grid = grid
        st.session_state.rps_mask = mask
        st.session_state.rps_time = 0
        st.session_state.rps_hist_time = []
        st.session_state.rps_hist_red = []
        st.session_state.rps_hist_blue = []
        st.session_state.rps_hist_green = []
        st.session_state.rps_initialized = True

    if not st.session_state.rps_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    col_vis, col_stats = st.columns([1, 1])

    with col_vis:
        st.write("### Petri Dish View")

        legend_html = """
        <style>
        .legend-item { display: inline-flex; align-items: center; margin-right: 15px; }
        .box { width: 12px; height: 12px; border: 1px solid #555; margin-right: 5px; }
        </style>
        <div style="margin-bottom: 10px;">
            <div class="legend-item">
                <span class="box" style="background-color:#FF3333;"></span>Toxic
            </div>
            <div class="legend-item">
                <span class="box" style="background-color:#33FF33;"></span>Sensitive
            </div>
            <div class="legend-item">
                <span class="box" style="background-color:#3366FF;"></span>Resistive
            </div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        dish_placeholder = st.empty()

    with col_stats:
        st.write("### Population Dynamics")
        chart_counts = st.empty()
        chart_fracs = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    if run_sim:
        grid = st.session_state.rps_grid
        mask = st.session_state.rps_mask

        for _ in range(steps_per_frame):
            # Random neighbor choice for local interactions
            dx = np.random.randint(-1, 2, size=(GRID, GRID))
            dy = np.random.randint(-1, 2, size=(GRID, GRID))

            x_indices, y_indices = np.indices((GRID, GRID))
            nx = (x_indices + dx) % GRID
            ny = (y_indices + dy) % GRID

            valid = mask & mask[nx, ny]
            rand_spread = np.random.rand(GRID, GRID)
            rand_eat = np.random.rand(GRID, GRID)

            neighbor = grid[nx, ny]
            self_state = grid

            # Reproduction: neighbor expands into empty space
            repro = (
                valid
                & (self_state == EMPTY)
                & (neighbor != EMPTY)
                & (rand_spread < spread_rate)
            )
            grid[repro] = neighbor[repro]

            # Dominance cycles (Toxic → Sensitive → Resistive → Toxic)
            eat_mask = valid & (rand_eat < eat_prob)

            toxic_kill = eat_mask & (neighbor == RED) & (self_state == GREEN)
            sensitive_win = eat_mask & (neighbor == GREEN) & (self_state == BLUE)
            resistive_win = eat_mask & (neighbor == BLUE) & (self_state == RED)

            replaced = toxic_kill | sensitive_win | resistive_win
            grid[replaced] = neighbor[replaced]

            grid[~mask] = EMPTY

        st.session_state.rps_time += steps_per_frame

        # Track population dynamics
        c_red = np.sum(grid == RED)
        c_blue = np.sum(grid == BLUE)
        c_green = np.sum(grid == GREEN)

        st.session_state.rps_hist_time.append(st.session_state.rps_time)
        st.session_state.rps_hist_red.append(c_red)
        st.session_state.rps_hist_blue.append(c_blue)
        st.session_state.rps_hist_green.append(c_green)

        st.session_state.rps_grid = grid
        st.rerun()

    # Render dish image from grid state
    grid = st.session_state.rps_grid
    mask = st.session_state.rps_mask

    img = np.zeros((GRID, GRID, 3))
    img[grid == RED] = [1.0, 0.2, 0.2]
    img[grid == BLUE] = [0.2, 0.4, 1.0]
    img[grid == GREEN] = [0.2, 1.0, 0.2]
    img[~mask] = 0.0

    dish_placeholder.image(
        img,
        caption=f"Time step: {st.session_state.rps_time}",
        clamp=True,
        use_column_width=True,
    )

    if len(st.session_state.rps_hist_time) > 0:
        df = pd.DataFrame(
            {
                "Time": st.session_state.rps_hist_time,
                "Toxic (Red)": st.session_state.rps_hist_red,
                "Sensitive (Green)": st.session_state.rps_hist_green,
                "Resistive (Blue)": st.session_state.rps_hist_blue,
            }
        )

        df_melt = df.melt("Time", var_name="Strain", value_name="Count")

        chart = (
            alt.Chart(df_melt)
            .mark_line()
            .encode(
                x=alt.X("Time"),
                y=alt.Y("Count"),
                color=alt.Color(
                    "Strain",
                    scale=alt.Scale(
                        domain=[
                            "Toxic (Red)",
                            "Sensitive (Green)",
                            "Resistive (Blue)",
                        ],
                        range=["#FF3333", "#33FF33", "#3366FF"],
                    ),
                ),
            )
            .properties(height=200)
        )

        chart_counts.altair_chart(chart, use_container_width=True)

        df["Total"] = (
            df["Toxic (Red)"]
            + df["Sensitive (Green)"]
            + df["Resistive (Blue)"]
        ).replace(0, 1)

        df_frac = df.copy()
        df_frac["Toxic (Red)"] /= df["Total"]
        df_frac["Sensitive (Green)"] /= df["Total"]
        df_frac["Resistive (Blue)"] /= df["Total"]

        df_frac_melt = df_frac.melt("Time", var_name="Strain", value_name="Fraction")

        chart2 = (
            alt.Chart(df_frac_melt)
            .mark_line()
            .encode(
                x=alt.X("Time"),
                y=alt.Y(
                    "Fraction",
                    title="Population Fraction",
                    scale=alt.Scale(domain=[0, 1])     # ← ← FIXED Y-AXIS HERE
                ),
                color=alt.Color(
                    "Strain",
                    scale=alt.Scale(
                        domain=[
                            "Toxic (Red)",
                            "Sensitive (Green)",
                            "Resistive (Blue)",
                        ],
                        range=["#FF3333", "#33FF33", "#3366FF"],
                    ),
                ),
            )
            .properties(height=200)
        )

        chart_fracs.altair_chart(chart2, use_container_width=True)



if __name__ == "__main__":
    app()


