import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    st.title("Cyclic Dominance (Rock-Paper-Scissors)")
    st.markdown("""
    **Simulation Details:**
    * **Model:** Spatial non-transitive competition (The "Colicin" Model).
    * **Dynamics:** * <span style='color:#FF3333'>**Toxic**</span> kills <span style='color:#33FF33'>**Sensitive**</span>.
        * <span style='color:#33FF33'>**Sensitive**</span> outgrows <span style='color:#3366FF'>**Resistive**</span>.
        * <span style='color:#3366FF'>**Resistive**</span> outcompetes <span style='color:#FF3333'>**Toxic**</span>.
    * **Outcome:** Species coexist in rotating spiral waves or go extinct depending on mobility.
    """, unsafe_allow_html=True)

    # -----------------------------
    # 1. PARAMETERS
    # -----------------------------
    st.sidebar.subheader("Ecosystem Parameters")
    
    SPREAD_RATE = st.sidebar.slider("Reproduction Rate", 0.0, 1.0, 0.25)
    EAT_PROB = st.sidebar.slider("Interaction Probability", 0.0, 1.0, 0.45)
    
    st.sidebar.subheader("Initial Density")
    INIT_RED = st.sidebar.slider("Init Toxic (Red)", 0.0, 0.3, 0.02)
    INIT_GREEN = st.sidebar.slider("Init Sensitive (Green)", 0.0, 0.3, 0.02)
    INIT_BLUE = st.sidebar.slider("Init Resistive (Blue)", 0.0, 0.3, 0.02)

    st.sidebar.subheader("System Settings")
    GRID = 200
    # Speed control
    STEPS_PER_FRAME = st.sidebar.slider("Steps per Frame", 1, 20, 5)

    # Constants
    EMPTY = 0
    RED = 1   # Toxic
    BLUE = 2  # Resistive
    GREEN = 3 # Sensitive

    # -----------------------------
    # 2. INITIALIZATION (Session State)
    # -----------------------------
    if 'rps_grid' not in st.session_state:
        st.session_state.rps_initialized = False

    def reset_simulation():
        # A. Circular Mask
        yy, xx = np.indices((GRID, GRID))
        center = GRID // 2
        radius = GRID // 2 - 2
        mask = (xx - center)**2 + (yy - center)**2 <= radius**2
        
        # B. Initial Grid
        grid = np.zeros((GRID, GRID), dtype=int)
        rand = np.random.rand(GRID, GRID)
        
        # Apply initial probabilities within mask
        mask_red = (rand < INIT_RED) & mask
        mask_blue = (rand >= INIT_RED) & (rand < INIT_RED + INIT_BLUE) & mask
        mask_green = (rand >= INIT_RED + INIT_BLUE) & (rand < INIT_RED + INIT_BLUE + INIT_GREEN) & mask
        
        grid[mask_red] = RED
        grid[mask_blue] = BLUE
        grid[mask_green] = GREEN

        # C. Save State
        st.session_state.rps_grid = grid
        st.session_state.rps_mask = mask
        st.session_state.rps_time = 0
        
        # History
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

    # -----------------------------
    # 3. MAIN SIMULATION LOOP
    # -----------------------------
    col_vis, col_stats = st.columns([1, 1])
    
    with col_vis:
        st.write("### Petri Dish View")
        
        # --- HTML LEGEND ---
        legend_html = """
        <style>
            .legend-item { display: inline-flex; align-items: center; margin-right: 15px; font-size: 13px; }
            .box { width: 12px; height: 12px; border: 1px solid #555; margin-right: 5px; }
        </style>
        <div style="margin-bottom: 10px;">
            <div class="legend-item">
                <span class="box" style="background-color: #FF3333;"></span>Toxic (Killer)
            </div>
            <div class="legend-item">
                <span class="box" style="background-color: #33FF33;"></span>Sensitive (Victim)
            </div>
            <div class="legend-item">
                <span class="box" style="background-color: #3366FF;"></span>Resistive (Immune)
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
        
        for _ in range(STEPS_PER_FRAME):
            # ---------------------------------------------------------
            # VECTORIZED UPDATE
            # ---------------------------------------------------------
            # 1. Select Random Neighbors
            dx = np.random.randint(-1, 2, size=(GRID, GRID))
            dy = np.random.randint(-1, 2, size=(GRID, GRID))
            
            x_indices, y_indices = np.indices((GRID, GRID))
            nx = (x_indices + dx) % GRID
            ny = (y_indices + dy) % GRID
            
            # 2. Validity Masks
            valid_mask = mask & mask[nx, ny]
            rand_spread = np.random.rand(GRID, GRID)
            rand_eat = np.random.rand(GRID, GRID)
            
            # INVERTED LOGIC (Look at self as 'Target' for stability)
            # T_inv = Self, S_inv = Neighbor
            S_inv = grid[nx, ny] 
            T_inv = grid         
            
            # 3. Reproduction: Neighbor fills Me (if I am empty)
            inv_repro = valid_mask & (T_inv == EMPTY) & (S_inv != EMPTY) & (rand_spread < SPREAD_RATE)
            grid[inv_repro] = S_inv[inv_repro]
            
            # 4. Interaction/Predation: Neighbor replaces Me
            # Red(1) eats Green(3) -> Toxic kills Sensitive
            # Green(3) eats Blue(2) -> Sensitive outgrows Resistive
            # Blue(2) eats Red(1) -> Resistive outcompetes Toxic
            
            inv_eat = valid_mask & (rand_eat < EAT_PROB)
            
            # Neighbor is Red, I am Green -> I become Red
            toxic_kill = inv_eat & (S_inv == RED) & (T_inv == GREEN)
            
            # Neighbor is Green, I am Blue -> I become Green
            sensitive_grow = inv_eat & (S_inv == GREEN) & (T_inv == BLUE)
            
            # Neighbor is Blue, I am Red -> I become Blue
            resistive_win = inv_eat & (S_inv == BLUE) & (T_inv == RED)
            
            mask_replaced = toxic_kill | sensitive_grow | resistive_win
            grid[mask_replaced] = S_inv[mask_replaced]
            
            # Cleanup
            grid[~mask] = EMPTY

        # Update Stats
        st.session_state.rps_time += STEPS_PER_FRAME
        c_red = np.sum(grid == RED)
        c_blue = np.sum(grid == BLUE)
        c_green = np.sum(grid == GREEN)
        
        st.session_state.rps_hist_time.append(st.session_state.rps_time)
        st.session_state.rps_hist_red.append(c_red)
        st.session_state.rps_hist_blue.append(c_blue)
        st.session_state.rps_hist_green.append(c_green)
        
        st.session_state.rps_grid = grid
        st.rerun()

    # -----------------------------
    # 4. RENDERING
    # -----------------------------
    # A. Build Image
    # Toxic(Red), Resistive(Blue), Sensitive(Green)
    grid = st.session_state.rps_grid
    mask = st.session_state.rps_mask
    
    img = np.zeros((GRID, GRID, 3))
    
    mask_r = (grid == RED)
    mask_b = (grid == BLUE)
    mask_g = (grid == GREEN)
    
    # Colors matching the Legend
    img[mask_r] = [1.0, 0.2, 0.2] # Red
    img[mask_b] = [0.2, 0.4, 1.0] # Blue
    img[mask_g] = [0.2, 1.0, 0.2] # Green
    
    img[~mask] = 0.0
    
    dish_placeholder.image(img, caption=f"Time step: {st.session_state.rps_time}", clamp=True, use_column_width=True)
    
    # B. ALTAIR Charts
    if len(st.session_state.rps_hist_time) > 0:
        
        # Prepare DataFrame
        df = pd.DataFrame({
            'Time': st.session_state.rps_hist_time,
            'Toxic (Red)': st.session_state.rps_hist_red,
            'Sensitive (Green)': st.session_state.rps_hist_green,
            'Resistive (Blue)': st.session_state.rps_hist_blue
        })
        
        # 1. Absolute Counts
        df_melt = df.melt('Time', var_name='Strain', value_name='Count')
        
        chart_c = alt.Chart(df_melt).mark_line().encode(
            x=alt.X('Time', axis=alt.Axis(title='Time (Generations)')),
            y=alt.Y('Count', axis=alt.Axis(title='Population Size')),
            color=alt.Color('Strain', scale=alt.Scale(
                domain=['Toxic (Red)', 'Sensitive (Green)', 'Resistive (Blue)'],
                range=['#FF3333', '#33FF33', '#3366FF']
            ))
        ).properties(height=200)
        
        chart_counts.altair_chart(chart_c, use_container_width=True)
        
        # 2. Fractions (Normalized)
        # Calculate fractions
        df['Total'] = df['Toxic (Red)'] + df['Sensitive (Green)'] + df['Resistive (Blue)']
        # Avoid div by zero
        df['Total'] = df['Total'].replace(0, 1) 
        
        df_frac = pd.DataFrame({
            'Time': df['Time'],
            'Toxic (Red)': df['Toxic (Red)'] / df['Total'],
            'Sensitive (Green)': df['Sensitive (Green)'] / df['Total'],
            'Resistive (Blue)': df['Resistive (Blue)'] / df['Total']
        })
        
        df_frac_melt = df_frac.melt('Time', var_name='Strain', value_name='Fraction')
        
        chart_f = alt.Chart(df_frac_melt).mark_line().encode(
            x=alt.X('Time', axis=alt.Axis(title='Time (Generations)')),
            y=alt.Y('Fraction', axis=alt.Axis(title='Relative Abundance', format='%')),
            color=alt.Color('Strain', scale=alt.Scale(
                domain=['Toxic (Red)', 'Sensitive (Green)', 'Resistive (Blue)'],
                range=['#FF3333', '#33FF33', '#3366FF']
            ))
        ).properties(height=200)
        
        chart_fracs.altair_chart(chart_f, use_container_width=True)

if __name__ == "__main__":
    app()
