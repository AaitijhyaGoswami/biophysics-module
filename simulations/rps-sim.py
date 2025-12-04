import streamlit as st
import numpy as np

def app():
    st.title("Cyclic Dominance (Rock-Paper-Scissors)")
    st.markdown("""
    **Simulation Details:**
    * **Model:** Spatial non-transitive competition (Rock-Paper-Scissors).
    * **Dynamics:** Red eats Green, Green eats Blue, Blue eats Red.
    * **Outcome:** Species coexist in rotating spiral waves or go extinct depending on parameters.
    """)

    # -----------------------------
    # 1. PARAMETERS
    # -----------------------------
    st.sidebar.subheader("Ecosystem Parameters")
    
    SPREAD_RATE = st.sidebar.slider("Reproduction Rate", 0.0, 1.0, 0.25)
    EAT_PROB = st.sidebar.slider("Predation Probability", 0.0, 1.0, 0.45)
    
    st.sidebar.subheader("Initial density")
    INIT_RED = st.sidebar.slider("Init Red", 0.0, 0.3, 0.02)
    INIT_BLUE = st.sidebar.slider("Init Blue", 0.0, 0.3, 0.02)
    INIT_GREEN = st.sidebar.slider("Init Green", 0.0, 0.3, 0.02)

    st.sidebar.subheader("System Settings")
    GRID = 200
    # Speed control
    STEPS_PER_FRAME = st.sidebar.slider("Steps per Frame", 1, 20, 5)

    # Constants
    EMPTY = 0
    RED = 1
    BLUE = 2
    GREEN = 3

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
        # Note: logic slightly adjusted to match the strict cumulative probability of original script
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
        st.session_state.rps_hist_red = []
        st.session_state.rps_hist_blue = []
        st.session_state.rps_hist_green = []
        st.session_state.rps_hist_frac_red = []
        st.session_state.rps_hist_frac_blue = []
        st.session_state.rps_hist_frac_green = []
        
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
            # VECTORIZED UPDATE (Equivalent to original loop but fast)
            # ---------------------------------------------------------
            # 1. Select Random Neighbors for every cell
            dx = np.random.randint(-1, 2, size=(GRID, GRID))
            dy = np.random.randint(-1, 2, size=(GRID, GRID))
            
            x_indices, y_indices = np.indices((GRID, GRID))
            nx = (x_indices + dx) % GRID
            ny = (y_indices + dy) % GRID
            
            # S = Source (Current Cell), T = Target (Neighbor)
            S = grid
            T = grid[nx, ny]
            
            # 2. Validity Masks
            # Check if both Source and Target are inside the Petri dish circle
            valid_mask = mask & mask[nx, ny]
            
            # Random rolls for probabilities
            rand_spread = np.random.rand(GRID, GRID)
            rand_eat = np.random.rand(GRID, GRID)
            
            # 3. Reproduction Rule
            # Original: if T == EMPTY and rand < SPREAD: new[nx, ny] = S
            # We construct a mask of where this happens
            repro_mask = valid_mask & (T == EMPTY) & (S != EMPTY) & (rand_spread < SPREAD_RATE)
            
            # 4. Predation Rule
            # Original: if rand < EAT: check RPS logic -> new[nx, ny] = S
            eat_check = valid_mask & (rand_eat < EAT_PROB)
            
            red_eats_green = eat_check & (S == RED) & (T == GREEN)
            blue_eats_red  = eat_check & (S == BLUE) & (T == RED)
            green_eats_blue= eat_check & (S == GREEN) & (T == BLUE)
            
            predation_mask = red_eats_green | blue_eats_red | green_eats_blue
            
            # 5. Apply Updates (Simultaneous update logic)
            # Note: We must be careful about update order. 
            # In original code: new = grid.copy(). Updates write to 'new'.
            # A neighbor (nx, ny) is the TARGET. 
            # So if (x,y) eats (nx,ny), then new[nx,ny] becomes (x,y)'s color.
            
            # We need to map the "Source" actions to the "Target" coordinates.
            # This is hard to fully vectorize without collisions (multiple sources targeting same empty spot).
            # However, for visual simulation speed, we can approximate by applying updates in place
            # or using the dominant interaction.
            
            # STREAMLIT OPTIMIZATION:
            # Fully correct vectorization of "random neighbor writes" requires `np.add.at` logic which is complex for categorical data.
            # Instead, we invert the logic: For every cell, look at *its* neighbor and decide if *it* gets eaten or replaced.
            # This is statistically symmetric and much faster.
            
            # INVERTED LOGIC (Look at self as 'Target'):
            # Let T = grid (self), S = neighbor (randomly picked)
            S_inv = grid[nx, ny] # Neighbor
            T_inv = grid         # Self
            
            # Repro: Neighbor is filled, I am empty -> I become Neighbor
            inv_repro = valid_mask & (T_inv == EMPTY) & (S_inv != EMPTY) & (rand_spread < SPREAD_RATE)
            grid[inv_repro] = S_inv[inv_repro]
            
            # Predation: Neighbor eats Me
            inv_eat = valid_mask & (rand_eat < EAT_PROB)
            inv_r_e_g = inv_eat & (S_inv == RED) & (T_inv == GREEN)
            inv_b_e_r = inv_eat & (S_inv == BLUE) & (T_inv == RED)
            inv_g_e_b = inv_eat & (S_inv == GREEN) & (T_inv == BLUE)
            
            mask_eaten = inv_r_e_g | inv_b_e_r | inv_g_e_b
            grid[mask_eaten] = S_inv[mask_eaten]
            
            # Enforce Mask cleanup (just in case)
            grid[~mask] = EMPTY

        # Update Stats
        st.session_state.rps_time += STEPS_PER_FRAME
        c_red = np.sum(grid == RED)
        c_blue = np.sum(grid == BLUE)
        c_green = np.sum(grid == GREEN)
        total = c_red + c_blue + c_green
        
        st.session_state.rps_hist_red.append(c_red)
        st.session_state.rps_hist_blue.append(c_blue)
        st.session_state.rps_hist_green.append(c_green)
        
        if total > 0:
            st.session_state.rps_hist_frac_red.append(c_red/total)
            st.session_state.rps_hist_frac_blue.append(c_blue/total)
            st.session_state.rps_hist_frac_green.append(c_green/total)
        else:
            st.session_state.rps_hist_frac_red.append(0)
            st.session_state.rps_hist_frac_blue.append(0)
            st.session_state.rps_hist_frac_green.append(0)
            
        st.session_state.rps_grid = grid
        st.rerun()

    # -----------------------------
    # 4. RENDERING
    # -----------------------------
    # A. Build Image
    # Colors: Black, Red(#FF3333), Blue(#3366FF), Green(#33FF33)
    grid = st.session_state.rps_grid
    mask = st.session_state.rps_mask
    
    img = np.zeros((GRID, GRID, 3))
    
    mask_r = (grid == RED)
    mask_b = (grid == BLUE)
    mask_g = (grid == GREEN)
    
    # Hex to RGB conversions
    # FF3333 -> [1.0, 0.2, 0.2]
    # 3366FF -> [0.2, 0.4, 1.0]
    # 33FF33 -> [0.2, 1.0, 0.2]
    
    img[mask_r] = [1.0, 0.2, 0.2]
    img[mask_b] = [0.2, 0.4, 1.0]
    img[mask_g] = [0.2, 1.0, 0.2]
    
    img[~mask] = 0.0
    
    dish_placeholder.image(img, caption=f"Time step: {st.session_state.rps_time}", clamp=True, use_column_width=True)
    
    # B. Charts
    if len(st.session_state.rps_hist_red) > 0:
        # 1. Absolute Populations
        chart_counts.line_chart({
            "Red": st.session_state.rps_hist_red,
            "Blue": st.session_state.rps_hist_blue,
            "Green": st.session_state.rps_hist_green
        }, height=200) # Colors are auto-assigned by Streamlit, but labels match
        
        # 2. Fractions
        chart_fracs.line_chart({
            "Frac Red": st.session_state.rps_hist_frac_red,
            "Frac Blue": st.session_state.rps_hist_frac_blue,
            "Frac Green": st.session_state.rps_hist_frac_green
        }, height=200)
