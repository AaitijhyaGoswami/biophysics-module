import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    # -----------------------
    # 0. CONFIGURATION & HELPERS
    # -----------------------
    # Fixed Physics Constants (Hidden from user for stability)
    D_DIFFUSION = 0.8   # Rate of metabolite diffusion
    DECAY_RATE = 0.02   # Rate at which metabolites break down
    FEED_RATE = 0.03    # Rate at which fresh Nutrient N enters system
    
    def laplacian(field):
        """Discrete Laplace operator for diffusion"""
        return (
            np.roll(field, 1, axis=0) + 
            np.roll(field, -1, axis=0) + 
            np.roll(field, 1, axis=1) + 
            np.roll(field, -1, axis=1) - 
            4 * field
        )

    st.title("Syntrophic Cross-Feeding & Oscillations")
    st.markdown("""
    **The Mechanism:**
    1.  [Image of bacterial cross feeding diagram]
    2.  **Species A (Red)** eats Nutrient N $\\to$ secretes Metabolite X.
    3.  **Species B (Green)** eats Metabolite X $\\to$ secretes Toxin Y.
    4.  **Toxin Y** inhibits Species A.
    
    **Result:** A 'Red' bloom creates food for 'Green'. 'Green' blooms and poisons 'Red'. 'Red' dies, 'Green' starves. The cycle repeats.
    """)

    # -----------------------
    # 1. SIMPLIFIED PARAMETERS
    # -----------------------
    with st.sidebar:
        st.header("Control Panel")
        
        # A. Simulation Speed
        GRID_SIZE = 150
        STEPS_PER_FRAME = st.slider("Simulation Speed", 1, 30, 8)
        
        st.subheader("Biological Rates")
        
        # 1. Growth/Death Balance
        repro_rate = st.slider("Reproduction Rate", 0.1, 1.0, 0.4, 
                             help="How fast bacteria multiply when food is abundant.")
        
        death_rate = st.slider("Death Rate", 0.001, 0.1, 0.02, format="%.3f",
                             help="Base mortality rate.")

        # 2. Interaction Strength
        st.subheader("Interactions")
        
        metabolic_yield = st.slider("Metabolic Yield", 0.1, 2.0, 0.8,
                                  help="How much food X is produced by A? High = B grows easier.")
        
        toxicity = st.slider("Inhibition Strength", 0.0, 5.0, 2.5,
                           help="How strongly B poisons A. High = Strong Oscillations.")

        # 3. System Energy
        nutrient_supply = st.slider("Nutrient Supply (Chemostat)", 0.0, 1.0, 0.5,
                                  help="Availability of base resource N.")

        if st.button("Reset Simulation"):
            st.session_state.cf_initialized = False
            st.rerun()

    # -----------------------
    # 2. STATE INITIALIZATION
    # -----------------------
    if 'cf_initialized' not in st.session_state:
        st.session_state.cf_initialized = False

    def init_simulation():
        # Grid: 0=Empty, 1=Species A, 2=Species B
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        
        # Seed random clusters
        num_blobs = 15
        for _ in range(num_blobs):
            rx, ry = np.random.randint(0, GRID_SIZE, 2)
            # Create a blob
            y, x = np.ogrid[-10:10, -10:10]
            mask = x*x + y*y <= 25
            
            # Place A and B near each other
            region_x = (np.arange(20) + rx - 10) % GRID_SIZE
            region_y = (np.arange(20)[:,None] + ry - 10) % GRID_SIZE
            
            # 50/50 mix in blobs
            blob_grid = np.random.choice([0, 1, 2], (20, 20), p=[0.2, 0.4, 0.4])
            grid[np.ix_(region_x, region_y[0])] = np.where(mask, blob_grid, 0)

        # Fields: N (Base), X (Food for B), Y (Toxin for A)
        field_n = np.ones((GRID_SIZE, GRID_SIZE)) * nutrient_supply
        field_x = np.zeros((GRID_SIZE, GRID_SIZE))
        field_y = np.zeros((GRID_SIZE, GRID_SIZE))

        # History
        hist = {"time": [], "pop_a": [], "pop_b": [], "res_x": []}

        st.session_state.cf_grid = grid
        st.session_state.cf_n = field_n
        st.session_state.cf_x = field_x
        st.session_state.cf_y = field_y
        st.session_state.cf_time = 0
        st.session_state.cf_hist = hist
        st.session_state.cf_initialized = True

    if not st.session_state.cf_initialized:
        init_simulation()

    # -----------------------
    # 3. CORE SIMULATION LOGIC
    # -----------------------
    def step_simulation():
        grid = st.session_state.cf_grid
        N = st.session_state.cf_n
        X = st.session_state.cf_x
        Y = st.session_state.cf_y
        
        # --- A. Field Dynamics (Reaction-Diffusion) ---
        
        # 1. Diffusion
        N += D_DIFFUSION * laplacian(N)
        X += D_DIFFUSION * laplacian(X)
        Y += D_DIFFUSION * laplacian(Y)
        
        # 2. Production/Consumption based on Bacteria Presence
        mask_A = (grid == 1)
        mask_B = (grid == 2)
        
        # A eats N -> makes X
        # Consumption rate depends on N availability
        uptake_N = mask_A * N * 0.1
        N -= uptake_N
        X += uptake_N * metabolic_yield # A converts N to X
        
        # B eats X -> makes Y
        uptake_X = mask_B * X * 0.1
        X -= uptake_X
        Y += uptake_X * metabolic_yield # B converts X to Y
        
        # 3. Decay & Chemostat Feed
        N += FEED_RATE * (nutrient_supply - N) # Replenish N
        X -= DECAY_RATE * X
        Y -= DECAY_RATE * Y
        
        # Clamp fields
        N = np.clip(N, 0, 1)
        X = np.clip(X, 0, 1)
        Y = np.clip(Y, 0, 1)

        # --- B. Cellular Automata (Birth/Death) ---
        
        # Probabilities
        rand_birth = np.random.random(grid.shape)
        rand_death = np.random.random(grid.shape)
        
        # 1. Death Rules
        # A dies from natural causes OR Toxin Y
        prob_death_A = death_rate + (Y * toxicity * 0.1)
        # B dies from natural causes (starvation included implicitly via lack of birth)
        prob_death_B = death_rate
        
        kill_A = mask_A & (rand_death < prob_death_A)
        kill_B = mask_B & (rand_death < prob_death_B)
        
        grid[kill_A] = 0
        grid[kill_B] = 0
        
        # 2. Birth Rules (into empty neighbors)
        # We simplify spatial spreading: random check of neighbor
        # Shift grid to find neighbors
        shifts = [(0,1), (0,-1), (1,0), (-1,0)]
        idx = np.random.randint(0, 4) # Pick one random direction for this frame to save compute
        sx, sy = shifts[idx]
        
        neighbor_grid = np.roll(grid, sx, axis=0)
        neighbor_grid = np.roll(neighbor_grid, sy, axis=1)
        
        # Empty spots
        mask_empty = (grid == 0)
        
        # If neighbor is A, can it reproduce into current empty spot?
        # A needs Nutrient N
        growth_A = repro_rate * N
        birth_A = mask_empty & (neighbor_grid == 1) & (rand_birth < growth_A)
        
        # If neighbor is B, can it reproduce into current empty spot?
        # B needs Metabolite X
        growth_B = repro_rate * X
        birth_B = mask_empty & (neighbor_grid == 2) & (rand_birth < growth_B)
        
        # Apply Births (A has priority in this simple model if collision, or just sequential)
        grid[birth_A] = 1
        grid[birth_B] = 2 # B overwrites A if collision (rare), effectively competition

        # Save State
        st.session_state.cf_grid = grid
        st.session_state.cf_n = N
        st.session_state.cf_x = X
        st.session_state.cf_y = Y
        st.session_state.cf_time += 1
        
        # Stats
        if st.session_state.cf_time % 2 == 0:
            hist = st.session_state.cf_hist
            hist["time"].append(st.session_state.cf_time)
            hist["pop_a"].append(int(np.sum(grid == 1)))
            hist["pop_b"].append(int(np.sum(grid == 2)))
            hist["res_x"].append(float(np.mean(X)))

    # -----------------------
    # 4. RUNNER & VISUALIZATION
    # -----------------------
    col_main, col_plots = st.columns([1.5, 1])
    
    with col_main:
        st.write("### Spatial Dynamics")
        # Custom Legend
        legend = """
        <div style='display: flex; gap: 20px; font-size: 14px;'>
            <div><span style='color:#FF4444; font-size:20px'>●</span> <b>Species A</b> (Producer)</div>
            <div><span style='color:#44FF44; font-size:20px'>●</span> <b>Species B</b> (Consumer)</div>
        </div>
        """
        st.markdown(legend, unsafe_allow_html=True)
        dish_container = st.empty()
        
    with col_plots:
        st.write("### Populations (A vs B)")
        chart_container = st.empty()
        st.write("### Metabolic Field (X)")
        field_container = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    if run_sim:
        for _ in range(STEPS_PER_FRAME):
            step_simulation()
        st.rerun()

    # RENDER IMAGE
    grid = st.session_state.cf_grid
    
    # Create RGB Image
    # R channel = Species A
    # G channel = Species B
    # B channel = Toxin Y (faint visualization)
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    
    mask_a = (grid == 1)
    mask_b = (grid == 2)
    
    img[mask_a] = [1.0, 0.2, 0.2] # Red
    img[mask_b] = [0.2, 1.0, 0.2] # Green
    
    # Background (traces of chemicals)
    # We overlay the metabolite X in blueish for visibility
    X = st.session_state.cf_x
    img[~(mask_a | mask_b), 2] = X[~(mask_a | mask_b)] 

    dish_container.image(img, use_column_width=True, clamp=True)

    # RENDER CHARTS
    hist = st.session_state.cf_hist
    if len(hist["time"]) > 2:
        df = pd.DataFrame({
            "Time": hist["time"],
            "Species A": hist["pop_a"],
            "Species B": hist["pop_b"]
        })
        
        melted = df.melt("Time", var_name="Species", value_name="Population")
        
        c = alt.Chart(melted).mark_line().encode(
            x="Time", 
            y="Population", 
            color=alt.Color("Species", scale=alt.Scale(range=['#FF4444', '#44FF44']))
        ).properties(height=200)
        
        chart_container.altair_chart(c, use_container_width=True)
        
        # Field strength chart
        df_x = pd.DataFrame({"Time": hist["time"], "Metabolite X": hist["res_x"]})
        c_x = alt.Chart(df_x).mark_area(opacity=0.3, color='blue').encode(
            x="Time", y="Metabolite X"
        ).properties(height=150)
        field_container.altair_chart(c_x, use_container_width=True)

if __name__ == "__main__":
    app()
