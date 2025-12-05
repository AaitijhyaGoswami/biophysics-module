import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def app():
    st.title("Spatial Lotka-Volterra (Predator-Prey)")
    st.markdown("""
    **Simulation Details:**
    * **Model:** Reaction-Diffusion on a 2D Petri dish.
    * **Dynamics:** Prey (Blue) consume Nutrient (Green); Predators (Red) consume Prey.
    * **Logic:** Finite Difference Method (Laplacian diffusion).
    """)

    # -----------------------------
    # 1. PARAMETERS
    # -----------------------------
    st.sidebar.subheader("Model Parameters")
    
    # Diffusion coefficients
    D_PREY = st.sidebar.slider("Diff Prey", 0.0, 0.1, 0.02, format="%.3f")
    D_PRED = st.sidebar.slider("Diff Predator", 0.0, 0.1, 0.03, format="%.3f")
    
    # Lotka-Volterra coefficients
    mu = st.sidebar.slider("Prey Growth (μ)", 0.0, 0.1, 0.05, format="%.3f")
    alpha = st.sidebar.slider("Nutrient Consump (α)", 0.0, 0.1, 0.05, format="%.3f")
    beta = st.sidebar.slider("Predation Rate (β)", 0.0, 0.1, 0.03, format="%.3f")
    gamma = st.sidebar.slider("Predator Eff (γ)", 0.0, 1.0, 0.8, format="%.2f")
    delta = st.sidebar.slider("Predator Death (δ)", 0.0, 0.01, 0.002, format="%.4f")

    st.sidebar.subheader("System Settings")
    GRID_SIZE = 200
    STEPS_PER_FRAME = st.sidebar.slider("Simulation Speed", 1, 50, 5)

    # -----------------------------
    # 2. HELPER FUNCTIONS
    # -----------------------------
    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1,1:-1] = (arr[:-2,1:-1] + arr[2:,1:-1] +
                          arr[1:-1,:-2] + arr[1:-1,2:] -
                          4 * arr[1:-1,1:-1])
        return lap

    def create_colonies(mask, num_colonies, radius, intensity):
        arr = np.zeros_like(mask, dtype=float)
        ys, xs = np.where(mask)
        for _ in range(num_colonies):
            if len(ys) > 0:
                idx = np.random.randint(len(ys))
                cy, cx = ys[idx], xs[idx]
                yy, xx = np.ogrid[:GRID_SIZE, :GRID_SIZE]
                dist_sq = (yy - cy)**2 + (xx - cx)**2
                arr[dist_sq <= radius**2] = intensity
        return arr

    # -----------------------------
    # 3. INITIALIZATION
    # -----------------------------
    if 'lv_prey' not in st.session_state:
        st.session_state.lv_initialized = False

    def reset_simulation():
        y, x = np.ogrid[-GRID_SIZE/2:GRID_SIZE/2, -GRID_SIZE/2:GRID_SIZE/2]
        mask = x**2 + y**2 <= (GRID_SIZE/2 - 2)**2
        
        np.random.seed(42)
        prey = create_colonies(mask, 20, 5, 0.5)
        predator = create_colonies(mask, 10, 4, 0.3)
        nutrient = np.ones((GRID_SIZE, GRID_SIZE))
        nutrient[~mask] = 0

        st.session_state.lv_prey = prey
        st.session_state.lv_predator = predator
        st.session_state.lv_nutrient = nutrient
        st.session_state.lv_mask = mask
        st.session_state.lv_time = 0
        
        # History lists
        st.session_state.lv_hist_time = []
        st.session_state.lv_hist_prey = []
        st.session_state.lv_hist_pred = []
        st.session_state.lv_hist_nutr = []
        st.session_state.lv_hist_ratio = []
        
        st.session_state.lv_initialized = True

    if not st.session_state.lv_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    # -----------------------------
    # 4. LAYOUT
    # -----------------------------
    col_main, col_graphs = st.columns([1, 1])
    
    with col_main:
        st.write("### Petri Dish View")
        petri_placeholder = st.empty()
    
    with col_graphs:
        st.write("### Real-time Dynamics")
        # We use a single placeholder for all matplotlib graphs
        graph_placeholder = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    # -----------------------------
    # 5. SIMULATION LOOP
    # -----------------------------
    if run_sim:
        prey = st.session_state.lv_prey
        predator = st.session_state.lv_predator
        nutrient = st.session_state.lv_nutrient
        mask = st.session_state.lv_mask
        
        for _ in range(STEPS_PER_FRAME):
            # Diffusion
            prey += D_PREY * laplacian(prey)
            predator += D_PRED * laplacian(predator)
            
            # Reactions
            delta_prey = mu * prey * nutrient - beta * prey * predator
            delta_pred = gamma * beta * prey * predator - delta * predator
            delta_nutrient = -alpha * prey * nutrient
            
            prey += delta_prey
            predator += delta_pred
            nutrient += delta_nutrient
            
            # Clamping
            prey = np.clip(prey, 0, 1)
            predator = np.clip(predator, 0, 1)
            nutrient = np.clip(nutrient, 0, 1)
            prey[~mask] = 0
            predator[~mask] = 0
            nutrient[~mask] = 0
            
            st.session_state.lv_time += 1
            
            # Record History
            if st.session_state.lv_time % 5 == 0:
                st.session_state.lv_hist_time.append(st.session_state.lv_time)
                st.session_state.lv_hist_prey.append(np.sum(prey))
                st.session_state.lv_hist_pred.append(np.sum(predator))
                st.session_state.lv_hist_nutr.append(np.sum(nutrient))
                
                # Safe ratio
                s_prey = np.sum(prey)
                st.session_state.lv_hist_ratio.append(
                    np.sum(predator) / s_prey if s_prey > 0 else 0
                )

        # Save State
        st.session_state.lv_prey = prey
        st.session_state.lv_predator = predator
        st.session_state.lv_nutrient = nutrient
        
        st.rerun()

    # -----------------------------
    # 6. RENDERING
    # -----------------------------
    # A. Image Update
    prey = st.session_state.lv_prey
    predator = st.session_state.lv_predator
    nutrient = st.session_state.lv_nutrient
    mask = st.session_state.lv_mask

    img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    img[..., 0] = np.clip(predator * 4, 0, 1) # Red
    img[..., 1] = np.clip(nutrient * 4, 0, 1) # Green
    img[..., 2] = np.clip(prey * 4, 0, 1)     # Blue
    img[~mask] = 0
    
    petri_placeholder.image(img, caption=f"Time: {st.session_state.lv_time} mins", use_column_width=True, clamp=True)

    # B. Graph Update (Matplotlib)
    if len(st.session_state.lv_hist_time) > 0:
        # Create a Matplotlib Figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        
        times = st.session_state.lv_hist_time
        
        # Plot 1: Population
        ax1.plot(times, st.session_state.lv_hist_prey, 'b-', label='Prey')
        ax1.plot(times, st.session_state.lv_hist_pred, 'r-', label='Predator')
        ax1.set_ylabel("Biomass")
        ax1.legend(loc="upper right", fontsize="small")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Population Dynamics")
        
        # Plot 2: Nutrient
        ax2.plot(times, st.session_state.lv_hist_nutr, 'g-', label='Nutrient')
        ax2.set_ylabel("Nutrient Level")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Ratio
        ax3.plot(times, st.session_state.lv_hist_ratio, 'm-', label='Pred/Prey Ratio')
        ax3.set_ylabel("Ratio")
        ax3.set_xlabel("Time (minutes)")  # <--- HERE IS YOUR LABEL
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        graph_placeholder.pyplot(fig)
