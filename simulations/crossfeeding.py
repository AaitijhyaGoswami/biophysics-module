import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    st.title("Microbial Cross-Feeding (Syntrophy)")
    st.markdown("""
    **Simulation Details:**
    * **Mechanism:** Obligate Cross-feeding (Commensalism/Mutualism).
    * **Metabolism:**
        1.  <span style='color:#00CCFF'>**Producer (Blue)**</span> eats **Glucose** $\\to$ excretes **Acetate**.
        2.  <span style='color:#FF00FF'>**Consumer (Magenta)**</span> eats **Acetate**.
    * **Dynamics:** Spatial "chasing" waves. The Consumer must physically follow the Producer to survive.
    """, unsafe_allow_html=True)

    # -----------------------------
    # 1. PARAMETERS
    # -----------------------------
    st.sidebar.subheader("Metabolic Parameters")

    # Diffusion
    D_BAC = 0.05  # Bacterial Diffusion
    D_NUT = 0.25  # Nutrient Diffusion (small molecules move faster)

    # Species 1 (Producer)
    mu1 = st.sidebar.slider("Producer Growth Rate", 0.1, 0.5, 0.3, 0.01)
    # Species 2 (Consumer)
    mu2 = st.sidebar.slider("Consumer Growth Rate", 0.1, 0.5, 0.2, 0.01)
    
    # Stoichiometry
    prod_rate = st.sidebar.slider("Byproduct Production", 0.1, 1.0, 0.6, help="How much Acetate is made per unit of Producer growth")
    
    st.sidebar.subheader("System Settings")
    GRID = 120
    STEPS_PER_FRAME = st.sidebar.slider("Simulation Speed", 1, 30, 10)
    
    # Constants for Monod Kinetics
    K_A = 0.1 # Half-saturation constant for Glucose
    K_B = 0.1 # Half-saturation constant for Acetate
    DEATH = 0.01 # Natural decay rate

    # -----------------------------
    # 2. HELPER FUNCTIONS
    # -----------------------------
    def laplacian(arr):
        """Finite difference approximation of diffusion in 2D"""
        lap = -4 * arr
        lap += np.roll(arr, 1, axis=0)
        lap += np.roll(arr, -1, axis=0)
        lap += np.roll(arr, 1, axis=1)
        lap += np.roll(arr, -1, axis=1)
        return lap

    # -----------------------------
    # 3. INITIALIZATION
    # -----------------------------
    if 'cf_state' not in st.session_state:
        st.session_state.cf_initialized = False

    def reset_simulation():
        # 1. Species (S1, S2)
        # Start with small random inoculations in the center
        s1 = np.zeros((GRID, GRID))
        s2 = np.zeros((GRID, GRID))
        
        center = GRID // 2
        r = 10
        y, x = np.ogrid[-center:GRID-center, -center:GRID-center]
        mask = x*x + y*y <= r*r
        
        # Seed random bacteria in center circle
        s1[mask] = np.random.rand(np.sum(mask)) * 0.5
        s2[mask] = np.random.rand(np.sum(mask)) * 0.5

        # 2. Nutrients (N_A = Glucose, N_B = Acetate)
        # Petri dish starts full of Glucose (A), empty of Acetate (B)
        n_a = np.ones((GRID, GRID)) * 1.0
        n_b = np.zeros((GRID, GRID))

        st.session_state.cf_s1 = s1
        st.session_state.cf_s2 = s2
        st.session_state.cf_na = n_a
        st.session_state.cf_nb = n_b
        st.session_state.cf_time = 0.0
        
        # History lists
        st.session_state.cf_hist_time = []
        st.session_state.cf_hist_s1 = []
        st.session_state.cf_hist_s2 = []
        st.session_state.cf_hist_na = []
        st.session_state.cf_hist_nb = []
        
        st.session_state.cf_initialized = True

    if not st.session_state.cf_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    # -----------------------------
    # 4. MAIN LOOP
    # -----------------------------
    col_vis, col_stats = st.columns([1, 1])

    with col_vis:
        st.write("### Metabolic Wavefronts")
        
        # HTML Legend
        legend_html = """
        <style>
            .legend-item { display: inline-flex; align-items: center; margin-right: 15px; font-size: 13px; }
            .box { width: 12px; height: 12px; border: 1px solid #555; margin-right: 5px; }
        </style>
        <div style="margin-bottom: 10px;">
            <div class="legend-item">
                <span class="box" style="background-color: #00CCFF;"></span>Producer (eats Glucose)
            </div>
            <div class="legend-item">
                <span class="box" style="background-color: #FF00FF;"></span>Consumer (eats Byproduct)
            </div>
            <div class="legend-item">
                <span class="box" style="background-color: #330033;"></span>Depleted Zone
            </div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        dish_placeholder = st.empty()

    with col_stats:
        st.write("### Chemostat Dynamics")
        chart_biomass = st.empty()
        chart_nutrient = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    if run_sim:
        S1 = st.session_state.cf_s1
        S2 = st.session_state.cf_s2
        Na = st.session_state.cf_na
        Nb = st.session_state.cf_nb
        dt = 0.1 # Time step

        for _ in range(STEPS_PER_FRAME):
            # 1. Diffusion
            # Nutrients diffuse faster than cells
            D_S1 = D_BAC * laplacian(S1)
            D_S2 = D_BAC * laplacian(S2)
            D_Na = D_NUT * laplacian(Na)
            D_Nb = D_NUT * laplacian(Nb)

            # 2. Reaction (Monod Kinetics)
            # Growth rates
            growth_1 = mu1 * (Na / (K_A + Na)) * S1
            growth_2 = mu2 * (Nb / (K_B + Nb)) * S2
            
            # 3. Update State
            
            # Species 1: Grows, Dies, Diffuses
            S1 += (growth_1 - DEATH * S1 + D_S1) * dt
            
            # Species 2: Grows, Dies, Diffuses
            S2 += (growth_2 - DEATH * S2 + D_S2) * dt
            
            # Nutrient A (Glucose): Consumed by S1, Diffuses
            # Yield assumed 1.0 for simplicity
            Na += (-growth_1 + D_Na) * dt
            
            # Nutrient B (Byproduct): Produced by S1, Consumed by S2, Diffuses
            Nb += (prod_rate * growth_1 - growth_2 + D_Nb) * dt

            # 4. Clamping (No negative concentrations)
            S1 = np.clip(S1, 0, 10) # Soft cap for stability
            S2 = np.clip(S2, 0, 10)
            Na = np.clip(Na, 0, 1)  # Normalized nutrient
            Nb = np.clip(Nb, 0, 1)

        # Update History
        st.session_state.cf_time += dt * STEPS_PER_FRAME
        
        # Log totals
        st.session_state.cf_hist_time.append(st.session_state.cf_time)
        st.session_state.cf_hist_s1.append(np.sum(S1))
        st.session_state.cf_hist_s2.append(np.sum(S2))
        st.session_state.cf_hist_na.append(np.mean(Na)) # Mean concentration
        st.session_state.cf_hist_nb.append(np.mean(Nb))

        # Save State
        st.session_state.cf_s1 = S1
        st.session_state.cf_s2 = S2
        st.session_state.cf_na = Na
        st.session_state.cf_nb = Nb
        st.rerun()

    # -----------------------------
    # 5. RENDERING
    # -----------------------------
    
    # A. Image Generation
    # We want to visualize S1 and S2 clearly.
    # S1 = Producer (Cyan/Blue-Green)
    # S2 = Consumer (Magenta/Pink)
    # Background = Black
    
    S1 = st.session_state.cf_s1
    S2 = st.session_state.cf_s2
    
    img = np.zeros((GRID, GRID, 3))
    
    # Normalize for display
    # We use a soft scaling factor so we can see low concentrations
    scale = 3.0
    
    # Red Channel: S2 (Consumer)
    img[..., 0] = np.clip(S2 * scale, 0, 1)
    
    # Green Channel: S1 (Producer)
    img[..., 1] = np.clip(S1 * scale * 0.8, 0, 1) # Slightly lower gain
    
    # Blue Channel: S1 + S2 (Shared)
    # This makes S1 appear Cyan (G+B) and S2 appear Magenta (R+B)
    img[..., 2] = np.clip((S1 + S2) * scale, 0, 1)
    
    dish_placeholder.image(img, caption=f"Time: {st.session_state.cf_time:.1f} hrs", clamp=True, use_column_width=True)

    # B. Charts
    if len(st.session_state.cf_hist_time) > 0:
        
        # 1. Biomass Chart
        df_bio = pd.DataFrame({
            'Time': st.session_state.cf_hist_time,
            'Producer': st.session_state.cf_hist_s1,
            'Consumer': st.session_state.cf_hist_s2
        })
        df_bio_melt = df_bio.melt('Time', var_name='Species', value_name='Biomass')
        
        chart_b = alt.Chart(df_bio_melt).mark_line().encode(
            x=alt.X('Time', axis=alt.Axis(title='Time (Hours)')),
            y=alt.Y('Biomass', axis=alt.Axis(title='Total Population')),
            color=alt.Color('Species', scale=alt.Scale(
                domain=['Producer', 'Consumer'],
                range=['#00CCFF', '#FF00FF']
            ))
        ).properties(height=200)
        chart_biomass.altair_chart(chart_b, use_container_width=True)
        
        # 2. Nutrient Chart
        df_nut = pd.DataFrame({
            'Time': st.session_state.cf_hist_time,
            'Glucose (Feed)': st.session_state.cf_hist_na,
            'Acetate (Byproduct)': st.session_state.cf_hist_nb
        })
        df_nut_melt = df_nut.melt('Time', var_name='Nutrient', value_name='Concentration')
        
        chart_n = alt.Chart(df_nut_melt).mark_line().encode(
            x=alt.X('Time', axis=alt.Axis(title='Time (Hours)')),
            y=alt.Y('Concentration', axis=alt.Axis(title='Avg Concentration (mM)')),
            strokeDash=alt.StrokeDash('Nutrient', legend=alt.Legend(title='Nutrient Type')), # Dashed lines for nutrients
            color=alt.Color('Nutrient', scale=alt.Scale(
                domain=['Glucose (Feed)', 'Acetate (Byproduct)'],
                range=['gray', 'orange']
            ))
        ).properties(height=200)
        chart_nutrient.altair_chart(chart_n, use_container_width=True)

if __name__ == "__main__":
    app()
