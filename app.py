import streamlit as st

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Biophysics Suite | IISc",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# MODULE IMPORTS
# -------------------------------------------------------------------------
# This dictionary will hold your simulations.
# The try/except blocks ensure the app still runs even if a file is missing.
modules = {}

try:
    from simulations import bacterial_growth
    modules["Bacterial Growth"] = bacterial_growth
except ImportError:
    pass

try:
    from simulations import lotka_volterra
    modules["Lotka-Volterra"] = lotka_volterra
except ImportError:
    pass

try:
    from simulations import megaplate
    modules["Mega Plate Evolution"] = megaplate
except ImportError:
    pass

try:
    from simulations import rps
    modules["Rock-Paper-Scissors"] = rps
except ImportError:
    pass

# -------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.title("Biophysics Suite")
st.sidebar.markdown("**IISc Fall of Code '25**")

# Define the navigation options
options = ["Home"] + list(modules.keys())
page = st.sidebar.radio("Select Simulation:", options)

st.sidebar.markdown("---")
st.sidebar.caption("Project Team:")
st.sidebar.info(
    "**Physics Major**\n*Simulation & Modeling*\n\n"
    "**Bio Major**\n*Theoretical Framework*"
)

# -------------------------------------------------------------------------
# MAIN ROUTING
# -------------------------------------------------------------------------
if page == "Home":
    st.title("Computational Biophysics Simulation Suite")
    st.markdown("### Stochastic & Deterministic Modeling of Biological Dynamics")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Welcome.** This interactive dashboard explores the intersection of statistical physics and biology.
        Developed as part of the **IISc Fall of Code** initiative, these modules visualize how simple mathematical rules give rise to complex biological phenomena.
        
        #### The Modules:
        
        **1. Bacterial Growth (Stochastic)**
        * *Concept:* Reaction-Diffusion systems with stochastic noise.
        * *Visuals:* Real-time colony morphology and nutrient depletion fields.
        
        **2. Lotka-Volterra (Predator-Prey)**
        * *Concept:* Coupled differential equations on a spatial grid.
        * *Visuals:* Phase-space dynamics and oscillating populations.
        
        **3. The Mega Plate (Evolution)**
        * *Concept:* Spatial evolutionary dynamics.
        * *Visuals:* Adaptation of bacteria to increasing antibiotic gradients (inspired by the Kishony Lab).
        
        **4. Rock-Paper-Scissors (Game Theory)**
        * *Concept:* Non-transitive cyclic dominance in ecosystems.
        * *Visuals:* Biodiversity preservation through spatial spiral waves.
        """)
        
        st.info("👈 **Select a simulation from the sidebar to begin.**")

    with col2:
        st.markdown("### Project Status")
        st.success("✅ **Simulations Live**")
        st.warning("📝 **Reports In Progress**")
        st.write("Full theoretical derivations and research reports are currently being drafted.")

else:
    # Run the selected simulation module
    if page in modules:
        modules[page].app()
