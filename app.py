import streamlit as st

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Biophysics Suite",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# MODULE IMPORTS
# -------------------------------------------------------------------------
modules = {}

try:
    from simulations import growth_sim
    modules["Bacterial Growth"] = growth_sim
except ImportError:
    pass

try:
    from simulations import lotka_volterra
    modules["Lotka-Volterra"] = lotka_volterra
except ImportError:
    pass

try:
    from simulations import mega_plate
    modules["MEGA Plate Evolution"] = mega_plate
except ImportError:
    pass

try:
    from simulations import rps_sim
    modules["Cyclic Dominance"] = rps_sim
except ImportError:
    pass

try:
    from simulations import cross_feeding
    modules["Cross-Feeding"] = cross_feeding
except ImportError:
    pass

# -------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.title("Biophysics Suite")

# Define the navigation options
options = ["Home"] + list(modules.keys())
page = st.sidebar.radio("Select Simulation:", options)

st.sidebar.markdown("---")
st.sidebar.caption("Project Team:")
st.sidebar.info(
    "**Aaitijhya Goswami**\n*Simulation & Modeling*\n\n"
    "**Ritaja Dutta**\n*Theoretical Framework*"
)

# -------------------------------------------------------------------------
# MAIN ROUTING
# -------------------------------------------------------------------------
if page == "Home":
    # HOME PAGE CONTENT
    st.title("Computational Biophysics Simulation Suite")
    st.markdown("### Stochastic & Deterministic Modeling of Biological Dynamics")
    st.markdown("---")

    # Abstracts and descriptions
    st.markdown("""
    #### Welcome

    This interactive dashboard facilitates the simulation of various biological processes inspired by 
    **statistical physics** and **ecological modeling**. The tools and visualizations aim to provide insight into
    complex biological dynamics using simplified mathematical models. These modules represent an integration
    of stochastic and deterministic principles to explore emergent behaviors of individual species and populations.
    """)

    st.markdown("## The Modules")
    st.markdown("---")

    # Abstracts, examples, and applications for each module
    st.markdown("""
    ### **1. Bacterial Growth Model**
    - **Abstract:** This model simulates *reaction-diffusion systems with stochastic noise*, showing bacterial colony growth and the interplay between nutrient availability and diffusion-limited aggregation. The results highlight phenomena such as branching and dendritic growth patterns.
    - **Applications:**  
        - Modeling biofilm formation and bacterial growth patterns.  
        - Studying environmental changes that affect colony formation.  
        - Antibiotic testing and resistance.  
    - **Examples:** Formation of patterns in *Bacillus subtilis* and *Pseudomonas aeruginosa* colonies.
    """)

    st.markdown("""
    ### **2. Lotka-Volterra Ecosystem Dynamics**
    - **Abstract:** The Lotka-Volterra model uses coupled differential equations to represent predator-prey 
      interactions on a spatial grid. These simulations go beyond traditional ODEs to incorporate 
      spatial effects, revealing complex phase-space dynamics and oscillatory behaviors.
    - **Applications:**  
        - Population dynamics and predator-prey systems in ecology.  
        - Biodiversity and conservation efforts.  
    - **Examples:** Fox and rabbit population fluctuations, microbial interactions, or even financial market modeling.
    """)

    st.markdown("""
    ### **3. MEGA Plate Evolution**
    - **Abstract:** Based on the famous *MEGA Plate Experiment* by the Kishony Lab, this model captures 
      *antibiotic resistance evolution under spatially distributed drug gradients*. Spatial dynamics, stochastic
      mutation rates, and stepwise resistance development are simulated to illustrate how bacteria adapt 
      under extreme selective pressure.
    - **Applications:**  
        - Antibiotic resistance research.  
        - Evolutionary biology studies on adaptation mechanisms.  
        - Informing pharmaceutical and healthcare strategies against resistant pathogens.  
    - **Examples:** Expanding microbe populations in the presence of increasing drug concentrations.
    """)

    st.markdown("""
    ### **4. Cyclic Dominance**
    - **Abstract:** This module captures non-transitive interactions (e.g., *A beats B, B beats C, C beats A*),
      typical of ecosystems characterized by cyclic dominance. The model uses stochastic spatial lattice 
      updates to reveal the emergence of *spiral waves* and the preservation of biodiversity via non-transitive
      interactions.
    - **Applications:**  
        - Studying biodiversity in ecosystems.  
        - Ecological modeling and population stability analysis.  
    - **Examples:** Rock-Paper-Scissors dynamics in colicin-producing *E. coli* strains or the mating
      strategies of *Uta stansburiana* lizards.
    """)

    st.markdown("""
    ### **5. Cross-Feeding (Syntrophy)**
    - **Abstract:** Cross-feeding, or metabolic interdependence, is a phenomenon where one organism produces a 
      resource that another consumes. This simulation illustrates spatiotemporal dynamics of producers (A) 
      secreting a nutrient (X), and consumers (B) that consume X while producing a toxin (Y) that affects A. The 
      interplay between nutrient production, consumption, and spatial dynamics leads to emergent behaviors 
      such as population waves.
    - **Applications:**  
        - Studying metabolic interdependence in microbial ecology.  
        - Applications in synthetic biology and bioengineering (e.g., co-culture fermentation).  
        - Understanding nutrient cycling in ecosystem ecology.  
    - **Examples:** Metabolic interactions in *E. coli* strains and gut microbiota community dynamics.
    """)

    # Closing message
    st.markdown("### Ready to start?")
    st.info("👈 **Select a simulation from the sidebar to begin exploring!**")

else:
    # Run the selected simulation module
    if page in modules:
        modules[page].app()
