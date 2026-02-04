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
    modules["Syntrophy"] = cross_feeding
except ImportError:
    pass

# -------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.title("Biophysics Suite")

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
    st.title("Computational Biophysics Simulation Suite")
    st.markdown("### Stochastic & Deterministic Modeling of Biological Dynamics")
    st.markdown("---")

    st.markdown("""
    #### Welcome

    This interactive dashboard facilitates the simulation of various biological processes inspired by 
    statistical physics and ecological modeling. The tools and visualizations aim to provide insight into
    complex biological dynamics using simplified mathematical models. These modules represent an integration
    of stochastic and deterministic principles to explore emergent behaviors of individual species and populations.
    """)

    st.info("**Click on the hyperlinks below to learn more!**")

    st.markdown("## The Modules")
    st.markdown("---")

    # 1. Bacterial Growth
    st.markdown("""
    ### **1. [Bacterial Growth Model](https://en.wikipedia.org/wiki/Bacterial_growth)**
    - **Abstract:** This model simulates reaction-diffusion systems with stochastic noise, showing bacterial colony growth and the interplay between nutrient availability and diffusion-limited aggregation. The results highlight phenomena such as branching and dendritic growth patterns.
    - **Applications:** - Modeling biofilm architecture and metabolic gradients.  
        - Studying chemotaxis and nutrient-seeking behaviors in heterogeneous environments.  
        - Predicting colony expansion rates in clinical diagnostics.  
    - **Examples:** Branching morphogenesis in *Bacillus subtilis* and swarming motility patterns in *Pseudomonas aeruginosa*.
    """)

    # 2. Lotka-Volterra
    st.markdown("""
    ### **2. [Lotka-Volterra Ecosystem Dynamics](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)**
    - **Abstract:** The Lotka-Volterra model uses coupled differential equations to represent predator-prey 
      interactions on a spatial grid. These simulations go beyond traditional ODEs to incorporate 
      spatial effects, revealing complex phase-space dynamics and oscillatory behaviors.
    - **Applications:** - Population dynamics and stability analysis in microbial ecology.  
        - Modeling phage-bacteria dynamics within the human microbiome.  
        - Analyzing oscillations in trophic levels for conservation biology.  
    - **Examples:** Predator-prey oscillations between *Didinium nasutum* and *Paramecium caudatum*, or the *Lynx canadensis* and *Lepus americanus* (Snowshoe hare) system.
    """)

    # 3. MEGA Plate Evolution
    st.markdown("""
    ### **3. [MEGA Plate Evolution](https://www.biorxiv.org/content/10.1101/2021.12.23.474071v1.full)**
    - **Abstract:** Based on the Microbial Evolution and Growth Arena (MEGA) plate experiment, this model captures 
      antibiotic resistance evolution under spatially distributed drug gradients. Spatial dynamics, stochastic
      mutation rates, and stepwise resistance development are simulated to illustrate how bacteria adapt 
      under extreme selective pressure.
    - **Applications:** - Predicting the emergence of Multi-Drug Resistance (MDR).  
        - Optimizing antibiotic dosing schedules to minimize evolutionary escape.  
        - Informing healthcare strategies against resistant pathogens.  
    - **Examples:** *Escherichia coli* evolving resistance to increasing concentrations of Trimethoprim and *Staphylococcus aureus* adaptation landscapes.
    """)
    

    # 4. Cyclic Dominance
    st.markdown("""
    ### **4. [Cyclic Dominance](https://en.wikipedia.org/wiki/Cyclic_succession)**
    - **Abstract:** This module captures non-transitive interactions (e.g., A beats B, B beats C, C beats A),
      typical of ecosystems characterized by cyclic dominance. The model uses stochastic spatial lattice 
      updates to reveal the emergence of spiral waves and the preservation of biodiversity via non-transitive
      interactions.
    - **Applications:** - Studying the maintenance of biodiversity in complex ecosystems.  
        - Game Theory applications in evolutionary biology and social dynamics.  
        - Stability analysis of polymorphic populations.
    - **Examples:** The triad of colicin-producing, sensitive, and resistant strains of *Escherichia coli*, and the mating strategies of the Side-blotched lizard (*Uta stansburiana*).
    """)

    # 5. Cross-Feeding (Syntrophy)
    st.markdown("""
    ### **5. [Syntrophy](https://en.wikipedia.org/wiki/Syntrophy)**
    - **Abstract:** Cross-feeding, or metabolic interdependence, is a phenomenon where one organism produces a 
      resource that another consumes. This simulation illustrates spatiotemporal dynamics of producers (A) 
      secreting a nutrient (X), and consumers (B) that consume X while producing a toxin (Y) that affects A. The 
      interplay between nutrient production, consumption, and spatial dynamics leads to emergent behaviors 
      such as population waves.
    - **Applications:** - Engineering synthetic microbial consortia for bioengineering and biofuels.  
        - Understanding nutrient cycling and metabolic syntrophy in anaerobic digestion.  
        - Studying human gut microbiota community stability.
    - **Examples:** Interspecies electron transfer between *Geobacter metallireducens* and *Methanosarcina barkeri*, or amino acid cross-feeding in auxotrophic *Saccharomyces cerevisiae*.
    """)

    # Closing message
    st.markdown("### Ready to start?")
    st.info("👈 **Select a simulation from the sidebar to begin exploring!**")

else:
    # Run the selected simulation module
    if page in modules:
        modules[page].app()
