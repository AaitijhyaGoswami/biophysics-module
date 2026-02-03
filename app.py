import streamlit as st

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Computational Biophysics Suite",
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
st.sidebar.caption("Research Team:")
st.sidebar.info(
    "**Aaitijhya Goswami**\n*Simulation & Modeling*\n\n"
    "**Ritaja Dutta**\n*Theoretical Framework*"
)

st.sidebar.markdown("---")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Science-logo.png", use_column_width=True)

# -------------------------------------------------------------------------
# MAIN ROUTING
# -------------------------------------------------------------------------
if page == "Home":
    # HOME PAGE CONTENT
    st.title("Computational Biophysics Simulation Suite")
    st.markdown("### Stochastic & Deterministic Modeling of Biological Dynamics")
    st.markdown("---")

    st.markdown("""
        **Welcome.** This interactive dashboard facilitates the visualization of mathematical models 
        aimed at decoding complex biological systems through computational physics. Each model presented in this suite explores 
        various aspects of population dynamics, ecosystem resilience, and biophysics.

        #### Introduction:
        This research project addresses fundamental principles of biophysics and applies them to simulate dynamic biological systems. 
        The modules use computational methods to simulate and visualize phenomena ranging from nutrient competition to predator-prey 
        interactions. With the ability to explore spatial structures, non-linear dynamics, and stochasticity, this platform offers insights 
        into emergent spatial and temporal patterns in biological systems.
        """)

        st.info("👈 **Use the sidebar to choose a simulation module.**")


    st.markdown("#### Research Modules")
    st.markdown("---")

    # ABSTRACT FOR EACH MODULE
    st.markdown("""
    ### **1. Bacterial Growth Model**
    - **Abstract:** This model simulates *reaction-diffusion systems with stochastic noise*, showing bacterial colony growth and the interplay between nutrient availability and diffusion-limited aggregation. The results highlight phenomena such as branching and dendritic growth patterns.
    - **Applications:** Biofilm growth modeling, antibiotic testing, and drug resistance mechanisms.
    - **Examples:** Formation of patterns in *Bacillus subtilis* colonies.
    """)

    st.markdown("""
    ### **2. Lotka-Volterra Dynamics**
    - **Abstract:** The Lotka-Volterra model uses coupled differential equations to represent predator-prey interactions. It is extended here to a spatial grid to simulate more realistic ecological scenarios, including spatial segregation and oscillatory dynamics.
    - **Applications:** Modeling predator-prey systems, ecological management, and evolutionary studies.
    - **Examples:** Rabbits and foxes interacting in an ecosystem; plankton population dynamics in marine ecosystems.
    """)

    st.markdown("""
    ### **3. MEGA Plate Evolution (Antibiotic Resistance)**
    - **Abstract:** This module replicates the experimental setup of the MEGA-plate experiment by Kishony Lab, where *E. coli* evolve under increasing antibiotic gradients. The model reveals stepwise evolution to higher resistance levels through stochastic reproduction and mutation.
    - **Applications:** Antibiotic resistance estimation, microbial evolution, and evolutionary biology education.
    - **Examples:** Observation of boundary expansion of resistant strains under changing environmental conditions.
    """)

    st.markdown("""
    ### **4. Cyclic Dominance (Rock-Paper-Scissors)**
    - **Abstract:** This model captures non-transitive interactions (e.g., *A beats B, B beats C, C beats A*), generating cyclic dominance patterns seen in microbial ecosystems. Spiral waves emerge from spatial lattice updates governed by stochastic birth and competition rules.
    - **Applications:** Ecosystem modeling, community ecology, and biodiversity studies.
    - **Examples:** Studies of colicin-producing *E. coli* strains and lizard mating strategies (*Uta stansburiana*).
    """)

    st.markdown("""
    ### **5. Cross-Feeding Metabolic Systems**
    - **Abstract:** Cross-feeding is a form of syntrophy where one species produces metabolites that are consumed by another species. This simulation visualizes two populations: a producer species generating a beneficial resource and a consumer species that metabolizes it while producing a toxic byproduct.
    - **Applications:** Studying cooperative behavior and syntrophy, biotechnology (co-culture systems), and microbial community dynamics.
    - **Examples:** Cooperation among *Escherichia coli* strains and evolution of division of labor.
    """)

    st.markdown("### Ready to explore?")
    st.info("��� **Select a simulation from the sidebar to dive deeper into these fascinating systems!**")

else:
    # Run the selected simulation module
    if page in modules:
        modules[page].app()
