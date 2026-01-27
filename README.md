# 🧬 Computational Ecology Suite
### Reaction-Diffusion & Spatial Dynamics Simulations

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> **Databased Fall of Code 2025** > **Authors:** Aaitijhya Goswami & Ritaja Dutta

---

## 📖 Overview

This project is an interactive **Computational Ecology Dashboard** developed to visualize complex biological systems through the lens of statistical physics and differential equations. 

By integrating **Reaction-Diffusion Partial Differential Equations (PDEs)** with **Probabilistic Cellular Automata**, the application simulates how microscopic local interactions—such as competition, predation, and mutation—drive emergent global phenomena like spiral waves, fractal branching, and evolutionary rescue.

The simulation suite runs on **Streamlit**, leveraging **NumPy** for high-performance vectorized grid operations to model thousands of agents in real-time.

---

## 🚀 Key Modules

### 1. Multi-Colony Competition
**Bio-Physical Concept:** Resource Depletion & Growth Phases  
Visualizes the competition between 12 distinct bacterial colonies for a single, finite nutrient source on an agar plate. It accurately models the three phases of bacterial growth:
* **Lag Phase:** Adaptation and synthesis of raw materials.
* **Log (Exponential) Phase:** Rapid division driven by nutrient abundance.
* **Stationary Phase:** Growth arrest due to resource exhaustion ($N \to 0$).

**Mathematics:**
Modeled as a consumption-diffusion system where biomass growth ($dB/dt$) is coupled to nutrient depletion ($-dN/dt$) via Monod kinetics.

### 2. Spatial Lotka-Volterra (Predator-Prey)
**Bio-Physical Concept:** Trophic Cascades & Diffusive Instability  
Simulates the spatiotemporal dynamics between a Prey species (e.g., *E. coli*) and a Predator species (e.g., *Bdellovibrio*) in a finite environment. Unlike standard ODE models, this spatial implementation reveals **traveling waves** and **phase shifts** (Prey peaks $\to$ Predator peaks $\to$ Collapse).

**Governing Equations:**
$$\frac{\partial P}{\partial t} = D_p \nabla^2 P + \mu P N - \beta P Q$$
$$\frac{\partial Q}{\partial t} = D_q \nabla^2 Q + \gamma \beta P Q - \delta Q$$
*(Where $P$ is Prey, $Q$ is Predator, $N$ is Nutrient)*

### 3. The MEGA Plate Experiment
**Bio-Physical Concept:** Evolutionary Rescue & Fitness Landscapes  
Inspired by the Harvard Medical School experiment, this module models **stepwise evolution** across a spatial antibiotic gradient. Bacteria must mutate to survive in concentric zones of increasing toxicity.
* **Zone 1 (Black):** Safe haven (Wild Type thrives).
* **Zone 2 (Grey):** Moderate toxicity (Single Mutant survives).
* **Zone 3 (White):** High toxicity (Superbug survives).

**Algorithm:** Stochastic Cellular Automata with probabilistic mutation events allowing lineages to "tunnel" through fitness barriers.

### 4. Cyclic Dominance (Rock-Paper-Scissors)
**Bio-Physical Concept:** Non-Transitive Competition  
Models the stability of biodiversity through intransitive competition loops, often observed in *E. coli* Colicin production.
* 🔵 **Sensitive (Paper):** Fast grower, no toxin. Beats Resistant.
* 🟢 **Resistant (Rock):** Immune to toxin, metabolic cost. Beats Toxic.
* 🔴 **Toxic (Scissors):** Produces poison, slow grower. Beats Sensitive.

**Outcome:** Low mobility leads to stable **spiral waves**, while high mobility causes stochastic extinction (biodiversity collapse).

### 5. Metabolic Cross-Feeding
**Bio-Physical Concept:** Syntrophy & Mutualism  
Simulates an obligate mutualistic relationship where species depend on each other's metabolic byproducts (e.g., the Human Gut Microbiome or Biofilms).
* **Syntrophy:** Species B eats the waste of Species A.
* **Toxicity Feedback:** Species B produces a byproduct that is toxic to A.
This creates complex "chasing" patterns where partners must stay close to feed but far enough to avoid poisoning.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Core Logic:** `NumPy` (Vectorized Finite Difference Method, Monte Carlo steps)
* **Visualization:** `Streamlit` (UI), `Altair` (Real-time Analytics), `Matplotlib`
* **Image Processing:** `SciPy` (Gaussian filters for chemical diffusion halos)

---

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AaitijhyaGoswami/biophysics-module.git](https://github.com/AaitijhyaGoswami/biophysics-module.git)
    cd biophysics-module
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```text
biophysics-module/
├── app.py               # Entry point for the Streamlit app
├── modules/
│   ├── colony.py        # Module 1: Multi-Colony Competition
│   ├── lotka.py         # Module 2: Predator-Prey
│   ├── mega_plate.py    # Module 3: Evolutionary Rescue
│   ├── rps.py           # Module 4: Cyclic Dominance
│   └── cross_feed.py    # Module 5: Cross-Feeding
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas for new biological models or optimization improvements.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Built with ❤️ for Science | Databased Fall of Code 2025
</p>
