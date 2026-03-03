import time
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    st.set_page_config(page_title="Chemotaxis Predator-Prey", layout="wide")
    st.title("Spatial Chemotaxis Predator–Prey")
    st.subheader("Reaction–Diffusion–Chemotaxis System in a Circular Domain")

    st.markdown("""
    This interactive simulator solves a **coupled nonlinear Lotka-Volterra PDE system**
    describing predator–prey interactions with **directed chemotactic motion**.
    Predators migrate up prey gradients, producing spiral hunting fronts
    and collapse zones.
    """)

    # ---------------- APPLICATIONS ----------------
    with st.expander("Explore Applications & Scientific Relevance", expanded=True):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            **Microbial Ecology & Biofilms**
            * **Protist-Bacteria Interaction:** Modeling how predatory ciliates or amoebae track and consume bacterial clusters.
            * **Myxobacteria Swarming:** Studying "wolf-pack" hunting behaviors where bacteria aggregate to digest prey.
            * **Bioremediation:** Predicting how predatory microbes move toward contaminant-degrading prey species.
            """)
        with col_info2:
            st.markdown("""
            **Mathematical Biology & Medicine**
            * **Angiogenesis:** Simulating how endothelial cells migrate toward high concentrations of growth factors (VEGF) secreted by tumors.
            * **Immunology:** Modeling leukocyte (white blood cell) chemotaxis toward pathogens or inflammatory signals.
            * **Pattern Formation:** Analyzing the stability of Turing-like patterns in environments with active transport.
            """)

    # ---------------- REFERENCES ----------------
    with st.expander("📚 Key Reference Papers & Further Reading", expanded=False):
        col_ref1, col_ref2 = st.columns(2)
        with col_ref1:
            st.markdown("""
            **Chemotaxis & Keller–Segel Models**
            * Keller, E. F. & Segel, L. A. (1971). [Model for Chemotaxis.](https://doi.org/10.1016/0022-5193(71)90050-6)
              *Journal of Theoretical Biology, 30(2).* — The original PDE framework for directed cell migration along chemical gradients.
            * Horstmann, D. (2003). [From 1970 Until Present: The Keller–Segel Model in Chemotaxis and Its Consequences.](https://doi.org/10.18452/7929)
              *Jahresbericht der DMV, 105.* — Comprehensive review of mathematical analysis and blow-up behaviour.
            * Painter, K. J. & Hillen, T. (2002). [Volume-Filling and Quorum-Sensing in Models for Chemosensitive Movement.](https://doi.org/10.1139/o02-027)
              *Canadian Applied Mathematics Quarterly, 10(4).* — Extensions preventing density blow-up via volume exclusion.

            **Lotka–Volterra Spatial Dynamics**
            * Turing, A. M. (1952). [The Chemical Basis of Morphogenesis.](https://doi.org/10.1098/rstb.1952.0012)
              *Phil. Trans. R. Soc. B.* — Foundational diffusion-driven instability theory underpinning spatial pattern formation.
            * Mimura, M. & Murray, J. D. (1978). [On a Diffusive Prey–Predator Model Which Exhibits Patchiness.](https://doi.org/10.1007/BF00276918)
              *Journal of Theoretical Biology, 75(3).* — Early demonstration of spatial heterogeneity arising from predator-prey diffusion.
            """)
        with col_ref2:
            st.markdown("""
            **Predator–Prey Pattern Formation**
            * Murray, J. D. (2003). [Mathematical Biology II: Spatial Models and Biomedical Applications.](https://doi.org/10.1007/b98869)
              *Springer.* — Definitive textbook treatment of reaction-diffusion systems and spatial ecological models.
            * Tyson, R. et al. (1999). [Models and Analysis of Chemotactic Bacterial Patterns in a Liquid Medium.](https://doi.org/10.1098/rspb.1999.0742)
              *Proc. Royal Society B, 266.* — Numerical and analytical study of chemotaxis-driven aggregation in bacterial systems.

            **Microbial Predation & Immunology**
            * Velicer, G. J. & Vos, M. (2009). [Sociobiology of the Myxobacteria.](https://doi.org/10.1146/annurev.micro.091208.073326)
              *Annual Review of Microbiology, 63.* — Review of cooperative predation strategies including coordinated swarming.
            * Lauffenburger, D. A. & Horwitz, A. F. (1996). [Cell Migration: A Physically Integrated Molecular Process.](https://doi.org/10.1016/S0092-8674(00)81179-X)
              *Cell, 84(3).* — Mechanistic framework connecting receptor-level chemosensing to directed cell movement.
            * Tranquillo, R. T. & Lauffenburger, D. A. (1987). [Stochastic Model of Leukocyte Chemosensory Movement.](https://doi.org/10.1007/BF02460024)
              *Journal of Mathematical Biology, 25.* — Stochastic derivation of macroscopic chemotaxis equations from receptor dynamics.
            """)

    # ---------------- THEORY ----------------
    st.markdown("### Governing Equations")
    st.latex(r"\frac{\partial P}{\partial t}=D_P\nabla^2P+\mu PN-\beta PQ")
    st.latex(r"\frac{\partial Q}{\partial t}=D_Q\nabla^2Q-\chi\nabla\cdot(Q\nabla P)+\gamma\beta PQ-\delta Q")
    st.latex(r"\frac{\partial N}{\partial t}=-\alpha PN")
    st.latex(r"""
    \begin{aligned}
    P(x,y,t) &:\ \text{Prey density} \\
    Q(x,y,t) &:\ \text{Predator density} \\
    N(x,y,t) &:\ \text{Nutrient concentration} \\
    \\
    D_P,\, D_Q &:\ \text{Diffusion coefficients} \\
    \chi &:\ \text{Chemotactic sensitivity} \\
    \mu &:\ \text{Prey growth rate} \\
    \alpha &:\ \text{Nutrient consumption rate} \\
    \beta &:\ \text{Predation rate} \\
    \gamma &:\ \text{Predator conversion efficiency} \\
    \delta &:\ \text{Predator death rate}
    \end{aligned}
    """)

    # ---------------- SIDEBAR ----------------
    st.sidebar.subheader("Diffusion")
    d_prey = st.sidebar.slider("Diff Prey",      0.0, 0.1, 0.02, format="%.3f")
    d_pred = st.sidebar.slider("Diff Predator",  0.0, 0.1, 0.03, format="%.3f")

    st.sidebar.subheader("Lotka–Volterra")
    mu    = st.sidebar.slider("Prey Growth (μ)",        0.0, 0.1,  0.05,  format="%.3f")
    alpha = st.sidebar.slider("Nutrient Consump (α)",   0.0, 0.1,  0.05,  format="%.3f")
    beta  = st.sidebar.slider("Predation Rate (β)",     0.0, 0.1,  0.03,  format="%.3f")
    gamma = st.sidebar.slider("Predator Eff (γ)",       0.0, 1.0,  0.8,   format="%.2f")
    delta = st.sidebar.slider("Predator Death (δ)",     0.0, 0.01, 0.002, format="%.4f")

    st.sidebar.subheader("Chemotaxis")
    chi = st.sidebar.slider("Chemotaxis Strength (χ)", 0.0, 2.0, 0.8, format="%.2f")

    st.sidebar.subheader("System")
    grid            = 200
    steps_per_frame = st.sidebar.slider("Simulation Speed", 1, 50, 5)
    dt              = 0.1

    # throttle constants — tuned for smoothness
    CHARTS_EVERY = 20  # redraw Altair charts every 20 frames

    # ---------------- OPERATORS ----------------
    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1, 1:-1] = (
            arr[:-2, 1:-1] + arr[2:, 1:-1] +
            arr[1:-1, :-2] + arr[1:-1, 2:] -
            4 * arr[1:-1, 1:-1]
        )
        return lap

    def gradient(arr):
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        gx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / 2
        gy[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / 2
        return gx, gy

    def divergence(fx, fy):
        div = np.zeros_like(fx)
        div[1:-1, 1:-1] = (
            (fx[1:-1, 2:] - fx[1:-1, :-2]) / 2 +
            (fy[2:, 1:-1] - fy[:-2, 1:-1]) / 2
        )
        return div

    def seed_colonies(mask, count, radius, intensity):
        arr = np.zeros_like(mask, float)
        ys, xs = np.where(mask)
        for _ in range(count):
            i = np.random.randint(len(ys))
            cy, cx = ys[i], xs[i]
            yy, xx = np.ogrid[:grid, :grid]
            arr[(yy-cy)**2 + (xx-cx)**2 <= radius**2] = intensity
        return arr

    # ---------------- INIT ----------------
    def reset():
        y, x = np.ogrid[-grid/2:grid/2, -grid/2:grid/2]
        mask = x**2 + y**2 <= (grid/2 - 2)**2
        np.random.seed(42)
        prey     = seed_colonies(mask, 20, 5, 0.5)
        predator = seed_colonies(mask, 10, 4, 0.3)
        nutrient = np.ones((grid, grid)) * mask

        st.session_state.p      = prey
        st.session_state.q      = predator
        st.session_state.n      = nutrient
        st.session_state.mask   = mask
        st.session_state.t      = 0
        st.session_state.frame  = 0
        st.session_state.hist_t = []
        st.session_state.hist_p = []
        st.session_state.hist_q = []
        st.session_state.hist_n = []
        st.session_state.hist_r = []
        st.session_state.lv_init = True

    if "lv_init" not in st.session_state:
        reset()

    if st.sidebar.button("Reset Simulation"):
        reset()

    # ---------------- LAYOUT ----------------
    col_main, col_graph = st.columns([1, 1])
    with col_main:
        st.markdown("### Figure 1 — Spatial Density Fields")
        petri_view = st.empty()
    with col_graph:
        st.markdown("### Figure 2 — Global Population Dynamics")
        chart_pop = st.empty()
        st.markdown("### Figure 3 — Nutrient Depletion")
        chart_nutr = st.empty()
        st.markdown("### Figure 4 — Predator/Prey Ratio")
        chart_ratio = st.empty()

    running = st.toggle("Run Simulation", value=False)

    st.markdown("---")
    st.markdown("**Numerics:** Forward Euler, 5-point finite differences, Keller–Segel chemotaxis + Lotka–Volterra reactions.")

    # ---------------- RENDER ----------------
    def render(P, Q, N, mask, t, frame):
        # Figure 1: spatial image (every frame)
        img = np.zeros((grid, grid, 3))
        img[..., 0] = np.clip(Q * 4, 0, 1)
        img[..., 1] = np.clip(N * 4, 0, 1)
        img[..., 2] = np.clip(P * 4, 0, 1)
        img[~mask]  = 0
        petri_view.image(img, caption=f"Time: {t}", use_column_width=True, clamp=True)

        # Figures 2–4: charts (throttled)
        if frame % CHARTS_EVERY == 0 and len(st.session_state.hist_t) > 0:
            df_pop = pd.DataFrame({
                "Time":     st.session_state.hist_t,
                "Prey":     st.session_state.hist_p,
                "Predator": st.session_state.hist_q,
            })
            chart_pop.altair_chart(
                alt.Chart(df_pop.melt("Time", var_name="Species", value_name="Population"))
                .mark_line().encode(x="Time", y="Population", color="Species")
                .properties(height=200),
                use_container_width=True
            )
            chart_nutr.altair_chart(
                alt.Chart(pd.DataFrame({"Time": st.session_state.hist_t, "Nutrient": st.session_state.hist_n}))
                .mark_line(color="green").encode(x="Time", y="Nutrient").properties(height=150),
                use_container_width=True
            )
            chart_ratio.altair_chart(
                alt.Chart(pd.DataFrame({"Time": st.session_state.hist_t, "Ratio": st.session_state.hist_r}))
                .mark_line(color="orange").encode(x="Time", y="Ratio").properties(height=150),
                use_container_width=True
            )

    # ---------------- LIVE SIMULATION LOOP ----------------
    if running:
        P    = st.session_state.p.copy()
        Q    = st.session_state.q.copy()
        N    = st.session_state.n.copy()
        mask = st.session_state.mask
        t    = st.session_state.t
        frame = st.session_state.frame

        hist_t = list(st.session_state.hist_t)
        hist_p = list(st.session_state.hist_p)
        hist_q = list(st.session_state.hist_q)
        hist_n = list(st.session_state.hist_n)
        hist_r = list(st.session_state.hist_r)

        while True:
            t0 = time.perf_counter()

            # --- physics ---
            for _ in range(steps_per_frame):
                gx, gy = gradient(P)
                chemo  = divergence(Q * gx, Q * gy)

                dP = mu * P * N - beta * P * Q
                dQ = gamma * beta * P * Q - delta * Q
                dN = -alpha * P * N

                P += dt * (dP + d_prey * laplacian(P))
                Q += dt * (dQ + d_pred * laplacian(Q) - chi * chemo)
                N += dt * dN

                P = np.clip(P, 0, 1)
                Q = np.clip(Q, 0, 1)
                N = np.clip(N, 0, 1)
                P[~mask] = Q[~mask] = N[~mask] = 0

                t += 1
                if t % 5 == 0:
                    hist_t.append(t)
                    hist_p.append(float(P.sum()))
                    hist_q.append(float(Q.sum()))
                    hist_n.append(float(N.sum()))
                    hist_r.append(float(Q.sum() / P.sum()) if P.sum() > 0 else 0.0)

            frame += 1

            # write history so render() can read it
            st.session_state.hist_t = hist_t
            st.session_state.hist_p = hist_p
            st.session_state.hist_q = hist_q
            st.session_state.hist_n = hist_n
            st.session_state.hist_r = hist_r

            # --- render ---
            render(P, Q, N, mask, t, frame)

            # --- persist ---
            st.session_state.p     = P.copy()
            st.session_state.q     = Q.copy()
            st.session_state.n     = N.copy()
            st.session_state.t     = t
            st.session_state.frame = frame

            # --- pace to ~20 fps ---
            elapsed = time.perf_counter() - t0
            wait    = (1 / 20) - elapsed
            if wait > 0:
                time.sleep(wait)

    else:
        render(
            st.session_state.p,
            st.session_state.q,
            st.session_state.n,
            st.session_state.mask,
            st.session_state.t,
            frame=0,
        )

if __name__ == "__main__":
    app()
