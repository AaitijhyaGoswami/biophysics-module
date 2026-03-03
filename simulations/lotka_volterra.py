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
    Predators migrate up prey gradients, producing spiral hunting fronts and collapse zones.
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
    d_prey = st.sidebar.slider("Diff Prey",     0.0, 0.1, 0.02, format="%.3f")
    d_pred = st.sidebar.slider("Diff Predator", 0.0, 0.1, 0.03, format="%.3f")

    st.sidebar.subheader("Lotka–Volterra")
    mu    = st.sidebar.slider("Prey Growth (μ)",       0.0, 0.1,  0.05,  format="%.3f")
    alpha = st.sidebar.slider("Nutrient Consump (α)",  0.0, 0.1,  0.05,  format="%.3f")
    beta  = st.sidebar.slider("Predation Rate (β)",    0.0, 0.1,  0.03,  format="%.3f")
    gamma = st.sidebar.slider("Predator Eff (γ)",      0.0, 1.0,  0.8,   format="%.2f")
    delta = st.sidebar.slider("Predator Death (δ)",    0.0, 0.01, 0.002, format="%.4f")

    st.sidebar.subheader("Chemotaxis")
    chi = st.sidebar.slider("Chemotaxis Strength (χ)", 0.0, 2.0, 0.8, format="%.2f")

    st.sidebar.subheader("System")
    grid            = 200
    steps_per_frame = st.sidebar.slider("Simulation Speed", 1, 50, 5)
    dt              = 0.1
    CHARTS_EVERY    = 20  # redraw charts every N frames

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
            arr[(yy - cy)**2 + (xx - cx)**2 <= radius**2] = intensity
        return arr

    # ---------------- INIT ----------------
    def reset():
        y, x = np.ogrid[-grid/2:grid/2, -grid/2:grid/2]
        mask     = x**2 + y**2 <= (grid/2 - 2)**2
        np.random.seed(42)
        prey     = seed_colonies(mask, 20, 5, 0.5)
        predator = seed_colonies(mask, 10, 4, 0.3)
        nutrient = np.ones((grid, grid)) * mask

        st.session_state.p       = prey
        st.session_state.q       = predator
        st.session_state.n       = nutrient
        st.session_state.mask    = mask
        st.session_state.t       = 0
        st.session_state.frame   = 0
        st.session_state.hist_t  = []
        st.session_state.hist_p  = []
        st.session_state.hist_q  = []
        st.session_state.hist_n  = []
        st.session_state.hist_r  = []
        st.session_state.lv_init = True

    if "lv_init" not in st.session_state:
        reset()

    if st.sidebar.button("Reset Simulation"):
        reset()

    # ---------------- LAYOUT ----------------
    # Static headers in columns for visual layout
    hdr_l, hdr_r = st.columns(2)
    hdr_l.markdown("### Figure 1 — Spatial Density Fields")
    hdr_r.markdown("### Figure 2 — Global Population Dynamics")

    # Placeholders defined at TOP LEVEL (not inside `with col:`) so the
    # while-loop websocket flush works correctly on every frame
    img_col, pop_col = st.columns(2)
    petri_view = img_col.empty()
    chart_pop  = pop_col.empty()

    nutr_spacer, nutr_col = st.columns(2)
    nutr_spacer.markdown("**RGB legend:** 🔴 Predator | 🟢 Nutrient | 🔵 Prey")
    nutr_col.markdown("### Figure 3 — Nutrient Depletion")
    nutr_sp2, chart_nutr_col = st.columns(2)
    chart_nutr = chart_nutr_col.empty()

    ratio_spacer, ratio_hdr_col = st.columns(2)
    ratio_hdr_col.markdown("### Figure 4 — Predator/Prey Ratio")
    ratio_sp2, chart_ratio_col = st.columns(2)
    chart_ratio = chart_ratio_col.empty()

    running = st.toggle("Run Simulation", value=False)

    st.markdown("---")
    st.markdown("**Numerics:** Forward Euler, 5-point finite differences, Keller–Segel chemotaxis + Lotka–Volterra reactions.")

    # ---------------- RENDER ----------------
    def render(P, Q, N, mask, t, frame):
        # Figure 1 — every frame
        img = np.zeros((grid, grid, 3))
        img[..., 0] = np.clip(Q * 4, 0, 1)
        img[..., 1] = np.clip(N * 4, 0, 1)
        img[..., 2] = np.clip(P * 4, 0, 1)
        img[~mask]  = 0
        petri_view.image(img, caption=f"Time: {t}", use_column_width=True, clamp=True)

        # Figures 2–4 — throttled
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
                alt.Chart(pd.DataFrame({
                    "Time": st.session_state.hist_t,
                    "Nutrient": st.session_state.hist_n
                })).mark_line(color="green").encode(x="Time", y="Nutrient").properties(height=150),
                use_container_width=True
            )
            chart_ratio.altair_chart(
                alt.Chart(pd.DataFrame({
                    "Time": st.session_state.hist_t,
                    "Ratio": st.session_state.hist_r
                })).mark_line(color="orange").encode(x="Time", y="Ratio").properties(height=150),
                use_container_width=True
            )

    # ---------------- SIMULATION LOOP ----------------
    if running:
        P     = st.session_state.p.copy()
        Q     = st.session_state.q.copy()
        N     = st.session_state.n.copy()
        mask  = st.session_state.mask
        t     = st.session_state.t
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
