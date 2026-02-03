import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    st.title("Spatial Chemotaxis Predator–Prey")
    st.markdown("""
    **Model:** Keller–Segel Chemotaxis + Lotka–Volterra  
    **Dynamics:** Predators actively climb prey gradients  
    **System:** 2D circular Petri dish with nutrient coupling
    """)

    # ---------------- Sidebar ----------------
    st.sidebar.subheader("Diffusion")
    d_prey = st.sidebar.slider("Diff Prey", 0.0, 0.1, 0.02, format="%.3f")
    d_pred = st.sidebar.slider("Diff Predator", 0.0, 0.1, 0.03, format="%.3f")

    st.sidebar.subheader("Lotka–Volterra")
    mu = st.sidebar.slider("Prey Growth (μ)", 0.0, 0.1, 0.05, format="%.3f")
    alpha = st.sidebar.slider("Nutrient Consump (α)", 0.0, 0.1, 0.05, format="%.3f")
    beta = st.sidebar.slider("Predation Rate (β)", 0.0, 0.1, 0.03, format="%.3f")
    gamma = st.sidebar.slider("Predator Eff (γ)", 0.0, 1.0, 0.8, format="%.2f")
    delta = st.sidebar.slider("Predator Death (δ)", 0.0, 0.01, 0.002, format="%.4f")

    st.sidebar.subheader("Chemotaxis")
    chi = st.sidebar.slider("Chemotaxis Strength (χ)", 0.0, 2.0, 0.8, format="%.2f")

    st.sidebar.subheader("System")
    grid = 200
    steps_per_frame = st.sidebar.slider("Simulation Speed", 1, 50, 5)

    dt = 0.1

    # ---------------- Operators ----------------
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
            idx = np.random.randint(len(ys))
            cy, cx = ys[idx], xs[idx]
            yy, xx = np.ogrid[:grid, :grid]
            arr[(yy-cy)**2 + (xx-cx)**2 <= radius**2] = intensity
        return arr

    # ---------------- Init ----------------
    if "lv_init" not in st.session_state:
        st.session_state.lv_init = False

    def reset():
        y, x = np.ogrid[-grid/2:grid/2, -grid/2:grid/2]
        mask = x**2 + y**2 <= (grid/2 - 2)**2

        np.random.seed(42)
        prey = seed_colonies(mask, 20, 5, 0.5)
        predator = seed_colonies(mask, 10, 4, 0.3)
        nutrient = np.ones((grid, grid)) * mask

        st.session_state.p = prey
        st.session_state.q = predator
        st.session_state.n = nutrient
        st.session_state.mask = mask
        st.session_state.t = 0

        st.session_state.hist_t = []
        st.session_state.hist_p = []
        st.session_state.hist_q = []
        st.session_state.hist_n = []
        st.session_state.hist_r = []

        st.session_state.lv_init = True

    if not st.session_state.lv_init:
        reset()

    if st.sidebar.button("Reset Simulation"):
        reset()
        st.rerun()

    col_main, col_graph = st.columns([1,1])
    petri_view = col_main.empty()

    with col_graph:
        chart_pop = st.empty()
        chart_nutr = st.empty()
        chart_ratio = st.empty()

    running = st.toggle("Run Simulation", value=False)

    # ---------------- Simulation ----------------
    if running:
        P = st.session_state.p
        Q = st.session_state.q
        N = st.session_state.n
        mask = st.session_state.mask

        for _ in range(steps_per_frame):
            gx, gy = gradient(P)
            Jx = Q * gx
            Jy = Q * gy
            chemo = divergence(Jx, Jy)

            dP = mu*P*N - beta*P*Q
            dQ = gamma*beta*P*Q - delta*Q
            dN = -alpha*P*N

            P += dt*(dP + d_prey*laplacian(P))
            Q += dt*(dQ + d_pred*laplacian(Q) - chi*chemo)
            N += dt*dN

            P = np.clip(P, 0, 1)
            Q = np.clip(Q, 0, 1)
            N = np.clip(N, 0, 1)

            P[~mask] = Q[~mask] = N[~mask] = 0

            st.session_state.t += 1
            if st.session_state.t % 5 == 0:
                st.session_state.hist_t.append(st.session_state.t)
                st.session_state.hist_p.append(P.sum())
                st.session_state.hist_q.append(Q.sum())
                st.session_state.hist_n.append(N.sum())
                st.session_state.hist_r.append(Q.sum()/P.sum() if P.sum()>0 else 0)

        st.session_state.p, st.session_state.q, st.session_state.n = P, Q, N
        st.rerun()

    # ---------------- Visualization ----------------
    P = st.session_state.p
    Q = st.session_state.q
    N = st.session_state.n
    mask = st.session_state.mask

    img = np.zeros((grid, grid, 3))
    img[...,0] = np.clip(Q*4,0,1)
    img[...,1] = np.clip(N*4,0,1)
    img[...,2] = np.clip(P*4,0,1)
    img[~mask] = 0

    petri_view.image(img, caption=f"Time: {st.session_state.t}", use_column_width=True, clamp=True)

    if len(st.session_state.hist_t) > 0:
        df_pop = pd.DataFrame({
            "Time": st.session_state.hist_t,
            "Prey": st.session_state.hist_p,
            "Predator": st.session_state.hist_q
        })
        df_m = df_pop.melt("Time", var_name="Species", value_name="Population")

        pop_plot = alt.Chart(df_m).mark_line().encode(
            x="Time", y="Population", color="Species"
        ).properties(height=200)
        chart_pop.altair_chart(pop_plot, use_container_width=True)

        df_n = pd.DataFrame({"Time": st.session_state.hist_t,"Nutrient": st.session_state.hist_n})
        chart_nutr.altair_chart(
            alt.Chart(df_n).mark_line(color="green").encode(x="Time", y="Nutrient").properties(height=150),
            use_container_width=True)

        df_r = pd.DataFrame({"Time": st.session_state.hist_t,"Ratio": st.session_state.hist_r})
        chart_ratio.altair_chart(
            alt.Chart(df_r).mark_line(color="orange").encode(x="Time", y="Ratio").properties(height=150),
            use_container_width=True)

if __name__ == "__main__":
    app()
