import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# ======================================================
# Spatial Chemotaxis Lotka–Volterra Simulator
# ======================================================

def app():
    st.title("Chemotaxis Predator–Prey (Reaction–Diffusion)")
    st.markdown("""
    **Model:** Keller–Segel chemotaxis + Lotka–Volterra  
    **Mechanism:** Predators move *up the prey gradient*  
    **Result:** Spiral hunting fronts & collapsing prey islands
    """)

    # ---------------- Sidebar ----------------
    st.sidebar.subheader("Diffusion")
    d_prey = st.sidebar.slider("Prey Diffusion", 0.0, 0.1, 0.02)
    d_pred = st.sidebar.slider("Predator Diffusion", 0.0, 0.1, 0.03)

    st.sidebar.subheader("Lotka–Volterra")
    mu = st.sidebar.slider("Prey Growth μ", 0.0, 0.1, 0.05)
    alpha = st.sidebar.slider("Nutrient Use α", 0.0, 0.1, 0.05)
    beta = st.sidebar.slider("Predation β", 0.0, 0.1, 0.03)
    gamma = st.sidebar.slider("Predator Eff γ", 0.0, 1.0, 0.8)
    delta = st.sidebar.slider("Predator Death δ", 0.0, 0.01, 0.002)

    st.sidebar.subheader("Chemotaxis")
    chi = st.sidebar.slider("Chemotaxis Strength χ", 0.0, 2.0, 0.8)

    st.sidebar.subheader("System")
    grid = 200
    steps_per_frame = st.sidebar.slider("Speed", 1, 50, 5)

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

    def seed(mask, n, r, val):
        arr = np.zeros_like(mask, float)
        ys, xs = np.where(mask)
        for _ in range(n):
            i = np.random.randint(len(ys))
            cy, cx = ys[i], xs[i]
            yy, xx = np.ogrid[:grid, :grid]
            arr[(yy - cy)**2 + (xx - cx)**2 <= r*r] = val
        return arr

    # ---------------- Reset ----------------
    if "lv_init" not in st.session_state:
        st.session_state.lv_init = False

    def reset():
        y, x = np.ogrid[-grid/2:grid/2, -grid/2:grid/2]
        mask = x**2 + y**2 <= (grid/2 - 2)**2

        prey = seed(mask, 20, 5, 0.6)
        predator = seed(mask, 10, 4, 0.4)
        nutrient = np.ones((grid, grid)) * mask

        st.session_state.p = prey
        st.session_state.q = predator
        st.session_state.n = nutrient
        st.session_state.mask = mask
        st.session_state.t = 0
        st.session_state.hist = {"t": [], "p": [], "q": [], "n": [], "r": []}
        st.session_state.lv_init = True

    if not st.session_state.lv_init:
        reset()

    if st.sidebar.button("Reset"):
        reset()
        st.rerun()

    col1, col2 = st.columns(2)
    petri = col1.empty()
    plot = col2.empty()

    run = st.toggle("Run Simulation")

    # ---------------- Simulation ----------------
    if run:
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
                st.session_state.hist["t"].append(st.session_state.t)
                st.session_state.hist["p"].append(P.sum())
                st.session_state.hist["q"].append(Q.sum())
                st.session_state.hist["n"].append(N.sum())
                st.session_state.hist["r"].append(Q.sum()/P.sum() if P.sum() > 0 else 0)

        st.session_state.p, st.session_state.q, st.session_state.n = P, Q, N
        st.rerun()

    # ---------------- Visualization ----------------
    P = st.session_state.p
    Q = st.session_state.q
    N = st.session_state.n
    mask = st.session_state.mask

    img = np.zeros((grid, grid, 3))
    img[...,0] = np.clip(Q*4, 0, 1)
    img[...,1] = np.clip(N*4, 0, 1)
    img[...,2] = np.clip(P*4, 0, 1)
    img[~mask] = 0

    petri.image(img, caption=f"Time: {st.session_state.t}", use_column_width=True)

    if len(st.session_state.hist["t"]) > 5:
        df = pd.DataFrame(st.session_state.hist)
        plot.altair_chart(
            alt.Chart(df.melt("t"))
            .mark_line()
            .encode(x="t", y="value", color="variable"),
            use_container_width=True
        )

if __name__ == "__main__":
    app()
