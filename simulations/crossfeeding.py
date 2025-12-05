# app.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(layout="wide", page_title="Cross-feeding A↔B", initial_sidebar_state="expanded")

# -----------------------
# Helper: laplacian (periodic)
# -----------------------
def laplacian(field):
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4 * field
    )

# -----------------------
# App UI / Parameters
# -----------------------
st.title("Cross-feeding (A ↔ B) — Spatial Reaction-Diffusion CA")
st.markdown("""
A simple spatial cross-feeding model:
- **A** consumes nutrient **N** and excretes **M_A**.
- **B** consumes **M_A** and excretes **M_B**, which inhibits **A**.
- Metabolites diffuse and decay. Local reproduction is probabilistic and depends on resource availability.
""")

with st.sidebar:
    st.header("Simulation parameters")

    GRID = st.slider("Grid size (N × N)", 100, 400, 220, step=20)
    PETRI_WIDTH = st.slider("Petri display width (px)", 300, 1000, 700, step=50)

    STEPS_PER_FRAME = st.slider("Steps per frame", 1, 25, 6)
    dt = st.number_input("Timestep (dt)", value=1.0, step=0.1)

    st.subheader("Species parameters")
    p_spread_A = st.slider("Base spread A (prob)", 0.0, 1.0, 0.20)
    p_spread_B = st.slider("Base spread B (prob)", 0.0, 1.0, 0.18)
    death_A = st.slider("Death prob A", 0.0, 0.1, 0.002)
    death_B = st.slider("Death prob B", 0.0, 0.1, 0.002)

    st.subheader("Resource / Metabolite physics")
    D_m = st.slider("Metabolite diffusion D_m", 0.0, 1.0, 0.6)
    decay_m = st.slider("Metabolite decay", 0.0, 1.0, 0.01)
    prod_A = st.slider("A produces M_A (per A per dt)", 0.0, 1.0, 0.12)
    prod_B = st.slider("B produces M_B (per B per dt)", 0.0, 1.0, 0.08)
    cons_N_by_A = st.slider("A consumption of N (per A per dt)", 0.0, 1.0, 0.06)
    cons_MA_by_B = st.slider("B consumption of M_A (scales growth)", 0.0, 1.0, 0.12)

    st.subheader("Interaction strengths")
    K_N = st.slider("Half-sat nutrient K_N (A growth)", 0.01, 5.0, 0.5)
    K_MA = st.slider("Half-sat M_A K_MA (B growth)", 0.01, 5.0, 0.4)
    inhib_MB_on_A = st.slider("Inhibition of A by M_B (0=no, 1=strong)", 0.0, 2.0, 1.2)

    st.subheader("Initial densities")
    init_A = st.slider("Init A fraction", 0.0, 0.5, 0.015)
    init_B = st.slider("Init B fraction", 0.0, 0.5, 0.015)
    init_empty = 1.0 - (init_A + init_B)
    st.caption(f"Init empty (rest) ≈ {init_empty:.3f}")

    if st.button("Reset and reinitialize"):
        st.session_state.clear()

# -----------------------
# Constants
# -----------------------
EMPTY = 0
A = 1
B = 2

# -----------------------
# Session-state initialization
# -----------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = False

def init_state():
    # Circular mask for petri dish
    yy, xx = np.indices((GRID, GRID))
    center = GRID // 2
    radius = GRID // 2 - 2
    mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2

    # species grid: EMPTY / A / B
    grid = np.zeros((GRID, GRID), dtype=np.int8)

    r = np.random.rand(GRID, GRID)
    grid[(r < init_A) & mask] = A
    grid[(r >= init_A) & (r < init_A + init_B) & mask] = B
    grid[~mask] = EMPTY

    # metabolite fields
    ma = np.zeros((GRID, GRID), dtype=float)  # M_A (produced by A, eaten by B)
    mb = np.zeros((GRID, GRID), dtype=float)  # M_B (produced by B, inhibits A)

    # nutrient field N (starts uniform inside dish)
    N = np.zeros((GRID, GRID), dtype=float)
    N[mask] = 1.0  # normalized nutrient

    # history
    hist = {
        "time": [],
        "count_A": [],
        "count_B": [],
        "mean_MA": [],
        "mean_MB": [],
        "mean_N": []
    }

    st.session_state.grid = grid
    st.session_state.mask = mask
    st.session_state.ma = ma
    st.session_state.mb = mb
    st.session_state.N = N
    st.session_state.t = 0
    st.session_state.hist = hist
    st.session_state.run_sim = False
    st.session_state.initialized = True

if not st.session_state.initialized:
    init_state()

# Controls
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    run = st.checkbox("Run simulation", value=st.session_state.run_sim)
    st.session_state.run_sim = run

with col2:
    step_once = st.button("Step once")

with col3:
    st.write(f"Time: {st.session_state.t}")

# Placeholders for visuals
col_vis, col_stats = st.columns([1.4, 1])

with col_vis:
    st.write("### Petri dish")
    legend_html = """
    <div style="display:flex; gap:12px; align-items:center; margin-bottom:6px;">
      <div style="display:inline-block; width:12px; height:12px; background:#FF6666; border:1px solid #555"></div> A (consumes N, makes M_A)
      <div style="display:inline-block; width:12px; height:12px; background:#66CC66; border:1px solid #555; margin-left:10px;"></div> B (consumes M_A, makes M_B)
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    dish_ph = st.empty()
    ma_ph = st.empty()
    mb_ph = st.empty()

with col_stats:
    st.write("### Population & metabolites")
    chart_counts_ph = st.empty()
    chart_frac_ph = st.empty()

# -----------------------
# Simulation core function: single step
# -----------------------
def sim_step(grid, ma, mb, N, mask):
    # Random neighbor selection
    dx = np.random.randint(-1, 2, size=(GRID, GRID))
    dy = np.random.randint(-1, 2, size=(GRID, GRID))
    x_idx, y_idx = np.indices((GRID, GRID))
    nx = (x_idx + dx) % GRID
    ny = (y_idx + dy) % GRID

    neighbor = grid[nx, ny]
    self = grid

    # reproduction and replacement probabilities
    # A spreads into empty if nutrient local is available and not strongly inhibited by mb
    N_local = N  # using local nutrient field
    mb_local = mb
    ma_local = ma

    # Effective A growth modifier: nutrient saturating function minus inhibition by M_B
    growth_A_local = (N_local / (N_local + K_N)) * (1.0 / (1.0 + inhib_MB_on_A * mb_local))
    growth_B_local = (ma_local / (ma_local + K_MA))

    # random matrices
    r_spread = np.random.rand(GRID, GRID)
    r_death = np.random.rand(GRID, GRID)

    # ---------- Death events
    death_events_A = (self == A) & (r_death < death_A)
    death_events_B = (self == B) & (r_death < death_B)
    grid_after_death = grid.copy()
    grid_after_death[death_events_A | death_events_B] = EMPTY

    # ---------- Reproduction attempts (neighbor tries to replace me)
    # Inverted perspective: neighbor (S) acts on me (T)
    S = neighbor
    T = grid_after_death  # target after deaths

    # Valid positions: inside petri mask and neighbor also inside
    valid = mask & mask[nx, ny]

    # A reproduction: neighbor is A, target is EMPTY
    # Probability depends on base spread * growth_A_local at neighbor position
    prob_repro_A = p_spread_A * growth_A_local[nx, ny]
    repro_A = valid & (S == A) & (T == EMPTY) & (r_spread < prob_repro_A)

    # B reproduction: neighbor is B, target is EMPTY
    prob_repro_B = p_spread_B * growth_B_local[nx, ny]
    repro_B = valid & (S == B) & (T == EMPTY) & (r_spread < prob_repro_B)

    # Additional competitive takeover: B can displace A if local MA is high (aggressive uptake)
    takeover_B_on_A = valid & (S == B) & (T == A) & (r_spread < (0.02 + 0.2 * (ma[nx, ny] / (ma[nx, ny] + 0.2))))

    # Apply reproduction / takeover
    new_grid = T.copy()
    new_grid[repro_A] = A
    new_grid[repro_B] = B
    new_grid[takeover_B_on_A] = B

    # ---------- Metabolite production/consumption & diffusion
    # Production: A produces M_A; B produces M_B
    prodA_field = prod_A * (new_grid == A).astype(float)
    prodB_field = prod_B * (new_grid == B).astype(float)

    # Nutrient consumption by A (approx proportional to local A)
    consN_field = cons_N_by_A * (new_grid == A).astype(float)

    # B consumes M_A when present -> we model as local uptake reducing MA and aiding B growth
    uptake_MA_by_B = cons_MA_by_B * (new_grid == B).astype(float) * (ma / (ma + 1e-6))

    # Diffuse metabolites (explicit Euler)
    lap_ma = laplacian(ma)
    lap_mb = laplacian(mb)
    ma = ma + dt * (D_m * lap_ma + prodA_field - uptake_MA_by_B - decay_m * ma)
    mb = mb + dt * (D_m * lap_mb + prodB_field - decay_m * mb)

    # Nutrient diffusion is neglected here; nutrient declines inside dish due to consumption
    N = N - dt * consN_field
    # prevent negatives
    ma = np.clip(ma, 0.0, None)
    mb = np.clip(mb, 0.0, None)
    N = np.clip(N, 0.0, 1.0)

    # enforce mask (outside dish remains zero / empty)
    new_grid[~mask] = EMPTY
    ma[~mask] = 0.0
    mb[~mask] = 0.0
    N[~mask] = 0.0

    return new_grid, ma, mb, N

# -----------------------
# Run simulation steps if requested
# -----------------------
if st.session_state.run_sim or step_once:
    steps = STEPS_PER_FRAME if st.session_state.run_sim else 1
    for _ in range(steps):
        g, ma, mb, N = sim_step(st.session_state.grid, st.session_state.ma, st.session_state.mb, st.session_state.N, st.session_state.mask)
        st.session_state.grid = g
        st.session_state.ma = ma
        st.session_state.mb = mb
        st.session_state.N = N
        st.session_state.t += 1

        # record history periodically
        if st.session_state.t % 1 == 0:
            hist = st.session_state.hist
            hist["time"].append(st.session_state.t)
            hist["count_A"].append(int(np.sum(g == A)))
            hist["count_B"].append(int(np.sum(g == B)))
            hist["mean_MA"].append(float(ma.mean()))
            hist["mean_MB"].append(float(mb.mean()))
            hist["mean_N"].append(float(N.mean()))
            st.session_state.hist = hist

    # rerun to update UI if running continuously
    if st.session_state.run_sim:
        st.experimental_rerun()

# -----------------------
# Rendering visuals
# -----------------------
grid = st.session_state.grid
ma = st.session_state.ma
mb = st.session_state.mb
mask = st.session_state.mask

# Build RGB image for dish
img = np.zeros((GRID, GRID, 3), dtype=float)
img[grid == A] = [1.0, 0.35, 0.35]  # A: reddish
img[grid == B] = [0.35, 1.0, 0.45]  # B: greenish
img[~mask] = 0.05  # background darker

dish_ph.image(img, caption=f"Petri dish — time {st.session_state.t}", use_column_width=False, width=PETRI_WIDTH)

# Metabolite heatmaps (normalized for display)
ma_disp = ma.copy()
mb_disp = mb.copy()
if ma_disp.max() > 0:
    ma_norm = ma_disp / (ma_disp.max())
else:
    ma_norm = ma_disp
if mb_disp.max() > 0:
    mb_norm = mb_disp / (mb_disp.max())
else:
    mb_norm = mb_disp

# Convert to RGB heatmap (simple colormap mapping)
def grey_to_rgb(arr):
    rgb = np.stack([arr, arr, arr], axis=-1)
    return rgb

ma_rgb = np.zeros_like(img)
mb_rgb = np.zeros_like(img)
ma_rgb[..., 0] = ma_norm  # red channel emphasizes M_A
mb_rgb[..., 1] = mb_norm  # green channel emphasizes M_B
ma_rgb[~mask] = 0.0
mb_rgb[~mask] = 0.0

ma_ph.image(ma_rgb, caption="M_A (produced by A)", use_column_width=False, width=PETRI_WIDTH//2)
mb_ph.image(mb_rgb, caption="M_B (produced by B)", use_column_width=False, width=PETRI_WIDTH//2)

# Charts
hist = st.session_state.hist
if len(hist["time"]) > 0:
    df = pd.DataFrame({
        "Time": hist["time"],
        "A": hist["count_A"],
        "B": hist["count_B"],
        "Mean MA": hist["mean_MA"],
        "Mean MB": hist["mean_MB"],
        "Mean N": hist["mean_N"]
    })

    df_melt = df.melt("Time", var_name="Species", value_name="Count")
    chart_c = alt.Chart(df_melt[df_melt["Species"].isin(["A", "B"])]).mark_line().encode(
        x=alt.X("Time", axis=alt.Axis(title="Time steps")),
        y=alt.Y("Count", axis=alt.Axis(title="Abundance")),
        color=alt.Color("Species", scale=alt.Scale(domain=["A", "B"], range=["#FF6666", "#66CC66"]))
    ).properties(height=220)
    chart_counts_ph.altair_chart(chart_c, use_container_width=True)

    # Fractions
    df["total"] = df["A"] + df["B"]
    df["total"] = df["total"].replace(0, 1)
    df_frac = pd.DataFrame({
        "Time": df["Time"],
        "A_frac": df["A"] / df["total"],
        "B_frac": df["B"] / df["total"]
    }).melt("Time", var_name="Species", value_name="Fraction")
    chart_f = alt.Chart(df_frac).mark_line().encode(
        x="Time",
        y=alt.Y("Fraction", axis=alt.Axis(format='%')),
        color=alt.Color("Species", scale=alt.Scale(domain=["A_frac", "B_frac"], range=["#FF6666", "#66CC66"]))
    ).properties(height=180)
    chart_frac_ph.altair_chart(chart_f, use_container_width=True)

st.write("### Controls / tips")
st.markdown("""
- Increase diffusion `D_m` and production `prod_A` / `prod_B` to make smoother fields.
- Increasing `inhib_MB_on_A` makes stronger negative feedback and can produce spatial-temporal oscillations.
- `STEPS per frame` controls how fast the simulation advances each UI update.
- If you want faster exploration, reduce `GRID` or `STEPS per frame`.
""")
