# ================================================================
# BattSim v4.2 — Digital Twin Co-Simulation Framework
# HuggingFace Spaces — Docker + Streamlit
#
# Machine 1: PyBaMM DFN (Physical Asset Emulator)
# Machine 2: 2-RC ECM + Single EKF (Digital Observer)
#
# Uncertainty Quantification:
#   - tr(P): state covariance propagation
#   - Cycle-by-cycle peak uncertainty growth
#   - Innovation residuals (whiteness test)
#   - ±2σ SOC confidence band
#
# Author : Eng. Thaer Abushawar
# Refs   : Plett (2004), Chen et al. (2020), Coman et al. (2022)
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import streamlit as st
import time

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="BattSim v4.2 — Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

.machine-box {
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    background: linear-gradient(135deg,#1a1f2e 0%,#1e2430 100%);
}
.machine-title {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 0.3rem;
}
.m1-color { color: #00b4d8; }
.m2-color { color: #f77f00; }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-label { color: #8b949e; font-size: 0.7rem; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { color: #00b4d8; font-size: 1.4rem; font-weight: 700;
                font-family: 'JetBrains Mono'; }
.metric-unit  { color: #6e7681; font-size: 0.7rem; }

.assess-ok   { background:#0d2818; border-left:3px solid #2dc653;
               padding:0.5rem 0.75rem; border-radius:4px; margin:4px 0;
               font-size:0.8rem; color:#7ee787; }
.assess-warn { background:#2a1f0d; border-left:3px solid #f77f00;
               padding:0.5rem 0.75rem; border-radius:4px; margin:4px 0;
               font-size:0.8rem; color:#ffa657; }
.assess-err  { background:#2a0d0d; border-left:3px solid #ef233c;
               padding:0.5rem 0.75rem; border-radius:4px; margin:4px 0;
               font-size:0.8rem; color:#ff7b72; }

.data-link { background:#0d1117; border:1px dashed #30363d; border-radius:6px;
             padding:0.4rem 0.75rem; font-family:'JetBrains Mono';
             font-size:0.72rem; color:#58a6ff; margin:3px 0; }

div[data-testid="stSidebar"] { background:#0d1117; }
.stButton>button {
    background:linear-gradient(135deg,#1f6feb,#1a4fd6);
    color:white; border:none; border-radius:8px;
    font-weight:600; letter-spacing:0.03em;
    transition: all 0.2s;
}
.stButton>button:hover {
    transform:translateY(-1px);
    box-shadow:0 4px 15px rgba(31,111,235,0.4);
}
.footer-bar {
    margin-top:2rem; padding:1rem 0 0.5rem;
    border-top:1px solid #21262d; text-align:center;
    font-size:0.75rem; color:#6e7681;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# Chemistry Library
# ================================================================
CHEM = {
    "NMC — Chen2020 (LG M50 21700)": {
        "Q": 5.0,
        "R0": 0.010, "R1": 0.015, "C1": 3000,
                     "R2": 0.008, "C2": 8000,
        "soc_lut": [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,
                    .5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1.0],
        "ocv_lut": [3.0,3.3,3.42,3.5,3.54,3.57,3.62,3.65,3.68,3.71,
                    3.74,3.77,3.8,3.84,3.88,3.92,3.96,4.01,4.06,4.13,4.2],
        "pybamm": "Chen2020",
        "color":  "#00b4d8",
        "desc":   "Tesla Model 3 LR / Performance",
        "v_min": 2.5, "v_max": 4.2,
    },
    "LFP — Prada2013 (A123 26650)": {
        "Q": 1.1,
        "R0": 0.020, "R1": 0.018, "C1": 2500,
                     "R2": 0.010, "C2": 6000,
        "soc_lut": [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,
                    .5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1.0],
        "ocv_lut": [3.0,3.1,3.2,3.25,3.28,3.30,3.31,3.32,3.325,3.33,
                    3.335,3.34,3.345,3.35,3.36,3.37,3.38,3.39,3.40,3.42,3.6],
        "pybamm": "Prada2013",
        "color":  "#2dc653",
        "desc":   "Tesla Model 3 SR / BYD Blade",
        "v_min": 2.0, "v_max": 3.6,
    },
}

NOISE_MAP = {"Low — 5 mV": 0.005, "Medium — 10 mV": 0.010, "High — 30 mV": 0.030}
PROTOCOL_MAP = {
    "Constant Current (CC)": "cc",
    "Pulse / HPPC":          "pulse",
}

# ================================================================
# Helpers
# ================================================================
def make_ocv(p):
    return interp1d(p["soc_lut"], p["ocv_lut"], kind="cubic", fill_value="extrapolate")

def docv_dsoc(fn, soc, h=1e-4):
    return (fn(soc+h) - fn(soc-h)) / (2*h)

# ================================================================
# Machine 1 — PyBaMM DFN
# ================================================================
@st.cache_data(show_spinner=False, max_entries=8)
def run_dfn(pset_name, n_cycles, c_rate, protocol, v_min, v_max):
    import pybamm
    model  = pybamm.lithium_ion.DFN()
    params = pybamm.ParameterValues(pset_name)

    if protocol == "cc":
        steps = [
            f"Discharge at {c_rate}C until {v_min} V",
            "Rest for 5 minutes",
            f"Charge at {c_rate/2:.1f}C until {v_max} V",
            "Rest for 5 minutes",
        ] * n_cycles
    else:
        steps = []
        for _ in range(n_cycles):
            steps += [
                f"Discharge at {c_rate}C for 10 seconds",
                "Rest for 40 seconds",
                f"Discharge at {c_rate*2:.1f}C for 10 seconds",
                "Rest for 40 seconds",
                f"Charge at {c_rate}C for 10 seconds",
                "Rest for 40 seconds",
                f"Discharge at {c_rate}C until {v_min} V",
                "Rest for 5 minutes",
                f"Charge at {c_rate/2:.1f}C until {v_max} V",
                "Rest for 5 minutes",
            ]

    exp = pybamm.Experiment(steps)
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sim.solve()
    sol = sim.solution

    def safe1d(arr):
        return np.asarray(arr).ravel()

    t   = safe1d(sol["Time [s]"].entries)
    V   = safe1d(sol["Terminal voltage [V]"].entries)
    I   = safe1d(sol["Current [A]"].entries)
    T   = safe1d(sol["Cell temperature [K]"].entries) - 273.15
    Q_n = float(params["Nominal cell capacity [A.h]"])

    _, unique_idx = np.unique(t, return_index=True)
    t = t[unique_idx]; V = V[unique_idx]
    I = I[unique_idx]; T = T[unique_idx]

    dt  = np.diff(t, prepend=t[0])
    soc = np.clip(1.0 - np.cumsum(I * dt)/3600.0/Q_n, 0.0, 1.0)

    t_u = np.arange(t[0], t[-1], 10.0)
    return (
        t_u,
        np.interp(t_u, t, V),
        np.interp(t_u, t, I),
        np.interp(t_u, t, soc),
        np.interp(t_u, t, T),
        Q_n,
    )

# ================================================================
# Machine 2 — Single EKF (2-RC ECM)
# State vector: x = [SOC, V_RC1, V_RC2]
# ================================================================
class EKF:
    """
    Extended Kalman Filter for 2-RC Thevenin ECM.
    Estimates: SOC, V_RC1, V_RC2
    Observes:  V_terminal (noisy)
    """
    def __init__(self, soc0, Q_nom, chem, noise_var, p0_scale, q_scale, r_scale):
        self.dt    = 10.0
        self.ocv   = make_ocv(chem)
        self.R0    = chem["R0"]
        self.R1, self.C1 = chem["R1"], chem["C1"]
        self.R2, self.C2 = chem["R2"], chem["C2"]
        self.Q_nom = Q_nom

        # State: [SOC, V_RC1, V_RC2]
        self.x = np.array([[soc0], [0.0], [0.0]])

        # Covariance matrices
        self.P = np.diag([p0_scale, p0_scale * 0.1, p0_scale * 0.1])
        self.Q = np.diag([q_scale * 1e-6, q_scale * 1e-5, q_scale * 1e-5])
        self.R = np.array([[r_scale * noise_var]])
        self.I3 = np.eye(3)

        self.last_Ck = np.zeros((1, 3))
        self.last_Kk = np.zeros((3, 1))

    def step(self, v_meas, current):
        dt = self.dt
        e1 = np.exp(-dt / (self.R1 * self.C1))
        e2 = np.exp(-dt / (self.R2 * self.C2))
        s, v1, v2 = self.x[:, 0]

        # ── Predict ──────────────────────────────────────────
        s_p  = s  - current * dt / (self.Q_nom * 3600.0)
        v1_p = v1 * e1 + current * self.R1 * (1 - e1)
        v2_p = v2 * e2 + current * self.R2 * (1 - e2)
        x_p  = np.array([[s_p], [v1_p], [v2_p]])

        A    = np.diag([1.0, e1, e2])
        P_p  = A @ self.P @ A.T + self.Q

        # ── Update ───────────────────────────────────────────
        dOCV = docv_dsoc(self.ocv, np.clip(s_p, 0.0, 1.0))
        Ck   = np.array([[dOCV, -1.0, -1.0]])
        self.last_Ck = Ck

        v_hat = (float(self.ocv(np.clip(s_p, 0.0, 1.0)))
                 - v1_p - v2_p - current * self.R0)
        nu    = v_meas - v_hat

        S  = Ck @ P_p @ Ck.T + self.R
        Kk = P_p @ Ck.T / S[0, 0]
        self.last_Kk = Kk

        self.x       = x_p + Kk * nu
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)

        IKC    = self.I3 - Kk @ Ck
        self.P = IKC @ P_p @ IKC.T + Kk @ self.R @ Kk.T

        soc_e = float(self.x[0, 0])
        v_est = (float(self.ocv(np.clip(soc_e, 0.0, 1.0)))
                 - float(self.x[1, 0]) - float(self.x[2, 0])
                 - current * self.R0)

        return (
            v_est,
            soc_e,
            float(np.trace(self.P)),   # tr(P) — total state uncertainty
            float(self.P[0, 0]),       # P_soc — SOC variance
            nu,                        # innovation residual
        )

# ================================================================
# Co-simulation driver
# ================================================================
def run_cosim(chem_name, n_cycles, c_rate, noise_std,
              p0_scale, q_scale, r_scale, protocol):
    chem = CHEM[chem_name]

    status_box = st.empty()
    prog       = st.progress(0)

    status_box.markdown(
        "<div class='data-link'>📡 [Machine 1] Initialising PyBaMM DFN solver...</div>",
        unsafe_allow_html=True)
    prog.progress(5)

    t, V_true, I_true, soc_true, T_true, Q_nom = run_dfn(
        chem["pybamm"], n_cycles, c_rate, protocol,
        chem["v_min"], chem["v_max"]
    )
    prog.progress(45)

    status_box.markdown(
        "<div class='data-link'>🔗 [Handshake] Streaming noisy measurements → Machine 2...</div>",
        unsafe_allow_html=True)
    time.sleep(0.3)

    ekf = EKF(float(soc_true[0]), Q_nom, chem,
              noise_std**2, p0_scale, q_scale, r_scale)

    N   = len(t)
    log = {k: np.empty(N) for k in
           ["V_meas", "V_est", "soc_est", "P_tr", "P_soc", "innov"]}
    log["t"]        = t
    log["V_true"]   = V_true
    log["soc_true"] = soc_true
    log["I_true"]   = I_true
    log["T_true"]   = T_true

    last_Ck = last_Kk = None
    ckpt = max(1, N // 40)

    status_box.markdown(
        f"<div class='data-link'>⚙️ [Machine 2] Running EKF — {N:,} steps @ dt=10s...</div>",
        unsafe_allow_html=True)

    np.random.seed(42)
    for k in range(N):
        vm  = V_true[k] + np.random.normal(0.0, noise_std)
        out = ekf.step(vm, I_true[k])
        log["V_meas"][k]  = vm
        log["V_est"][k]   = out[0]
        log["soc_est"][k] = out[1]
        log["P_tr"][k]    = out[2]
        log["P_soc"][k]   = out[3]
        log["innov"][k]   = out[4]
        last_Ck = ekf.last_Ck
        last_Kk = ekf.last_Kk
        if k % ckpt == 0:
            prog.progress(45 + int(50 * k / N))

    prog.progress(100)
    status_box.empty()
    prog.empty()

    return log, Q_nom, chem, last_Ck, last_Kk

# ================================================================
# Assessment engine
# ================================================================
def engineering_assessment(log, noise_std, n_cyc, cyc_pk):
    msgs      = []
    innov_rms = np.sqrt(np.mean(log["innov"]**2))
    ratio     = innov_rms / noise_std

    if ratio > 3.0:
        msgs.append(("err",  f"EKF DIVERGING — Innovation/Noise = {ratio:.1f}× "
                             f"(limit 3×). Reduce sensor noise or increase R."))
    elif ratio > 1.5:
        msgs.append(("warn", f"EKF Under Stress — Innovation/Noise = {ratio:.1f}× "
                             f"(ideal < 1.5×). Consider retuning Q or R."))
    else:
        msgs.append(("ok",   f"EKF Healthy — Innovation/Noise = {ratio:.2f}×. "
                             f"Filter consistent with noise model."))

    p_ratio = log["P_tr"][-1] / log["P_tr"][0]
    if p_ratio > 0.3:
        msgs.append(("err",  f"EKF Did Not Converge — tr(P) final = {p_ratio:.1%} of initial."))
    elif p_ratio > 0.05:
        msgs.append(("warn", f"Partial Convergence — tr(P) = {p_ratio:.1%} of initial. "
                             f"More cycles or better P₀ would help."))
    else:
        msgs.append(("ok",   f"EKF Converged — tr(P) dropped to {p_ratio:.2%} of initial."))

    if n_cyc > 1:
        growth = (cyc_pk[-1] - cyc_pk[0]) / cyc_pk[0] * 100
        if growth > 20:
            msgs.append(("warn", f"Uncertainty Propagation Detected — Peak tr(P) grew "
                                 f"{growth:.1f}% over {n_cyc} cycles. "
                                 f"ECM model error accumulates."))
        else:
            msgs.append(("ok",  f"Uncertainty Stable — Peak tr(P) growth = {growth:.1f}% "
                                f"across {n_cyc} cycles. Observer tracking asset well."))

    return msgs

# ================================================================
# Color palette
# ================================================================
DARK  = "#0d1117"; PLOT  = "#161b22"
C_TEA = "#00b4d8"; C_ORG = "#f77f00"
C_GRN = "#2dc653"; C_RED = "#ef233c"
C_PUR = "#c77dff"; C_YLW = "#ffd60a"
PALS  = [C_TEA, C_ORG, C_GRN, C_RED, C_PUR]

BASE_LAYOUT = dict(
    paper_bgcolor=DARK, plot_bgcolor=PLOT,
    font=dict(color="#c9d1d9", family="Inter", size=11),
    margin=dict(t=50, b=36, l=58, r=24),
    height=340,
    legend=dict(bgcolor="rgba(22,27,34,0.95)", bordercolor="#30363d",
                borderwidth=1, font=dict(size=10),
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

def ax(title, fmt=None):
    d = dict(title=title, gridcolor="#21262d", showgrid=True,
             zeroline=False, linecolor="#30363d")
    if fmt: d["tickformat"] = fmt
    return d

# ================================================================
# Sidebar
# ================================================================
st.sidebar.markdown("""
<div style='text-align:center;padding:1rem 0 0.5rem'>
  <span style='font-size:2.2rem'>🔋</span><br>
  <span style='font-weight:700;font-size:1.15rem;color:#00b4d8'>BattSim</span>
  <span style='font-size:0.75rem;color:#6e7681'> v4.2</span><br>
  <span style='font-size:0.68rem;color:#6e7681'>DFN ↔ EKF · Digital Twin Co-Sim</span>
</div>
<hr style='border-color:#21262d;margin:0.5rem 0'>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🖥️ Asset Configuration  *(Machine 1)*")
chem_name     = st.sidebar.selectbox("Cell Chemistry", list(CHEM.keys()), label_visibility="collapsed")
noise_label   = st.sidebar.select_slider("Sensor Noise σ", list(NOISE_MAP.keys()), value="Medium — 10 mV")
noise_std     = NOISE_MAP[noise_label]
protocol_label= st.sidebar.selectbox("Test Protocol", list(PROTOCOL_MAP.keys()))
protocol      = PROTOCOL_MAP[protocol_label]
n_cycles      = st.sidebar.slider("Number of Cycles", 1, 35, 3)
c_rate        = st.sidebar.select_slider("C-Rate", [0.5, 1.0, 1.5, 2.0, 3.0], value=1.0)

st.sidebar.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🎛️ Observer Tuning  *(Machine 2)*")
p0_scale = st.sidebar.select_slider("P₀ scale (initial uncertainty)",
    [1e-4, 1e-3, 1e-2, 1e-1], value=1e-3,
    format_func=lambda x: f"{x:.0e}")
q_scale  = st.sidebar.select_slider("Q scale (process noise)",
    [0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
r_scale  = st.sidebar.select_slider("R scale (meas. noise weight)",
    [0.5, 1.0, 2.0, 5.0], value=1.0)

st.sidebar.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)
chem = CHEM[chem_name]
st.sidebar.markdown("**System Architecture**")
st.sidebar.code(f"""Machine 1 — Physical Asset
  Model  : PyBaMM DFN
  Params : {chem_name.split("—")[0].strip()}
  Output : V_true, I, T + σ_noise

Machine 2 — Digital Observer
  Model  : 2-RC Thevenin ECM
  Filter : Single EKF
  State  : [SOC, V_RC1, V_RC2]
  Input  : V_noisy, I  [observable only]

UQ Channel
  tr(P)  → state covariance
  ±2σ    → SOC confidence band
  ν(k)   → innovation residuals
  Cycle peaks → propagation
""", language="")

run_btn = st.sidebar.button("▶  Run Co-Simulation", use_container_width=True, type="primary")

# ================================================================
# Header
# ================================================================
st.markdown("## 🔋 BattSim v4.2 — Digital Twin Co-Simulation")
st.markdown(
    "**Machine 1:** PyBaMM DFN — full electrochemical model (physical asset emulator)  \n"
    "**Machine 2:** 2-RC ECM + Single EKF — state observer (digital twin)  \n"
    "**Goal:** Quantify uncertainty propagation as the battery cycles"
)
st.markdown("---")

# ================================================================
# Main simulation
# ================================================================
if run_btn:
    log, Q_nom, chem, last_Ck, last_Kk = run_cosim(
        chem_name, n_cycles, c_rate, noise_std,
        p0_scale, q_scale, r_scale, protocol
    )
    st.session_state.update({
        "log": log, "Q_nom": Q_nom, "chem": chem,
        "chem_name": chem_name, "n_cycles": n_cycles,
        "noise_std": noise_std, "c_rate": c_rate,
        "p0_scale": p0_scale, "q_scale": q_scale, "r_scale": r_scale,
        "protocol_label": protocol_label,
        "last_Ck": last_Ck, "last_Kk": last_Kk,
    })

if "log" not in st.session_state:
    st.info("👈  Configure parameters in the sidebar and press **Run Co-Simulation**.")
    st.stop()

# ── Unpack ───────────────────────────────────────────────────────
log           = st.session_state["log"]
Q_nom         = st.session_state["Q_nom"]
chem          = st.session_state["chem"]
chem_name     = st.session_state["chem_name"]
n_cycles      = st.session_state["n_cycles"]
noise_std     = st.session_state["noise_std"]
p0_scale      = st.session_state["p0_scale"]
q_scale       = st.session_state["q_scale"]
r_scale       = st.session_state["r_scale"]
protocol_label= st.session_state["protocol_label"]
last_Ck       = st.session_state["last_Ck"]
last_Kk       = st.session_state["last_Kk"]

t_h    = log["t"] / 3600.0
N      = len(t_h)
cl     = max(1, N // n_cycles)
cyc_pk = [log["P_tr"][c*cl:min((c+1)*cl, N)].max() for c in range(n_cycles)]

v_rmse  = np.sqrt(np.mean((log["V_true"] - log["V_est"])**2)) * 1000
s_rmse  = np.sqrt(np.mean((log["soc_true"] - log["soc_est"])**2)) * 100
sigma   = np.sqrt(log["P_soc"]) * 100
cc      = chem["color"]

# ================================================================
# KPI Row (4 cards only — no SOH)
# ================================================================
k1, k2, k3, k4 = st.columns(4)

def kpi(col, label, value, unit, color="#00b4d8"):
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{color}'>{value}</div>
      <div class='metric-unit'>{unit}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Voltage RMSE",  f"{v_rmse:.2f}",  "mV",  C_TEA)
kpi(k2, "SOC RMSE",      f"{s_rmse:.3f}",  "%",   C_ORG)
kpi(k3, "Peak tr(P)",    f"{log['P_tr'].max():.2e}", "state UQ", C_PUR)
kpi(k4, "Cycles Run",    f"{n_cycles}",    f"× {c_rate}C {protocol_label.split()[0]}", C_YLW)

st.markdown("---")

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3 = st.tabs([
    "🖥️ Machine 1 — Physical Asset",
    "📡 Machine 2 — Digital Twin",
    "📊 Uncertainty Analytics",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — Physical Asset
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='machine-title m1-color'>🖥️ Machine 1 — PyBaMM DFN Physical Asset Emulator</div>",
                unsafe_allow_html=True)
    st.caption("Ground-truth electrochemical simulation. Machine 2 only sees V_noisy + I — not the internal states below.")

    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["T_true"], mode="lines",
            line=dict(color=C_RED, width=2), fill="tozeroy",
            fillcolor="rgba(239,35,60,0.08)", showlegend=False))
        f.update_layout(title="🌡️ Internal Cell Temperature  *(hidden from Observer)*",
            xaxis=ax("Time [h]"), yaxis=ax("Temperature [°C]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with c2:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["V_meas"], mode="lines",
            name="V_noisy → Machine 2", line=dict(color="#6e7681", width=0.6), opacity=0.5))
        f.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
            name="V_true (DFN)", line=dict(color=cc, width=2.2)))
        f.update_layout(title="📶 Noisy Measurements  *(streamed to Machine 2)*",
            xaxis=ax("Time [h]"), yaxis=ax("Voltage [V]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["soc_true"]*100, mode="lines",
            line=dict(color=cc, width=2.2), showlegend=False))
        f.update_layout(title="⚡ True SOC (DFN)",
            xaxis=ax("Time [h]"), yaxis=ax("SOC [%]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)
    with c4:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["I_true"], mode="lines",
            line=dict(color=C_YLW, width=1.8), showlegend=False))
        f.update_layout(title="🔌 Applied Current Profile",
            xaxis=ax("Time [h]"), yaxis=ax("Current [A]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 2 — Digital Twin
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='machine-title m2-color'>📡 Machine 2 — 2-RC ECM + Single EKF Digital Observer</div>",
                unsafe_allow_html=True)
    st.caption("Observer reconstructs battery states using only V_noisy and I. No access to DFN internals.")

    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
            name="V_true (DFN)", line=dict(color=cc, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["V_est"], mode="lines",
            name="V_est (EKF)", line=dict(color=C_ORG, width=1.8, dash="dash")))
        f.update_layout(title="① Voltage Tracking — DFN vs EKF",
            xaxis=ax("Time [h]"), yaxis=ax("Voltage [V]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with c2:
        upper = log["soc_est"]*100 + 2*sigma
        lower = log["soc_est"]*100 - 2*sigma
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=np.concatenate([t_h, t_h[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(247,127,0,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±2σ confidence"))
        f.add_trace(go.Scatter(x=t_h, y=log["soc_true"]*100, mode="lines",
            name="SOC true (DFN)", line=dict(color=cc, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["soc_est"]*100, mode="lines",
            name="SOC est (EKF)", line=dict(color=C_ORG, width=1.8, dash="dash")))
        f.update_layout(title="② SOC Estimation ± 2σ Uncertainty Band",
            xaxis=ax("Time [h]"), yaxis=ax("SOC [%]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        ve = np.abs(log["V_true"] - log["V_est"]) * 1000
        f  = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=ve, mode="lines",
            line=dict(color=C_RED, width=1.5),
            fill="tozeroy", fillcolor="rgba(239,35,60,0.12)", showlegend=False))
        f.add_hline(y=v_rmse, line_dash="dash", line_color=C_YLW,
            annotation_text=f"RMSE = {v_rmse:.2f} mV",
            annotation_position="top right")
        f.update_layout(title="③ Voltage Estimation Error |V_true − V_est|",
            xaxis=ax("Time [h]"), yaxis=ax("|Error| [mV]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with c4:
        soc_err = np.abs(log["soc_true"] - log["soc_est"]) * 100
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=soc_err, mode="lines",
            line=dict(color=C_PUR, width=1.5),
            fill="tozeroy", fillcolor="rgba(199,125,255,0.10)", showlegend=False))
        f.add_hline(y=s_rmse, line_dash="dash", line_color=C_YLW,
            annotation_text=f"RMSE = {s_rmse:.3f}%",
            annotation_position="top right")
        f.update_layout(title="④ SOC Estimation Error |SOC_true − SOC_est|",
            xaxis=ax("Time [h]"), yaxis=ax("|Error| [%]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 3 — Uncertainty Analytics
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Uncertainty Propagation Analysis")
    st.caption("Core research question: does uncertainty grow as the battery cycles? How fast?")

    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        for cyc in range(n_cycles):
            s, e = cyc*cl, min((cyc+1)*cl, N)
            f.add_trace(go.Scatter(
                x=t_h[s:e], y=log["P_tr"][s:e], mode="lines",
                name=f"Cycle {cyc+1}",
                line=dict(color=PALS[cyc % len(PALS)], width=2)))
        f.update_layout(title="⑤ State Covariance tr(P) — All Cycles",
            xaxis=ax("Time [h]"), yaxis=ax("tr(P)"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with c2:
        growth_pct  = [(p - cyc_pk[0]) / cyc_pk[0] * 100 for p in cyc_pk]
        bar_colors  = [C_GRN if g <= 10 else (C_YLW if g <= 25 else C_RED)
                       for g in growth_pct]
        f = go.Figure()
        f.add_trace(go.Bar(
            x=[f"Cycle {i+1}" for i in range(n_cycles)],
            y=cyc_pk,
            marker_color=bar_colors,
            text=["Baseline" if i == 0 else f"{g:+.1f}%"
                  for i, g in enumerate(growth_pct)],
            textposition="outside",
        ))
        f.update_layout(
            title="⑥ Cycle-by-Cycle Peak tr(P)  ← Key UQ Result",
            xaxis=ax("Cycle"), yaxis=ax("Peak tr(P)"),
            **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["innov"]*1000, mode="lines",
            line=dict(color=C_TEA, width=1.0), opacity=0.7,
            name="Innovation ν(k)"))
        f.add_hline(y=+2*noise_std*1000, line_dash="dash", line_color=C_RED,
            annotation_text="+2σ", annotation_position="top right")
        f.add_hline(y=-2*noise_std*1000, line_dash="dash", line_color=C_RED,
            annotation_text="−2σ")
        f.update_layout(
            title="⑦ Innovation Residuals ν(k)  [white noise if EKF is tuned]",
            xaxis=ax("Time [h]"), yaxis=ax("ν(k) [mV]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with c4:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=sigma, mode="lines",
            line=dict(color=C_ORG, width=2),
            fill="tozeroy", fillcolor="rgba(247,127,0,0.10)", showlegend=False))
        f.update_layout(title="⑧ SOC Standard Deviation σ_SOC(t)",
            xaxis=ax("Time [h]"), yaxis=ax("σ_SOC [%]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    # Summary table
    st.markdown("#### Cycle-by-Cycle Uncertainty Summary")
    df_cyc = pd.DataFrame({
        "Cycle":         [f"Cycle {i+1}" for i in range(n_cycles)],
        "Peak tr(P)":    [f"{p:.3e}" for p in cyc_pk],
        "Δ vs Cycle 1":  ["Baseline"] + [f"{(p-cyc_pk[0])/cyc_pk[0]*100:+.1f}%"
                          for p in cyc_pk[1:]],
        "Peak σ_SOC [%]":[f"{np.sqrt(log['P_soc'][c*cl:min((c+1)*cl,N)]).max()*100:.4f}"
                          for c in range(n_cycles)],
        "Status":        ["Baseline"] + [
            "🟢 Stable"   if (p-cyc_pk[0])/cyc_pk[0]*100 <= 10 else
            ("🟡 Growing" if (p-cyc_pk[0])/cyc_pk[0]*100 <= 25 else "🔴 High Growth")
            for p in cyc_pk[1:]
        ],
    })
    st.dataframe(df_cyc, use_container_width=True, hide_index=True)

# ================================================================
# Engineering Assessment
# ================================================================
st.markdown("---")
st.markdown("### 🔍 Engineering Assessment")
msgs = engineering_assessment(log, noise_std, n_cycles, cyc_pk)
for kind, msg in msgs:
    css  = {"ok": "assess-ok", "warn": "assess-warn", "err": "assess-err"}[kind]
    icon = {"ok": "✅", "warn": "⚠️", "err": "🔴"}[kind]
    st.markdown(f"<div class='{css}'>{icon} {msg}</div>", unsafe_allow_html=True)

# ================================================================
# MACHINE 3 — Physics-Informed Neural Network (PINN) Corrector
# ================================================================
st.markdown("---")
st.markdown("""
<div class='machine-box'>
  <div class='machine-title' style='color:#c77dff'>
    🧠 Machine 3 — PINN Residual Corrector
  </div>
  <small style='color:#8b949e'>
    Learns the DFN→ECM model mismatch and corrects EKF estimates online.
    Inputs: [I, SOC_est, t_norm] · Target: residual = V_true − V_ecm
  </small>
</div>
""", unsafe_allow_html=True)

pinn_col1, pinn_col2, pinn_col3, pinn_col4 = st.columns(4)
with pinn_col1:
    pinn_epochs  = st.slider("Training Epochs", 100, 2000, 500, step=100)
with pinn_col2:
    pinn_hidden  = st.select_slider("Hidden Layers",
        options=[1, 2, 3, 4], value=2)
with pinn_col3:
    pinn_neurons = st.select_slider("Neurons/Layer",
        options=[16, 32, 64, 128], value=64)
with pinn_col4:
    pinn_lambda  = st.select_slider("Physics λ",
        options=[0.0, 0.01, 0.1, 1.0], value=0.1)

if st.button("🚀  Train PINN Corrector", type="primary", use_container_width=False):

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # ── Build training data from current simulation log ───────
    t_norm  = (log["t"] - log["t"].min()) / (log["t"].max() - log["t"].min())
    ocv_fn  = make_ocv(chem)
    V_ocv   = np.array([float(ocv_fn(np.clip(s,0,1))) for s in log["soc_est"]])
    V_ecm   = V_ocv - log["I_true"] * chem["R0"]
    residual = log["V_true"] - V_ecm

    # Features: [t_norm, I, SOC_est, V_ecm_norm]
    V_ecm_norm = (V_ecm - V_ecm.mean()) / (V_ecm.std() + 1e-8)
    I_norm     = (log["I_true"] - log["I_true"].mean()) / (log["I_true"].std() + 1e-8)
    res_mean   = residual.mean()
    res_std    = residual.std() + 1e-8
    res_norm   = (residual - res_mean) / res_std

    X = np.stack([
        t_norm,
        I_norm,
        log["soc_est"],
        V_ecm_norm,
    ], axis=1).astype(np.float32)

    y = res_norm.astype(np.float32).reshape(-1, 1)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    # ── PINN Architecture ─────────────────────────────────────
    class PINN(nn.Module):
        def __init__(self, hidden, neurons):
            super().__init__()
            layers = [nn.Linear(4, neurons), nn.Tanh()]
            for _ in range(hidden - 1):
                layers += [nn.Linear(neurons, neurons), nn.Tanh()]
            layers += [nn.Linear(neurons, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model_pinn = PINN(pinn_hidden, pinn_neurons)
    optimizer  = optim.Adam(model_pinn.parameters(), lr=1e-3)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pinn_epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────
    loss_hist  = []
    phys_hist  = []
    prog3      = st.progress(0)
    status3    = st.empty()

    dt_tensor  = torch.tensor(10.0)
    Q_tensor   = torch.tensor(float(Q_nom) * 3600.0)

    for epoch in range(pinn_epochs):
        model_pinn.train()
        optimizer.zero_grad()

        # Data loss
        pred      = model_pinn(X_t)
        data_loss = nn.MSELoss()(pred, y_t)

        # Physics loss: d(SOC)/dt ≈ -I/Q
        # Approximate with finite difference on SOC_est
        soc_pred  = torch.tensor(log["soc_est"].astype(np.float32))
        dsoc_dt   = (soc_pred[1:] - soc_pred[:-1]) / 10.0
        i_over_q  = torch.tensor(
            (log["I_true"][:-1] / (Q_nom * 3600)).astype(np.float32))
        phys_loss = nn.MSELoss()(dsoc_dt, -i_over_q)

        # Total loss
        loss = data_loss + pinn_lambda * phys_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(float(data_loss))
        phys_hist.append(float(phys_loss))

        if epoch % max(1, pinn_epochs // 20) == 0:
            prog3.progress((epoch + 1) / pinn_epochs)
            status3.markdown(
                f"<div class='data-link'>🧠 Epoch {epoch+1}/{pinn_epochs} — "
                f"Data Loss: {float(data_loss):.5f} | "
                f"Physics Loss: {float(phys_loss):.6f}</div>",
                unsafe_allow_html=True)

    prog3.empty()
    status3.empty()

    # ── PINN Inference ────────────────────────────────────────
    model_pinn.eval()
    with torch.no_grad():
        res_pred_norm = model_pinn(X_t).numpy().ravel()

    res_predicted = res_pred_norm * res_std + res_mean
    V_corrected   = V_ecm + res_predicted

    # Corrected SOC via OCV inversion (simple lookup)
    ocv_arr = np.array(chem["ocv_lut"])
    dV_correction = V_corrected - log["V_est"]
    dSOC_correction = dV_correction / np.array(
        [float(docv_dsoc(make_ocv(chem), np.clip(s,0.05,0.95)))
         for s in log["soc_est"]])  
    soc_corrected = np.clip(log["soc_est"] + dSOC_correction * 0.1, 0, 1)


    # ── Metrics comparison ────────────────────────────────────
    v_rmse_ecm   = np.sqrt(np.mean((log["V_true"] - V_ecm)**2)) * 1000
    v_rmse_pinn  = np.sqrt(np.mean((log["V_true"] - V_corrected)**2)) * 1000
    s_rmse_ekf   = np.sqrt(np.mean((log["soc_true"] - log["soc_est"])**2)) * 100
    s_rmse_pinn  = np.sqrt(np.mean((log["soc_true"] - soc_corrected)**2)) * 100
    improv_v     = (v_rmse_ecm - v_rmse_pinn) / v_rmse_ecm * 100
    improv_s     = (s_rmse_ekf - s_rmse_pinn) / s_rmse_ekf * 100

    # Store in session
    st.session_state["pinn_V_corrected"]   = V_corrected
    st.session_state["pinn_soc_corrected"] = soc_corrected
    st.session_state["pinn_loss_hist"]     = loss_hist
    st.session_state["pinn_phys_hist"]     = phys_hist
    st.session_state["pinn_metrics"] = {
        "v_rmse_ecm":  v_rmse_ecm,
        "v_rmse_pinn": v_rmse_pinn,
        "s_rmse_ekf":  s_rmse_ekf,
        "s_rmse_pinn": s_rmse_pinn,
        "improv_v":    improv_v,
        "improv_s":    improv_s,
    }

# ── Show results if trained ───────────────────────────────────
if "pinn_V_corrected" in st.session_state:

    V_corrected   = st.session_state["pinn_V_corrected"]
    soc_corrected = st.session_state["pinn_soc_corrected"]
    loss_hist     = st.session_state["pinn_loss_hist"]
    phys_hist     = st.session_state["pinn_phys_hist"]
    m             = st.session_state["pinn_metrics"]

    # ── KPI row ───────────────────────────────────────────────
    st.markdown("#### 📊 Correction Results")
    pk1, pk2, pk3, pk4 = st.columns(4)

    def kpi_delta(col, label, before, after, unit):
        delta = after - before
        sign  = "▲" if delta > 0 else "▼"
        color = C_GRN if delta < 0 else C_RED
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value' style='color:{color}'>{after:.3f}</div>
          <div class='metric-unit'>{unit}
            <span style='color:{color}'>{sign}{abs(delta):.3f}</span>
            vs EKF {before:.3f}
          </div>
        </div>""", unsafe_allow_html=True)

    kpi_delta(pk1, "V RMSE (PINN)",
              m["v_rmse_ecm"], m["v_rmse_pinn"], "mV")
    kpi_delta(pk2, "SOC RMSE (PINN)",
              m["s_rmse_ekf"], m["s_rmse_pinn"], "%")
    pk3.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Voltage Improvement</div>
      <div class='metric-value' style='color:{C_GRN}'>{m["improv_v"]:.1f}%</div>
      <div class='metric-unit'>vs ECM baseline</div>
    </div>""", unsafe_allow_html=True)
    pk4.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>SOC Improvement</div>
      <div class='metric-value' style='color:{C_GRN}'>{m["improv_s"]:.1f}%</div>
      <div class='metric-unit'>vs EKF baseline</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 4 charts ──────────────────────────────────────────────
    pc1, pc2 = st.columns(2)

    with pc1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
            name="V_true (DFN)", line=dict(color=cc, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["V_est"], mode="lines",
            name=f"V_est EKF ({v_rmse:.1f} mV)",
            line=dict(color=C_ORG, width=1.5, dash="dash")))
        f.add_trace(go.Scatter(x=t_h, y=V_corrected, mode="lines",
            name=f"V_PINN ({m['v_rmse_pinn']:.1f} mV)",
            line=dict(color=C_PUR, width=2.0)))
        f.update_layout(
            title="① Voltage: DFN vs EKF vs PINN Corrected",
            xaxis=ax("Time [h]"), yaxis=ax("Voltage [V]"),
            **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with pc2:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["soc_true"]*100, mode="lines",
            name="SOC_true (DFN)", line=dict(color=cc, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["soc_est"]*100, mode="lines",
            name=f"SOC EKF ({s_rmse:.2f}%)",
            line=dict(color=C_ORG, width=1.5, dash="dash")))
        f.add_trace(go.Scatter(x=t_h, y=soc_corrected*100, mode="lines",
            name=f"SOC PINN ({m['s_rmse_pinn']:.2f}%)",
            line=dict(color=C_PUR, width=2.0)))
        f.update_layout(
            title="② SOC: DFN vs EKF vs PINN Corrected",
            xaxis=ax("Time [h]"), yaxis=ax("SOC [%]"),
            **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    pc3, pc4 = st.columns(2)

    with pc3:
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=list(range(1, len(loss_hist)+1)), y=loss_hist,
            mode="lines", name="Data Loss",
            line=dict(color=C_PUR, width=2)))
        f.add_trace(go.Scatter(
            x=list(range(1, len(phys_hist)+1)), y=phys_hist,
            mode="lines", name="Physics Loss",
            line=dict(color=C_ORG, width=1.5, dash="dash")))
        f.update_layout(
            title="③ PINN Training Loss Convergence",
            xaxis=ax("Epoch"), yaxis=ax("Loss", ".2e"),
            **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with pc4:
        ocv_fn  = make_ocv(chem)
        V_ocv   = np.array([float(ocv_fn(np.clip(s,0,1))) for s in log["soc_est"]])
        V_ecm_b = V_ocv - log["I_true"] * chem["R0"]
        res_true = log["V_true"] - V_ecm_b
        res_pred = V_corrected - V_ecm_b
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=res_true*1000, mode="lines",
            name="True Residual (DFN - ECM)",
            line=dict(color=cc, width=1.8)))
        f.add_trace(go.Scatter(x=t_h, y=res_pred*1000, mode="lines",
            name="PINN Predicted Residual",
            line=dict(color=C_PUR, width=1.8, dash="dash")))
        f.update_layout(
            title="④ Residual: True vs PINN Prediction",
            xaxis=ax("Time [h]"), yaxis=ax("Residual [mV]"),
            **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    # ── Architecture summary ──────────────────────────────────
    st.markdown("#### 🧠 PINN Architecture")
    st.code(f"""
Input Layer    : 4 features [t_norm, I_norm, SOC_est, V_ecm_norm]
Hidden Layers  : {pinn_hidden} × {pinn_neurons} neurons (Tanh activation)
Output Layer   : 1 → residual correction [mV]
Physics Loss   : d(SOC)/dt = -I / Q_nom  (Coulomb counting constraint)
Total Loss     : L = MSE(data) + {pinn_lambda} × MSE(physics)
Optimizer      : Adam + Cosine Annealing LR
Epochs         : {pinn_epochs}
─────────────────────────────────────────────────
ECM Baseline V RMSE  : {m['v_rmse_ecm']:.2f} mV
PINN Corrected RMSE  : {m['v_rmse_pinn']:.2f} mV
Improvement          : {m['improv_v']:.1f}%
EKF SOC RMSE         : {m['s_rmse_ekf']:.3f}%
PINN SOC RMSE        : {m['s_rmse_pinn']:.3f}%
""", language="")

# ================================================================
# PDF Report — ReportLab (true PDF, embedded charts)
# ================================================================
st.markdown("---")

def generate_pdf_report(log, t_h, sigma, cyc_pk, noise_std, n_cycles, cl, N,
                         v_rmse, s_rmse, chem, chem_name, Q_nom, protocol_label,
                         c_rate, p0_scale, q_scale, r_scale, last_Ck, last_Kk, msgs):
    import io
    import plotly.graph_objects as go
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, PageBreak, KeepTogether
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    W_PAGE, H_PAGE = A4
    CW = W_PAGE - 36 * mm

    # ── Colours ──────────────────────────────────────────────
    TEAL  = colors.HexColor("#00b4d8")
    NAVY  = colors.HexColor("#0f172a")
    SLATE = colors.HexColor("#64748b")
    LIGHT = colors.HexColor("#f8fafc")
    GRN_B = colors.HexColor("#dcfce7")
    YLW_B = colors.HexColor("#fef9c3")
    RED_B = colors.HexColor("#fee2e2")
    GRN_T = colors.HexColor("#166534")
    YLW_T = colors.HexColor("#854d0e")
    RED_T = colors.HexColor("#991b1b")

    # ── Paragraph styles ─────────────────────────────────────
    def S(name, **kw): return ParagraphStyle(name, **kw)

    sCover   = S("cov",  fontSize=26, textColor=NAVY,  fontName="Helvetica-Bold",
                         alignment=TA_CENTER, spaceAfter=6)
    sSubCov  = S("sub",  fontSize=13, textColor=TEAL,  fontName="Helvetica",
                         alignment=TA_CENTER, spaceAfter=4)
    sH2      = S("h2",   fontSize=13, textColor=NAVY,  fontName="Helvetica-Bold",
                         spaceBefore=10, spaceAfter=5, leftIndent=8)
    sBody    = S("bd",   fontSize=9,  textColor=colors.HexColor("#374151"),
                         fontName="Helvetica", leading=14, spaceAfter=3)
    sCaption = S("cap",  fontSize=8,  textColor=SLATE, fontName="Helvetica-Oblique",
                         alignment=TA_CENTER, spaceAfter=4, spaceBefore=2)
    sSmall   = S("sm",   fontSize=7.5,textColor=SLATE, fontName="Helvetica",
                         alignment=TA_CENTER, spaceAfter=2)
    sKpiLbl  = S("kl",   fontSize=7,  textColor=SLATE, fontName="Helvetica",
                         alignment=TA_CENTER)
    sKpiVal  = S("kv",   fontSize=17, textColor=TEAL,  fontName="Helvetica-Bold",
                         alignment=TA_CENTER)
    sKpiUnt  = S("ku",   fontSize=7,  textColor=SLATE, fontName="Helvetica",
                         alignment=TA_CENTER)
    sCfgLbl  = S("cfl",  fontSize=8.5,textColor=SLATE, fontName="Helvetica")
    sCfgVal  = S("cfv",  fontSize=8.5,textColor=NAVY,  fontName="Courier-Bold")
    sTblHd   = S("th",   fontSize=8.5,textColor=colors.white,
                         fontName="Helvetica-Bold", alignment=TA_CENTER)
    sTblCl   = S("tc",   fontSize=8,  textColor=colors.HexColor("#374151"),
                         fontName="Courier", leading=11)

    BASE_TBL = TableStyle([
        ("BACKGROUND",    (0,0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [LIGHT, colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ])

    def hr(): return HRFlowable(width="100%", thickness=0.5,
                                 color=colors.HexColor("#e2e8f0"),
                                 spaceBefore=6, spaceAfter=6)

    def h2(txt):
        return Paragraph(f'<font color="#00b4d8">▌</font>  <b>{txt}</b>', sH2)

    def tbl(headers, rows, widths):
        data = [[Paragraph(h, sTblHd) for h in headers]]
        for row in rows:
            data.append([Paragraph(str(c), sTblCl) for c in row])
        t = Table(data, colWidths=widths)
        t.setStyle(BASE_TBL)
        return t

    def small_kv_tbl(rows):
        data = [[Paragraph(r[0], sCfgLbl), Paragraph(r[1], sCfgVal)] for r in rows]
        t = Table(data, colWidths=[CW*0.23, CW*0.27])
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT, colors.white]),
            ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#e2e8f0")),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("TOPPADDING",(0,0),(-1,-1),3), ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ]))
        return t

    # ── Chart helpers ─────────────────────────────────────────
    CHART_W, CHART_H = 680, 255
    cc = chem["color"]
    C_ORG = "#f77f00"; C_RED = "#ef233c"; C_GRN = "#2dc653"
    C_YLW = "#ffd60a"; C_PUR = "#c77dff"
    PALS  = [cc, C_ORG, C_GRN, C_RED, C_PUR, "#a8dadc", "#457b9d"]

    LBASE = dict(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font=dict(color="#1e293b", family="Arial", size=10),
        margin=dict(t=44, b=42, l=58, r=18),
        width=CHART_W, height=CHART_H,
        legend=dict(bgcolor="rgba(248,250,252,0.9)", bordercolor="#cbd5e1",
                    borderwidth=1, orientation="h", yanchor="bottom",
                    y=1.04, xanchor="center", x=0.5, font=dict(size=9)),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", showgrid=True, zeroline=False),
    )

    def T(txt): return dict(text=txt, font=dict(size=11, color="#0f172a"),
                             x=0.5, xanchor="center")

    def fig_to_rl(fig, w_mm=None):
        buf = io.BytesIO()
        fig.write_image(buf, format="png", scale=2)
        buf.seek(0)
        rw = (w_mm * mm) if w_mm else CW
        rh = rw * CHART_H / CHART_W
        return RLImage(buf, width=rw, height=rh)

    # ── Build 5 charts ────────────────────────────────────────
    # Fig 1 — Voltage Tracking
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
        name="V_true (DFN)", line=dict(color=cc, width=2.2)))
    f1.add_trace(go.Scatter(x=t_h, y=log["V_est"], mode="lines",
        name=f"V_est (EKF)  RMSE={v_rmse:.1f} mV",
        line=dict(color=C_ORG, width=1.8, dash="dash")))
    f1.update_layout(**LBASE, title=T("① Voltage Tracking — DFN vs EKF Observer"))
    f1.update_xaxes(title_text="Time [h]")
    f1.update_yaxes(title_text="Voltage [V]")
    img_f1 = fig_to_rl(f1)

    # Fig 2 — SOC ±2σ
    upper = log["soc_est"]*100 + 2*sigma
    lower = log["soc_est"]*100 - 2*sigma
    f2 = go.Figure()
    f2.add_trace(go.Scatter(
        x=t_h.tolist() + t_h.tolist()[::-1],
        y=upper.tolist() + lower.tolist()[::-1],
        fill="toself", fillcolor="rgba(247,127,0,0.13)",
        line=dict(color="rgba(0,0,0,0)"), name="±2σ Band"))
    f2.add_trace(go.Scatter(x=t_h, y=log["soc_true"]*100, mode="lines",
        name="SOC_true (DFN)", line=dict(color=cc, width=2.2)))
    f2.add_trace(go.Scatter(x=t_h, y=log["soc_est"]*100, mode="lines",
        name=f"SOC_est (EKF)  RMSE={s_rmse:.2f}%",
        line=dict(color=C_ORG, width=1.8, dash="dash")))
    f2.update_layout(**LBASE, title=T("② SOC Estimation with ±2σ Uncertainty Band"))
    f2.update_xaxes(title_text="Time [h]")
    f2.update_yaxes(title_text="SOC [%]")
    img_f2 = fig_to_rl(f2)

    # Fig 3 — tr(P) convergence
    f3 = go.Figure()
    shown = sorted(set([0,1]+list(range(0,n_cycles,max(1,n_cycles//5)))+[n_cycles-1]))[:7]
    for i, c in enumerate(shown):
        s, e = c*cl, min((c+1)*cl, N)
        f3.add_trace(go.Scatter(x=t_h[s:e], y=log["P_tr"][s:e], mode="lines",
            name=f"Cycle {c+1}", line=dict(color=PALS[i%len(PALS)], width=1.9)))
    f3.update_layout(**LBASE, title=T("③ State Covariance tr(P) — EKF Convergence"))
    f3.update_xaxes(title_text="Time [h]")
    f3.update_yaxes(title_text="tr(P)", tickformat=".1e")
    img_f3 = fig_to_rl(f3)

    # Fig 4 — cycle peak tr(P)
    growth_arr = [(p-cyc_pk[0])/cyc_pk[0]*100 for p in cyc_pk]
    mk_col = ["#94a3b8" if i==0 else
              (C_GRN if g<=10 else (C_YLW if g<=25 else C_RED))
              for i,g in enumerate(growth_arr)]
    f4 = go.Figure()
    f4.add_trace(go.Scatter(
        x=list(range(1, n_cycles+1)), y=cyc_pk,
        mode="lines+markers",
        line=dict(color="#0369a1", width=2.0),
        marker=dict(color=mk_col, size=7, line=dict(width=1, color="#0f172a")),
        name="Peak tr(P)"))
    f4.add_hline(y=cyc_pk[0], line_dash="dot", line_color="#94a3b8",
        annotation_text="Cycle 1 baseline",
        annotation_font=dict(size=9, color="#64748b"))
    f4.update_layout(**LBASE, title=T("④ Cycle-by-Cycle Uncertainty Growth — Peak tr(P)"))
    f4.update_xaxes(title_text="Cycle Number", dtick=max(1,n_cycles//8))
    f4.update_yaxes(title_text="Peak tr(P)", tickformat=".2e")
    img_f4 = fig_to_rl(f4)

    # Fig 5 — Innovation residuals
    innov_rms = float(np.sqrt(np.mean(log["innov"]**2))*1000)
    f5 = go.Figure()
    f5.add_trace(go.Scatter(x=t_h, y=log["innov"]*1000, mode="lines",
        line=dict(color="#1d4ed8", width=0.9), opacity=0.8,
        name="Innovation ν(k)"))
    f5.add_hline(y=+2*noise_std*1000, line_dash="dash", line_color="#ef4444",
        annotation_text=f"+2σ={2*noise_std*1000:.0f} mV",
        annotation_font=dict(size=9))
    f5.add_hline(y=-2*noise_std*1000, line_dash="dash", line_color="#ef4444",
        annotation_text=f"−2σ", annotation_font=dict(size=9))
    f5.add_hline(y=0, line_color="#94a3b8", line_width=0.8)
    f5.update_layout(**LBASE, title=T("⑤ Innovation Residuals ν(k) — Whiteness Test"))
    f5.update_xaxes(title_text="Time [h]")
    f5.update_yaxes(title_text="ν(k) [mV]")
    img_f5 = fig_to_rl(f5)

    # ── Assemble PDF ──────────────────────────────────────────
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm, bottomMargin=15*mm,
        title="BattSim v4.2 Report",
        author="Eng. Thaer Abushawar")

    story = []

    # ══ PAGE 1 — COVER ═══════════════════════════════════════
    story += [
        Spacer(1, 14*mm),
        Paragraph("BattSim v4.2", S("ltx", fontSize=30, textColor=TEAL,
            fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=5)),
        Paragraph("Digital Twin Co-Simulation Report", sSubCov),
        Paragraph("DFN Physical Asset  ↔  EKF Digital Observer", sSmall),
        Spacer(1, 8*mm), hr(), Spacer(1, 4*mm),
    ]

    # Config + KPI side by side
    cfg_rows = [
        ["Cell Chemistry",  chem_name.split("(")[0].strip()],
        ["Test Protocol",   protocol_label],
        ["Cycles / C-Rate", f"{n_cycles} cyc @ {c_rate}C"],
        ["Sensor Noise σ",  f"{noise_std*1000:.0f} mV"],
        ["P₀ / Q / R",      f"{p0_scale:.0e} / {q_scale} / {r_scale}"],
        ["Data Points",     f"{N:,} @ dt=10 s"],
        ["Date",            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
        ["Author",          "Eng. Thaer Abushawar"],
    ]
    cfg_tbl = Table(
        [[Paragraph(r[0], sCfgLbl), Paragraph(r[1], sCfgVal)] for r in cfg_rows],
        colWidths=[CW*0.36, CW*0.64])
    cfg_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT,colors.white]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#e2e8f0")),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story += [cfg_tbl, Spacer(1, 7*mm)]

    # KPI cards
    kpi_vals = [
        ("VOLTAGE RMSE", f"{v_rmse:.2f}", "mV"),
        ("SOC RMSE",     f"{s_rmse:.3f}", "%"),
        ("PEAK tr(P)",   f"{log['P_tr'].max():.1e}", "state UQ"),
        ("CYCLES RUN",   f"{n_cycles}", f"@ {c_rate}C"),
    ]
    kpi_row1 = [Paragraph(k[0], sKpiLbl) for k in kpi_vals]
    kpi_row2 = [Paragraph(k[1], sKpiVal) for k in kpi_vals]
    kpi_row3 = [Paragraph(k[2], sKpiUnt) for k in kpi_vals]
    kpi_tbl = Table([kpi_row1, kpi_row2, kpi_row3], colWidths=[CW/4]*4)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LIGHT),
        ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#e2e8f0")),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))
    story += [kpi_tbl, Spacer(1,5*mm), hr(),
              Paragraph(
                  "Refs: Plett (2004) J. Power Sources 134 · "
                  "Chen et al. (2020) J. Electrochem. Soc. 167 · "
                  "Coman et al. (2022)", sSmall),
              PageBreak()]

    # ══ PAGE 2 — MACHINE 1 + FIG 1 + FIG 2 ══════════════════
    story.append(h2("Machine 1 — PyBaMM DFN Physical Asset"))
    story.append(Paragraph(
        f"The DFN model provides ground-truth electrochemical states. "
        f"Machine 2 observes only V_noisy = V_true + η  (σ = {noise_std*1000:.0f} mV).",
        sBody))
    story.append(Spacer(1, 3*mm))

    left_rows = [
        ["Q nominal", f"{Q_nom:.2f} Ah"],
        ["R₀",        f"{chem['R0']*1000:.1f} mΩ"],
        ["R₁/C₁",    f"{chem['R1']*1000:.0f} mΩ / {chem['C1']:.0f} F"],
        ["R₂/C₂",    f"{chem['R2']*1000:.0f} mΩ / {chem['C2']:.0f} F"],
        ["V range",  f"{chem['v_min']}–{chem['v_max']} V"],
    ]
    right_rows = [
        ["Total time",  f"{log['t'][-1]/3600:.2f} h"],
        ["Data points", f"{N:,}"],
        ["V_true range",f"{log['V_true'].min():.3f}–{log['V_true'].max():.3f} V"],
        ["I range",     f"{log['I_true'].min():.2f}–{log['I_true'].max():.2f} A"],
        ["T range",     f"{log['T_true'].min():.1f}–{log['T_true'].max():.1f} °C"],
    ]
    two = Table([[small_kv_tbl(left_rows), small_kv_tbl(right_rows)]],
                colWidths=[CW/2, CW/2])
    two.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),
                              ("RIGHTPADDING",(0,0),(-1,-1),0),
                              ("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [two, Spacer(1,4*mm), hr(),
              img_f1,
              Paragraph("Fig. 1 — Voltage tracking: DFN ground-truth (solid) vs EKF reconstruction (dashed).", sCaption),
              Spacer(1,3*mm),
              img_f2,
              Paragraph("Fig. 2 — SOC estimation with ±2σ uncertainty band (orange shading). Method: Plett (2004).", sCaption),
              PageBreak()]

    # ══ PAGE 3 — MACHINE 2 + FIG 3 + FIG 5 ══════════════════
    story.append(h2("Machine 2 — 2-RC ECM + Extended Kalman Filter"))
    story.append(Paragraph(
        "The EKF reconstructs battery states [SOC, V_RC1, V_RC2] "
        "from noisy voltage and current only — no access to DFN internals.", sBody))
    story.append(Spacer(1, 3*mm))

    _ir     = innov_rms / (noise_std * 1000)
    _im_abs = abs(float(log["innov"].mean()) * 1000)
    p_ratio = float(log["P_tr"][-1] / log["P_tr"][0] * 100)

    ekf_left = [
        ["V RMSE",       f"{v_rmse:.3f} mV"],
        ["SOC RMSE",     f"{s_rmse:.4f} %"],
        ["Max SOC err",  f"{np.abs(log['soc_true']-log['soc_est']).max()*100:.3f} %"],
        ["tr(P) initial",f"{log['P_tr'][0]:.3e}"],
        ["tr(P) final",  f"{log['P_tr'][-1]:.3e}"],
        ["Convergence",  f"{p_ratio:.2f}% of P₀"],
    ]
    ekf_right = [
        ["∂h/∂SOC",  f"{float(last_Ck[0,0]):.5f}"],
        ["∂h/∂VRC1", f"{float(last_Ck[0,1]):.5f}"],
        ["∂h/∂VRC2", f"{float(last_Ck[0,2]):.5f}"],
        ["K_SOC",    f"{float(last_Kk[0,0]):.6f}"],
        ["K_VRC1",   f"{float(last_Kk[1,0]):.6f}"],
        ["K_VRC2",   f"{float(last_Kk[2,0]):.6f}"],
    ]
    two2 = Table([[small_kv_tbl(ekf_left), small_kv_tbl(ekf_right)]],
                 colWidths=[CW/2, CW/2])
    two2.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),
                               ("RIGHTPADDING",(0,0),(-1,-1),0),
                               ("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [two2, Spacer(1, 3*mm), hr()]

    # Innovation whiteness table
    def _b(ok_c, w_c): return "PASS" if ok_c else ("REVIEW" if w_c else "FAIL")
    inn_rows = [
        ["Innovation RMS",   f"{innov_rms:.3f} mV", f"< {noise_std*1500:.1f} mV",
         _b(innov_rms<noise_std*1500, innov_rms<noise_std*3000)],
        ["Inn./Noise ratio", f"{_ir:.3f}×",          "< 1.5×",
         _b(_ir<1.5, _ir<3.0)],
        ["Innovation mean",  f"{_im_abs:.4f} mV",   "≈ 0",
         _b(_im_abs<noise_std*500, _im_abs<noise_std*1000)],
    ]
    story += [tbl(["Metric","Value","Target","Status"], inn_rows,
                  [CW*0.33, CW*0.22, CW*0.22, CW*0.23]),
              Spacer(1, 4*mm),
              img_f3,
              Paragraph("Fig. 3 — State covariance tr(P) convergence. tr(P) → 0 confirms EKF convergence.", sCaption),
              Spacer(1, 3*mm),
              img_f5,
              Paragraph("Fig. 5 — Innovation residuals ν(k). White noise within ±2σ bounds → well-tuned EKF.", sCaption),
              PageBreak()]

    # ══ PAGE 4 — UQ ANALYTICS + ASSESSMENT ═══════════════════
    story.append(h2("Uncertainty Propagation Analytics"))
    story.append(Paragraph(
        "Cycle-by-cycle peak tr(P) quantifies how state uncertainty evolves across "
        "charge–discharge cycles. Growth > 20% signals ECM model error accumulation.", sBody))
    story += [Spacer(1, 2*mm),
              img_f4,
              Paragraph(
                  "Fig. 4 — Cycle-by-cycle peak tr(P). "
                  "Colour: green ≤10%, yellow ≤25%, red >25% growth vs Cycle 1.", sCaption),
              Spacer(1, 3*mm), hr()]

    # Cycle summary table
    cyc_hdr = ["Cycle","SOC₀[%]","SOC_min[%]","SOC_end[%]","Dur[min]",
               "Peak tr(P)","Δ vs C1","Peak σ_SOC[%]","Status"]
    cyc_data = []
    for c in range(n_cycles):
        g    = (cyc_pk[c]-cyc_pk[0])/cyc_pk[0]*100 if c>0 else 0
        lbl  = "Baseline" if c==0 else ("Stable" if g<=10 else ("Growing" if g<=25 else "High"))
        psoc = np.sqrt(log["P_soc"][c*cl:min((c+1)*cl,N)]).max()*100
        s_i  = c*cl; e_i = min((c+1)*cl,N)
        dur  = (log["t"][e_i-1]-log["t"][s_i])/60
        cyc_data.append([
            f"Cycle {c+1}",
            f"{log['soc_true'][s_i]*100:.1f}",
            f"{log['soc_true'][s_i:e_i].min()*100:.1f}",
            f"{log['soc_true'][e_i-1]*100:.1f}",
            f"{dur:.1f}",
            f"{cyc_pk[c]:.3e}",
            "Baseline" if c==0 else f"{g:+.1f}%",
            f"{psoc:.4f}",
            lbl
        ])
    cw9 = [CW*w for w in [0.10,0.09,0.10,0.10,0.09,0.14,0.10,0.14,0.14]]
    story += [tbl(cyc_hdr, cyc_data, cw9), Spacer(1, 4*mm), hr()]

    # Engineering assessment
    story.append(h2("Engineering Assessment"))
    story.append(Spacer(1, 2*mm))
    for kind, msg in msgs:
        icon = "✅" if kind=="ok" else ("⚠️" if kind=="warn" else "🔴")
        bg   = GRN_B if kind=="ok" else (YLW_B if kind=="warn" else RED_B)
        tc   = GRN_T if kind=="ok" else (YLW_T if kind=="warn" else RED_T)
        row  = Table(
            [[Paragraph(f"{icon}  {msg}",
                         S("am", fontSize=8.5, textColor=tc,
                           fontName="Helvetica", leading=13))]],
            colWidths=[CW])
        row.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),bg),
            ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING",(0,0),(-1,-1),5),  ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]))
        story += [row, Spacer(1, 2*mm)]

    story += [
        Spacer(1, 8*mm), hr(),
        Paragraph("BattSim v4.2 — Digital Twin Co-Simulation Framework", sSmall),
        Paragraph("Designed & Developed by Eng. Thaer Abushawar", sSmall),
        Paragraph(
            "Plett (2004) J. Power Sources 134  ·  "
            "Chen et al. (2020) J. Electrochem. Soc. 167  ·  "
            "Coman et al. (2022)", sSmall),
    ]

    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf.read()


if st.button("📥  Generate PDF Report", use_container_width=False, type="secondary"):
    with st.spinner("Building charts and PDF..."):
        pdf_bytes = generate_pdf_report(
            log, t_h, sigma, cyc_pk, noise_std, n_cycles, cl, N,
            v_rmse, s_rmse, chem, chem_name, Q_nom, protocol_label,
            c_rate, p0_scale, q_scale, r_scale, last_Ck, last_Kk,
            engineering_assessment(log, noise_std, n_cycles, cyc_pk)
        )
    st.download_button(
        label="✅  Download PDF Report",
        data=pdf_bytes,
        file_name=f"BattSim_{chem_name.split('—')[0].strip().replace(' ','_')}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=False,
    )

# ================================================================
# Footer
# ================================================================
st.markdown("""
<div class='footer-bar'>
  🔋 <b>BattSim v4.2</b> — DFN ↔ EKF Digital Twin Co-Simulation &nbsp;|&nbsp;
  Eng. Thaer Abushawar &nbsp;|&nbsp;
  Plett (2004) · Chen et al. (2020) · Coman et al. (2022)
</div>""", unsafe_allow_html=True)




# ================================================================
# Footer
# ================================================================
st.markdown("""
<div class='footer-bar'>
  🔋 <strong>BattSim v4.2</strong> · DFN ↔ EKF Digital Twin Co-Simulation ·
  Eng. Thaer Abushawar ·
  <em>Plett (2004) · Chen et al. (2020) · Coman et al. (2022)</em>
</div>
""", unsafe_allow_html=True)
