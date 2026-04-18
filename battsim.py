# ================================================================
# BattSim — v4.1 (Complete & Research-Grade)
# ================================================================
#
# Two-machine co-simulation framework:
#
#   Machine 1 — PyBaMM DFN
#       Full electrochemical model (Doyle-Fuller-Newman).
#       Acts as the physical asset / ground truth.
#
#   Machine 2 — Dual EKF (DEKF) with 2-RC Thevenin ECM
#       EKF-1 estimates states:  x = [SOC, V_rc1, V_rc2]
#       EKF-2 estimates params:  theta = [Q_nom, R0]
#       Both filters run alternately at 1-second intervals.
#
#   Uncertainty Quantification:
#       tr(P1) tracks state-estimation uncertainty per cycle.
#       tr(P2) tracks parameter-estimation uncertainty.
#       Sensitivity sweeps noise level and P1 initialisation,
#       using the session's actual noise as the fixed baseline
#       when sweeping P1 (session-aware, not hardcoded).
#
#   Supported chemistries: NMC (Chen2020), LFP (Prada2013), NCA (Ecker2015)
#
# Author : Eng. Thaer Abushawar — Thaer199@gmail.com
# Refs   : Plett (2004) J. Power Sources 134
#           Chen et al. (2020) J. Electrochem. Soc. 167
#           Coman et al. (2022) J. Electrochem. Soc. 169
# ================================================================

import warnings
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import streamlit as st

warnings.filterwarnings("ignore")


# ── Streamlit page setup ─────────────────────────────────────────
st.set_page_config(
    page_title="BattSim v4.1",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

  .metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
  }
  .metric-label { color: #a0aec0; font-size: 0.75rem; font-weight: 500;
                  text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { color: #00b4d8; font-size: 1.6rem; font-weight: 700;
                  font-family: 'JetBrains Mono'; }
  .metric-sub   { color: #718096; font-size: 0.7rem; margin-top: 2px; }

  .badge-ok   { background:#1a3a2a; color:#2dc653; padding:3px 10px;
                border-radius:20px; font-size:0.72rem; font-weight:600; }
  .badge-warn { background:#3a2a1a; color:#f77f00; padding:3px 10px;
                border-radius:20px; font-size:0.72rem; font-weight:600; }

  .section-hdr {
    border-left: 3px solid #00b4d8;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.75rem;
    font-size: 1rem; font-weight: 600; color: #e2e8f0;
  }

  .footer-bar {
    margin-top: 2.5rem;
    padding: 1.2rem 0 0.8rem;
    border-top: 1px solid #21262d;
    text-align: center;
    font-size: 0.78rem;
    color: #6e7681;
    font-family: 'Inter', sans-serif;
  }
  .footer-bar .app-name { color: #00b4d8; font-weight: 700; font-size: 0.9rem; }
  .footer-bar .author   { color: #c9d1d9; font-weight: 600; }
  .footer-bar a         { color: #00b4d8; text-decoration: none; }
  .footer-bar a:hover   { text-decoration: underline; }

  div[data-testid="stSidebar"] { background: #0d1117; }
  .stButton>button {
    background: linear-gradient(135deg,#2b6cb0,#2c5282);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; letter-spacing: 0.03em; transition: all 0.2s;
  }
  .stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(66,153,225,0.4);
  }
</style>
""", unsafe_allow_html=True)


# ================================================================
# Chemistry parameter library
# ================================================================

CHEM = {
    "NMC — LG M50 (Chen2020)": {
        "Q":   5.0,
        "R0":  0.010, "R1": 0.015, "C1": 3000,
                       "R2": 0.008, "C2": 8000,
        "soc_lut": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "ocv_lut": [3.0, 3.3, 3.42, 3.5, 3.54, 3.57, 3.62, 3.65, 3.68, 3.71,
                    3.74, 3.77, 3.8, 3.84, 3.88, 3.92, 3.96, 4.01, 4.06, 4.13, 4.2],
        "pybamm": "Chen2020",
        "color":  "#00b4d8",
        "desc":   "Tesla Model 3 LR / Performance cell",
    },
    "LFP — Prada2013": {
        "Q":   1.1,
        "R0":  0.020, "R1": 0.018, "C1": 2500,
                       "R2": 0.010, "C2": 6000,
        "soc_lut": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "ocv_lut": [3.0, 3.1, 3.2, 3.25, 3.28, 3.30, 3.31, 3.32, 3.325, 3.33,
                    3.335, 3.34, 3.345, 3.35, 3.36, 3.37, 3.38, 3.39, 3.40, 3.42, 3.6],
        "pybamm": "Prada2013",
        "color":  "#2dc653",
        "desc":   "Tesla Model 3 SR / BYD Blade cell",
    },
    "NCA — Ecker2015": {
        "Q":   7.5,
        "R0":  0.008, "R1": 0.012, "C1": 4000,
                       "R2": 0.006, "C2": 10000,
        "soc_lut": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "ocv_lut": [2.8, 3.2, 3.4, 3.52, 3.58, 3.62, 3.66, 3.70, 3.73, 3.76,
                    3.79, 3.82, 3.86, 3.90, 3.94, 3.98, 4.02, 4.07, 4.12, 4.17, 4.2],
        "pybamm": "Ecker2015",
        "color":  "#f77f00",
        "desc":   "Panasonic 18650 / early Tesla packs",
    },
}


def make_ocv(p):
    return interp1d(p["soc_lut"], p["ocv_lut"], kind="cubic", fill_value="extrapolate")


def docv_dsoc(ocv_fn, soc, h=1e-4):
    return (ocv_fn(soc + h) - ocv_fn(soc - h)) / (2.0 * h)


# ================================================================
# Machine 1 — PyBaMM DFN
# ================================================================

def run_dfn(pset_name, n_cycles, c_rate, prog, status):
    import pybamm

    status.markdown(f"**[Machine 1 — DFN]** Loading parameter set `{pset_name}`...")
    prog.progress(5)

    model  = pybamm.lithium_ion.DFN()
    params = pybamm.ParameterValues(pset_name)

    exp = pybamm.Experiment(
        [
            f"Discharge at {c_rate}C until 2.5 V",
            "Rest for 5 minutes",
            f"Charge at {c_rate / 2:.1f}C until 4.2 V",
            "Rest for 5 minutes",
        ] * n_cycles
    )

    status.markdown(f"**[Machine 1 — DFN]** Solving {n_cycles} cycle(s) at {c_rate}C ...")
    prog.progress(12)

    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sim.solve()
    sol = sim.solution
    prog.progress(42)

    t     = sol["Time [s]"].entries
    V     = sol["Terminal voltage [V]"].entries
    I     = sol["Current [A]"].entries
    Q_nom = float(params["Nominal cell capacity [A.h]"])

    dt  = np.diff(t, prepend=t[0])
    soc = np.clip(1.0 - np.cumsum(I * dt) / 3600.0 / Q_nom, 0.0, 1.0)

    t_u = np.arange(t[0], t[-1], 1.0)
    return (
        t_u,
        np.interp(t_u, t, V),
        np.interp(t_u, t, I),
        np.interp(t_u, t, soc),
        Q_nom,
    )


# ================================================================
# Machine 2 — Dual EKF (DEKF)
# ================================================================

class DEKF:
    def __init__(self, soc0, Q_nom, R0, R1, C1, R2, C2, noise_var, chem):
        self.dt  = 1.0
        self.ocv = make_ocv(chem)
        self.R1, self.C1 = R1, C1
        self.R2, self.C2 = R2, C2

        self.x1  = np.array([[soc0], [0.0], [0.0]])
        self.P1  = np.diag([1e-3, 1e-4, 1e-4])
        self.Q1  = np.diag([1e-8, 1e-6, 1e-6])
        self.R1m = np.array([[noise_var]])

        self.x2  = np.array([[Q_nom], [R0]])
        self.P2  = np.diag([0.01, 1e-5])
        self.Q2  = np.diag([1e-8, 1e-10])
        self.R2m = np.array([[noise_var * 4.0]])

        self.I2 = np.eye(2)
        self.I3 = np.eye(3)

    @property
    def Q_est(self):
        return max(float(self.x2[0, 0]), 0.1)

    @property
    def R0_est(self):
        return max(float(self.x2[1, 0]), 1e-4)

    def step(self, v_meas, current):
        dt     = self.dt
        R1, C1 = self.R1, self.C1
        R2, C2 = self.R2, self.C2
        e1     = np.exp(-dt / (R1 * C1))
        e2     = np.exp(-dt / (R2 * C2))

        s, v1, v2 = self.x1[:, 0]
        s_p  = s  - current * dt / (self.Q_est * 3600.0)
        v1_p = v1 * e1 + current * R1 * (1.0 - e1)
        v2_p = v2 * e2 + current * R2 * (1.0 - e2)
        x1p  = np.array([[s_p], [v1_p], [v2_p]])

        A1  = np.diag([1.0, e1, e2])
        P1p = A1 @ self.P1 @ A1.T + self.Q1

        dOCV = docv_dsoc(self.ocv, s_p)
        C1m  = np.array([[dOCV, -1.0, -1.0]])
        vh_1 = float(self.ocv(s_p)) - v1_p - v2_p - current * self.R0_est
        nu_1 = v_meas - vh_1
        S1   = C1m @ P1p @ C1m.T + self.R1m
        K1   = P1p @ C1m.T / S1[0, 0]

        self.x1       = x1p + K1 * nu_1
        self.x1[0, 0] = np.clip(self.x1[0, 0], 0.0, 1.0)
        IKC1    = self.I3 - K1 @ C1m
        self.P1 = IKC1 @ P1p @ IKC1.T + K1 @ self.R1m @ K1.T

        P2p = self.P2 + self.Q2

        soc_e   = float(self.x1[0, 0])
        dOCV_e  = docv_dsoc(self.ocv, soc_e)
        dSOC_dQ = (current * dt / 3600.0) / (self.Q_est ** 2)
        C2m     = np.array([[dOCV_e * dSOC_dQ, -current]])

        vh_2 = (float(self.ocv(soc_e))
                - float(self.x1[1, 0])
                - float(self.x1[2, 0])
                - current * self.R0_est)
        nu_2 = v_meas - vh_2

        S2 = C2m @ P2p @ C2m.T + self.R2m
        if abs(S2[0, 0]) > 1e-15:
            K2 = P2p @ C2m.T / S2[0, 0]
            self.x2       = self.x2 + K2 * nu_2
            self.x2[0, 0] = max(float(self.x2[0, 0]), 0.1)
            self.x2[1, 0] = max(float(self.x2[1, 0]), 1e-4)
            IKC2    = self.I2 - K2 @ C2m
            self.P2 = IKC2 @ P2p @ IKC2.T + K2 @ self.R2m @ K2.T

        v_est = (float(self.ocv(soc_e))
                 - float(self.x1[1, 0])
                 - float(self.x1[2, 0])
                 - current * self.R0_est)

        return (
            v_est,
            soc_e,
            float(np.trace(self.P1)),
            float(self.P1[0, 0]),
            self.Q_est,
            self.R0_est,
            float(np.trace(self.P2)),
        )


# ================================================================
# Co-simulation driver
# ================================================================

def run_cosim(chem_name, n_cycles, c_rate, noise_std, prog, status):
    chem = CHEM[chem_name]
    t, V_true, I_true, soc_true, Q_nom = run_dfn(
        chem["pybamm"], n_cycles, c_rate, prog, status
    )

    status.markdown("**[Machine 2 — DEKF]** Initialising Dual EKF (2-RC + online param ID) ...")
    prog.progress(47)

    ekf = DEKF(
        float(soc_true[0]), Q_nom,
        chem["R0"], chem["R1"], chem["C1"],
        chem["R2"], chem["C2"],
        noise_std ** 2, chem,
    )

    N   = len(t)
    log = {k: np.empty(N) for k in [
        "V_meas", "V_est", "soc_est",
        "P1_tr", "P1_soc", "Q_est", "R0_est", "P2_tr",
    ]}
    log["t"]        = t
    log["V_true"]   = V_true
    log["soc_true"] = soc_true
    log["I_true"]   = I_true

    status.markdown(f"**[Co-Sim]** Running {N:,} time steps — DEKF online estimation ...")
    ckpt = max(1, N // 40)

    for k in range(N):
        vm = V_true[k] + np.random.normal(0.0, noise_std)
        v_est, soc_e, p1tr, p1s, q_e, r0_e, p2tr = ekf.step(vm, I_true[k])
        log["V_meas"][k]  = vm
        log["V_est"][k]   = v_est
        log["soc_est"][k] = soc_e
        log["P1_tr"][k]   = p1tr
        log["P1_soc"][k]  = p1s
        log["Q_est"][k]   = q_e
        log["R0_est"][k]  = r0_e
        log["P2_tr"][k]   = p2tr
        if k % ckpt == 0:
            prog.progress(47 + int(48 * k / N))

    prog.progress(97)
    return log, Q_nom, chem


# ================================================================
# Sensitivity analysis
# ================================================================

def sensitivity_analysis(chem_name, base_log, session_noise=0.010):
    chem = CHEM[chem_name]
    N    = len(base_log["t"])

    factors = {
        "Sensor noise sigma [V]": [0.005, 0.010, 0.020, 0.030, 0.040],
        "Initial P1 diagonal":    [1e-4,  1e-3,  1e-2,  1e-1,  0.5  ],
    }

    results = {}
    for label, vals in factors.items():
        peaks = []
        for v in vals:
            noise = v          if "noise"   in label else session_noise
            p0    = v          if "Initial" in label else 1e-3

            ekf_s = DEKF(
                float(base_log["soc_true"][0]), chem["Q"],
                chem["R0"], chem["R1"], chem["C1"],
                chem["R2"], chem["C2"],
                noise ** 2, chem,
            )
            ekf_s.P1 = np.diag([p0, p0 * 0.1, p0 * 0.1])

            p_max = 0.0
            for k in range(min(N, 3600)):
                vm = base_log["V_true"][k] + np.random.normal(0.0, noise)
                _, _, p1tr, *_ = ekf_s.step(vm, base_log["I_true"][k])
                p_max = max(p_max, p1tr)
            peaks.append(p_max)
        results[label] = {"vals": vals, "peaks": peaks}

    return results


# ================================================================
# Color palette
# ================================================================

C_TEAL = "#00b4d8"
C_ORG  = "#f77f00"
C_GRN  = "#2dc653"
C_RED  = "#ef233c"
C_PUR  = "#c77dff"
C_YLW  = "#ffd60a"
DARK   = "#0d1117"
PLOT   = "#161b22"

LAYOUT_BASE = dict(
    paper_bgcolor=DARK,
    plot_bgcolor=PLOT,
    font=dict(color="#c9d1d9", family="Inter", size=11),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d",
                borderwidth=1, font=dict(size=10)),
    margin=dict(t=48, b=36, l=58, r=24),
    height=380,
)


# ================================================================
# Dashboard — 8 independent figures
# ================================================================

def render_dashboard(log, chem, n_cyc, chem_name):
    t_h  = log["t"] / 3600.0
    ve   = np.abs(log["V_true"] - log["V_est"]) * 1000.0
    se   = np.abs(log["soc_true"] - log["soc_est"]) * 100.0
    cc   = chem["color"]
    N    = len(log["t"])
    cl   = N // n_cyc
    pals = [C_TEAL, C_ORG, C_GRN, C_RED, C_PUR]

    def base_fig(title, ytitle, xtitle="Time [h]"):
        f = go.Figure()
        f.update_layout(
            title=dict(text=title, font=dict(size=12, color="#e2e8f0")),
            xaxis=dict(title=xtitle, gridcolor="#21262d", showgrid=True, zeroline=False),
            yaxis=dict(title=ytitle, gridcolor="#21262d", showgrid=True, zeroline=False),
            **LAYOUT_BASE,
        )
        return f

    c1, c2 = st.columns(2)
    with c1:
        f = base_fig("① Terminal Voltage — DFN vs DEKF", "V [V]")
        f.add_trace(go.Scatter(x=t_h, y=log["V_meas"], mode="lines",
            name="Measured", line=dict(color="#6e7681", width=0.5), opacity=0.4))
        f.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
            name="DFN truth", line=dict(color=cc, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["V_est"], mode="lines",
            name="DEKF est.", line=dict(color=C_ORG, width=1.8, dash="dash")))
        st.plotly_chart(f, use_container_width=True)

    with c2:
        f = base_fig("② State of Charge Estimation", "SOC [%]")
        f.add_trace(go.Scatter(x=t_h, y=log["soc_true"] * 100, mode="lines",
            name="SOC — DFN", line=dict(color=cc, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["soc_est"] * 100, mode="lines",
            name="SOC — DEKF", line=dict(color=C_ORG, width=1.8, dash="dash")))
        f.add_trace(go.Scatter(
            x=np.concatenate([t_h, t_h[::-1]]),
            y=np.concatenate([log["soc_true"]*100+se, (log["soc_true"]*100-se)[::-1]]),
            fill="toself", fillcolor="rgba(247,127,0,0.10)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False))
        st.plotly_chart(f, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        f = base_fig("③ DEKF State Covariance  tr(P1)", "tr(P1)")
        f.add_trace(go.Scatter(x=t_h, y=log["P1_tr"], mode="lines",
            line=dict(color=C_GRN, width=2.2),
            fill="tozeroy", fillcolor="rgba(45,198,83,0.12)", showlegend=False))
        st.plotly_chart(f, use_container_width=True)

    with c4:
        f = base_fig("④ Online Parameter Identification", "Value")
        f.add_trace(go.Scatter(x=t_h, y=log["Q_est"], mode="lines",
            name="Q_nom est [Ah]", line=dict(color=C_PUR, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["R0_est"] * 1000, mode="lines",
            name="R0 est [mΩ]", line=dict(color=C_YLW, width=2.2)))
        st.plotly_chart(f, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        f = base_fig("⑤ Voltage Estimation Error", "|Error| [mV]")
        f.add_trace(go.Scatter(x=t_h, y=ve, mode="lines",
            line=dict(color=C_ORG, width=1.5),
            fill="tozeroy", fillcolor="rgba(247,127,0,0.15)", showlegend=False))
        st.plotly_chart(f, use_container_width=True)

    with c6:
        f = base_fig("⑥ SOC Estimation Error", "|Error| [%]")
        f.add_trace(go.Scatter(x=t_h, y=se, mode="lines",
            line=dict(color=C_RED, width=1.5),
            fill="tozeroy", fillcolor="rgba(239,35,60,0.15)", showlegend=False))
        st.plotly_chart(f, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        f = base_fig("⑦ Uncertainty Propagation per Cycle", "tr(P1)", "Time in cycle [h]")
        for c in range(n_cyc):
            s, e = c * cl, min((c + 1) * cl, N)
            tc   = (log["t"][s:e] - log["t"][s]) / 3600.0
            f.add_trace(go.Scatter(x=tc, y=log["P1_tr"][s:e], mode="lines",
                name=f"Cycle {c + 1}",
                line=dict(color=pals[c % len(pals)], width=2.2)))
        st.plotly_chart(f, use_container_width=True)

    with c8:
        f = base_fig("⑧ Parameter Covariance  tr(P2)", "tr(P2)")
        f.add_trace(go.Scatter(x=t_h, y=log["P2_tr"], mode="lines",
            line=dict(color=C_TEAL, width=2.2),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.12)", showlegend=False))
        st.plotly_chart(f, use_container_width=True)


# ================================================================
# Sensitivity tornado chart
# ================================================================

def tornado_chart(sens_data):
    fig     = go.Figure()
    palette = [C_TEAL, C_ORG, C_GRN, C_PUR]

    for i, (lbl, d) in enumerate(sens_data.items()):
        lo, hi = min(d["peaks"]), max(d["peaks"])
        base   = d["peaks"][len(d["peaks"]) // 2]
        delta  = (hi - lo) / base * 100 if base > 0 else 0

        fig.add_trace(go.Bar(
            y=[lbl], x=[hi - lo], base=[lo],
            orientation="h",
            marker_color=palette[i % len(palette)],
            showlegend=False,
            text=[f"Δ {delta:.0f}%"],
            textposition="inside",
            insidetextanchor="middle",
        ))

    fig.update_layout(
        title="Sensitivity of Peak tr(P1) to Observer Tuning Parameters",
        paper_bgcolor=DARK, plot_bgcolor=PLOT,
        font=dict(color="#c9d1d9", family="Inter"),
        xaxis=dict(title="Peak tr(P1) range", gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        height=300,
        margin=dict(t=55, b=35, l=210, r=30),
    )
    return fig


# ================================================================
# Summary statistics
# ================================================================

def compute_stats(log, n_cyc, Q_nom):
    N  = len(log["t"])
    cl = N // n_cyc

    v_rmse  = np.sqrt(np.mean((log["V_true"] - log["V_est"]) ** 2)) * 1000.0
    s_rmse  = np.sqrt(np.mean((log["soc_true"] - log["soc_est"]) ** 2)) * 100.0
    p_peak  = log["P1_tr"].max()
    p_final = log["P1_tr"][-1]
    soh     = log["Q_est"][-1] / Q_nom * 100.0

    cyc_pk = [
        log["P1_tr"][c * cl : min((c + 1) * cl, N)].max()
        for c in range(n_cyc)
    ]
    return v_rmse, s_rmse, p_peak, p_final, soh, cyc_pk


# ================================================================
# Sidebar
# ================================================================

st.sidebar.markdown("""
<div style='text-align:center;padding:1rem 0 0.5rem'>
  <span style='font-size:2rem'>🔋</span><br>
  <span style='font-weight:700;font-size:1.1rem;color:#00b4d8'>BattSim</span><br>
  <span style='font-size:0.7rem;color:#6e7681'>v4.1 · DFN ↔ DEKF · Research Grade</span>
</div>
<hr style='border-color:#30363d;margin:0.5rem 0'>
""", unsafe_allow_html=True)

chem_name = st.sidebar.selectbox("🧪 Chemistry", list(CHEM.keys()))
n_cycles  = st.sidebar.slider("🔁 Cycles", 1, 5, 3)
c_rate    = st.sidebar.select_slider("⚡ C-Rate", [0.5, 1.0, 1.5, 2.0], value=1.0)
noise_db  = st.sidebar.select_slider(
    "📡 Sensor noise",
    ["Low (5 mV)", "Medium (10 mV)", "High (30 mV)"],
    value="Medium (10 mV)",
)
noise_map = {"Low (5 mV)": 0.005, "Medium (10 mV)": 0.010, "High (30 mV)": 0.030}
noise_std = noise_map[noise_db]

st.sidebar.markdown("---")
run_btn  = st.sidebar.button("▶  Run co-simulation",    use_container_width=True, type="primary")
sens_btn = st.sidebar.button("📊  Sensitivity analysis", use_container_width=True)

st.sidebar.markdown("""
<hr style='border-color:#30363d'>
<div style='font-size:0.72rem;color:#6e7681;line-height:1.8'>
<b style='color:#8b949e'>Machine 1 — PyBaMM DFN</b><br>
Doyle-Fuller-Newman electrochemical model<br>
Ground-truth voltage, current, SOC<br><br>
<b style='color:#8b949e'>Machine 2 — DEKF (2-RC)</b><br>
EKF-1 states  : SOC, V1, V2<br>
EKF-2 params  : Q_nom, R0<br>
Chain-rule Jacobian · exact delta-t<br>
Session-consistent sensitivity sweeps<br><br>
<b style='color:#8b949e'>UQ metrics</b><br>
tr(P1) — state uncertainty<br>
tr(P2) — parameter uncertainty<br>
Sensitivity — observer tuning only<br><br>
<i>Plett 2004 · Chen 2020 · Coman 2022</i>
</div>
""", unsafe_allow_html=True)


# ================================================================
# Page header
# ================================================================

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 🔋 BattSim — v4.1")
    st.markdown(
        "**Machine 1:** PyBaMM DFN (physical asset) &nbsp;|&nbsp; "
        "**Machine 2:** Dual EKF 2-RC + online parameter identification  \n"
        "**UQ:** tr(P) covariance propagation per cycle · "
        "multi-chemistry · sensitivity analysis"
    )

chem_data = CHEM[chem_name]
with col_h2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Selected chemistry</div>
      <div class='metric-value' style='font-size:1rem;color:{chem_data["color"]}'>
        {chem_name.split("—")[0].strip()}
      </div>
      <div class='metric-sub'>{chem_data["desc"]}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ================================================================
# Run simulation
# ================================================================

if run_btn:
    np.random.seed(42)
    pbar = st.progress(0)
    stat = st.empty()
    try:
        log, Q_nom, chem = run_cosim(
            chem_name, n_cycles, c_rate, noise_std, pbar, stat
        )
        pbar.progress(100)
        stat.success("✅ Co-simulation complete.")
        st.session_state.update({
            "log":       log,
            "Q_nom":     Q_nom,
            "chem":      chem,
            "chem_name": chem_name,
            "n_cyc":     n_cycles,
            "noise_std": noise_std,
        })
    except Exception as ex:
        import traceback
        pbar.empty()
        stat.error(f"❌ {ex}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# ================================================================
# Display results
# ================================================================

if "log" in st.session_state:
    log        = st.session_state["log"]
    Q_nom      = st.session_state["Q_nom"]
    chem       = st.session_state["chem"]
    cname      = st.session_state["chem_name"]
    n_cyc      = st.session_state["n_cyc"]
    sess_noise = st.session_state.get("noise_std", 0.010)

    v_rmse, s_rmse, p_peak, p_final, soh, cyc_pk = compute_stats(log, n_cyc, Q_nom)
    v_err_max = np.abs(log["V_true"] - log["V_est"]).max() * 1000.0

    # KPI row
    st.markdown("<div class='section-hdr'>📈 Key Performance Indicators</div>",
                unsafe_allow_html=True)

    def kpi(col, label, val, sub="", warn=False):
        badge_cls = "badge-warn" if warn else "badge-ok"
        badge     = f"<span class='{badge_cls}'>{sub}</span>" if sub else ""
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value'>{val}</div>
          {badge}
        </div>""", unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi(k1, "Voltage RMSE",  f"{v_rmse:.2f} mV",  "EKF-1")
    kpi(k2, "SOC RMSE",      f"{s_rmse:.4f} %",    "EKF-1")
    kpi(k3, "Peak tr(P1)",   f"{p_peak:.2e}",       "UQ state")
    kpi(k4, "Final tr(P1)",  f"{p_final:.2e}",      "UQ state")
    kpi(k5, "SOH (DEKF)",    f"{soh:.1f} %",        "EKF-2", warn=soh < 90)
    kpi(k6, "Max V error",   f"{v_err_max:.1f} mV", "Peak")

    # Dashboard
    st.markdown("<div class='section-hdr'>📊 Co-simulation Dashboard</div>",
                unsafe_allow_html=True)
    render_dashboard(log, chem, n_cyc, cname)

    # Cycle UQ table
    st.markdown("<div class='section-hdr'>🔁 Uncertainty Propagation per Cycle</div>",
                unsafe_allow_html=True)

    df_cyc = pd.DataFrame({
        "Cycle": [f"Cycle {i + 1}" for i in range(n_cyc)],
        "Peak tr(P1)": [f"{p:.3e}" for p in cyc_pk],
        "Δ vs Cycle 1": [
            "—" if i == 0
            else f"{(p - cyc_pk[0]) / cyc_pk[0] * 100:+.1f}%"
            for i, p in enumerate(cyc_pk)
        ],
        "Status": [
            "🟢 Baseline" if i == 0
            else ("🟡 Growing" if p > cyc_pk[0] * 1.05 else "🟢 Stable")
            for i, p in enumerate(cyc_pk)
        ],
    })
    st.dataframe(df_cyc, use_container_width=True, hide_index=True)

    trend = cyc_pk[-1] / cyc_pk[0] if n_cyc > 1 else 1.0
    if trend > 1.05:
        st.warning(
            f"⚠️ **tr(P1) grew {(trend - 1) * 100:.1f}%** over {n_cyc} cycles — "
            "the model-reality gap is widening, consistent with ongoing degradation."
        )
    else:
        st.success("✅ **tr(P1) is stable** — the DEKF is tracking well across all cycles.")

    # Sensitivity analysis
    if sens_btn or st.session_state.get("sens_done"):
        st.markdown(
            "<div class='section-hdr'>🌪️ Sensitivity Analysis — Observer Tuning</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Running observer-parameter sweeps ..."):
            sens = sensitivity_analysis(cname, log, sess_noise)
            st.session_state["sens_done"] = True

        st.plotly_chart(tornado_chart(sens), use_container_width=True)

        with st.expander("📋 Sensitivity raw data"):
            rows = []
            for lbl, d in sens.items():
                for v, p in zip(d["vals"], d["peaks"]):
                    rows.append({"Factor": lbl, "Value": v, "Peak tr(P1)": f"{p:.3e}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # CSV download
    st.markdown("---")
    df_full = pd.DataFrame({
        "Time [h]":       log["t"] / 3600.0,
        "V_true [V]":     log["V_true"],
        "V_measured [V]": log["V_meas"],
        "V_DEKF [V]":     log["V_est"],
        "SOC_true":       log["soc_true"],
        "SOC_DEKF":       log["soc_est"],
        "tr(P1)":         log["P1_tr"],
        "P1_soc":         log["P1_soc"],
        "Q_est [Ah]":     log["Q_est"],
        "R0_est [Ohm]":   log["R0_est"],
        "tr(P2)":         log["P2_tr"],
    })
    buf = io.BytesIO()
    df_full.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        "⬇️  Download full results (CSV)",
        buf,
        file_name="battsim_v4_1_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem;color:#8b949e'>
      <div style='font-size:4rem;margin-bottom:1rem'>🔋</div>
      <h3 style='color:#c9d1d9'>Ready to run</h3>
      <p>Select a chemistry and parameters in the sidebar,
         then click <b>Run co-simulation</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
Machine 1 — PyBaMM DFN          noisy V, I        Machine 2 — DEKF (2-RC)
  Doyle-Fuller-Newman       ─────────────────►   EKF-1  x = [SOC, V1, V2]
  Chen2020 / Prada2013                           EKF-2  θ = [Q_nom, R0]
  Ecker2015                                      Chain-rule Jacobian dV/dQ
  (ground truth)                                 Exact Δt via np.diff
                                                 Session-consistent sensitivity
                                                 UQ: tr(P1) / tr(P2) per cycle
    """, language="")


# ================================================================
# Footer
# ================================================================

st.markdown("""
<div class='footer-bar'>
  <span class='app-name'>🔋 BattSim v4.1</span>
  &nbsp;·&nbsp; DFN ↔ DEKF · Research Grade
  &nbsp;·&nbsp; Multi-chemistry · UQ · Sensitivity Analysis
  <br><br>
  Designed &amp; Developed by &nbsp;
  <span class='author'>Eng. Thaer Abushawar</span>
  &nbsp;·&nbsp;
  <a href='mailto:Thaer199@gmail.com'>Thaer199@gmail.com</a>
  <br>
  <span style='font-size:0.70rem;color:#484f58;margin-top:4px;display:block'>
    Plett (2004) · Chen et al. (2020) · Coman et al. (2022)
  </span>
</div>
""", unsafe_allow_html=True)
