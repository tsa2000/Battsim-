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
import base64
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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


def degraded_capacity(Q_init, n_cyc, c_rate, alpha=0.002, beta=1.3):
    """Physical degradation — C-Rate stress + cycle count only."""
    fade = alpha * (c_rate ** beta) * n_cyc
    return Q_init * max(1.0 - fade, 0.65)


# ================================================================
# Machine 1 — PyBaMM DFN
# ================================================================

@st.cache_resource
def load_pybamm_model(pset_name):
    import pybamm
    model  = pybamm.lithium_ion.DFN()
    params = pybamm.ParameterValues(pset_name)
    return model, params


def run_dfn(pset_name, n_cycles, c_rate, prog, status):
    import pybamm

    status.markdown(f"**[Machine 1 — DFN]** Loading parameter set `{pset_name}`...")
    prog.progress(5)

    model, params = load_pybamm_model(pset_name)

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
        self.Q_nom_degraded = Q_nom

        self.x1  = np.array([[soc0], [0.0], [0.0]])
        self.P1  = np.diag([1e-3, 1e-4, 1e-4])
        self.Q1  = np.diag([1e-8, 1e-6, 1e-6])
        self.R1m = np.array([[noise_var]])

        self.x2  = np.array([[Q_nom], [R0]])
        self.P2  = np.diag([0.01, 1e-5])

        dOCV_init = abs(float(docv_dsoc(self.ocv, np.clip(soc0, 1e-4, 1 - 1e-4))))
        q2_scale  = max(dOCV_init, 1e-3)
        self.Q2   = np.diag([1e-13 * q2_scale, 1e-6])

        self.R2m = np.array([[noise_var * 4.0]])

        self.Q_min = 0.65 * Q_nom
        self.Q_max = Q_nom

        self.I2 = np.eye(2)
        self.I3 = np.eye(3)

    @property
    def Q_est(self):
        return float(np.clip(self.x2[0, 0], self.Q_min, self.Q_max))

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
        if abs(S2[0, 0]) > 1e-15 and abs(current) > 0.01:  # ← أضف هذا الشرط فقط
            K2 = P2p @ C2m.T / S2[0, 0]
            self.x2       = self.x2 + K2 * nu_2
            self.x2[0, 0] = np.clip(float(self.x2[0, 0]), self.Q_min, self.Q_max)
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
    Q_degraded = degraded_capacity(Q_nom, n_cycles, c_rate)

    status.markdown("**[Machine 2 — DEKF]** Initialising Dual EKF (2-RC + online param ID) ...")
    prog.progress(47)

    if "NCA" in chem_name and n_cycles > 5:
        status.warning("⚠️ NCA + cycles > 5: flat OCV curve reduces EKF-2 observability — SOH estimate may drift.")

    ekf = DEKF(
        float(soc_true[0]), Q_degraded,
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
    innovations = []
    trP_history = []

    for k in range(N):
        vm = V_true[k] + np.random.normal(0.0, noise_std)
        v_est, soc_e, p1tr, p1s, q_e, r0_e, p2tr = ekf.step(vm, I_true[k])
        innovations.append(vm - v_est)
        trP_history.append(p1tr)
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
    return log, Q_nom, Q_degraded, chem, innovations, trP_history


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
# PDF Export
# ================================================================

def generate_pdf(log, chem, chem_name, n_cyc, Q_nom, c_rate,
                 v_rmse, s_rmse, p_peak, p_final, soh,
                 cyc_pk, innovations, trP_history, noise_std):
    import plotly.io as pio
    from io import BytesIO
    from datetime import datetime

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm)

    styles = getSampleStyleSheet()
    S = {
        "title": ParagraphStyle('t', fontSize=18, textColor=colors.HexColor('#00b4d8'),
            fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=4),
        "sub":   ParagraphStyle('s', fontSize=9,  textColor=colors.HexColor('#718096'),
            alignment=TA_CENTER, spaceAfter=2),
        "h2":    ParagraphStyle('h', fontSize=12, textColor=colors.HexColor('#1a202c'),
            fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6),
        "ok":    ParagraphStyle('ok',  fontSize=8.5, textColor=colors.HexColor('#276749'), leading=13),
        "warn":  ParagraphStyle('wn',  fontSize=8.5, textColor=colors.HexColor('#c05621'), leading=13),
        "err":   ParagraphStyle('er',  fontSize=8.5, textColor=colors.HexColor('#9b2c2c'), leading=13),
        "foot":  ParagraphStyle('f',   fontSize=7.5, textColor=colors.HexColor('#a0aec0'),
            alignment=TA_CENTER),
    }

    story = []

    story.append(Paragraph("🔋 BattSim v4.1 — Simulation Report", S["title"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M')}  |  "
        f"Chemistry: {chem_name.split('—')[0].strip()}  |  "
        f"Cycles: {n_cyc}  |  C-Rate: {c_rate}C  |  "
        f"Noise: {noise_std*1000:.0f} mV",
        S["sub"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor('#00b4d8'), spaceAfter=10))

    story.append(Paragraph("Key Performance Indicators", S["h2"]))
    kpi_data = [
        ["Metric", "Value", "Filter"],
        ["Voltage RMSE",  f"{v_rmse:.3f} mV",  "EKF-1"],
        ["SOC RMSE",      f"{s_rmse:.4f} %",   "EKF-1"],
        ["Peak tr(P1)",   f"{p_peak:.3e}",     "UQ State"],
        ["Final tr(P1)",  f"{p_final:.3e}",    "UQ State"],
        ["SOH (DEKF)",    f"{soh:.3f} %",      "EKF-2"],
        ["Max V Error",   f"{np.abs(log['V_true']-log['V_est']).max()*1000:.2f} mV", "Peak"],
    ]
    t = Table(kpi_data, colWidths=[6*cm, 4.5*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#00b4d8')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1),
            [colors.HexColor('#f7fafc'), colors.HexColor('#edf2f7')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#cbd5e0')),
        ('ALIGN',         (1,1), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Simulation Diagnostics", S["h2"]))
    diag = diagnose_simulation(innovations, trP_history, soh / 100.0, noise_std, s_rmse)
    for icon, title, msg in diag:
        st_s = S["ok"] if icon == "✅" else (S["warn"] if icon == "🟡" else S["err"])
        story.append(Paragraph(f"<b>{icon} {title}</b> — {msg}", st_s))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Uncertainty Propagation per Cycle", S["h2"]))
    cyc_data = [["Cycle", "Peak tr(P1)", "Δ vs Cycle 1", "Status"]]
    for i, p in enumerate(cyc_pk):
        delta  = "—" if i == 0 else f"{(p-cyc_pk[0])/cyc_pk[0]*100:+.1f}%"
        status = "Baseline" if i == 0 else ("Growing" if p > cyc_pk[0]*1.05 else "Stable")
        cyc_data.append([f"Cycle {i+1}", f"{p:.3e}", delta, status])
    ct = Table(cyc_data, colWidths=[3.5*cm, 4.5*cm, 4*cm, 3.5*cm])
    ct.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#2d3748')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8.5),
        ('ROWBACKGROUNDS',(0,1), (-1,-1),
            [colors.HexColor('#f7fafc'), colors.HexColor('#edf2f7')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#cbd5e0')),
        ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(ct)
    story.append(Spacer(1, 10))

    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=colors.HexColor('#e2e8f0'), spaceAfter=8))
    story.append(Paragraph("Co-Simulation Plots", S["h2"]))

    t_h  = log["t"] / 3600.0
    cc   = chem["color"]
    pals = ["#00b4d8","#f77f00","#2dc653","#ef233c","#c77dff"]
    N    = len(log["t"])
    cl   = max(1, N // n_cyc)
    ve   = np.abs(log["V_true"] - log["V_est"]) * 1000.0
    se   = np.abs(log["soc_true"] - log["soc_est"]) * 100.0

    CL = dict(paper_bgcolor='white', plot_bgcolor='#f8fafc',
              font=dict(color='#2d3748', family='Arial', size=10),
              margin=dict(t=40, b=36, l=58, r=16),
              width=760, height=300,
              legend=dict(bgcolor='white', bordercolor='#e2e8f0',
                          borderwidth=1, font=dict(size=9)))

    def ax(title):
        return dict(title=title, gridcolor='#e2e8f0',
                    showgrid=True, zeroline=False, linecolor='#cbd5e0')

    def to_img(fig):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_mpl, axm = plt.subplots(figsize=(10, 3.5))
        fig_mpl.patch.set_facecolor('#f8fafc')
        axm.set_facecolor('#f8fafc')

        for trace in fig.data:
            x = list(trace.x) if trace.x is not None else []
            y = list(trace.y) if trace.y is not None else []
            if not x or not y:
                continue
            color  = trace.line.color if hasattr(trace, 'line') and trace.line and trace.line.color else "#00b4d8"
            lw     = trace.line.width if hasattr(trace, 'line') and trace.line and trace.line.width else 1.5
            ls     = "--" if hasattr(trace, 'line') and trace.line and trace.line.dash == "dash" else "-"
            alpha  = getattr(trace, 'opacity', 1.0) or 1.0
            label  = trace.name if hasattr(trace, 'name') and trace.showlegend is not False else None

            if trace.mode == "lines" or trace.mode is None:
                if trace.fill == "toself":
                    continue
                axm.plot(x, y, color=color, linewidth=lw, linestyle=ls,
                         alpha=alpha, label=label)
            if hasattr(trace, 'fill') and trace.fill == "tozeroy":
                axm.fill_between(x, y, alpha=0.15, color=color)

        title_text = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
        axm.set_title(title_text, fontsize=10, color="#1a202c", pad=6)

        xaxis = fig.layout.xaxis
        yaxis = fig.layout.yaxis
        if xaxis and xaxis.title and xaxis.title.text:
            axm.set_xlabel(xaxis.title.text, fontsize=8, color="#4a5568")
        if yaxis and yaxis.title and yaxis.title.text:
            axm.set_ylabel(yaxis.title.text, fontsize=8, color="#4a5568")

        axm.tick_params(colors="#4a5568", labelsize=7)
        axm.grid(True, color="#e2e8f0", linewidth=0.5)
        for spine in axm.spines.values():
            spine.set_edgecolor("#e2e8f0")

        handles, labels = axm.get_legend_handles_labels()
        if handles:
            axm.legend(handles, labels, fontsize=7.5, framealpha=0.9,
                       edgecolor="#cbd5e0", facecolor="white",
                       loc="upper right", ncol=min(len(handles), 3))

        plt.tight_layout(pad=0.5)
        img_buf = BytesIO()
        fig_mpl.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig_mpl)
        img_buf.seek(0)
        return Image(img_buf, width=17.5*cm, height=7*cm)

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=t_h, y=log["V_meas"], mode="lines",
        name="Measured", line=dict(color="#a0aec0", width=0.6), opacity=0.5))
    f1.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
        name="DFN truth", line=dict(color=cc, width=2)))
    f1.add_trace(go.Scatter(x=t_h, y=log["V_est"], mode="lines",
        name="DEKF est.", line=dict(color="#f77f00", width=1.8, dash="dash")))
    f1.update_layout(title="① Terminal Voltage", xaxis=ax("Time [h]"), yaxis=ax("V [V]"), **CL)
    story.append(to_img(f1)); story.append(Spacer(1, 6))

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=t_h, y=log["soc_true"]*100, mode="lines",
        name="SOC — DFN", line=dict(color=cc, width=2)))
    f2.add_trace(go.Scatter(x=t_h, y=log["soc_est"]*100, mode="lines",
        name="SOC — DEKF", line=dict(color="#f77f00", width=1.8, dash="dash")))
    f2.update_layout(title="② State of Charge", xaxis=ax("Time [h]"), yaxis=ax("SOC [%]"), **CL)
    story.append(to_img(f2)); story.append(Spacer(1, 6))

    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=t_h, y=log["P1_tr"], mode="lines",
        line=dict(color="#2dc653", width=2),
        fill="tozeroy", fillcolor="rgba(45,198,83,0.12)", showlegend=False))
    f3.update_layout(title="③ State Covariance tr(P1)", xaxis=ax("Time [h]"), yaxis=ax("tr(P1)"), **CL)
    story.append(to_img(f3)); story.append(Spacer(1, 6))

    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=t_h, y=log["Q_est"], mode="lines",
        name="Q_nom [Ah]", line=dict(color="#c77dff", width=2)))
    f4.add_trace(go.Scatter(x=t_h, y=log["R0_est"]*1000, mode="lines",
        name="R0 [mΩ]", line=dict(color="#ffd60a", width=2)))
    f4.update_layout(title="④ Online Parameter ID", xaxis=ax("Time [h]"), yaxis=ax("Value"), **CL)
    story.append(to_img(f4)); story.append(Spacer(1, 6))

    f5 = go.Figure()
    f5.add_trace(go.Scatter(x=t_h, y=ve, mode="lines",
        line=dict(color="#f77f00", width=1.5),
        fill="tozeroy", fillcolor="rgba(247,127,0,0.15)", showlegend=False))
    f5.update_layout(title="⑤ Voltage Error", xaxis=ax("Time [h]"), yaxis=ax("|Error| [mV]"), **CL)
    story.append(to_img(f5)); story.append(Spacer(1, 6))

    f6 = go.Figure()
    f6.add_trace(go.Scatter(x=t_h, y=se, mode="lines",
        line=dict(color="#ef233c", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,35,60,0.15)", showlegend=False))
    f6.update_layout(title="⑥ SOC Error", xaxis=ax("Time [h]"), yaxis=ax("|Error| [%]"), **CL)
    story.append(to_img(f6)); story.append(Spacer(1, 6))

    f7 = go.Figure()
    for c in range(n_cyc):
        s, e = c*cl, min((c+1)*cl, N)
        tc   = (log["t"][s:e] - log["t"][s]) / 3600.0
        f7.add_trace(go.Scatter(x=tc, y=log["P1_tr"][s:e], mode="lines",
            name=f"Cycle {c+1}", line=dict(color=pals[c % len(pals)], width=2)))
    f7.update_layout(title="⑦ Uncertainty per Cycle",
        xaxis=ax("Time in cycle [h]"), yaxis=ax("tr(P1)"), **CL)
    story.append(to_img(f7)); story.append(Spacer(1, 6))

    f8 = go.Figure()
    f8.add_trace(go.Scatter(x=t_h, y=log["P2_tr"], mode="lines",
        line=dict(color="#00b4d8", width=2),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.12)", showlegend=False))
    f8.update_layout(title="⑧ Parameter Covariance tr(P2)",
        xaxis=ax("Time [h]"), yaxis=ax("tr(P2)"), **CL)
    story.append(to_img(f8))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=colors.HexColor('#e2e8f0'), spaceAfter=6))
    story.append(Paragraph(
        "BattSim v4.1 · Designed &amp; Developed by Eng. Thaer Abushawar · "
        "Plett (2004) · Chen et al. (2020) · Coman et al. (2022)",
        S["foot"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


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
    legend=dict(
        bgcolor="rgba(22,27,34,0.95)",
        bordercolor="#00b4d8",
        borderwidth=1,
        font=dict(size=11, color="#e2e8f0"),
        itemsizing="constant",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(t=60, b=36, l=58, r=24),
    height=400,
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

def compute_stats(log, n_cyc, Q_nom, Q_degraded):
    N  = len(log["t"])
    cl = N // n_cyc

    v_rmse  = np.sqrt(np.mean((log["V_true"] - log["V_est"]) ** 2)) * 1000.0
    s_rmse  = np.sqrt(np.mean((log["soc_true"] - log["soc_est"]) ** 2)) * 100.0
    p_peak  = log["P1_tr"].max()
    p_final = log["P1_tr"][-1]
    soh = log["Q_est"][-1] / Q_nom * 100.0

    cyc_pk = [
        log["P1_tr"][c * cl : min((c + 1) * cl, N)].max()
        for c in range(n_cyc)
    ]
    return v_rmse, s_rmse, p_peak, p_final, soh, cyc_pk


def diagnose_simulation(innovations, trP_history, soh, noise_std, s_rmse):
    diagnostics = []
    innov_rms   = np.sqrt(np.mean(np.array(innovations)**2))
    innov_ratio = innov_rms / noise_std

    if innov_ratio > 3.0:
        diagnostics.append(("🔴", "EKF-1 Diverging",
            f"Innovation/Noise = {innov_ratio:.1f}× (limit: 3×). Noise overwhelms EKF-1 "
            f"or DFN dynamics at this C-Rate exceed the filter design envelope. "
            f"In a real BMS this triggers a sensor fault alarm."))
    elif innov_ratio > 1.5:
        diagnostics.append(("🟡", "EKF-1 Under Stress",
            f"Innovation/Noise = {innov_ratio:.1f}× (ideal < 1.5×). "
            f"Filter tracks but Q or R matrix may need retuning."))
    else:
        diagnostics.append(("✅", "EKF-1 Healthy",
            f"Innovation/Noise = {innov_ratio:.2f}× — consistent with noise model."))

    trP_ratio = trP_history[-1] / trP_history[0]
    if trP_ratio > 0.5:
        diagnostics.append(("🔴", "EKF-1 Did Not Converge",
            f"Final tr(P1) = {trP_ratio:.1%} of initial. Filter never gained confidence — "
            f"noise too large or too few cycles to observe sufficient SOC dynamics."))
    elif trP_ratio > 0.1:
        diagnostics.append(("🟡", "Partial Convergence",
            f"Final tr(P1) = {trP_ratio:.1%} of initial. "
            f"More cycles or better P0 initialisation would help."))
    else:
        diagnostics.append(("✅", "EKF-1 Converged",
            f"tr(P1) dropped to {trP_ratio:.1%} of initial — high confidence in state estimates."))

    if s_rmse > 10.0:
        diagnostics.append(("🔴", "High SOC Error",
            f"SOC RMSE = {s_rmse:.2f}% > 10%. Poor OCV observability "
            f"(flat curve — NCA/LFP at mid-SOC) or Coulomb counting drift."))
    elif s_rmse > 5.0:
        diagnostics.append(("🟡", "Moderate SOC Error",
            f"SOC RMSE = {s_rmse:.2f}%. Acceptable for short cycles."))
    else:
        diagnostics.append(("✅", "SOC Accuracy Good",
            f"SOC RMSE = {s_rmse:.2f}% — within BMS-grade tolerance."))

    if soh > 1.0:
        diagnostics.append(("🔴", "SOH Physically Impossible",
            f"SOH = {soh*100:.1f}% exceeds 100%. EKF-2 diverged — high noise caused it to "
            f"interpret voltage drops as capacity gains. A real battery cannot gain capacity."))
    elif soh < 0.65:
        diagnostics.append(("🔴", "SOH Unrealistically Low",
            f"SOH = {soh*100:.1f}% — EKF-2 diverged in the opposite direction."))
    elif soh > 0.98:
        diagnostics.append(("🟡", "SOH Suspiciously High",
            f"SOH = {soh*100:.1f}%. DFN-based runs here do not include explicit cycle-aging "
            f"degradation, so Q_nom stays nearly constant and this reflects model limitation, not a real aged cell."))
    else:
        diagnostics.append(("✅", "SOH Physically Valid",
            f"SOH = {soh*100:.1f}% — within realistic range [65%–98%]."))

    return diagnostics


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
n_cycles  = st.sidebar.slider("🔁 Cycles", 1, 35, 3)
c_rate    = st.sidebar.select_slider("⚡ C-Rate", [0.5, 1.0, 1.5, 2.0, 3.0], value=1.0)
noise_db  = st.sidebar.select_slider(
    "📡 Sensor noise",
    ["Low (5 mV)", "Medium (10 mV)", "High (30 mV)"],
    value="Medium (10 mV)",
)
noise_map = {"Low (5 mV)": 0.005, "Medium (10 mV)": 0.010, "High (30 mV)": 0.030}
noise_std = noise_map[noise_db]

if "NCA" in chem_name and n_cycles > 5:
    st.sidebar.warning("⚠️ NCA > 5 cycles: flat OCV curve can weaken EKF-2 capacity observability.")

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
        log, Q_nom, Q_degraded, chem, innovations, trP_history = run_cosim(
            chem_name, n_cycles, c_rate, noise_std, pbar, stat
        )
        pbar.progress(100)
        stat.success("✅ Co-simulation complete.")
        st.session_state.update({
            "log":          log,
            "Q_nom":        Q_nom,
            "Q_degraded":   Q_degraded,
            "chem":         chem,
            "chem_name":    chem_name,
            "n_cyc":        n_cycles,
            "noise_std":    noise_std,
            "c_rate":       c_rate,
            "innovations":  innovations,
            "trP_history":  trP_history,
        })
    except Exception as e:
        stat.error(f"Simulation failed: {e}")

if sens_btn:
    if "log" not in st.session_state:
        st.warning("Run a co-simulation first.")
    else:
        sens = sensitivity_analysis(
            st.session_state["chem_name"],
            st.session_state["log"],
            session_noise=st.session_state["noise_std"],
        )
        st.session_state["sens"] = sens

if "log" in st.session_state:
    log         = st.session_state["log"]
    Q_nom       = st.session_state["Q_nom"]
    Q_degraded  = st.session_state["Q_degraded"]
    chem        = st.session_state["chem"]
    chem_name   = st.session_state["chem_name"]
    n_cyc       = st.session_state["n_cyc"]
    noise_std_s = st.session_state["noise_std"]
    c_rate_s    = st.session_state["c_rate"]
    innovations = st.session_state["innovations"]
    trP_history = st.session_state["trP_history"]

    v_rmse, s_rmse, p_peak, p_final, soh, cyc_pk = compute_stats(log, n_cyc, Q_nom, Q_degraded)

    st.markdown("### 📈 Key Performance Indicators")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("VOLTAGE RMSE", f"{v_rmse:.2f} mV", help="EKF-1")
        st.metric("PEAK TR(P1)", f"{p_peak:.2e}", help="UQ state")
    with c2:
        st.metric("SOC RMSE", f"{s_rmse:.4f} %", help="EKF-1")
        st.metric("FINAL TR(P1)", f"{p_final:.2e}", help="UQ state")
    with c3:
        st.metric("SOH (DEKF)", f"{soh:.3f} %", help="EKF-2")
        st.metric("MAX V ERROR", f"{np.abs(log['V_true']-log['V_est']).max()*1000:.1f} mV", help="Peak")

    st.markdown("### 🔬 Simulation Diagnostics")
    for icon, title, msg in diagnose_simulation(innovations, trP_history, soh / 100.0, noise_std_s, s_rmse):
        if icon == "✅":
            st.success(f"{icon} {title}\n\n{msg}")
        elif icon == "🟡":
            st.warning(f"{icon} {title}\n\n{msg}")
        else:
            st.error(f"{icon} {title}\n\n{msg}")

    render_dashboard(log, chem, n_cyc, chem_name)

    if "sens" in st.session_state:
        st.markdown("### 📊 Sensitivity Analysis")
        st.plotly_chart(tornado_chart(st.session_state["sens"]), use_container_width=True)

    st.markdown("### 📄 Export")
    pdf_bytes = generate_pdf(
        log, chem, chem_name, n_cyc, Q_nom, c_rate_s,
        v_rmse, s_rmse, p_peak, p_final, soh,
        cyc_pk, innovations, trP_history, noise_std_s
    )
    st.download_button(
        "⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="battsim_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

st.markdown("""
<div class="footer-bar">
  <span class="app-name">BattSim v4.1</span> ·
  <span class="author">Eng. Thaer Abushawar</span> ·
  DFN ↔ DEKF · Research-grade battery co-simulation
</div>
""", unsafe_allow_html=True)
