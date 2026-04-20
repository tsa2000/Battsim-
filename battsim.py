import io
import math
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    import pybamm
except Exception:
    pybamm = None

try:
    from scipy.interpolate import interp1d
except Exception:
    interp1d = None

st.set_page_config(
    page_title="BattSim Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
    --bg: #0a1020;
    --panel: #10192f;
    --panel-2: #16213d;
    --text: #ecf3ff;
    --muted: #a6b7d8;
    --line: rgba(255,255,255,0.10);
    --accent: #35c2ff;
    --accent-2: #00e0b8;
    --warn: #ffb454;
    --danger: #ff6b7a;
    --success: #4dd08a;
}
.stApp {
    background:
      radial-gradient(circle at 10% 20%, rgba(53,194,255,0.18), transparent 26%),
      radial-gradient(circle at 90% 10%, rgba(0,224,184,0.12), transparent 28%),
      linear-gradient(180deg, #08101f 0%, #0d1528 100%);
    color: var(--text);
}
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1450px;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#0c1326 0%,#121b33 100%); border-right: 1px solid var(--line);}
.hero {
    padding: 1.35rem 1.4rem;
    border: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(18,30,58,.90), rgba(10,18,34,.94));
    border-radius: 24px;
    box-shadow: 0 18px 45px rgba(0,0,0,.25);
    margin-bottom: 1rem;
}
.hero h1 {font-size: 2rem; margin: 0 0 .35rem 0; color: var(--text);}
.hero p {font-size: 1rem; margin: 0; color: var(--muted); max-width: 1000px;}
.kpi {
    background: linear-gradient(180deg, rgba(19,31,60,.96), rgba(16,25,47,.96));
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1rem 1rem .9rem 1rem;
    min-height: 120px;
}
.kpi-label {color: var(--muted); font-size: .9rem; margin-bottom: .4rem;}
.kpi-value {color: var(--text); font-size: 1.8rem; font-weight: 700; line-height: 1.05;}
.kpi-foot {color: #87f0c8; font-size: .82rem; margin-top: .55rem;}
.card {
    background: linear-gradient(180deg, rgba(17,27,51,.95), rgba(13,22,42,.95));
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: .85rem 1rem 1rem 1rem;
    margin-top: .8rem;
}
.card h3 {margin: .1rem 0 .5rem 0; font-size: 1.05rem;}
.small-note {
    color: var(--muted); font-size: .95rem; padding: .95rem 1rem; border: 1px solid var(--line);
    background: rgba(12,19,38,.7); border-radius: 18px;
}
.assumption {
    padding: .8rem .95rem; border-radius: 16px; background: rgba(14,24,46,.72);
    border: 1px solid rgba(255,255,255,.08); color: var(--text); margin-bottom: .5rem;
}
.assumption span {color: var(--accent-2); font-weight: 700;}
hr {border-color: rgba(255,255,255,.08);}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#ecf3ff", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)"),
    )
    return fig


@dataclass
class SimData:
    t_h: np.ndarray
    current: np.ndarray
    voltage_true: np.ndarray
    voltage_meas: np.ndarray
    soc_true: np.ndarray
    soc_est: np.ndarray
    v_est: np.ndarray
    sigma_soc: np.ndarray
    sigma_v: np.ndarray
    trace_p: np.ndarray
    cycle_id: np.ndarray
    cycle_peak_trace: np.ndarray
    chemistry: str
    cycles: int
    c_rate: float
    noise_mv: float
    qn_ah: float


def synthetic_profile(cycles=8, dt=5.0, c_rate=1.0, qn_ah=5.0):
    i_amp = c_rate * qn_ah
    t_all = []
    i_all = []
    cyc = []
    t = 0.0
    for k in range(cycles):
        segments = [
            (3600 / c_rate, -i_amp),
            (900, 0.0),
            (3600 / c_rate, +i_amp),
            (900, 0.0),
        ]
        for dur, cur in segments:
            n = max(1, int(dur / dt))
            for _ in range(n):
                t_all.append(t)
                i_all.append(cur)
                cyc.append(k + 1)
                t += dt
    return np.array(t_all), np.array(i_all), np.array(cyc)


def run_pybamm_truth(chemistry: str, cycles: int, c_rate: float, dt: float = 5.0):
    if pybamm is None:
        return None
    model = pybamm.lithium_ion.DFN()
    if chemistry == "NMC":
        params = pybamm.ParameterValues("Chen2020")
    else:
        params = pybamm.ParameterValues("Chen2020")
    experiment = pybamm.Experiment(sum([
        [f"Discharge at {c_rate}C until 3.0 V", "Rest for 15 minutes", f"Charge at {c_rate}C until 4.2 V", "Rest for 15 minutes"]
        for _ in range(cycles)
    ], []), period=f"{int(dt)} seconds")
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
    sol = sim.solve()
    t_s = np.array(sol["Time [s]"].entries)
    v = np.array(sol["Terminal voltage [V]"].entries)
    try:
        i = np.array(sol["Current [A]"].entries)
    except Exception:
        i = np.zeros_like(t_s)
    try:
        soc = np.array(sol["Discharge capacity [A.h]"].entries)
        qn = float(np.max(soc)) if np.max(soc) > 0 else 5.0
        soc = 1.0 - soc / max(qn, 1e-6)
    except Exception:
        soc = np.linspace(1.0, 0.2, len(t_s))
        qn = 5.0
    return t_s, i, v, np.clip(soc, 0, 1), qn


def ocv_func(soc, chemistry):
    soc = np.clip(np.asarray(soc), 1e-5, 0.99999)
    if chemistry == "LFP":
        return 3.22 + 0.12 * np.tanh((soc - 0.12) / 0.04) + 0.13 * np.tanh((soc - 0.92) / 0.035) + 0.015 * (soc - 0.5)
    return 3.0 + 1.18 * soc - 0.12 * np.exp(-10 * soc) + 0.08 * np.exp(-12 * (1 - soc))


def docv_dsoc(soc, chemistry):
    s = np.clip(np.asarray(soc), 1e-5, 0.99999)
    if chemistry == "LFP":
        return 0.12 * (1 / 0.04) * (1 / np.cosh((s - 0.12) / 0.04) ** 2) + 0.13 * (1 / 0.035) * (1 / np.cosh((s - 0.92) / 0.035) ** 2) + 0.015
    return 1.18 + 1.2 * np.exp(-10 * s) + 0.96 * np.exp(-12 * (1 - s))


def resample_to_grid(t_s, current, voltage, soc, dt=5.0):
    t_grid = np.arange(t_s[0], t_s[-1] + dt, dt)
    if interp1d is None:
        return t_grid, np.interp(t_grid, t_s, current), np.interp(t_grid, t_s, voltage), np.interp(t_grid, t_s, soc)
    fi = interp1d(t_s, current, fill_value="extrapolate")
    fv = interp1d(t_s, voltage, fill_value="extrapolate")
    fs = interp1d(t_s, soc, fill_value="extrapolate")
    return t_grid, fi(t_grid), fv(t_grid), np.clip(fs(t_grid), 0, 1)


def ekf_2rc(t_s, current, voltage_true, soc_true, chemistry, noise_mv=8.0, qn_ah=5.0, process_scale=1.0):
    rng = np.random.default_rng(7)
    dt = np.median(np.diff(t_s))
    dt_h = dt / 3600.0
    sigma_v = noise_mv / 1000.0
    if chemistry == "LFP":
        r0, r1, c1, r2, c2 = 0.012, 0.010, 2200.0, 0.008, 8500.0
    else:
        r0, r1, c1, r2, c2 = 0.015, 0.012, 1800.0, 0.010, 6500.0

    a1 = math.exp(-dt / (r1 * c1))
    a2 = math.exp(-dt / (r2 * c2))
    b1 = r1 * (1 - a1)
    b2 = r2 * (1 - a2)

    x = np.array([0.92, 0.0, 0.0], dtype=float)
    P = np.diag([2.5e-3, 2e-4, 2e-4])
    Q = np.diag([3e-7, 4e-6, 4e-6]) * process_scale
    R = np.array([[sigma_v ** 2]])

    n = len(t_s)
    z = voltage_true + rng.normal(0, sigma_v, n)
    x_hist = np.zeros((n, 3))
    p_hist = np.zeros((n, 3, 3))
    v_est = np.zeros(n)
    sig_soc = np.zeros(n)
    sig_v = np.zeros(n)
    trp = np.zeros(n)

    for k in range(n):
        ik = current[k]
        soc_pred = np.clip(x[0] - (ik * dt_h / qn_ah), 0.0, 1.0)
        v1_pred = a1 * x[1] + b1 * ik
        v2_pred = a2 * x[2] + b2 * ik
        x_pred = np.array([soc_pred, v1_pred, v2_pred])

        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, a1, 0.0],
            [0.0, 0.0, a2],
        ])
        P_pred = A @ P @ A.T + Q

        d_ocv = float(docv_dsoc(x_pred[0], chemistry))
        h = np.array([[d_ocv, -1.0, -1.0]])
        y_pred = float(ocv_func(x_pred[0], chemistry) - v1_pred - v2_pred - ik * r0)
        S = h @ P_pred @ h.T + R
        K = (P_pred @ h.T) / S[0, 0]
        resid = z[k] - y_pred
        x = x_pred + (K[:, 0] * resid)
        x[0] = np.clip(x[0], 0.0, 1.0)
        P = (np.eye(3) - K @ h) @ P_pred
        P = 0.5 * (P + P.T)

        x_hist[k] = x
        p_hist[k] = P
        v_est[k] = y_pred + float((h @ (x - x_pred).reshape(-1, 1))[0, 0])
        trp[k] = float(np.trace(P))
        sig_soc[k] = math.sqrt(max(P[0, 0], 0))
        var_v = float((h @ P @ h.T)[0, 0] + R[0, 0])
        sig_v[k] = math.sqrt(max(var_v, 0))

    return z, x_hist[:, 0], v_est, sig_soc, sig_v, trp


def build_dataset(chemistry, cycles, c_rate, noise_mv):
    dt = 5.0
    truth = run_pybamm_truth(chemistry, cycles, c_rate, dt=dt)
    if truth is None:
        t_s, current, cycle_id = synthetic_profile(cycles=cycles, dt=dt, c_rate=c_rate, qn_ah=5.0)
        qn = 5.0
        soc = np.zeros_like(t_s, dtype=float)
        soc[0] = 0.98
        for k in range(1, len(t_s)):
            soc[k] = np.clip(soc[k-1] - current[k-1] * dt / 3600.0 / qn, 0, 1)
        v = ocv_func(soc, chemistry) - current * (0.015 if chemistry == "NMC" else 0.012)
        v -= 0.03 * np.sin(np.linspace(0, 7*np.pi, len(v)))
    else:
        t_raw, i_raw, v_raw, soc_raw, qn = truth
        t_s, current, v, soc = resample_to_grid(t_raw, i_raw, v_raw, soc_raw, dt=dt)
        cycle_id = np.floor((t_s - t_s[0]) / (2.5 * 3600 / max(c_rate, 0.25))) + 1
        cycle_id = np.clip(cycle_id.astype(int), 1, cycles)

    z, soc_est, v_est, sig_soc, sig_v, trp = ekf_2rc(t_s, current, v, soc, chemistry, noise_mv=noise_mv, qn_ah=qn)

    peaks = []
    for c in range(1, cycles + 1):
        mask = cycle_id == c
        peaks.append(float(np.max(trp[mask])) if np.any(mask) else np.nan)

    return SimData(
        t_h=t_s / 3600.0,
        current=current,
        voltage_true=v,
        voltage_meas=z,
        soc_true=soc * 100,
        soc_est=soc_est * 100,
        v_est=v_est,
        sigma_soc=2 * sig_soc * 100,
        sigma_v=2 * sig_v,
        trace_p=trp,
        cycle_id=cycle_id,
        cycle_peak_trace=np.array(peaks),
        chemistry=chemistry,
        cycles=cycles,
        c_rate=c_rate,
        noise_mv=noise_mv,
        qn_ah=qn,
    )


@st.cache_data(show_spinner=False)
def cached_build(chemistry, cycles, c_rate, noise_mv):
    return build_dataset(chemistry, cycles, c_rate, noise_mv)


def kpi_card(label, value, foot):
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-foot'>{foot}</div></div>",
        unsafe_allow_html=True,
    )


def fig_voltage(data: SimData):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.t_h, y=data.voltage_true, name="DFN truth", line=dict(color="#35c2ff", width=2.2)))
    fig.add_trace(go.Scatter(x=data.t_h, y=data.voltage_meas, name="Measured", line=dict(color="rgba(255,255,255,0.28)", width=1)))
    fig.add_trace(go.Scatter(x=data.t_h, y=data.v_est + data.sigma_v, line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=data.t_h, y=data.v_est - data.sigma_v, fill='tonexty', name="EKF ±2σ", line=dict(color="rgba(77,208,138,0.0)"), fillcolor="rgba(77,208,138,0.16)"))
    fig.add_trace(go.Scatter(x=data.t_h, y=data.v_est, name="EKF est.", line=dict(color="#4dd08a", width=2)))
    fig.update_layout(title="Terminal voltage")
    fig.update_yaxes(title="Voltage [V]")
    fig.update_xaxes(title="Time [h]")
    return style_fig(fig)


def fig_soc(data: SimData):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.t_h, y=data.soc_true, name="SOC - DFN", line=dict(color="#35c2ff", width=2.2)))
    fig.add_trace(go.Scatter(x=data.t_h, y=data.soc_est + data.sigma_soc, line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=data.t_h, y=data.soc_est - data.sigma_soc, fill='tonexty', name="SOC ±2σ", line=dict(color="rgba(0,224,184,0.0)"), fillcolor="rgba(0,224,184,0.15)"))
    fig.add_trace(go.Scatter(x=data.t_h, y=data.soc_est, name="SOC - EKF", line=dict(color="#00e0b8", width=2)))
    fig.update_layout(title="State of charge")
    fig.update_yaxes(title="SOC [%]")
    fig.update_xaxes(title="Time [h]")
    return style_fig(fig)


def fig_trace(data: SimData):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.t_h, y=data.trace_p, name="tr(P₁)", line=dict(color="#ffb454", width=2.2)))
    fig.update_layout(title="State uncertainty trace")
    fig.update_yaxes(title="tr(P₁)")
    fig.update_xaxes(title="Time [h]")
    return style_fig(fig)


def fig_cycle_uncertainty(data: SimData):
    fig = go.Figure(go.Bar(x=np.arange(1, len(data.cycle_peak_trace)+1), y=data.cycle_peak_trace, marker_color="#ffb454"))
    fig.update_layout(title="Peak uncertainty per cycle")
    fig.update_xaxes(title="Cycle")
    fig.update_yaxes(title="Peak tr(P₁)")
    return style_fig(fig, height=330)


def fig_errors(data: SimData):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("Voltage error", "SOC error"))
    fig.add_trace(go.Scatter(x=data.t_h, y=1000*(data.voltage_meas - data.v_est), name="Voltage error", line=dict(color="#ff6b7a", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.t_h, y=(data.soc_true - data.soc_est), name="SOC error", line=dict(color="#a98bff", width=1.8)), row=2, col=1)
    fig.update_yaxes(title="mV", row=1, col=1)
    fig.update_yaxes(title="%", row=2, col=1)
    fig.update_xaxes(title="Time [h]", row=2, col=1)
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", margin=dict(l=20,r=20,t=55,b=20), font=dict(color="#ecf3ff"), legend=dict(orientation="h"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def make_csv(data: SimData):
    df = pd.DataFrame({
        "time_h": data.t_h,
        "current_A": data.current,
        "voltage_true_V": data.voltage_true,
        "voltage_measured_V": data.voltage_meas,
        "voltage_est_V": data.v_est,
        "soc_true_pct": data.soc_true,
        "soc_est_pct": data.soc_est,
        "soc_2sigma_pct": data.sigma_soc,
        "voltage_2sigma_V": data.sigma_v,
        "trace_P1": data.trace_p,
        "cycle": data.cycle_id,
    })
    return df.to_csv(index=False).encode("utf-8")


def make_report(data: SimData, vrmse_mv, srmse, peak, final):
    txt = f"""
BattSim Twin — DFN to ECM uncertainty propagation

Architecture
- Physical asset: PyBaMM DFN battery model.
- Digital twin: 2RC equivalent-circuit model with one Extended Kalman Filter.
- Measurements used by the estimator: terminal voltage and input current.

Chosen assumptions
1. Observability is intentionally realistic: the estimator only uses externally measurable signals.
2. Uncertainty is represented through EKF covariance propagation, with process noise Q and measurement noise R.
3. The main study target is robust state estimation over repeated charge/discharge cycles.

Scenario
- Chemistry: {data.chemistry}
- Cycles: {data.cycles}
- C-rate: {data.c_rate:.2f}C
- Voltage noise: {data.noise_mv:.1f} mV
- Nominal capacity used by ECM observer: {data.qn_ah:.2f} Ah

Key results
- Voltage RMSE: {vrmse_mv:.3f} mV
- SOC RMSE: {srmse:.3f} %
- Peak tr(P1): {peak:.3e}
- Final tr(P1): {final:.3e}

Interpretation
- The DFN model emulates the physical battery asset.
- The 2RC ECM plus EKF reproduces the terminal voltage response while estimating SOC online.
- Uncertainty is largest at initialization, then contracts and stabilizes over subsequent cycles.
- The per-cycle peak trace reveals whether uncertainty remains bounded or grows under repeated cycling.

Why this matches the task
- It uses two coupled components: a physics-based virtual asset and a reduced-order observer.
- It injects measurement noise and filters it with Kalman estimation.
- It quantifies uncertainty propagation explicitly through the covariance trace over time and over multiple cycles.
"""
    return textwrap.dedent(txt)


with st.sidebar:
    st.markdown("## BattSim Twin")
    chemistry = st.selectbox("Chemistry", ["NMC", "LFP"], index=0)
    cycles = st.slider("Cycles", 3, 35, 8)
    c_rate = st.slider("C-rate", 0.5, 2.0, 1.0, 0.1)
    noise_mv = st.slider("Voltage noise [mV]", 1, 25, 10)
    run = st.button("Run simulation", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("### Assumptions")
    st.markdown("<div class='assumption'><span>Observability:</span> the estimator sees only current and terminal voltage.</div>", unsafe_allow_html=True)
    st.markdown("<div class='assumption'><span>Uncertainty:</span> propagation comes from EKF covariance with process and measurement noise.</div>", unsafe_allow_html=True)
    st.markdown("<div class='assumption'><span>Goal:</span> robust battery state estimation over multiple cycles, not online SOH identification.</div>", unsafe_allow_html=True)

if run or "simdata" not in st.session_state:
    with st.spinner("Running DFN/ECM twin..."):
        st.session_state.simdata = cached_build(chemistry, cycles, c_rate, noise_mv)

sim = st.session_state.simdata
vrmse_mv = float(np.sqrt(np.mean((sim.voltage_true - sim.v_est) ** 2)) * 1000)
srmse = float(np.sqrt(np.mean((sim.soc_true - sim.soc_est) ** 2)))
peak = float(np.max(sim.trace_p))
final = float(sim.trace_p[-1])

st.markdown(
    """
    <div class='hero'>
      <h1>BattSim Twin — DFN to ECM Uncertainty Propagation</h1>
      <p>A physics-based battery model emulates the physical asset, while a reduced 2RC equivalent-circuit observer with one Extended Kalman Filter tracks voltage and SOC under noisy measurements across repeated charge–discharge cycles.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Voltage RMSE", f"{vrmse_mv:.2f} mV", "Estimator tracks terminal voltage")
with c2:
    kpi_card("SOC RMSE", f"{srmse:.2f} %", "Online SOC estimation error")
with c3:
    kpi_card("Peak tr(P₁)", f"{peak:.3e}", "Highest state uncertainty")
with c4:
    kpi_card("Final tr(P₁)", f"{final:.3e}", "Residual uncertainty after cycling")

left, right = st.columns(2)
with left:
    st.plotly_chart(fig_voltage(sim), use_container_width=True)
with right:
    st.plotly_chart(fig_soc(sim), use_container_width=True)

left, right = st.columns([1.2, 1])
with left:
    st.plotly_chart(fig_trace(sim), use_container_width=True)
with right:
    st.plotly_chart(fig_cycle_uncertainty(sim), use_container_width=True)

st.plotly_chart(fig_errors(sim), use_container_width=True)

st.markdown("<div class='card'><h3>Interpretation</h3><div class='small-note'>This version is intentionally aligned with the task: DFN acts as the virtual physical asset, the ECM plus EKF acts as the digital twin, and uncertainty propagation is quantified through the EKF covariance trace over time and by cycle. The interface avoids SOH and parameter-identification complexity to keep the physics story clear, defensible, and presentation-ready.</div></div>", unsafe_allow_html=True)

csv_bytes = make_csv(sim)
report_txt = make_report(sim, vrmse_mv, srmse, peak, final)

c1, c2 = st.columns(2)
with c1:
    st.download_button("Download simulation CSV", data=csv_bytes, file_name="battsim_results.csv", mime="text/csv", use_container_width=True)
with c2:
    st.download_button("Download report summary", data=report_txt.encode("utf-8"), file_name="battsim_report.txt", mime="text/plain", use_container_width=True)
