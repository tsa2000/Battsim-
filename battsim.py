import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# NMC OCV CURVE (Polynomial fit for NMC622, validated from Chen2020)
# OCV = f(SOC), SOC ∈ [0, 1]
# ═══════════════════════════════════════════════════════════════════════════════
def ocv_nmc(soc):
    """
    Polynomial OCV model for NMC cathode (Graphite | NMC622).
    Fitted from Chen2020 dataset. Valid for SOC ∈ [0, 1].
    """
    z = np.clip(soc, 0.0, 1.0)
    return (
        3.4043
        + 1.6227 * z
        - 8.6635 * z**2
        + 21.3955 * z**3
        - 25.7324 * z**4
        + 12.0032 * z**5
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL ASSET: DFN with NMC (Chen2020) — Multi-Output
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def run_asset(cycles, c_rate, noise_v, noise_t):
    model  = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues("Chen2020")   # NMC622 | Graphite

    exp = pybamm.Experiment(
        [f"Discharge at {c_rate}C until 2.5 V",
         "Rest for 5 minutes",
         "Charge at 1C until 4.2 V",
         "Rest for 5 minutes"] * cycles
    )
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sol = sim.solve()

    t      = sol["Time [s]"].entries
    V_true = sol["Terminal voltage [V]"].entries
    T_true = sol["Cell temperature [K]"].entries
    I      = sol["Current [A]"].entries
    Q_dis  = sol["Discharge capacity [A.h]"].entries

    rng    = np.random.default_rng(42)
    V_meas = V_true + rng.normal(0, noise_v, len(t))
    T_meas = T_true + rng.normal(0, noise_t, len(t))

    Q_nom  = float(params["Nominal cell capacity [A.h]"])
    return t, V_meas, T_meas, V_true, T_true, I, Q_dis, Q_nom

# ═══════════════════════════════════════════════════════════════════════════════
# DIGITAL TWIN: MIMO-AEKF — 2RC Thermal ECM for NMC
#
# State vector:  x = [SOC, V_RC1, V_RC2, T_core]  (4×1)
# Inputs:        u = [I, T_amb]
# Measurements:  y = [V_terminal, T_cell]          (2×1)
#
# Discrete state equations (Euler, dt):
#   SOC[k+1]   = SOC[k]  − (I·dt) / (Q_nom·3600)
#   V_RC1[k+1] = exp(−dt/(R1·C1))·V_RC1[k] + R1·(1−exp(−dt/(R1·C1)))·I[k]
#   V_RC2[k+1] = exp(−dt/(R2·C2))·V_RC2[k] + R2·(1−exp(−dt/(R2·C2)))·I[k]
#   T_core[k+1] = T_core[k] + (dt/C_th)·(I²·R0 − (T_core−T_amb)/R_th)
#
# Measurement equation:
#   V_term = OCV(SOC) − V_RC1 − V_RC2 − I·R0
#   T_cell = T_core
# ═══════════════════════════════════════════════════════════════════════════════
class MIMOAdaptiveTwin:
    def __init__(self, Q_nom, R0, R1, C1, R2, C2, R_th, C_th, T_amb,
                 P0_diag, Q_diag, R_diag):
        self.Q_nom = Q_nom
        # ECM Parameters (NMC defaults, user-overridable)
        self.R0   = R0
        self.R1   = R1;  self.C1 = C1
        self.R2   = R2;  self.C2 = C2
        # Thermal Parameters
        self.R_th = R_th
        self.C_th = C_th
        self.T_amb = T_amb

        # State: [SOC, V_RC1, V_RC2, T_core]
        self.x = np.array([1.0, 0.0, 0.0, T_amb])

        # Covariance matrices
        self.P  = np.diag(P0_diag)
        self.Qn = np.diag(Q_diag)
        self.Rn = np.diag(R_diag)

        # SOH tracker
        self.capacity_fade = 0.0

    def step(self, V_m, T_m, I, dt):
        SOC, V1, V2, T_core = self.x

        # ── Adaptive Parameters (SOC + Temperature) ──────────────────────────
        # Arrhenius correction for resistance (activation energy ~30 kJ/mol for NMC)
        T_ref  = 298.15  # K
        arr    = np.exp(3600.0 * (1/T_core - 1/T_ref))   # Ea/R ≈ 3600 K
        R0_eff = self.R0 * (1 + 0.4*(1 - SOC)**2) * arr
        R1_eff = self.R1 * arr
        R2_eff = self.R2 * arr

        # ── State Transition (Exact ZOH for RC branches) ─────────────────────
        tau1 = R1_eff * self.C1
        tau2 = R2_eff * self.C2
        a1   = np.exp(-dt / tau1)
        a2   = np.exp(-dt / tau2)

        x_pred = np.array([
            SOC - (I * dt) / (self.Q_nom * 3600),          # Coulomb counting
            a1 * V1 + R1_eff * (1 - a1) * I,               # RC1 exact ZOH
            a2 * V2 + R2_eff * (1 - a2) * I,               # RC2 exact ZOH
            T_core + (dt / self.C_th) * (                   # Newton cooling
                I**2 * R0_eff - (T_core - self.T_amb) / self.R_th
            )
        ])

        # ── Jacobian F (4×4) ─────────────────────────────────────────────────
        F = np.array([
            [1.0,  0.0,  0.0,  0.0],
            [0.0,   a1,  0.0,  0.0],
            [0.0,  0.0,   a2,  0.0],
            [0.0,  0.0,  0.0,  1.0 - dt/(self.C_th * self.R_th)]
        ])

        P_pred = F @ self.P @ F.T + self.Qn

        # ── Measurement Equation ─────────────────────────────────────────────
        SOC_p, V1_p, V2_p, T_p = x_pred
        OCV_val = ocv_nmc(SOC_p)
        V_est   = OCV_val - V1_p - V2_p - I * R0_eff
        T_est   = T_p

        # Jacobian H (2×4): ∂y/∂x
        dOCV_dSOC = (ocv_nmc(SOC_p + 1e-5) - ocv_nmc(SOC_p - 1e-5)) / 2e-5
        H = np.array([
            [dOCV_dSOC, -1.0, -1.0, 0.0],  # Voltage row
            [0.0,        0.0,  0.0, 1.0],  # Temperature row
        ])

        # Innovation (2×1)
        nu = np.array([V_m - V_est, T_m - T_est])

        # Kalman Gain (4×2)
        S = H @ P_pred @ H.T + self.Rn
        K = P_pred @ H.T @ np.linalg.inv(S)

        # ── Update ───────────────────────────────────────────────────────────
        self.x = x_pred + K @ nu
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)    # SOC ∈ [0,1]
        self.x[3] = np.clip(self.x[3], 250.0, 350.0) # T ∈ physical range
        self.P = (np.eye(4) - K @ H) @ P_pred

        # NIS (Normalized Innovation Squared) — scalar
        nis = float(nu @ np.linalg.inv(S) @ nu)

        return (
            self.x[0],           # SOC
            self.x[3],           # T_core
            np.sqrt(self.P[0,0]),# σ_SOC
            np.sqrt(self.P[3,3]),# σ_T
            nis
        )

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="NMC MIMO Digital Twin", layout="wide")
st.title("🔋 NMC622 MIMO Digital Twin — Research Grade")

# ── Sidebar: Manual Settings ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Settings")

    with st.expander("📡 Asset (DFN) Settings", expanded=True):
        cycles  = st.number_input("Cycles",       1, 50,   5)
        c_rate  = st.slider("C-Rate",             0.5, 3.0, 1.0, 0.1)
        noise_v = st.slider("Voltage Noise (V)",  0.000, 0.050, 0.005, 0.001)
        noise_t = st.slider("Temp Noise (K)",     0.0,   2.0,   0.1,  0.05)

    with st.expander("🔧 ECM Parameters (NMC)", expanded=True):
        R0 = st.number_input("R0 — Ohmic Resistance (Ω)",     0.001, 0.200, 0.015, 0.001, format="%.3f")
        R1 = st.number_input("R1 — RC1 Resistance (Ω)",       0.001, 0.100, 0.010, 0.001, format="%.3f")
        C1 = st.number_input("C1 — RC1 Capacitance (F)",      100.0, 5000.0, 3000.0, 100.0)
        R2 = st.number_input("R2 — RC2 Resistance (Ω)",       0.001, 0.100, 0.005, 0.001, format="%.3f")
        C2 = st.number_input("C2 — RC2 Capacitance (F)",      100.0, 5000.0, 2000.0, 100.0)

    with st.expander("🌡️ Thermal Parameters", expanded=True):
        R_th  = st.number_input("R_th — Thermal Resistance (K/W)", 1.0, 20.0, 3.0,  0.5)
        C_th  = st.number_input("C_th — Thermal Capacity (J/K)",   100.0, 2000.0, 500.0, 50.0)
        T_amb = st.number_input("T_amb — Ambient Temp (K)",        273.15, 323.15, 298.15, 1.0)

    with st.expander("📊 EKF Noise Tuning", expanded=False):
        st.markdown("**Process Noise Q** (uncertainty in model)")
        q_soc = st.number_input("Q — SOC",    1e-9, 1e-4, 1e-7, format="%.2e")
        q_v1  = st.number_input("Q — V_RC1",  1e-9, 1e-4, 1e-8, format="%.2e")
        q_v2  = st.number_input("Q — V_RC2",  1e-9, 1e-4, 1e-8, format="%.2e")
        q_t   = st.number_input("Q — T_core", 1e-5, 1e-1, 1e-3, format="%.2e")
        st.markdown("**Measurement Noise R** (sensor trust)")
        r_v   = st.number_input("R — Voltage", 1e-5, 1e-1, 1e-3, format="%.2e")
        r_t   = st.number_input("R — Temp",    1e-3, 1.0,  0.1,  format="%.3f")
        st.markdown("**Initial Covariance P0**")
        p_soc = st.number_input("P0 — SOC",    1e-4, 1.0,  0.01, format="%.4f")
        p_t   = st.number_input("P0 — T_core", 0.1,  5.0,  0.5,  format="%.2f")

    run_btn = st.button("▶ Execute Co-Simulation", type="primary")

# ── Main: Run & Display ───────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running NMC DFN Asset (Physics)..."):
        t, V_m, T_m, V_true, T_true, I, Q_dis, Q_nom = run_asset(
            cycles, c_rate, noise_v, noise_t
        )

    with st.spinner("Running MIMO-AEKF Twin..."):
        twin = MIMOAdaptiveTwin(
            Q_nom  = Q_nom,
            R0=R0, R1=R1, C1=C1, R2=R2, C2=C2,
            R_th=R_th, C_th=C_th, T_amb=T_amb,
            P0_diag = [p_soc, 1e-4, 1e-4, p_t],
            Q_diag  = [q_soc, q_v1, q_v2, q_t],
            R_diag  = [r_v, r_t]
        )
        dt      = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
        results = [twin.step(V_m[k], T_m[k], I[k], dt) for k in range(len(t))]

    soc_est  = np.array([r[0] for r in results])
    temp_est = np.array([r[1] for r in results])
    sig_soc  = np.array([r[2] for r in results])
    sig_t    = np.array([r[3] for r in results])
    nis_arr  = np.array([r[4] for r in results])
    soc_true = 1.0 - (Q_dis / Q_nom)
    rmse_soc = np.sqrt(np.mean((soc_est - soc_true)**2))

    # ── Plots ──────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("SOC Estimation vs True", "Core Temperature (K)", "NIS Consistency"),
        vertical_spacing=0.10
    )

    # SOC
    fig.add_trace(go.Scatter(x=t, y=soc_true, name="SOC True (DFN)",
                             line=dict(color='steelblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_est,  name="SOC Estimated (AEKF)",
                             line=dict(color='orange', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_est + 2*sig_soc,
                             line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_est - 2*sig_soc,
                             fill='tonexty', fillcolor='rgba(255,165,0,0.15)',
                             name="95% CI SOC"), row=1, col=1)

    # Temperature
    fig.add_trace(go.Scatter(x=t, y=T_true,    name="T True",
                             line=dict(color='firebrick', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=temp_est,  name="T Estimated",
                             line=dict(color='salmon', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=temp_est + 2*sig_t,
                             line=dict(width=0), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=temp_est - 2*sig_t,
                             fill='tonexty', fillcolor='rgba(255,0,0,0.10)',
                             name="95% CI Temp"), row=2, col=1)

    # NIS
    fig.add_trace(go.Scatter(x=t, y=nis_arr, name="NIS",
                             line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=5.99, line_dash="dot", line_color="red",
                  annotation_text="χ²(0.95, df=2) = 5.99", row=3, col=1)

    fig.update_layout(height=900, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics Dashboard ──────────────────────────────────────────────────
    st.subheader("📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SOC RMSE",    f"{rmse_soc*100:.3f} %")
    c2.metric("Final SOC",   f"{soc_est[-1]*100:.1f} %")
    c3.metric("Max T_core",  f"{temp_est.max():.2f} K")
    c4.metric("Mean NIS",    f"{nis_arr.mean():.3f}",
              delta="✅ Consistent" if nis_arr.mean() < 5.99 else "⚠️ Diverging")

    st.markdown(f"""
    ### 📝 Analysis Report
    | Parameter | Value |
    |---|---|
    | Battery Chemistry | NMC622 / Graphite (Chen2020) |
    | Cycles Simulated | {cycles} |
    | C-Rate | {c_rate} C |
    | Q_nom | {Q_nom:.3f} Ah |
    | SOC RMSE | {rmse_soc*100:.3f} % |
    | Mean σ_SOC | {sig_soc.mean():.4f} |
    | Mean σ_T | {sig_t.mean():.3f} K |
    | NIS Consistency | {"✅ Filter is consistent (NIS < χ²)" if nis_arr.mean() < 5.99 else "⚠️ Filter needs retuning"} |
    | R0 (Ω) | {R0} |
    | τ₁ = R1·C1 (s) | {R1*C1:.1f} |
    | τ₂ = R2·C2 (s) | {R2*C2:.1f} |
    | T_amb | {T_amb} K |
    """)
