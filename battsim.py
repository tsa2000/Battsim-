import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# OCV: GITT-derived LUT for NMC622 / Graphite (Chen2020)
# ═══════════════════════════════════════════════════════════════════════════════
_SOC_LUT = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                     0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                     0.80, 0.85, 0.90, 0.95, 1.00])
_OCV_LUT = np.array([2.500, 3.100, 3.300, 3.440, 3.500, 3.540, 3.570, 3.600,
                     3.620, 3.645, 3.670, 3.700, 3.730, 3.760, 3.800, 3.850,
                     3.920, 4.000, 4.080, 4.150, 4.200])

def ocv_nmc(soc):
    return np.interp(np.clip(soc, 0.0, 1.0), _SOC_LUT, _OCV_LUT)

def docv_dsoc(soc, eps=1e-4):
    return (ocv_nmc(soc + eps) - ocv_nmc(soc - eps)) / (2 * eps)

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL ASSET: DFN NMC (Chen2020) — Multi-Output
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def run_asset(cycles, c_rate, noise_v, noise_t):
    model  = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues("Chen2020")
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
# DIGITAL TWIN — AEKF ENGINE
# State: x = [SOC, V_RC1, V_RC2, T_core]  (4×1)
# Measurements: y = [V_terminal, T_cell]   (2×1)
# ═══════════════════════════════════════════════════════════════════════════════
class AEKF:
    def __init__(self, Q_nom, R0, R1, C1, R2, C2, R_th, C_th, T_amb,
                 P0_diag, Q_diag, R_diag):
        self.Q_nom = Q_nom
        self.R0 = R0; self.R1 = R1; self.C1 = C1
        self.R2 = R2; self.C2 = C2
        self.R_th = R_th; self.C_th = C_th; self.T_amb = T_amb
        self.x  = np.array([1.0, 0.0, 0.0, T_amb])
        self.P  = np.diag(P0_diag)
        self.Qn = np.diag(Q_diag)
        self.Rn = np.diag(R_diag)

    def _params(self, soc, T):
        arr    = np.exp(3600.0 * (1/max(T, 250) - 1/298.15))
        R0_eff = self.R0 * (1 + 0.4*(1-soc)**2) * arr
        R1_eff = self.R1 * arr
        R2_eff = self.R2 * arr
        return R0_eff, R1_eff, R2_eff

    def _fx(self, x, I, dt):
        soc, V1, V2, T = x
        R0, R1, R2 = self._params(soc, T)
        a1 = np.exp(-dt / (R1 * self.C1))
        a2 = np.exp(-dt / (R2 * self.C2))
        return np.array([
            soc - (I * dt) / (self.Q_nom * 3600),
            a1 * V1 + R1 * (1 - a1) * I,
            a2 * V2 + R2 * (1 - a2) * I,
            T + (dt / self.C_th) * (I**2 * R0 - (T - self.T_amb) / self.R_th)
        ])

    def _hx(self, x, I):
        soc, V1, V2, T = x
        R0, _, _ = self._params(soc, T)
        return np.array([ocv_nmc(soc) - V1 - V2 - I * R0, T])

    def step(self, V_m, T_m, I, dt):
        soc, V1, V2, T = self.x
        R0, R1, R2 = self._params(soc, T)
        a1 = np.exp(-dt / (R1 * self.C1))
        a2 = np.exp(-dt / (R2 * self.C2))

        # Jacobian F (4×4)
        F = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0,  a1, 0.0, 0.0],
            [0.0, 0.0,  a2, 0.0],
            [0.0, 0.0, 0.0, 1.0 - dt/(self.C_th * self.R_th)]
        ])

        x_pred = self._fx(self.x, I, dt)
        P_pred = F @ self.P @ F.T + self.Qn

        # Jacobian H (2×4)
        H = np.array([
            [docv_dsoc(x_pred[0]), -1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        y_pred = self._hx(x_pred, I)
        nu     = np.array([V_m - y_pred[0], T_m - y_pred[1]])
        S      = H @ P_pred @ H.T + self.Rn
        K      = P_pred @ H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ nu
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.x[3] = np.clip(self.x[3], 250.0, 350.0)
        self.P = (np.eye(4) - K @ H) @ P_pred

        nis = float(nu @ np.linalg.inv(S) @ nu)
        return self.x[0], self.x[3], np.sqrt(self.P[0,0]), nu, nis

# ═══════════════════════════════════════════════════════════════════════════════
# UNSCENTED TRANSFORM — 7 Sigma Points (Julier & Uhlmann 1997)
# Propagates uncertainty through nonlinear ECM without linearisation
# ═══════════════════════════════════════════════════════════════════════════════
class UnscentedTransform:
    """
    Merwe Scaled Unscented Transform.
    n=4 states → 2n+1 = 9 sigma points.
    α=1e-3, β=2 (Gaussian prior), κ=0  →  λ = α²(n+κ) - n
    """
    def __init__(self, n=4, alpha=1e-3, beta=2.0, kappa=0.0):
        self.n = n
        lam    = alpha**2 * (n + kappa) - n
        self.Wm = np.full(2*n+1, 1/(2*(n+lam)))
        self.Wc = np.full(2*n+1, 1/(2*(n+lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
        self.gamma  = np.sqrt(n + lam)

    def sigma_points(self, x, P):
        S   = np.linalg.cholesky(P + 1e-10 * np.eye(self.n))
        pts = [x]
        for i in range(self.n):
            pts.append(x + self.gamma * S[:, i])
            pts.append(x - self.gamma * S[:, i])
        return np.array(pts)   # (2n+1, n)

    def propagate(self, sigmas, fx, *args):
        Y = np.array([fx(s, *args) for s in sigmas])
        x_mean = np.einsum('i,ij->j', self.Wm, Y)
        P_out  = sum(self.Wc[i] * np.outer(Y[i]-x_mean, Y[i]-x_mean)
                     for i in range(len(self.Wm)))
        return x_mean, P_out, Y

    def update(self, x_pred, P_pred, sigmas_prop, hx, y_meas, Rn, *args):
        Z = np.array([hx(s, *args) for s in sigmas_prop])
        z_mean = np.einsum('i,ij->j', self.Wm, Z)
        Pzz = sum(self.Wc[i] * np.outer(Z[i]-z_mean, Z[i]-z_mean)
                  for i in range(len(self.Wm))) + Rn
        Pxz = sum(self.Wc[i] * np.outer(sigmas_prop[i]-x_pred, Z[i]-z_mean)
                  for i in range(len(self.Wm)))
        K      = Pxz @ np.linalg.inv(Pzz)
        nu     = y_meas - z_mean
        x_upd  = x_pred + K @ nu
        P_upd  = P_pred - K @ Pzz @ K.T
        sigma_SOC = np.sqrt(abs(P_upd[0,0]))
        return x_upd, P_upd, sigma_SOC, nu

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED RUNNER: AEKF + UT per timestep
# ═══════════════════════════════════════════════════════════════════════════════
def run_twin(t, V_m, T_m, I, Q_nom,
             R0, R1, C1, R2, C2, R_th, C_th, T_amb,
             P0_diag, Q_diag, R_diag):
    dt   = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    ekf  = AEKF(Q_nom, R0, R1, C1, R2, C2, R_th, C_th, T_amb,
                P0_diag, Q_diag, R_diag)
    ut   = UnscentedTransform(n=4)
    x_ut = np.array([1.0, 0.0, 0.0, T_amb])
    P_ut = np.diag(P0_diag)
    Qn   = np.diag(Q_diag)
    Rn   = np.diag(R_diag)

    soc_ekf=[]; soc_ut=[]; sig_ekf=[]; sig_ut=[]
    temp_ekf=[]; nis_arr=[]; innov=[]

    for k in range(len(t)):
        Ik = I[k]
        y  = np.array([V_m[k], T_m[k]])

        # ── AEKF step ────────────────────────────────────────────────────────
        s, T_e, sg, nu, nis = ekf.step(V_m[k], T_m[k], Ik, dt)
        soc_ekf.append(s); sig_ekf.append(sg)
        temp_ekf.append(T_e); nis_arr.append(nis); innov.append(nu[0]*1000)

        # ── UT step ───────────────────────────────────────────────────────────
        sigmas = ut.sigma_points(x_ut, P_ut)

        def fx_ut(xi, Ik=Ik, dt=dt):
            soc_i, V1_i, V2_i, T_i = xi
            arr   = np.exp(3600*(1/max(T_i,250) - 1/298.15))
            R0e   = R0*(1+0.4*(1-soc_i)**2)*arr
            R1e   = R1*arr; R2e = R2*arr
            a1    = np.exp(-dt/(R1e*C1)); a2 = np.exp(-dt/(R2e*C2))
            return np.array([
                soc_i - (Ik*dt)/(Q_nom*3600),
                a1*V1_i + R1e*(1-a1)*Ik,
                a2*V2_i + R2e*(1-a2)*Ik,
                T_i + (dt/C_th)*(Ik**2*R0e - (T_i-T_amb)/R_th)
            ])

        def hx_ut(xi, Ik=Ik):
            soc_i, V1_i, V2_i, T_i = xi
            arr = np.exp(3600*(1/max(T_i,250) - 1/298.15))
            R0e = R0*(1+0.4*(1-soc_i)**2)*arr
            return np.array([ocv_nmc(soc_i) - V1_i - V2_i - Ik*R0e, T_i])

        x_pred_ut, P_pred_ut, sigmas_prop = ut.propagate(sigmas, fx_ut)
        P_pred_ut += Qn
        x_ut, P_ut, sg_ut, _ = ut.update(
            x_pred_ut, P_pred_ut, sigmas_prop, hx_ut, y, Rn
        )
        x_ut[0] = np.clip(x_ut[0], 0.0, 1.0)
        x_ut[3] = np.clip(x_ut[3], 250.0, 350.0)
        soc_ut.append(x_ut[0]); sig_ut.append(sg_ut)

    return (np.array(soc_ekf), np.array(soc_ut),
            np.array(sig_ekf), np.array(sig_ut),
            np.array(temp_ekf), np.array(nis_arr), np.array(innov))

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="NMC MIMO Digital Twin", layout="wide")
st.title("🔋 NMC622 MIMO Digital Twin — AEKF + Unscented Transform")

with st.sidebar:
    st.header("⚙️ Simulation Settings")

    with st.expander("📡 Asset (DFN) Settings", expanded=True):
        cycles  = st.number_input("Cycles",            1,  50,   5)
        c_rate  = st.slider("C-Rate",                  0.5, 3.0, 1.0, 0.1)
        noise_v = st.slider("Voltage Noise (V)",        0.000, 0.050, 0.005, 0.001)
        noise_t = st.slider("Temp Noise (K)",           0.0,   2.0,   0.1,  0.05)

    with st.expander("🔧 ECM Parameters (NMC622)", expanded=True):
        R0 = st.number_input("R0 — Ohmic Resistance (Ω)",  0.001, 0.200, 0.015, 0.001, format="%.3f")
        R1 = st.number_input("R1 — RC1 Resistance (Ω)",    0.001, 0.100, 0.010, 0.001, format="%.3f")
        C1 = st.number_input("C1 — RC1 Capacitance (F)",   100.0, 5000.0, 3000.0, 100.0)
        R2 = st.number_input("R2 — RC2 Resistance (Ω)",    0.001, 0.100, 0.005, 0.001, format="%.3f")
        C2 = st.number_input("C2 — RC2 Capacitance (F)",   100.0, 5000.0, 2000.0, 100.0)

    with st.expander("🌡️ Thermal Parameters", expanded=True):
        R_th  = st.number_input("R_th — Thermal Resistance (K/W)", 1.0, 20.0, 3.0,  0.5)
        C_th  = st.number_input("C_th — Thermal Capacity (J/K)",   100.0, 2000.0, 500.0, 50.0)
        T_amb = st.number_input("T_amb — Ambient Temp (K)",        273.15, 323.15, 298.15, 1.0)

    with st.expander("📊 EKF / UT Noise Tuning", expanded=False):
        st.markdown("**Process Noise Q**")
        q_soc = st.number_input("Q — SOC",    1e-9, 1e-4, 1e-7, format="%.2e")
        q_v1  = st.number_input("Q — V_RC1",  1e-9, 1e-4, 1e-8, format="%.2e")
        q_v2  = st.number_input("Q — V_RC2",  1e-9, 1e-4, 1e-8, format="%.2e")
        q_t   = st.number_input("Q — T_core", 1e-5, 1e-1, 1e-3, format="%.2e")
        st.markdown("**Measurement Noise R**")
        r_v   = st.number_input("R — Voltage", 1e-5, 1e-1, 1e-3, format="%.2e")
        r_t   = st.number_input("R — Temp",    1e-3, 1.0,  0.1,  format="%.3f")
        st.markdown("**Initial Covariance P0**")
        p_soc = st.number_input("P0 — SOC",    1e-4, 1.0,  0.01, format="%.4f")
        p_t   = st.number_input("P0 — T_core", 0.1,  5.0,  0.5,  format="%.2f")

    run_btn = st.button("▶ Execute Co-Simulation", type="primary")

# ── Main ─────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running DFN Asset (Physics)..."):
        t, V_m, T_m, V_true, T_true, I, Q_dis, Q_nom = run_asset(
            cycles, c_rate, noise_v, noise_t
        )

    with st.spinner("Running AEKF + Unscented Transform..."):
        P0  = [p_soc, 1e-4, 1e-4, p_t]
        Qd  = [q_soc, q_v1, q_v2, q_t]
        Rd  = [r_v, r_t]
        (soc_ekf, soc_ut, sig_ekf, sig_ut,
         temp_ekf, nis_arr, innov) = run_twin(
            t, V_m, T_m, I, Q_nom,
            R0, R1, C1, R2, C2, R_th, C_th, T_amb, P0, Qd, Rd
        )

    soc_true = 1.0 - (Q_dis / Q_nom)
    rmse_ekf = np.sqrt(np.mean((soc_ekf - soc_true)**2)) * 100
    rmse_ut  = np.sqrt(np.mean((soc_ut  - soc_true)**2)) * 100
    chi2_95  = 5.991   # χ²(0.95, df=2)

    # ── Plots ────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "SOC Estimation — DFN Truth vs AEKF vs UT",
            "σ_SOC: UT vs EKF Linearisation Check",
            "Core Temperature (K)",
            "NIS Consistency  |  Innovation Sequence (mV)"
        ),
        vertical_spacing=0.08
    )

    # Row 1: SOC
    fig.add_trace(go.Scatter(x=t, y=soc_true, name="DFN Truth",
                             line=dict(color='steelblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_ekf, name="AEKF",
                             line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_ut, name="UT",
                             line=dict(color='orange', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_ekf+2*sig_ekf,
                             line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_ekf-2*sig_ekf, fill='tonexty',
                             fillcolor='rgba(255,0,0,0.10)', name="EKF 95% CI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_ut+2*sig_ut,
                             line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_ut-2*sig_ut, fill='tonexty',
                             fillcolor='rgba(255,165,0,0.15)', name="UT 95% CI"), row=1, col=1)

    # Row 2: σ comparison
    fig.add_trace(go.Scatter(x=t, y=sig_ekf, name="σ EKF (Jacobian)",
                             line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=sig_ut, name="σ UT (Sigma pts)",
                             line=dict(color='orange', dash='dot')), row=2, col=1)

    # Row 3: Temperature
    fig.add_trace(go.Scatter(x=t, y=T_true, name="T True",
                             line=dict(color='firebrick', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=temp_ekf, name="T Estimated",
                             line=dict(color='salmon', dash='dash')), row=3, col=1)

    # Row 4: NIS + Innovation
    fig.add_trace(go.Scatter(x=t, y=nis_arr, name="NIS",
                             line=dict(color='purple', width=1), opacity=0.5), row=4, col=1)
    window = min(50, len(nis_arr)//4)
    if window > 1:
        nis_ma = np.convolve(nis_arr, np.ones(window)/window, mode='same')
        fig.add_trace(go.Scatter(x=t, y=nis_ma, name="NIS (MA-50)",
                                 line=dict(color='purple', width=2)), row=4, col=1)
    fig.add_hline(y=chi2_95, line_dash="dot", line_color="red",
                  annotation_text="χ²(0.95, df=2)=5.99", row=4, col=1)

    fig.update_layout(height=1100, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics Dashboard ───────────────────────────────────────────────
    st.subheader("📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SOC RMSE — EKF",  f"{rmse_ekf:.3f} %")
    c2.metric("SOC RMSE — UT",   f"{rmse_ut:.3f} %")
    c3.metric("Mean NIS",        f"{nis_arr.mean():.3f}",
              delta="✅ Consistent" if nis_arr.mean() < chi2_95 else "⚠️ Retuning needed")
    c4.metric("Max T_core",      f"{temp_ekf.max():.2f} K")

    # ── Report ──────────────────────────────────────────────────────────
    st.markdown(f"""
    ### 📝 Analysis Report
    | Parameter | Value |
    |---|---|
    | Battery Chemistry | NMC622 / Graphite (Chen2020) |
    | Cycles Simulated | {cycles} |
    | C-Rate | {c_rate} C |
    | Q_nom | {Q_nom:.3f} Ah |
    | SOC RMSE (EKF) | {rmse_ekf:.3f} % |
    | SOC RMSE (UT)  | {rmse_ut:.3f} % |
    | Mean σ_SOC EKF | {sig_ekf.mean():.4f} |
    | Mean σ_SOC UT  | {sig_ut.mean():.4f} |
    | NIS Consistency | {"✅ Filter is consistent (NIS < χ²)" if nis_arr.mean() < chi2_95 else "⚠️ Filter needs retuning"} |
    | τ₁ = R1·C1 (s) | {R1*C1:.1f} |
    | τ₂ = R2·C2 (s) | {R2*C2:.1f} |
    | OCV Model | GITT LUT (NMC622) |
    | UQ Method | Unscented Transform — 9 sigma points |
    | References | Julier & Uhlmann 1997, Plett 2004 |
    """)
