import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# OCV — Chen2020 NMC811/Graphite polynomial (101 نقطة للدقة القصوى)
# المرجع: Chen et al. 2020, J. Electrochem. Soc. 167, 080534
# ═══════════════════════════════════════════════════════════════════════════════
_SOC_LUT = np.linspace(0.0, 1.0, 101)
_OCV_LUT = np.clip(
    3.4043
    + 1.6227 * _SOC_LUT
    - 8.6635 * _SOC_LUT**2
    + 21.3955 * _SOC_LUT**3
    - 25.7324 * _SOC_LUT**4
    + 12.0032 * _SOC_LUT**5,
    2.5, 4.25
)

def ocv_nmc(soc):
    return np.interp(np.clip(soc, 0.0, 1.0), _SOC_LUT, _OCV_LUT)

def docv_dsoc(soc, eps=1e-5):
    return (ocv_nmc(soc + eps) - ocv_nmc(soc - eps)) / (2 * eps)

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL ASSET: DFN NMC622 — Chen2020 (LG M50 21700, 5 Ah)
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
# DIGITAL TWIN — AEKF
#
# State:        x = [SOC, V_RC1, V_RC2, T_core]   (4×1)
# Measurement:  y = [V_terminal, T_cell]           (2×1)
#
# State equations (ZOH exact discretisation — Plett 2004):
#   SOC[k+1]    = SOC[k] − I·dt / (Q_nom·3600)
#   V_RC1[k+1]  = exp(−dt/τ1)·V_RC1[k] + R1·(1−exp(−dt/τ1))·I
#   V_RC2[k+1]  = exp(−dt/τ2)·V_RC2[k] + R2·(1−exp(−dt/τ2))·I
#   T[k+1]      = T[k] + (dt/C_th)·(I²·R0 − (T−T_amb)/R_th)
#
# Measurement equation:
#   V_t  = OCV(SOC) − V_RC1 − V_RC2 − I·R0
#   T_out = T_core
#
# Resistance — Arrhenius correction (activation energy Ea/R ≈ 3600 K for NMC):
#   R0_eff = R0·(1 + 0.4·(1−SOC)²)·exp(3600·(1/T − 1/T_ref))
# ═══════════════════════════════════════════════════════════════════════════════
class AEKF:
    def __init__(self, Q_nom, R0, R1, C1, R2, C2,
                 R_th, C_th, T_amb, P0_diag, Q_diag, R_diag):
        self.Q_nom = Q_nom
        self.R0 = R0; self.R1 = R1; self.C1 = C1
        self.R2 = R2; self.C2 = C2
        self.R_th = R_th; self.C_th = C_th; self.T_amb = T_amb
        self.x  = np.array([1.0, 0.0, 0.0, T_amb])
        self.P  = np.diag(P0_diag)
        self.Qn = np.diag(Q_diag)
        self.Rn = np.diag(R_diag)

    def _arr(self, T):
        return np.exp(3600.0 * (1.0/max(T, 250.0) - 1.0/298.15))

    def _params(self, soc, T):
        a   = self._arr(T)
        R0e = self.R0 * (1.0 + 0.4*(1.0 - soc)**2) * a
        R1e = self.R1 * a
        R2e = self.R2 * a
        return R0e, R1e, R2e, a

    def step(self, V_m, T_m, I, dt):
        soc, V1, V2, T = self.x
        R0e, R1e, R2e, a = self._params(soc, T)
        tau1 = R1e * self.C1;  tau2 = R2e * self.C2
        e1   = np.exp(-dt / tau1);  e2 = np.exp(-dt / tau2)

        # ── Predict ──────────────────────────────────────────────────────────
        x_p = np.array([
            soc - (I * dt) / (self.Q_nom * 3600),
            e1 * V1 + R1e * (1 - e1) * I,
            e2 * V2 + R2e * (1 - e2) * I,
            T + (dt / self.C_th) * (I**2 * R0e - (T - self.T_amb) / self.R_th)
        ])

        # ── Jacobian F (4×4) — CORRECTED: includes ∂V_RC/∂T ─────────────────
        # ∂R_eff/∂T = R_eff · (−Ea/R) / T²  →  Arrhenius derivative
        darr_dT  = -3600.0 / max(T, 250.0)**2 * a
        dR1e_dT  = self.R1 * darr_dT
        dR2e_dT  = self.R2 * darr_dT

        F = np.array([
            [1.0,  0.0,  0.0,  0.0],
            [0.0,   e1,  0.0,  dR1e_dT * (1.0 - e1) * I * dt],
            [0.0,  0.0,   e2,  dR2e_dT * (1.0 - e2) * I * dt],
            [0.0,  0.0,  0.0,  1.0 - dt / (self.C_th * self.R_th)]
        ])

        P_p = F @ self.P @ F.T + self.Qn

        # ── Measurement Update ────────────────────────────────────────────────
        soc_p, V1_p, V2_p, T_p = x_p
        R0p, _, _, _ = self._params(soc_p, T_p)

        # Jacobian H (2×4): ∂y/∂x — numerical dOCV/dSOC
        H = np.array([
            [docv_dsoc(soc_p), -1.0, -1.0, 0.0],
            [0.0,               0.0,  0.0, 1.0]
        ])

        V_est = ocv_nmc(soc_p) - V1_p - V2_p - I * R0p
        nu    = np.array([V_m - V_est, T_m - T_p])
        S     = H @ P_p @ H.T + self.Rn
        K     = P_p @ H.T @ np.linalg.inv(S)

        self.x = x_p + K @ nu
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.x[3] = np.clip(self.x[3], 250.0, 350.0)
        self.P = (np.eye(4) - K @ H) @ P_p

        nis = float(nu @ np.linalg.inv(S) @ nu)
        return self.x[0], self.x[3], np.sqrt(self.P[0,0]), nu[0]*1000, nis

# ═══════════════════════════════════════════════════════════════════════════════
# UNSCENTED TRANSFORM — Merwe Scaled (9 Sigma Points, n=4)
# Julier & Uhlmann 1997 | Van der Merwe & Wan 2000
# α=1e-3, β=2 (Gaussian), κ=0  →  λ = α²(n+κ) − n
# ═══════════════════════════════════════════════════════════════════════════════
class UnscentedTransform:
    def __init__(self, n=4, alpha=1e-3, beta=2.0, kappa=0.0):
        self.n  = n
        lam     = alpha**2 * (n + kappa) - n
        c       = 1.0 / (2.0 * (n + lam))
        self.Wm = np.full(2*n+1, c)
        self.Wc = np.full(2*n+1, c)
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
        self.gamma  = np.sqrt(n + lam)

    def sigma_points(self, x, P):
        # Cholesky with jitter for numerical stability
        S   = np.linalg.cholesky(P + 1e-9 * np.eye(self.n))
        pts = [x.copy()]
        for i in range(self.n):
            pts.append(x + self.gamma * S[:, i])
            pts.append(x - self.gamma * S[:, i])
        return np.array(pts)   # shape: (2n+1, n)

    def propagate(self, sigmas, fx, *args):
        Y      = np.array([fx(s, *args) for s in sigmas])
        x_mean = np.einsum('i,ij->j', self.Wm, Y)
        P_out  = sum(self.Wc[i] * np.outer(Y[i]-x_mean, Y[i]-x_mean)
                     for i in range(len(self.Wm)))
        return x_mean, P_out, Y

    def update(self, x_pred, P_pred, sigmas_prop, hx, y_meas, Rn, *args):
        Z      = np.array([hx(s, *args) for s in sigmas_prop])
        z_mean = np.einsum('i,ij->j', self.Wm, Z)
        Pzz    = sum(self.Wc[i] * np.outer(Z[i]-z_mean, Z[i]-z_mean)
                     for i in range(len(self.Wm))) + Rn
        Pxz    = sum(self.Wc[i] * np.outer(sigmas_prop[i]-x_pred, Z[i]-z_mean)
                     for i in range(len(self.Wm)))
        K      = Pxz @ np.linalg.inv(Pzz)
        nu     = y_meas - z_mean
        x_upd  = x_pred + K @ nu
        P_upd  = P_pred - K @ Pzz @ K.T
        return x_upd, P_upd, np.sqrt(abs(P_upd[0,0]))

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED RUNNER: AEKF + UT per timestep
# ═══════════════════════════════════════════════════════════════════════════════
def run_twin(t, V_m, T_m, I, Q_nom,
             R0, R1, C1, R2, C2, R_th, C_th, T_amb,
             P0_diag, Q_diag, R_diag):

    length = min(len(t), len(V_m), len(T_m), len(I))
    t = t[:length]
    V_m = V_m[:length]
    T_m = T_m[:length]
    I = I[:length]

    dt = float(np.mean(np.diff(t))) if length > 1 else 1.0
    Qn = np.diag(Q_diag)
    Rn = np.diag(R_diag)

    ekf  = AEKF(Q_nom, R0, R1, C1, R2, C2, R_th, C_th, T_amb,
                P0_diag, Q_diag, R_diag)
    ut   = UnscentedTransform(n=4)
    x_ut = np.array([1.0, 0.0, 0.0, T_amb])
    P_ut = np.diag(P0_diag)

    soc_ekf=[]; soc_ut=[]; sig_ekf=[]; sig_ut=[]
    temp_ekf=[]; nis_arr=[]; innov_arr=[]

    def arr_fn(T): return np.exp(3600*(1/max(T,250)-1/298.15))

    def fx_ut(xi, Ik=0.0):
        s, V1, V2, Tc = xi
        a   = arr_fn(Tc)
        R0e = R0*(1+0.4*(1-s)**2)*a
        R1e = R1*a; R2e = R2*a
        e1  = np.exp(-dt/(R1e*C1)); e2 = np.exp(-dt/(R2e*C2))
        return np.array([
            s - (Ik*dt)/(Q_nom*3600),
            e1*V1 + R1e*(1-e1)*Ik,
            e2*V2 + R2e*(1-e2)*Ik,
            Tc + (dt/C_th)*(Ik**2*R0e - (Tc-T_amb)/R_th)
        ])

    def hx_ut(xi, Ik=0.0):
        s, V1, V2, Tc = xi
        R0e = R0*(1+0.4*(1-s)**2)*arr_fn(Tc)
        return np.array([ocv_nmc(s) - V1 - V2 - Ik*R0e, Tc])

    for k in range(len(t)):
        Ik = float(I[k])
        y  = np.array([V_m[k], T_m[k]])

        # AEKF step
        s_e, T_e, sg_e, nu_v, nis = ekf.step(V_m[k], T_m[k], Ik, dt)
        soc_ekf.append(s_e); sig_ekf.append(sg_e)
        temp_ekf.append(T_e); nis_arr.append(nis); innov_arr.append(nu_v)

        # UT step
        sigmas              = ut.sigma_points(x_ut, P_ut)
        x_p, P_p, sig_prop = ut.propagate(sigmas, fx_ut, Ik)
        P_p                += Qn
        x_ut, P_ut, sg_ut  = ut.update(x_p, P_p, sig_prop, hx_ut, y, Rn, Ik)
        x_ut[0] = np.clip(x_ut[0], 0.0, 1.0)
        x_ut[3] = np.clip(x_ut[3], 250.0, 350.0)
        soc_ut.append(x_ut[0]); sig_ut.append(sg_ut)

    return (np.array(soc_ekf), np.array(soc_ut),
            np.array(sig_ekf), np.array(sig_ut),
            np.array(temp_ekf), np.array(nis_arr), np.array(innov_arr))

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="NMC MIMO Digital Twin", layout="wide")
st.title("🔋 NMC622 MIMO Digital Twin — AEKF + Unscented Transform")
st.caption("Chen2020 | DFN PyBaMM | 2-RC ECM | AEKF | UT (9 σ-points) | Plett 2004 | Julier & Uhlmann 1997")

with st.sidebar:
    st.header("⚙️ Simulation Settings")

    with st.expander("📡 Asset (DFN) Settings", expanded=True):
        cycles  = st.number_input("Cycles",           1,  50,   5)
        c_rate  = st.slider("C-Rate",                 0.5, 3.0, 1.0, 0.1)
        noise_v = st.slider("Voltage Noise σ (V)",    0.000, 0.050, 0.005, 0.001)
        noise_t = st.slider("Temp Noise σ (K)",       0.0,   2.0,   0.1,  0.05)

    with st.expander("🔧 ECM Parameters (NMC622)", expanded=True):
        R0 = st.number_input("R0 — Ohmic Resistance (Ω)", 0.001, 0.200, 0.015, 0.001, format="%.3f")
        R1 = st.number_input("R1 — RC1 Resistance (Ω)",   0.001, 0.100, 0.010, 0.001, format="%.3f")
        C1 = st.number_input("C1 — RC1 Capacitance (F)",  100.0, 5000.0, 3000.0, 100.0)
        R2 = st.number_input("R2 — RC2 Resistance (Ω)",   0.001, 0.100, 0.005, 0.001, format="%.3f")
        C2 = st.number_input("C2 — RC2 Capacitance (F)",  100.0, 5000.0, 2000.0, 100.0)

    with st.expander("🌡️ Thermal Parameters", expanded=True):
        R_th  = st.number_input("R_th — Thermal Resistance (K/W)", 1.0,   20.0,   3.0,  0.5)
        C_th  = st.number_input("C_th — Thermal Capacity (J/K)",   100.0, 2000.0, 500.0, 50.0)
        T_amb = st.number_input("T_amb — Ambient Temp (K)",        273.15, 323.15, 298.15, 1.0)

    with st.expander("📊 EKF / UT Noise Tuning", expanded=False):
        st.markdown("**Process Noise Q** (model uncertainty)")
        q_soc = st.number_input("Q — SOC",    1e-9, 1e-4, 1e-7, format="%.2e")
        q_v1  = st.number_input("Q — V_RC1",  1e-9, 1e-4, 1e-8, format="%.2e")
        q_v2  = st.number_input("Q — V_RC2",  1e-9, 1e-4, 1e-8, format="%.2e")
        q_t   = st.number_input("Q — T_core", 1e-5, 1e-1, 1e-3, format="%.2e")
        st.markdown("**Measurement Noise R** (sensor trust)")
        r_v   = st.number_input("R — Voltage (V²)", 1e-5, 1e-1, 1e-3, format="%.2e")
        r_t   = st.number_input("R — Temp (K²)",    1e-3, 1.0,  0.1,  format="%.3f")
        st.markdown("**Initial Covariance P0**")
        p_soc = st.number_input("P0 — SOC",    1e-4, 1.0,  0.01, format="%.4f")
        p_t   = st.number_input("P0 — T_core", 0.1,  5.0,  0.5,  format="%.2f")

    run_btn = st.button("▶ Execute Co-Simulation", type="primary")

# ── Main ─────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("⚙️ Running DFN Asset (Physics Engine)..."):
        t, V_m, T_m, V_true, T_true, I, Q_dis, Q_nom = run_asset(
            cycles, c_rate, noise_v, noise_t
        )

    with st.spinner("🧠 Running AEKF + Unscented Transform..."):
        P0 = [p_soc, 1e-4, 1e-4, p_t]
        Qd = [q_soc, q_v1, q_v2, q_t]
        Rd = [r_v, r_t]
        (soc_ekf, soc_ut, sig_ekf, sig_ut,
         temp_ekf, nis_arr, innov) = run_twin(
            t, V_m, T_m, I, Q_nom,
            R0, R1, C1, R2, C2, R_th, C_th, T_amb, P0, Qd, Rd
        )

    soc_true = 1.0 - (Q_dis / Q_nom)
    rmse_ekf = np.sqrt(np.mean((soc_ekf - soc_true)**2)) * 100
    rmse_ut  = np.sqrt(np.mean((soc_ut  - soc_true)**2)) * 100
    chi2_95  = 5.991   # χ²(0.95, df=2) — 2 measurements

    # ── Plots ────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "SOC Estimation — DFN Truth vs AEKF vs UT",
            "σ_SOC: UT vs EKF Linearisation Check",
            "Core Temperature (K)",
            "NIS Consistency  |  Innovation Sequence ν (mV)"
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
                             fillcolor='rgba(255,0,0,0.10)',
                             name="EKF 95% CI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_ut+2*sig_ut,
                             line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=soc_ut-2*sig_ut, fill='tonexty',
                             fillcolor='rgba(255,165,0,0.15)',
                             name="UT 95% CI"), row=1, col=1)

    # Row 2: σ comparison (Linearisation Check)
    fig.add_trace(go.Scatter(x=t, y=sig_ekf, name="σ EKF (Jacobian)",
                             line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=sig_ut, name="σ UT (Sigma pts)",
                             line=dict(color='orange', dash='dot')), row=2, col=1)

    # Row 3: Temperature
    fig.add_trace(go.Scatter(x=t, y=T_true, name="T True (DFN)",
                             line=dict(color='firebrick', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=temp_ekf, name="T Estimated",
                             line=dict(color='salmon', dash='dash')), row=3, col=1)

    # Row 4: NIS + Innovation
    fig.add_trace(go.Scatter(x=t, y=nis_arr, name="NIS (raw)",
                             line=dict(color='purple', width=1), opacity=0.4), row=4, col=1)
    w = min(50, max(2, len(nis_arr)//20))
    nis_ma = np.convolve(nis_arr, np.ones(w)/w, mode='same')
    fig.add_trace(go.Scatter(x=t, y=nis_ma, name=f"NIS MA-{w}",
                             line=dict(color='purple', width=2)), row=4, col=1)
    fig.add_hline(y=chi2_95, line_dash="dot", line_color="red",
                  annotation_text="χ²(0.95, df=2) = 5.991", row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=innov, name="Innovation ν (mV)",
                             line=dict(color='green', width=1), opacity=0.5,
                             yaxis="y8"), row=4, col=1)

    fig.update_layout(height=1100, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics Dashboard ────────────────────────────────────────────────
    st.subheader("📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SOC RMSE — EKF",  f"{rmse_ekf:.3f} %")
    c2.metric("SOC RMSE — UT",   f"{rmse_ut:.3f} %")
    c3.metric("Mean NIS",
              f"{nis_arr.mean():.3f}",
              delta="✅ Consistent" if nis_arr.mean() < chi2_95 else "⚠️ Retuning needed")
    c4.metric("Max T_core",  f"{temp_ekf.max():.2f} K")

    # ── Analysis Report ──────────────────────────────────────────────────
    st.markdown(f"""
    ### 📝 Analysis Report
    | Parameter | Value |
    |---|---|
    | Battery Chemistry | NMC622/Graphite — Chen2020 (LG M50 21700) |
    | Cycles Simulated | {cycles} |
    | C-Rate | {c_rate} C |
    | Q_nom | {Q_nom:.3f} Ah |
    | **SOC RMSE — EKF** | **{rmse_ekf:.3f} %** |
    | **SOC RMSE — UT** | **{rmse_ut:.3f} %** |
    | Mean σ_SOC (EKF) | {sig_ekf.mean():.5f} |
    | Mean σ_SOC (UT) | {sig_ut.mean():.5f} |
    | NIS Consistency | {"✅ Consistent — filter well-calibrated" if nis_arr.mean() < chi2_95 else "⚠️ Retuning needed"} |
    | τ₁ = R1·C1 (s) | {R1*C1:.1f} |
    | τ₂ = R2·C2 (s) | {R2*C2:.1f} |
    | OCV Model | Chen2020 Polynomial — 101-point LUT |
    | UQ Method | Merwe Scaled UT — 9 sigma points |
    | Jacobian F | Corrected (includes ∂V_RC/∂T Arrhenius) |
    | References | Plett 2004 · Julier & Uhlmann 1997 · Van der Merwe 2000 |
    """)
