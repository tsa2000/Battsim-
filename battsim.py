
import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# MACHINE 1: PHYSICAL ASSET (DFN - High Fidelity)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def simulate_physical_asset(c_rate, cycles):
    model  = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues("Chen2020")
    exp    = pybamm.Experiment(
        [f"Discharge at {c_rate}C until 2.5V",
         "Rest for 10 min",
         f"Charge at {c_rate/2}C until 4.2V",
         "Rest for 10 min"] * cycles
    )
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sol = sim.solve(initial_soc=1.0)

    t       = sol["Time [s]"].entries
    v_true  = sol["Terminal voltage [V]"].entries
    i_true  = sol["Current [A]"].entries
    q_dis   = sol["Discharge capacity [A.h]"].entries
    v_noisy = v_true + np.random.normal(0, 0.005, size=v_true.shape)
    Q_nom   = float(params["Nominal cell capacity [A.h]"])
    soc_true = np.clip(1.0 - q_dis / Q_nom, 0, 1)

    return t, v_noisy, i_true, v_true, soc_true, Q_nom

# ─────────────────────────────────────────────────────────────────────────────
# MACHINE 2: DIGITAL TWIN (ECM + AEKF + Uncertainty Propagation)
# ─────────────────────────────────────────────────────────────────────────────
class DigitalTwin:
    def __init__(self, Q_nom):
        self.Q_nom = Q_nom
        self.x = np.array([1.0, 0.0, 0.0])   # [SOC, V1, V2]
        self.P = np.diag([0.01, 1e-4, 1e-4])  # Covariance
        self.Q = np.diag([1e-6, 1e-7, 1e-7])  # Process noise
        self.R = np.array([[1e-3]])            # Measurement noise

    def step(self, V_m, I, dt):
        # ── Predict ──────────────────────────────
        A = np.eye(3)
        A[0, 0] = 1.0
        self.x[0] -= (I * dt) / (self.Q_nom * 3600)
        Pp = A @ self.P @ A.T + self.Q

        # ── Jacobian H (dV/dx) ───────────────────
        H = np.array([[1.0, -1.0, -1.0]])

        # ── Update ───────────────────────────────
        V_hat = 3.7 - self.x[1] - self.x[2] - I * 0.01
        nu    = V_m - V_hat
        S     = H @ Pp @ H.T + self.R
        K     = Pp @ H.T / S[0, 0]
        self.x     = self.x + K.flatten() * nu
        self.P     = (np.eye(3) - K.reshape(3,1) @ H) @ Pp
        self.x[0]  = np.clip(self.x[0], 0.0, 1.0)

        sigma = np.sqrt(self.P[0, 0])
        nis   = float(nu**2 / S[0, 0])
        return self.x[0], sigma, nis

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔬 Digital Twin Research Framework: DFN Asset vs ECM Twin")

with st.sidebar:
    c_rate = st.slider("C-Rate", 0.2, 3.0, 1.0, 0.1)
    cycles = st.slider("Cycles", 1, 5, 3)
    run    = st.button("▶ Run Co-Simulation", type="primary")

if run:
    with st.spinner("Machine 1: Solving DFN physics..."):
        t, v_noisy, i, v_true, soc_true, Q_nom = simulate_physical_asset(c_rate, cycles)

    with st.spinner("Machine 2: AEKF Twin estimating..."):
        twin    = DigitalTwin(Q_nom)
        dt      = np.mean(np.diff(t))
        results = [twin.step(v_noisy[k], i[k], dt) for k in range(len(t))]
        soc_est = np.array([r[0] for r in results])
        sigma   = np.array([r[1] for r in results])
        nis     = np.array([r[2] for r in results])

    rmse = np.sqrt(np.mean((soc_est - soc_true)**2)) * 100
    mae  = np.mean(np.abs(soc_est - soc_true)) * 100

    # ── Metrics ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SOC RMSE",   f"{rmse:.2f} %")
    c2.metric("SOC MAE",    f"{mae:.2f} %")
    c3.metric("Mean σ",     f"±{sigma.mean()*100:.2f} %")
    c4.metric("Mean NIS",   f"{nis.mean():.3f}")

    # ── Plot 1: SOC Comparison ────────────────────────────────────────────────
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=soc_true, name="DFN Truth (Asset)", line=dict(color="royalblue")))
    fig1.add_trace(go.Scatter(x=t, y=soc_est,  name="ECM-AEKF (Twin)", line=dict(color="orange", dash="dash")))
    fig1.add_trace(go.Scatter(x=t, y=soc_est + 2*sigma, line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=t, y=soc_est - 2*sigma, fill="tonexty",
                              fillcolor="rgba(255,165,0,0.15)", line=dict(width=0),
                              name="95% Confidence Interval (UQ)"))
    fig1.update_layout(title="SOC: Asset vs Twin + Uncertainty Propagation",
                       xaxis_title="Time [s]", yaxis_title="SOC")
    st.plotly_chart(fig1, use_container_width=True)

    # ── Plot 2: Voltage Prediction ────────────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=v_true,  name="DFN Voltage (Asset)", line=dict(color="green")))
    fig2.add_trace(go.Scatter(x=t, y=v_noisy, name="Noisy Measurement",   line=dict(color="grey", dash="dot")))
    fig2.update_layout(title="Voltage: Physical Asset vs Noisy Sensor",
                       xaxis_title="Time [s]", yaxis_title="Voltage [V]")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Plot 3: NIS (Filter Consistency) ─────────────────────────────────────
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t, y=nis, name="NIS", line=dict(color="red")))
    fig3.add_hline(y=1.0, line_dash="dash", annotation_text="Ideal NIS = 1")
    fig3.update_layout(title="NIS: Filter Consistency Over Time",
                       xaxis_title="Time [s]", yaxis_title="NIS")
    st.plotly_chart(fig3, use_container_width=True)

    st.success("Co-simulation complete.")
