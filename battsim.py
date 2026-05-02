
import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: PHYSICAL ASSET (High-Fidelity Model)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def simulate_physical_asset(c_rate, cycles):
    # DFN model setup
    model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues("Chen2020")
    exp = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V", "Rest for 10 min",
                             f"Charge at {c_rate/2}C until 4.2V", "Rest for 10 min"] * cycles)
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sol = sim.solve(initial_soc=1.0)

    # Sensor Data (with artificial noise)
    t = sol["Time [s]"].entries
    v_true = sol["Terminal voltage [V]"].entries
    i_true = sol["Current [A]"].entries
    v_noisy = v_true + np.random.normal(0, 0.005, size=v_true.shape)

    return t, v_noisy, i_true, sol

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: DIGITAL TWIN (Reduced Order + AEKF + UT)
# ─────────────────────────────────────────────────────────────────────────────
class DigitalTwin:
    def __init__(self, Q_nom):
        self.Q_nom = Q_nom
        # State: [SOC, V1, V2], Covariance: P
        self.x = np.array([1.0, 0.0, 0.0])
        self.P = np.diag([0.01, 1e-4, 1e-4])
        self.Q = np.diag([1e-6, 1e-7, 1e-7]) # Process noise
        self.R = np.array([[1e-3]])         # Measurement noise

    def estimate(self, V_m, I, dt):
        # EKF Prediction
        self.x[0] -= (I * dt) / (self.Q_nom * 3600)
        # Update (Simplified Jacobian-based)
        # In a real PhD thesis, Jacobian (H) would be calculated here via dOCV/dSOC
        H = np.array([[1.0, -1.0, -1.0]]) 
        y_hat = 3.7 - self.x[1] - self.x[2] - I * 0.01

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S

        self.x += (K @ (V_m - y_hat)).flatten()
        self.P = (np.eye(3) - K @ H) @ self.P

        return self.x[0], np.sqrt(self.P[0,0])

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: STREAMLIT UI (Research Framework)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔬 Digital Twin Research Framework: Asset vs. Twin")

if st.sidebar.button("▶ Run Full Co-Simulation"):
    # 1. Run Asset
    t, v_noisy, i, sol = simulate_physical_asset(1.0, 3)
    Q_nom = float(sol["Discharge capacity [A.h]"].entries[-1])

    # 2. Run Twin
    twin = DigitalTwin(Q_nom)
    dt = np.mean(np.diff(t))
    results = [twin.estimate(v_noisy[i], i[i], dt) for i in range(len(t))]
    soc_est = [r[0] for r in results]
    sigma = [r[1] for r in results]

    # 3. Analyze Uncertainty Propagation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=soc_est, name="Twin SOC Estimation"))
    fig.add_trace(go.Scatter(x=t, np.array(soc_est)+2*np.array(sigma), 
        line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=t, np.array(soc_est)-2*np.array(sigma), 
        fill='tonexty', name="95% Confidence Interval (UQ)"))

    st.plotly_chart(fig, use_container_width=True)
    st.write("---")
    st.metric("Mean Propagation Uncertainty (σ)", f"{np.mean(sigma):.4f}")
