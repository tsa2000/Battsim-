import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# 1. PHYSICAL ASSET: 50-CYCLE DFN ENGINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_multi_cycle_asset(cycles, noise):
    model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues("Chen2020")
    # محاكاة 50 دورة كاملة شحن وتفريغ
    exp = pybamm.Experiment(["Discharge at 1C until 2.5V", "Charge at 1C until 4.2V"] * cycles)
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sol = sim.solve()
    
    t = sol["Time [s]"].entries
    v_meas = sol["Terminal voltage [V]"].entries + np.random.normal(0, noise, len(t))
    i_meas = sol["Current [A]"].entries
    return t, v_meas, i_meas, float(params["Nominal cell capacity [A.h]"])

# ─────────────────────────────────────────────────────────────────────────────
# 2. DIGITAL TWIN: AEKF ENGINE (Adaptive State Estimation)
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveTwin:
    def __init__(self, Q_nom):
        self.Q_nom = Q_nom
        self.x = np.array([1.0, 0.0, 0.0]) # [SOC, V1, V2]
        self.P = np.diag([0.01, 1e-4, 1e-4]) 
        self.Q = np.diag([1e-7, 1e-8, 1e-8]) 
        self.R = np.array([[1e-3]])

    def step(self, V_m, I, dt):
        # Adaptive Parameter: المقاومة تتغير مع الـ SOC
        R0 = 0.05 + 0.02 * (1 - self.x[0])
        
        # Predict
        self.x[0] -= (I * dt) / (self.Q_nom * 3600)
        self.P = self.P + self.Q
        
        # Update
        H = np.array([[1.0, -1.0, -1.0]])
        V_est = 3.7 - self.x[1] - self.x[2] - I * R0
        nu = V_m - V_est
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S[0,0]
        
        self.x += (K @ nu).flatten()
        self.P = (np.eye(3) - K.reshape(3,1) @ H) @ self.P
        
        # Uncertainty Propagation
        return self.x[0], np.sqrt(self.P[0,0])

# ─────────────────────────────────────────────────────────────────────────────
# 3. STREAMLIT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Battery Digital Twin", layout="wide")
st.title("🔋 Research-Grade 50-Cycle Digital Twin")

with st.sidebar:
    st.header("Settings")
    cycles = st.number_input("Number of Cycles", 1, 50, 50)
    noise = st.slider("Sensor Noise", 0.0, 0.05, 0.005, 0.001)
    run_btn = st.button("▶ Execute Co-Simulation")

if run_btn:
    with st.spinner("Processing 50 cycles (Physics + Twin)..."):
        t, v_m, i, Q_nom = run_multi_cycle_asset(cycles, noise)
        twin = AdaptiveTwin(Q_nom)
        
        dt = 1.0 
        results = [twin.step(v_m[k], i[k], dt) for k in range(len(t))]
        
        soc_est = np.array([r[0] for r in results])
        uncertainty = np.array([r[1] for r in results])

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=soc_est, name="Estimated SOC"))
        fig.add_trace(go.Scatter(x=t, y=soc_est + 2*uncertainty, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=t, y=soc_est - 2*uncertainty, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', name="95% CI"))
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Successfully simulated {cycles} cycles with adaptive estimation.")
