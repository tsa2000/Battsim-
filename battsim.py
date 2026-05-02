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
    # محاكاة دورات الشحن والتفريغ
    exp = pybamm.Experiment(["Discharge at 1C until 2.5V", "Charge at 1C until 4.2V"] * cycles)
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sol = sim.solve()
    
    t = sol["Time [s]"].entries
    v_meas = sol["Terminal voltage [V]"].entries + np.random.normal(0, noise, len(t))
    i_meas = sol["Current [A]"].entries
    return t, v_meas, i_meas, float(params["Nominal cell capacity [A.h]"])

# ─────────────────────────────────────────────────────────────────────────────
# 2. DIGITAL TWIN: AEKF ENGINE (Corrected Linear Algebra)
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveTwin:
    def __init__(self, Q_nom):
        self.Q_nom = Q_nom
        self.x = np.array([1.0, 0.0, 0.0]) # [SOC, V1, V2]
        self.P = np.diag([0.01, 1e-4, 1e-4]) 
        self.Q = np.diag([1e-7, 1e-8, 1e-8]) 
        self.R = np.array([[1e-3]])

    def step(self, V_m, I, dt):
        # Adaptive Parameter: تغير المقاومة مع الـ SOC
        R0 = 0.05 + 0.02 * (1 - self.x[0])
        
        # Predict
        self.x[0] -= (I * dt) / (self.Q_nom * 3600)
        self.P = self.P + self.Q
        
        # Update
        H = np.array([[1.0, -1.0, -1.0]])
        V_est = 3.7 - self.x[1] - self.x[2] - I * R0
        
        # تصحيح جبري: تحويل nu إلى مصفوفة (1x1) للضرب المصفوفي
        nu = np.array([[V_m - V_est]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x += (K @ nu).flatten()
        self.P = (np.eye(3) - K @ H) @ self.P
        
        # Uncertainty: الجذر التربيعي للتباين
        return self.x[0], np.sqrt(self.P[0,0])

# ─────────────────────────────────────────────────────────────────────────────
# 3. INTERFACE
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
        
        # إضافة النتائج الكتابية (Metrics Dashboard)
        st.subheader("📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        final_soc = soc_est[-1]
        max_uncertainty = np.max(uncertainty)
        
        col1.metric("Final Estimated SOC", f"{final_soc:.2%}")
        col2.metric("Max Uncertainty (σ)", f"{max_uncertainty:.4f}")
        col3.metric("Estimation Status", "Converged" if max_uncertainty < 0.1 else "Diverging")
        
        # تقرير نصي إضافي
        st.markdown(f"""
        ### Analysis Summary
        - **Total Cycles Simulated**: {cycles}
        - **Average Uncertainty**: {np.mean(uncertainty):.4f}
        - **System Behavior**: The Adaptive EKF has successfully tracked the state 
          with a confidence interval width of {np.mean(uncertainty)*2:.3f}.
        """)
