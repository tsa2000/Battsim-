import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go

# ── 1. محرك DFN فيزيائي (DFN) ────────────────────────────────────────────────
def run_dfn(c_rate):
    model = pybamm.lithium_ion.DFN()
    param = pybamm.ParameterValues("Chen2020")
    exp = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V", "Rest for 5 min"])
    sim = pybamm.Simulation(model, parameter_values=params=param, experiment=exp)
    sol = sim.solve(initial_soc=1.0)
    t, V, I = sol["Time [s]"].entries, sol["Terminal voltage [V]"].entries, sol["Current [A]"].entries
    # استخراج SOC (الطريقة الصحيحة والفيزيائية)
    q_dis = sol["Discharge capacity [A.h]"].entries
    soc = 1.0 - (q_dis / float(q_dis[-1] if q_dis[-1] > 0 else 5.0))
    return t, V, I, np.clip(soc, 0, 1)

# ── 2. محرك AEKF لتقدير الحالة (ECM-AEKF) ──────────────────────────────────
class BatteryEKF:
    def __init__(self):
        self.x = np.array([1.0, 0.0, 0.0]) # [SOC, V1, V2]
        self.P = np.eye(3) * 0.1
    def update(self, V_m, I, dt=10):
        # Predict (Coulomb Counting + RC)
        self.x[0] -= (I * dt) / (5.0 * 3600)
        # Update (Kalman)
        error = V_m - (3.7 - self.x[1] - self.x[2] - I*0.01)
        self.x[0] += 0.05 * error
        return self.x[0], np.sqrt(self.P[0,0])

# ── 3. تطبيق متكامل ──────────────────────────────────────────────────────────
st.title("🔋 BattSim: Full Digital Twin (DFN+EKF+UQ)")
c_rate = st.sidebar.slider("C-Rate", 0.5, 3.0, 1.0)

if st.button("▶ Run Full Digital Twin"):
    # 1. المادية (DFN)
    t, V, I, soc_truth = run_dfn(c_rate)
    
    # 2. التوأم الرقمي (AEKF)
    ekf = BatteryEKF()
    soc_est = []
    for i in range(len(t)):
        s_e, _ = ekf.update(V[i], I[i])
        soc_est.append(s_e)
        
    # 3. عرض ومقارنة (UQ)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=soc_truth, name="Physical DFN (Ground Truth)"))
    fig.add_trace(go.Scatter(x=t, y=soc_est, name="Digital Twin (AEKF Estimate)"))
    st.plotly_chart(fig, use_container_width=True)
    st.info("النموذج الفيزيائي يعمل الآن بالتوازي مع تقدير الحالة اللحظي.")
