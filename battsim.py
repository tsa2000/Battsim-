
import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

@st.cache_data
def run_dfn_truth(c_rate):
    model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues("Chen2020")
    exp = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V", "Rest for 10 min"])
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sol = sim.solve(initial_soc=1.0)

    t = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    I = sol["Current [A]"].entries
    q_dis = sol["Discharge capacity [A.h]"].entries
    soc = 1.0 - (q_dis / float(q_dis[-1] if q_dis[-1] > 0 else 5.0))

    # إصلاح استخراج OCV: نستخدم المتغيرات من الـ model المحلول
    # هذا يضمن توافق الاسم مع الإصدار الحالي
    soc_pts = np.linspace(0.1, 0.9, 50)
    # الحصول على OCV كـ processed variable
    ocv_fn = sol.get_processed_variable("Open-circuit voltage [V]")
    ocv_pts = ocv_fn(soc_pts)
    dOCV_dSOC = np.gradient(ocv_pts, soc_pts)

    return t, V, I, soc, soc_pts, ocv_pts, dOCV_dSOC

class ResearchTwin:
    def __init__(self, soc_lut, ocv_lut, dOCV_dSOC):
        self.dOCV_fn = interp1d(soc_lut, dOCV_dSOC, kind='linear', fill_value="extrapolate")
        self.ocv_fn  = interp1d(soc_lut, ocv_lut, kind='cubic', fill_value="extrapolate")
        self.x = np.array([1.0, 0.0, 0.0])
        self.P = np.diag([0.01, 1e-4, 1e-4])
        self.Q = np.diag([1e-6, 1e-6, 1e-6])
        self.R = 1e-3

    def step(self, V_meas, I, dt):
        H = np.array([[float(self.dOCV_fn(np.clip(self.x[0], 0.01, 0.99))), -1.0, -1.0]])
        self.x[0] -= (I * dt) / (5.0 * 3600)
        Pp = self.P + self.Q
        ocv = float(self.ocv_fn(np.clip(self.x[0], 0.01, 0.99)))
        V_est = ocv - self.x[1] - self.x[2] - I * 0.01
        nu = V_meas - V_est
        S = H @ Pp @ H.T + self.R
        K = Pp @ H.T / S
        self.x += (K @ nu).flatten()
        self.P = (np.eye(3) - K @ H) @ Pp
        return self.x[0], np.sqrt(self.P[0,0]), nu**2 / S

st.set_page_config(page_title="BattSim Research", layout="wide")
st.title("🔋 BattSim: Research-Grade Digital Twin")

if st.sidebar.button("▶ Start Research Sim"):
    with st.spinner("Solving Physics..."):
        t, V, I, soc_truth, soc_l, ocv_l, dOCV = run_dfn_truth(1.0)
    with st.spinner("Twin Estimating..."):
        twin = ResearchTwin(soc_l, ocv_l, dOCV)
        dt = np.mean(np.diff(t))
        results = [twin.step(V[i], I[i], dt) for i in range(len(t))]
        soc_est = [r[0] for r in results]
        sigma   = [r[1] for r in results]
        nis     = [r[2] for r in results]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=soc_truth, name="Physical DFN"))
    fig.add_trace(go.Scatter(x=t, y=soc_est, name="AEKF Twin", line=dict(dash="dash")))
    st.plotly_chart(fig, use_container_width=True)
