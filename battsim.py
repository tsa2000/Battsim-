
import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────────────────
# 1. المادية (Asset): محرك DFN عالي الدقة
# ─────────────────────────────────────────────────────────────────────────────
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
    soc = 1.0 - (q_dis / float(q_dis[-1]))

    # استخراج OCV للبحث
    soc_pts = np.linspace(0.01, 0.99, 50)
    ocv_pts = params["Open-circuit voltage [V]"](soc_pts)
    dOCV_dSOC = np.gradient(ocv_pts, soc_pts)

    return t, V, I, soc, soc_pts, ocv_pts, dOCV_dSOC

# ─────────────────────────────────────────────────────────────────────────────
# 2. التوأم (Twin): EKF مع Adaptive Jacobian
# ─────────────────────────────────────────────────────────────────────────────
class ResearchTwin:
    def __init__(self, soc_lut, ocv_lut, dOCV_dSOC):
        self.dOCV_fn = interp1d(soc_lut, dOCV_dSOC, kind='linear')
        self.ocv_fn  = interp1d(soc_lut, ocv_lut, kind='cubic')
        self.x = np.array([1.0, 0.0, 0.0]) # [SOC, V1, V2]
        self.P = np.diag([0.01, 1e-4, 1e-4])
        self.Q = np.diag([1e-6, 1e-6, 1e-6])
        self.R = 1e-3

    def step(self, V_meas, I, dt):
        # 1. Jacobian (Physics-informed H_k)
        H = np.array([[float(self.dOCV_fn(np.clip(self.x[0], 0.01, 0.99))), -1.0, -1.0]])

        # 2. Predict
        self.x[0] -= (I * dt) / (5.0 * 3600)
        Pp = self.P + self.Q

        # 3. Update
        ocv = float(self.ocv_fn(np.clip(self.x[0], 0.01, 0.99)))
        V_est = ocv - self.x[1] - self.x[2] - I * 0.01
        nu = V_meas - V_est
        S = H @ Pp @ H.T + self.R
        K = Pp @ H.T / S
        self.x += (K @ nu).flatten()
        self.P = (np.eye(3) - K @ H) @ Pp

        return self.x[0], np.sqrt(self.P[0,0]), nu**2 / S # SOC, Sigma, NIS

# ─────────────────────────────────────────────────────────────────────────────
# 3. الواجهة (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
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

    # عرض النتائج البحثية
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=soc_truth, name="Physical DFN"))
    fig.add_trace(go.Scatter(x=t, y=soc_est, name="AEKF Twin", line=dict(dash="dash")))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Avg NIS (Consistency)", f"{np.mean(nis):.3f}")
    col2.metric("RMSE SOC", f"{np.sqrt(np.mean((np.array(soc_est)-soc_truth)**2))*100:.2f}%")
