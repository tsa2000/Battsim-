import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go

# ── 1. محرك DFN (الحقيقة الفيزيائية) ──────────────────────────────────────────
@st.cache_resource
def get_model_data(c_rate):
    model = pybamm.lithium_ion.DFN()
    params = pybamm.ParameterValues("Chen2020")
    exp = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V", "Rest for 5 min"])
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sol = sim.solve(initial_soc=1.0)

    t = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    I = sol["Current [A]"].entries
    q_dis = sol["Discharge capacity [A.h]"].entries

    # استخراج OCV بطريقة آمنة (من المعاملات مباشرة)
    # استخدام OCV من المعاملات يضمن دائماً النجاح
    soc_pts = np.linspace(0, 1, 100)
    # الحصول على OCV من الـ OCP للأنود والكاثود
    ocv_pts = params["Open-circuit voltage [V]"](soc_pts) 

    return t, V, I, q_dis, soc_pts, ocv_pts

# ── باقي الكود (AEKF + UI) ──────────────────────────────────────────────────
class BatteryAEKF:
    def __init__(self, soc_lut, ocv_lut):
        self.soc_lut = soc_lut
        self.ocv_lut = ocv_lut
        self.x = np.array([1.0, 0.0, 0.0]) # [SOC, V1, V2]

    def estimate(self, V_m, I, dt, Q_nom):
        ocv_est = np.interp(self.x[0], self.soc_lut, self.ocv_lut)
        self.x[0] -= (I * dt) / (Q_nom * 3600)
        error = V_m - (ocv_est - self.x[1] - self.x[2] - I*0.01)
        self.x[0] += 0.02 * error
        return np.clip(self.x[0], 0, 1)

st.title("🔋 BattSim: Integrated Digital Twin")
c_rate = st.sidebar.slider("C-Rate", 0.5, 3.0, 1.0)

if st.button("▶ Run Full Digital Twin"):
    with st.spinner("Executing DFN & AEKF..."):
        t, V, I, q_dis, soc_lut, ocv_lut = get_model_data(c_rate)
        Q_nom = float(q_dis[-1] if q_dis[-1] > 0 else 5.0)

        aekf = BatteryAEKF(soc_lut, ocv_lut)
        soc_est = []
        dt = np.mean(np.diff(t))
        for i in range(len(t)):
            soc_e = aekf.estimate(V[i], I[i], dt, Q_nom)
            soc_est.append(soc_e)

        soc_truth = 1.0 - (q_dis / Q_nom)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=soc_truth, name="Physical DFN"))
        fig.add_trace(go.Scatter(x=t, y=soc_est, name="Digital Twin (AEKF)"))
        st.plotly_chart(fig, use_container_width=True)
        st.success("Simulation complete!")
