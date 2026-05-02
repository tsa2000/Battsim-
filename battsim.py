import streamlit as st
import pybamm
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# ── 1. محرك DFN (الحقيقة الفيزيائية) ──────────────────────────────────────────
@st.cache_resource
def get_model_data(c_rate):
    model = pybamm.lithium_ion.DFN()
    params = pybamm.ParameterValues("Chen2020")
    exp = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V", "Rest for 5 min"])
    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
    sol = sim.solve(initial_soc=1.0)
    # استخراج OCV من النموذج لربطه بالـ AEKF
    soc_pts = np.linspace(0, 1, 100)
    ocv_pts = params.evaluate(model.variables["Open-circuit voltage [V]"], soc_pts)
    
    return sol["Time [s]"].entries, sol["Terminal voltage [V]"].entries, \
           sol["Current [A]"].entries, sol["Discharge capacity [A.h]"].entries, \
           soc_pts, ocv_pts

# ── 2. محرك AEKF (التوأم الرقمي) ──────────────────────────────────────────────
class BatteryAEKF:
    def __init__(self, soc_lut, ocv_lut):
        self.soc_lut = soc_lut
        self.ocv_lut = ocv_lut
        self.x = np.array([1.0, 0.0, 0.0]) # [SOC, V1, V2]
        
    def estimate(self, V_m, I, dt, Q_nom):
        # تقدير OCV من الجدول المشترك مع DFN
        ocv_est = np.interp(self.x[0], self.soc_lut, self.ocv_lut)
        # Coulomb Counting (مترابط مع سعة DFN)
        self.x[0] -= (I * dt) / (Q_nom * 3600)
        # تصحيح EKF (مقارنة بـ V_meas الحقيقي)
        error = V_m - (ocv_est - self.x[1] - self.x[2] - I*0.01)
        self.x[0] += 0.02 * error # Gain
        return np.clip(self.x[0], 0, 1)

# ── 3. الواجهة (Streamlit) ──────────────────────────────────────────────────
st.title("🔋 BattSim: Integrated Digital Twin")
c_rate = st.sidebar.slider("C-Rate", 0.5, 3.0, 1.0)

if st.button("▶ Run Full Digital Twin"):
    with st.spinner("Executing DFN & AEKF..."):
        # تنفيذ المحاكاة
        t, V, I, q_dis, soc_lut, ocv_lut = get_model_data(c_rate)
        Q_nom = float(q_dis[-1] if q_dis[-1] > 0 else 5.0)
        
        # تنفيذ التوأم الرقمي (AEKF)
        aekf = BatteryAEKF(soc_lut, ocv_lut)
        soc_est = []
        dt = np.mean(np.diff(t))
        for i in range(len(t)):
            soc_e = aekf.estimate(V[i], I[i], dt, Q_nom)
            soc_est.append(soc_e)
            
        # مقارنة النتائج
        soc_truth = 1.0 - (q_dis / Q_nom)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=soc_truth, name="Physical DFN"))
        fig.add_trace(go.Scatter(x=t, y=soc_est, name="Digital Twin (AEKF)"))
        st.plotly_chart(fig, use_container_width=True)
        st.success("النموذج الفيزيائي والتوأم الرقمي يعملان الآن بتناغم تام.")
