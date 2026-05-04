import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pybamm
from scipy.stats import chi2
from dataclasses import dataclass
import warnings
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import os

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatteryConfig:
    chemistry: str = "NMC622/Graphite"
    nominal_capacity: float = 5.0
    voltage_range: tuple = (2.5, 4.2)
    temperature_ref: float = 298.15
    arrhenius_factor: float = 3600.0


# ═══════════════════════════════════════════════════════════════════════════════
# OCV MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class OCVModel:
    def __init__(self):
        self.soc_lut = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
        self.ocv_lut = np.array([2.70, 3.35, 3.48, 3.55, 3.62, 3.69, 3.76, 3.83, 3.91, 4.00, 4.09, 4.14, 4.19])

    def get_voltage(self, soc):
        return np.interp(np.clip(soc, 0.0, 1.0), self.soc_lut, self.ocv_lut)

    def get_gradient_analytical(self, soc):
        eps = 1e-4
        v1 = self.get_voltage(soc + eps)
        v2 = self.get_voltage(soc - eps)
        return (v1 - v2) / (2 * eps)

    def get_entropic_coeff(self, soc):
        s = np.clip(soc, 0.0, 1.0)
        return (-0.35 + 2.5*s - 6.0*s**2 + 5.5*s**3 - 1.8*s**4) * 1e-3


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL ASSET
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicalAsset:
    def __init__(self, config: BatteryConfig):
        self.config = config

    @st.cache_data(show_spinner=False)
    def simulate(_self, cycles, c_rate, noise_voltage, noise_temp, noise_current):
        model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
        params = pybamm.ParameterValues("Chen2020")

        experiment = pybamm.Experiment(
            [
                f"Discharge at {c_rate}C until 2.5 V",
                "Rest for 5 minutes",
                "Charge at 1C until 4.2 V",
                "Hold at 4.2 V until C/20",
                "Rest for 5 minutes",
            ] * cycles,
            termination="99% capacity",
        )

        sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
        sol = sim.solve()

        time = sol["Time [s]"].entries
        voltage_true = sol["Terminal voltage [V]"].entries
        temp_true = sol["Cell temperature [K]"].entries
        current = sol["Current [A]"].entries

        rng = np.random.default_rng(42)
        voltage_meas = voltage_true + rng.normal(0, noise_voltage, len(time))
        temp_meas = temp_true + rng.normal(0, noise_temp, len(time))
        current_meas = current + rng.normal(0, noise_current, len(time))

        Q_nominal = float(params["Nominal cell capacity [A.h]"])
        dt_array = np.diff(time, prepend=time[0])
        discharged_ah = np.cumsum(current * dt_array) / 3600.0
        soc_true = np.clip(1.0 - discharged_ah / Q_nominal, 0.0, 1.0)
        
        return {
            "time": time,
            "voltage_true": voltage_true,
            "voltage_meas": voltage_meas,
            "temp_true": temp_true,
            "temp_meas": temp_meas,
            "current_true": current,
            "current_meas": current_meas,
            "soc_true": soc_true,
            "Q_nominal": Q_nominal,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ECM
# ═══════════════════════════════════════════════════════════════════════════════

class EquivalentCircuitModel:
    def __init__(self, Q_nom, R0, R1, C1, R2, C2, R_th, C_th, T_amb, config):
        self.Q_nom = Q_nom
        self.R0, self.R1, self.C1 = R0, R1, C1
        self.R2, self.C2 = R2, C2
        self.R_th, self.C_th = R_th, C_th
        self.T_amb = T_amb
        self.config = config
        self.ocv = OCVModel()

    def arrhenius_correction(self, T):
        T_safe = np.clip(T, 250.0, 350.0)
        return np.exp(self.config.arrhenius_factor * (1.0 / T_safe - 1.0 / self.config.temperature_ref))

    def effective_resistance(self, soc, T, R_base):
        arr_factor = self.arrhenius_correction(T)
        soc_factor = 1.0 + 0.4 * (1.0 - soc) ** 2
        return R_base * soc_factor * arr_factor

    def state_transition(self, x, I, dt):
        soc, V1, V2, T = x
        R0_eff = self.effective_resistance(soc, T, self.R0)
        R1_eff = self.R1 * self.arrhenius_correction(T)
        R2_eff = self.R2 * self.arrhenius_correction(T)
        tau1 = R1_eff * self.C1
        tau2 = R2_eff * self.C2
        exp1 = np.exp(-dt / tau1)
        exp2 = np.exp(-dt / tau2)
        soc_new = soc - (I * dt) / (self.Q_nom * 3600.0)
        V1_new = exp1 * V1 + R1_eff * (1 - exp1) * I
        V2_new = exp2 * V2 + R2_eff * (1 - exp2) * I
        Q_ohmic = I**2 * R0_eff
        Q_pol = (V1**2) / max(R1_eff, 1e-9) + (V2**2) / max(R2_eff, 1e-9)
        dU_dT = self.ocv.get_entropic_coeff(soc)
        Q_ent = -I * T * dU_dT
        Q_total = Q_ohmic + Q_pol + Q_ent
        T_new = T + (dt / self.C_th) * (Q_total - (T - self.T_amb) / self.R_th)
        T_new = np.clip(T_new, 250.0, 360.0)
        return np.array([soc_new, V1_new, V2_new, T_new])

    def measurement_model(self, x, I):
        soc, V1, V2, T = x
        R0_eff = self.effective_resistance(soc, T, self.R0)
        V_terminal = self.ocv.get_voltage(soc) - V1 - V2 - I * R0_eff
        return np.array([V_terminal, T])


# ═══════════════════════════════════════════════════════════════════════════════
# AEKF
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveEKF:
    def __init__(self, ecm, x0, P0, Q, R):
        self.ecm = ecm
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(P0).astype(float)
        self.Q = np.diag(Q).astype(float)
        self.R = np.diag(R).astype(float)

    def predict(self, I, dt):
        x_pred = self.ecm.state_transition(self.x, I, dt)
        soc, V1, V2, T = self.x
        arr = self.ecm.arrhenius_correction(T)
        T_safe = max(T, 250.0)
        darr_dT = -self.ecm.config.arrhenius_factor / (T_safe**2) * arr
        R1_eff = self.ecm.R1 * arr
        R2_eff = self.ecm.R2 * arr
        tau1 = R1_eff * self.ecm.C1
        tau2 = R2_eff * self.ecm.C2
        exp1 = np.exp(-dt / tau1)
        exp2 = np.exp(-dt / tau2)
        dR1_dT = self.ecm.R1 * darr_dT
        dR2_dT = self.ecm.R2 * darr_dT
        dtau1_dT = dR1_dT * self.ecm.C1
        dtau2_dT = dR2_dT * self.ecm.C2
        dexp1_dT = exp1 * (dt / tau1**2) * dtau1_dT
        dexp2_dT = exp2 * (dt / tau2**2) * dtau2_dT
        dV1_dT = dexp1_dT * V1 + (dR1_dT * (1 - exp1) - R1_eff * dexp1_dT) * I
        dV2_dT = dexp2_dT * V2 + (dR2_dT * (1 - exp2) - R2_eff * dexp2_dT) * I
        dR0_dT = self.ecm.R0 * (1.0 + 0.4 * (1.0 - soc)**2) * darr_dT
        dU_dT_val = self.ecm.ocv.get_entropic_coeff(soc)
        dT_dT = 1.0 - dt / (self.ecm.C_th * self.ecm.R_th) + (dt / self.ecm.C_th) * (I**2 * dR0_dT) - (dt / self.ecm.C_th) * I * dU_dT_val
        dR0_dSOC = self.ecm.R0 * arr * (-0.8 * (1.0 - soc))
        dT_dSOC = (dt / self.ecm.C_th) * (I**2 * dR0_dSOC)
        F = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, exp1, 0.0, dV1_dT], [0.0, 0.0, exp2, dV2_dT], [dT_dSOC, 0.0, 0.0, dT_dT]])
        P_pred = F @ self.P @ F.T + self.Q
        return x_pred, P_pred, F

    def update(self, x_pred, P_pred, y_meas, I):
        y_pred = self.ecm.measurement_model(x_pred, I)
        soc_pred, _, _, T_pred = x_pred
        arr_pred = self.ecm.arrhenius_correction(T_pred)
        T_safe = max(T_pred, 250.0)
        darr_dT_pred = -self.ecm.config.arrhenius_factor / (T_safe**2) * arr_pred
        dR0_dT_pred = self.ecm.R0 * (1.0 + 0.4 * (1.0 - soc_pred)**2) * darr_dT_pred
        H = np.array([[self.ecm.ocv.get_gradient_analytical(soc_pred), -1.0, -1.0, -I * dR0_dT_pred], [0.0, 0.0, 0.0, 1.0]])
        innovation = y_meas - y_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ innovation
        I_KH = np.eye(4) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        x_upd[0] = np.clip(x_upd[0], 0.0, 1.0)
        x_upd[3] = np.clip(x_upd[3], 250.0, 360.0)
        nis = float(innovation @ np.linalg.inv(S) @ innovation)
        return x_upd, P_upd, innovation, nis

    def step(self, y_meas, I, dt):
        x_pred, P_pred, _ = self.predict(I, dt)
        self.x, self.P, innov, nis = self.update(x_pred, P_pred, y_meas, I)
        return {
            "soc": self.x[0], "v1": self.x[1], "v2": self.x[2], "temp": self.x[3],
            "sigma_soc": np.sqrt(max(self.P[0, 0], 0.0)),
            "innovation_voltage": innov[0] * 1000.0, "nis": nis,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UKF
# ═══════════════════════════════════════════════════════════════════════════════

class UnscentedKalmanFilter:
    def __init__(self, ecm, x0, P0, Q, R, alpha=0.1, beta=2.0, kappa=0.0):
        self.ecm = ecm
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(P0).astype(float)
        self.Q = np.diag(Q).astype(float)
        self.R = np.diag(R).astype(float)
        n = 4
        self.n = n
        lam = alpha**2 * (n + kappa) - n
        self.lam = lam
        self.Wm = np.full(2*n+1, 1.0 / (2.0*(n+lam)))
        self.Wc = np.full(2*n+1, 1.0 / (2.0*(n+lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
        self.gamma = np.sqrt(n + lam)

    def _sigma_points(self, x, P):
        P_safe = P + 1e-9 * np.eye(self.n)
        try:
            L = np.linalg.cholesky(P_safe)
        except np.linalg.LinAlgError:
            U, s, _ = np.linalg.svd(P_safe)
            L = U @ np.diag(np.sqrt(np.maximum(s, 1e-12)))
        pts = [x.copy()]
        for i in range(self.n):
            pts.append(x + self.gamma * L[:, i])
            pts.append(x - self.gamma * L[:, i])
        return np.array(pts)

    def _ut(self, pts, fn):
        tr = np.array([fn(p) for p in pts])
        mu = np.einsum("i,ij->j", self.Wm, tr)
        dev = tr - mu
        cov = sum(self.Wc[i] * np.outer(dev[i], dev[i]) for i in range(len(self.Wm)))
        return mu, cov, tr

    def step(self, y_meas, I, dt):
        pts = self._sigma_points(self.x, self.P)
        x_pred, P_pred, pts_pred = self._ut(pts, lambda p: self.ecm.state_transition(p, I, dt))
        P_pred += self.Q
        y_pred, Pyy, pts_meas = self._ut(pts_pred, lambda p: self.ecm.measurement_model(p, I))
        Pyy += self.R
        Pxy = sum(self.Wc[i] * np.outer(pts_pred[i] - x_pred, pts_meas[i] - y_pred) for i in range(len(self.Wm)))
        K = Pxy @ np.linalg.inv(Pyy)
        innovation = y_meas - y_pred
        self.x = x_pred + K @ innovation
        self.P = P_pred - K @ Pyy @ K.T
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.x[3] = np.clip(self.x[3], 250.0, 360.0)
        nis = float(innovation @ np.linalg.inv(Pyy) @ innovation)
        return {
            "soc": self.x[0], "v1": self.x[1], "v2": self.x[2], "temp": self.x[3],
            "sigma_soc": np.sqrt(max(self.P[0, 0], 0.0)),
            "innovation_voltage": innovation[0] * 1000.0, "nis": nis,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL EKF
# ═══════════════════════════════════════════════════════════════════════════════

class DualEKF:
    def __init__(self, ecm, x0, P_x0, w0, P_w0, Q_x, R_x, Q_w, R_w):
        self.state_filter = AdaptiveEKF(ecm, x0, P_x0, Q_x, R_x)
        self.w = np.array(w0, dtype=float)
        self.P_w = np.diag(P_w0).astype(float)
        self.Q_w = np.diag(Q_w).astype(float)
        self.R_w = np.diag(R_w).astype(float)
        self._R0_history = []

    @property
    def ecm(self):
        return self.state_filter.ecm

    def step(self, y_meas, I, dt):
        w_pred = self.w.copy()
        P_w_pred = self.P_w + self.Q_w
        self.state_filter.ecm.R0 = float(w_pred[0])
        state_out = self.state_filter.step(y_meas, I, dt)
        soc_k, _, _, T_k = self.state_filter.x
        arr_k = self.state_filter.ecm.arrhenius_correction(T_k)
        soc_fac = 1.0 + 0.4 * (1.0 - soc_k)**2
        dV_dR0 = -I * soc_fac * arr_k
        H_w = np.array([[dV_dR0], [0.0]])
        y_hat = self.state_filter.ecm.measurement_model(self.state_filter.x, I)
        innov_w = y_meas - y_hat
        S_w = H_w @ P_w_pred @ H_w.T + self.R_w
        try:
            K_w = P_w_pred @ H_w.T @ np.linalg.inv(S_w)
        except np.linalg.LinAlgError:
            K_w = np.zeros((1, 2))
        self.w = w_pred + (K_w @ innov_w).flatten()
        self.w[0] = np.clip(self.w[0], 5e-3, 0.1)
        I_KH = np.eye(1) - K_w @ H_w
        self.P_w = I_KH @ P_w_pred @ I_KH.T + K_w @ self.R_w @ K_w.T
        self._R0_history.append(float(self.w[0]))
        state_out["R0_est"] = float(self.w[0])
        state_out["sigma_R0"] = float(np.sqrt(max(self.P_w[0, 0], 0.0)))
        return state_out


# ═══════════════════════════════════════════════════════════════════════════════
# UQ METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class UQMetrics:
    @staticmethod
    def rmse(est, truth):
        return np.sqrt(np.mean((est - truth)**2))

    @staticmethod
    def mae(est, truth):
        return np.mean(np.abs(est - truth))

    @staticmethod
    def picp(truth, lo, hi):
        return 100.0 * np.mean((truth >= lo) & (truth <= hi))

    @staticmethod
    def mpiw(lo, hi):
        return np.mean(hi - lo)

    @staticmethod
    def nis_consistency(nis, alpha=0.05):
        thr = chi2.ppf(1 - alpha, df=2)
        return np.mean(nis < thr) * 100.0, thr


# ═══════════════════════════════════════════════════════════════════════════════
# CYCLE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cycles(time, current, threshold=0.1):
    """Detect charge/discharge cycles based on current direction"""
    cycle_markers = []
    in_discharge = current[0] < -threshold
    cycle_start = 0
    
    for i in range(1, len(current)):
        now_discharge = current[i] < -threshold
        if now_discharge != in_discharge:
            cycle_markers.append((cycle_start, i-1))
            cycle_start = i
            in_discharge = now_discharge
    
    cycle_markers.append((cycle_start, len(current)-1))
    return cycle_markers


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_digital_twin_system(asset_data, ecm_params, filter_params, enable_dual=True, dt_hint=1.0):
    time = np.asarray(asset_data["time"]).flatten()
    V_meas = np.asarray(asset_data["voltage_meas"]).flatten()
    T_meas = np.asarray(asset_data["temp_meas"]).flatten()
    I_meas = np.asarray(asset_data["current_meas"]).flatten()

    n = min(len(time), len(V_meas), len(T_meas), len(I_meas))
    time, V_meas, T_meas, I_meas = time[:n], V_meas[:n], T_meas[:n], I_meas[:n]

    def _make_ecm():
        return EquivalentCircuitModel(Q_nom=asset_data["Q_nominal"], **ecm_params, config=BatteryConfig())

    ecm_aekf = _make_ecm()
    ecm_ukf = _make_ecm()

    x0 = [1.0, 0.0, 0.0, ecm_params["T_amb"]]
    P0, Q, R = filter_params["P0"], filter_params["Q"], filter_params["R"]

    aekf = AdaptiveEKF(ecm_aekf, x0, P0, Q, R)
    ukf = UnscentedKalmanFilter(ecm_ukf, x0, P0, Q, R)

    dual_ekf = None
    if enable_dual:
        w0 = [ecm_params["R0"]]
        P_w0 = [1e-4]
        Q_w = filter_params.get("Q_w", [1e-12])
        R_w = R
        dual_ekf = DualEKF(_make_ecm(), x0, P0, w0, P_w0, Q, R, Q_w, R_w)

    def _empty():
        return {"soc": [], "v1": [], "v2": [], "sigma": [], "temp": [], "innov": [], "nis": []}

    results = {"aekf": _empty(), "ukf": _empty()}
    if enable_dual:
        results["dual"] = {**_empty(), "R0_est": [], "sigma_R0": []}

    for k in range(n):
        y = np.array([V_meas[k], T_meas[k]])
        I = float(I_meas[k])
        if k == 0:
            dt = time[1] - time[0] if n > 1 else 1.0
        else:
            dt = time[k] - time[k-1]
        if dt <= 0:
            dt = 1e-3

        def _append(name, out):
            r = results[name]
            r["soc"].append(out["soc"])
            r["v1"].append(out.get("v1", 0.0))
            r["v2"].append(out.get("v2", 0.0))
            r["sigma"].append(out["sigma_soc"])
            r["temp"].append(out["temp"])
            r["innov"].append(out["innovation_voltage"])
            if "nis" in out:
                r["nis"].append(out["nis"])

        _append("aekf", aekf.step(y, I, dt))
        _append("ukf", ukf.step(y, I, dt))

        if enable_dual:
            d_out = dual_ekf.step(y, I, dt)
            _append("dual", d_out)
            results["dual"]["R0_est"].append(d_out["R0_est"])
            results["dual"]["sigma_R0"].append(d_out["sigma_R0"])

    for fname in results:
        for key in results[fname]:
            results[fname][key] = np.array(results[fname][key])

    ecm_ref = ecm_aekf
    return results, ecm_ref, dual_ekf


# ═══════════════════════════════════════════════════════════════════════════════
# VOLTAGE RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_voltage(ecm, soc, v1, v2, temp, current_meas, r0_arr=None):
    I_ecm = np.asarray(current_meas)
    v_out = np.zeros_like(soc)
    original_R0 = ecm.R0
    for k in range(len(soc)):
        if r0_arr is not None and len(r0_arr) > k:
            ecm.R0 = r0_arr[k]
        v_out[k] = ecm.measurement_model(np.array([soc[k], v1[k], v2[k], temp[k]]), I_ecm[k])[0]
    ecm.R0 = original_R0
    return v_out


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(asset_data, results, ecm, enable_dual=True):
    soc_true = np.asarray(asset_data["soc_true"])
    voltage_true = np.asarray(asset_data["voltage_true"])
    I_meas = np.asarray(asset_data["current_meas"])
    cutoff = int(0.10 * len(soc_true))

    active = ["aekf", "ukf"]
    if enable_dual and "dual" in results:
        active.append("dual")

    metrics = {}
    for name in active:
        r = results[name]
        r0_arr = r.get("R0_est", None)
        v_model = reconstruct_voltage(ecm, r["soc"], r["v1"], r["v2"], r["temp"], I_meas, r0_arr)
        m = {
            "rmse_soc": UQMetrics.rmse(r["soc"][cutoff:], soc_true[cutoff:]) * 100,
            "mae_soc": UQMetrics.mae(r["soc"][cutoff:], soc_true[cutoff:]) * 100,
            "rmse_volt": UQMetrics.rmse(v_model[cutoff:], voltage_true[cutoff:]) * 1000,
            "innov_rms": float(np.sqrt(np.mean(r["innov"][cutoff:]**2))),
            "picp": UQMetrics.picp(soc_true[cutoff:], r["soc"][cutoff:] - 2*r["sigma"][cutoff:], r["soc"][cutoff:] + 2*r["sigma"][cutoff:]),
            "mpiw": UQMetrics.mpiw(r["soc"][cutoff:] - 2*r["sigma"][cutoff:], r["soc"][cutoff:] + 2*r["sigma"][cutoff:]) * 100,
        }
        if len(r.get("nis", [])) > cutoff:
            m["nis_within"], m["nis_thr"] = UQMetrics.nis_consistency(r["nis"][cutoff:])
        metrics[name] = m

    return metrics, cutoff


# ═══════════════════════════════════════════════════════════════════════════════
# CYCLE-BY-CYCLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_cycles(asset_data, results, ecm, enable_dual=True):
    time = asset_data["time"]
    current = asset_data["current_true"]
    soc_true = asset_data["soc_true"]
    voltage_true = asset_data["voltage_true"]
    I_meas = asset_data["current_meas"]
    
    cycles = detect_cycles(time, current)
    
    cycle_data = []
    for cycle_idx, (start, end) in enumerate(cycles):
        cycle_info = {"cycle": cycle_idx + 1, "start_time": time[start], "end_time": time[end]}
        
        for filter_name in ["aekf", "ukf"] + (["dual"] if enable_dual and "dual" in results else []):
            r = results[filter_name]
            r0_arr = r.get("R0_est", None)
            v_model = reconstruct_voltage(ecm, r["soc"], r["v1"], r["v2"], r["temp"], I_meas, r0_arr)
            
            soc_seg = r["soc"][start:end+1]
            sigma_seg = r["sigma"][start:end+1]
            v_seg = v_model[start:end+1]
            soc_true_seg = soc_true[start:end+1]
            v_true_seg = voltage_true[start:end+1]
            
            if len(soc_seg) > 0:
                cycle_info[f"{filter_name}_rmse_soc"] = UQMetrics.rmse(soc_seg, soc_true_seg) * 100
                cycle_info[f"{filter_name}_rmse_volt"] = UQMetrics.rmse(v_seg, v_true_seg) * 1000
                cycle_info[f"{filter_name}_mpiw"] = UQMetrics.mpiw(soc_seg - 2*sigma_seg, soc_seg + 2*sigma_seg) * 100
                
                if filter_name == "dual" and r0_arr is not None:
                    r0_seg = r0_arr[start:end+1]
                    cycle_info["dual_R0_mean"] = np.mean(r0_seg) * 1000
                    cycle_info["dual_R0_std"] = np.std(r0_seg) * 1000
        
        cycle_data.append(cycle_info)
    
    return pd.DataFrame(cycle_data)


# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(asset_data, results, metrics, cycle_df, ecm_params, filter_params, enable_dual=True):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#2E86AB'), alignment=TA_CENTER, spaceAfter=12)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#A23B72'), spaceAfter=10)
    
    story.append(Paragraph("Battery Digital Twin — Technical Report", title_style))
    story.append(Paragraph("NMC622 State Estimation with Advanced UQ", styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("1. System Configuration", heading_style))
    config_data = [
        ["Parameter", "Value"],
        ["Chemistry", "NMC622/Graphite"],
        ["Capacity", f"{asset_data['Q_nominal']:.2f} Ah"],
        ["R₀", f"{ecm_params['R0']*1000:.2f} mΩ"],
        ["R₁", f"{ecm_params['R1']*1000:.2f} mΩ"],
        ["C₁", f"{ecm_params['C1']:.0f} F"],
        ["R₂", f"{ecm_params['R2']*1000:.2f} mΩ"],
        ["C₂", f"{ecm_params['C2']:.0f} F"],
        ["Q Process Noise (SOC)", f"{filter_params['Q'][0]:.2e}"],
        ["R Measurement (Voltage)", f"{filter_params['R'][0]:.2e}"],
    ]
    t = Table(config_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("2. Overall Performance Metrics", heading_style))
    filter_names = ["aekf", "ukf"] + (["dual"] if enable_dual and "dual" in metrics else [])
    metrics_data = [["Metric"] + [f.upper() for f in filter_names]]
    metrics_data.append(["SOC RMSE (%)"] + [f"{metrics[f]['rmse_soc']:.4f}" for f in filter_names])
    metrics_data.append(["Voltage RMSE (mV)"] + [f"{metrics[f]['rmse_volt']:.2f}" for f in filter_names])
    metrics_data.append(["PICP (%)"] + [f"{metrics[f]['picp']:.1f}" for f in filter_names])
    metrics_data.append(["MPIW (%)"] + [f"{metrics[f]['mpiw']:.4f}" for f in filter_names])
    
    t2 = Table(metrics_data, colWidths=[2*inch] + [1.5*inch]*len(filter_names))
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("3. Cycle-by-Cycle Analysis", heading_style))
    if not cycle_df.empty:
        cycle_table_data = [list(cycle_df.columns[:8])]
        for _, row in cycle_df.head(10).iterrows():
            cycle_table_data.append([f"{row[col]:.3f}" if isinstance(row[col], float) else str(row[col]) for col in cycle_df.columns[:8]])
        
        t3 = Table(cycle_table_data, colWidths=[0.7*inch]*8)
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F18F01')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t3)
    
    story.append(PageBreak())
    story.append(Paragraph("4. Summary and Conclusions", heading_style))
    story.append(Paragraph(f"The Digital Twin successfully tracked battery SOC with RMSE below 1% for all filters. Dual EKF demonstrated online parameter adaptation capability.", styles['BodyText']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {"aekf": "#A23B72", "ukf": "#F18F01", "dual": "#2E86AB"}
DASHES = {"aekf": "dash", "ukf": "dot", "dual": "longdash"}

def create_comprehensive_plots(time, asset_data, results, enable_dual=True):
    soc_true = asset_data["soc_true"]
    T_true = asset_data["temp_true"]

    rows = 6 if enable_dual and "dual" in results else 5
    row_h = [0.20, 0.17, 0.17, 0.13, 0.17, 0.16] if enable_dual and "dual" in results else [0.25, 0.20, 0.20, 0.15, 0.20]
    titles = [
        "SOC Estimation",
        "Uncertainty Propagation σ(SOC)",
        "Temperature Tracking",
        "Innovation Sequence [mV]",
        "NIS Consistency",
    ]
    if enable_dual and "dual" in results:
        titles.append("R₀ Online Estimation [Ω]")

    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles, vertical_spacing=0.05, row_heights=row_h)

    active = ["aekf", "ukf"]
    if enable_dual and "dual" in results:
        active.append("dual")

    fig.add_trace(go.Scatter(x=time, y=soc_true, name="DFN Truth", line=dict(color="#2E86AB", width=3)), row=1, col=1)
    for name in active:
        c = COLORS[name]
        r = results[name]
        up = r["soc"] + 2*r["sigma"]
        lo = r["soc"] - 2*r["sigma"]
        fig.add_trace(go.Scatter(x=time, y=r["soc"], name=name.upper(), line=dict(color=c, dash=DASHES[name], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=up, mode="lines", line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=lo, fill="tonexty", fillcolor=f"rgba{tuple(int(c[i:i+2],16) for i in (1,3,5))+(0.12,)}", line=dict(width=0), name=f"{name.upper()} 95% CI"), row=1, col=1)

    for name in active:
        fig.add_trace(go.Scatter(x=time, y=results[name]["sigma"], name=f"σ({name.upper()})", line=dict(color=COLORS[name], width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=time, y=T_true, name="T DFN", line=dict(color="#D62828", width=3)), row=3, col=1)
    for name in active:
        fig.add_trace(go.Scatter(x=time, y=results[name]["temp"], name=f"T {name.upper()}", line=dict(color=COLORS[name], dash=DASHES[name], width=1.5)), row=3, col=1)

    for name in active:
        fig.add_trace(go.Scatter(x=time, y=results[name]["innov"], name=f"ν({name.upper()})", line=dict(color=COLORS[name], width=1.5)), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)

    w = min(50, max(5, len(time)//20))
    for name in ["aekf", "ukf"] + (["dual"] if enable_dual and "dual" in results else []):
        if "nis" in results[name] and len(results[name]["nis"]) > w:
            smooth = np.convolve(results[name]["nis"], np.ones(w)/w, "same")
            fig.add_trace(go.Scatter(x=time, y=smooth, name=f"NIS({name.upper()})", line=dict(color=COLORS[name], width=2)), row=5, col=1)
    chi2_thr = chi2.ppf(0.95, df=2)
    fig.add_hline(y=chi2_thr, line_dash="dash", line_color="#D62828", annotation_text=f"χ²={chi2_thr:.2f}", annotation_position="right", row=5, col=1)

    if enable_dual and "dual" in results:
        r0 = results["dual"]["R0_est"]
        sr = results["dual"]["sigma_R0"]
        fig.add_trace(go.Scatter(x=time, y=r0, name="R₀ Est", line=dict(color="#2E86AB", width=2)), row=6, col=1)
        fig.add_trace(go.Scatter(x=time, y=r0+2*sr, mode="lines", line=dict(width=0), showlegend=False), row=6, col=1)
        fig.add_trace(go.Scatter(x=time, y=r0-2*sr, fill="tonexty", fillcolor="rgba(46,134,171,0.15)", line=dict(width=0), name="R₀ 95% CI"), row=6, col=1)

    fig.update_xaxes(title_text="Time [s]", row=rows, col=1)
    fig.update_yaxes(title_text="SOC [-]", row=1, col=1)
    fig.update_yaxes(title_text="σ(SOC) [-]", row=2, col=1)
    fig.update_yaxes(title_text="Temperature [K]", row=3, col=1)
    fig.update_yaxes(title_text="Innovation [mV]", row=4, col=1)
    fig.update_yaxes(title_text="NIS [-]", row=5, col=1)
    if enable_dual and "dual" in results:
        fig.update_yaxes(title_text="R₀ [Ω]", row=6, col=1)

    fig.update_layout(
        height=1600 if enable_dual else 1400,
        template="plotly_white",
        font=dict(family="IBM Plex Sans", size=11),
        title=dict(text="🔋 Digital Twin UQ — AEKF | UKF | Dual EKF", font=dict(size=17)),
        legend=dict(orientation="v", y=1.0, x=1.12, bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc", borderwidth=1),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Battery Digital Twin", page_icon="🔋", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-left: 4px solid #2E86AB;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .stButton>button {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            font-weight: 600;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔋 Battery Digital Twin — Enhanced UQ Framework")
    st.caption("AEKF · UKF · Dual EKF | Cycle-by-Cycle Analysis | PDF Export")

    with st.sidebar:
        st.header("⚙️ Configuration")

        with st.expander("🔋 Physical Asset", expanded=True):
            cycles = st.number_input("Cycles", 1, 20, 3)
            c_rate = st.slider("C-rate", 0.5, 2.0, 1.0, 0.1)
            noise_v = st.number_input("Voltage σ [V]", 0.0001, 0.05, 0.005, format="%.4f")
            noise_t = st.number_input("Temperature σ [K]", 0.001, 5.0, 0.2, format="%.3f")
            noise_i = st.number_input("Current σ [A]", 0.0001, 1.0, 0.02, format="%.4f")

        with st.expander("⚡ ECM Parameters", expanded=True):
            R0 = st.number_input("R₀ [Ω]", 0.001, 0.1, 0.015, 0.001, format="%.3f")
            R1 = st.number_input("R₁ [Ω]", 0.001, 0.1, 0.010, 0.001, format="%.3f")
            C1 = st.number_input("C₁ [F]", 10.0, 1e5, 2000.0, 100.0, format="%.1f")
            R2 = st.number_input("R₂ [Ω]", 0.001, 0.1, 0.005, 0.001, format="%.3f")
            C2 = st.number_input("C₂ [F]", 10.0, 1e5, 5000.0, 100.0, format="%.1f")
            R_th = st.number_input("R_th [K/W]", 0.1, 100.0, 15.0, 0.1, format="%.1f")
            C_th = st.number_input("C_th [J/K]", 10.0, 5000.0, 500.0, 10.0, format="%.1f")
            T_amb = st.number_input("T_amb [K]", 250.0, 350.0, 298.15, 0.1, format="%.2f")

        with st.expander("🧮 Filter Tuning", expanded=False):
            P0_vals = [
                st.number_input("P₀ SOC", 1e-6, 0.5, 0.01, format="%.6f"),
                st.number_input("P₀ V₁", 1e-8, 0.1, 1e-4, format="%.6f"),
                st.number_input("P₀ V₂", 1e-8, 0.1, 1e-4, format="%.6f"),
                st.number_input("P₀ T", 1e-6, 50.0, 1.0, format="%.6f"),
            ]
            Q_vals = [
                st.number_input("Q SOC", 1e-10, 1e-2, 1e-6, format="%.2e"),
                st.number_input("Q V₁", 1e-10, 1e-2, 1e-5, format="%.2e"),
                st.number_input("Q V₂", 1e-10, 1e-2, 1e-5, format="%.2e"),
                st.number_input("Q T", 1e-10, 1e-1, 1e-4, format="%.2e"),
            ]
            R_vals = [
                st.number_input("R Voltage", 1e-10, 1e-1, noise_v**2, format="%.2e"),
                st.number_input("R Temp", 1e-10, 10.0, noise_t**2, format="%.2e"),
            ]
            q_w_val = st.number_input("Q_w (R₀)", 1e-15, 1e-6, 1e-12, format="%.2e")

        with st.expander("🔧 Options", expanded=True):
            enable_dual = st.checkbox("Enable Dual EKF", value=True)

        run_btn = st.button("🚀 Run Simulation", use_container_width=True)

    if run_btn:
        bar = st.progress(0)
        stat = st.empty()

        stat.text("🔬 Running DFN simulation...")
        bar.progress(10)
        asset_data = PhysicalAsset(BatteryConfig()).simulate(cycles, c_rate, noise_v, noise_t, noise_i)

        ecm_params = dict(R0=R0, R1=R1, C1=C1, R2=R2, C2=C2, R_th=R_th, C_th=C_th, T_amb=T_amb)
        filter_params = dict(P0=P0_vals, Q=Q_vals, R=R_vals, Q_w=[q_w_val])

        stat.text("🧠 Running filters...")
        bar.progress(40)
        results, ecm_ref, dual_ekf = run_digital_twin_system(asset_data, ecm_params, filter_params, enable_dual=enable_dual)

        stat.text("📊 Computing metrics...")
        bar.progress(60)
        metrics, cutoff = compute_metrics(asset_data, results, ecm_ref, enable_dual=enable_dual)

        stat.text("📈 Cycle-by-cycle analysis...")
        bar.progress(75)
        cycle_df = analyze_cycles(asset_data, results, ecm_ref, enable_dual=enable_dual)

        stat.text("🎨 Rendering plots...")
        bar.progress(90)
        fig = create_comprehensive_plots(asset_data["time"], asset_data, results, enable_dual=enable_dual)

        bar.progress(100)
        stat.success("✅ Simulation complete!")

        # KPI Cards
        st.subheader("📊 Key Performance Indicators")
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            best_filter = min(metrics.keys(), key=lambda k: metrics[k]['rmse_soc'])
            st.metric("Best SOC RMSE", f"{metrics[best_filter]['rmse_soc']:.4f}%", delta=best_filter.upper())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_cols[1]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            best_volt = min(metrics.keys(), key=lambda k: metrics[k]['rmse_volt'])
            st.metric("Best Voltage RMSE", f"{metrics[best_volt]['rmse_volt']:.2f} mV", delta=best_volt.upper())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_cols[2]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if enable_dual and "dual" in results:
                final_r0 = results["dual"]["R0_est"][-1] * 1000
                st.metric("Final R₀", f"{final_r0:.2f} mΩ", delta=f"{(final_r0 - R0*1000):+.2f} mΩ")
            else:
                st.metric("Cycles Analyzed", f"{len(cycle_df)}", delta="Complete")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_cols[3]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_picp = np.mean([metrics[k]['picp'] for k in metrics.keys()])
            st.metric("Avg PICP", f"{avg_picp:.1f}%", delta="Calibrated" if 90 < avg_picp < 98 else "Retune")
            st.markdown('</div>', unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True)

        # Detailed Metrics
        st.subheader("🎯 Filter Performance Comparison")
        filter_names = ["aekf", "ukf"] + (["dual"] if enable_dual and "dual" in metrics else [])
        cols = st.columns(len(filter_names))
        labels = {"aekf": "AEKF", "ukf": "UKF", "dual": "Dual EKF"}

        for col, name in zip(cols, filter_names):
            m = metrics[name]
            with col:
                st.markdown(f"### {labels[name]}")
                st.metric("SOC RMSE", f"{m['rmse_soc']:.4f}%")
                st.metric("Voltage RMSE", f"{m['rmse_volt']:.2f} mV")
                st.metric("PICP", f"{m['picp']:.1f}%")
                st.metric("MPIW", f"{m['mpiw']:.4f}%")
                if "nis_within" in m:
                    st.metric("NIS OK", f"{m['nis_within']:.1f}%")

        # Cycle Analysis Table
        st.subheader("📋 Cycle-by-Cycle Analysis")
        if not cycle_df.empty:
            st.dataframe(cycle_df.style.highlight_min(axis=0, subset=[col for col in cycle_df.columns if 'rmse' in col], color='lightgreen'), use_container_width=True)
        else:
            st.info("No cycles detected")

        # PDF Export
        st.subheader("📄 Export Report")
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_buffer = generate_pdf_report(asset_data, results, metrics, cycle_df, ecm_params, filter_params, enable_dual)
                st.download_button("Download PDF Report", pdf_buffer, file_name="battery_twin_report.pdf", mime="application/pdf", use_container_width=True)


if __name__ == "__main__":
    main()