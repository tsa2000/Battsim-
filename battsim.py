import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pybamm
from scipy.stats import chi2
from dataclasses import dataclass
import warnings
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
    def __init__(self, n_points=201):
        self.soc_lut = np.linspace(0.0, 1.0, n_points)
        self.ocv_lut = np.clip(
            3.4043
            + 1.6227 * self.soc_lut
            - 8.6635 * self.soc_lut**2
            + 21.3955 * self.soc_lut**3
            - 25.7324 * self.soc_lut**4
            + 12.0032 * self.soc_lut**5,
            2.5,
            4.25,
        )

    def get_voltage(self, soc):
        return np.interp(np.clip(soc, 0.0, 1.0), self.soc_lut, self.ocv_lut)

    def get_gradient(self, soc, eps=1e-6):
        return (self.get_voltage(soc + eps) - self.get_voltage(soc - eps)) / (2 * eps)

    def get_gradient_analytical(self, soc):
        s = np.clip(soc, 0.0, 1.0)
        return (
            1.6227
            - 2 * 8.6635 * s
            + 3 * 21.3955 * s**2
            - 4 * 25.7324 * s**3
            + 5 * 12.0032 * s**4
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL ASSET — DFN
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
        discharge_capacity = sol["Discharge capacity [A.h]"].entries

        rng = np.random.default_rng(42)
        voltage_meas = voltage_true + rng.normal(0, noise_voltage, len(time))
        temp_meas = temp_true + rng.normal(0, noise_temp, len(time))
        current_meas = current + rng.normal(0, noise_current, len(time))

        Q_nominal = float(params["Nominal cell capacity [A.h]"])
        soc_true = np.clip(1.0 - discharge_capacity / Q_nominal, 0.0, 1.0)

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
        return np.exp(
            self.config.arrhenius_factor
            * (1.0 / T_safe - 1.0 / self.config.temperature_ref)
        )

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
        T_new = T + (dt / self.C_th) * (I**2 * R0_eff - (T - self.T_amb) / self.R_th)

        return np.array([soc_new, V1_new, V2_new, T_new])

    def measurement_model(self, x, I):
        soc, V1, V2, T = x
        R0_eff = self.effective_resistance(soc, T, self.R0)
        V_terminal = self.ocv.get_voltage(soc) - V1 - V2 - I * R0_eff
        T_surface = T
        return np.array([V_terminal, T_surface])


# ═══════════════════════════════════════════════════════════════════════════════
# AEKF
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveEKF:
    def __init__(self, ecm: EquivalentCircuitModel, x0, P0, Q, R):
        self.ecm = ecm
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(P0).astype(float)
        self.Q = np.diag(Q).astype(float)
        self.R = np.diag(R).astype(float)

    def predict(self, I, dt):
        """Prediction step with full linearized dynamics and chain rule"""

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

        dR0_dT = self.ecm.R0 * (1.0 + 0.4 * (1.0 - soc) ** 2) * darr_dT
        dT_dT = (
            1.0
            - dt / (self.ecm.C_th * self.ecm.R_th)
            + (dt / self.ecm.C_th) * (I**2 * dR0_dT)
        )

        dR0_dSOC = self.ecm.R0 * arr * (-0.8 * (1.0 - soc))
        dT_dSOC = (dt / self.ecm.C_th) * (I**2 * dR0_dSOC)

        F = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, exp1, 0.0, dV1_dT],
                [0.0, 0.0, exp2, dV2_dT],
                [dT_dSOC, 0.0, 0.0, dT_dT],
            ]
        )

        P_pred = F @ self.P @ F.T + self.Q
        return x_pred, P_pred, F

    def update(self, x_pred, P_pred, y_meas, I):
        """Update step with fully coupled measurement innovation"""

        y_pred = self.ecm.measurement_model(x_pred, I)

        soc_pred, _, _, T_pred = x_pred

        arr_pred = self.ecm.arrhenius_correction(T_pred)
        T_safe = max(T_pred, 250.0)
        darr_dT_pred = -self.ecm.config.arrhenius_factor / (T_safe**2) * arr_pred
        dR0_dT_pred = self.ecm.R0 * (1.0 + 0.4 * (1.0 - soc_pred) ** 2) * darr_dT_pred

        H = np.array(
            [
                [self.ecm.ocv.get_gradient_analytical(soc_pred), -1.0, -1.0, -I * dR0_dT_pred],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        innovation = y_meas - y_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ innovation

        I_KH = np.eye(4) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        x_upd[0] = np.clip(x_upd[0], 0.0, 1.0)
        x_upd[3] = np.clip(x_upd[3], 250.0, 350.0)

        nis = float(innovation @ np.linalg.inv(S) @ innovation)

        return x_upd, P_upd, innovation, nis

    def step(self, y_meas, I, dt):
        x_pred, P_pred, _ = self.predict(I, dt)
        self.x, self.P, innov, nis = self.update(x_pred, P_pred, y_meas, I)

        return {
            "soc": self.x[0],
            "v1": self.x[1],
            "v2": self.x[2],
            "temp": self.x[3],
            "sigma_soc": np.sqrt(max(self.P[0, 0], 0.0)),
            "innovation_voltage": innov[0] * 1000.0,
            "nis": nis,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UKF
# ═══════════════════════════════════════════════════════════════════════════════

class UnscentedKalmanFilter:
    def __init__(self, ecm: EquivalentCircuitModel, x0, P0, Q, R, alpha=0.1, beta=2.0, kappa=0.0):
        self.ecm = ecm
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(P0).astype(float)
        self.Q = np.diag(Q).astype(float)
        self.R = np.diag(R).astype(float)

        n = 4
        self.n = n
        lam = alpha**2 * (n + kappa) - n
        self.lam = lam

        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
        self.gamma = np.sqrt(n + lam)

    def generate_sigma_points(self, x, P):
        P_safe = P + 1e-9 * np.eye(self.n)
        try:
            L = np.linalg.cholesky(P_safe)
        except np.linalg.LinAlgError:
            U, s, _ = np.linalg.svd(P_safe)
            L = U @ np.diag(np.sqrt(np.maximum(s, 1e-12)))

        sigma_points = [x.copy()]
        for i in range(self.n):
            sigma_points.append(x + self.gamma * L[:, i])
            sigma_points.append(x - self.gamma * L[:, i])

        return np.array(sigma_points)

    def unscented_transform(self, sigma_points, transform_func):
        transformed = np.array([transform_func(sp) for sp in sigma_points])
        mean = np.einsum("i,ij->j", self.Wm, transformed)
        deviations = transformed - mean
        cov = sum(self.Wc[i] * np.outer(deviations[i], deviations[i]) for i in range(len(self.Wm)))
        return mean, cov, transformed

    def step(self, y_meas, I, dt):
        sigma_points = self.generate_sigma_points(self.x, self.P)

        x_pred, P_pred, sigma_pred = self.unscented_transform(
            sigma_points, lambda sp: self.ecm.state_transition(sp, I, dt)
        )
        P_pred += self.Q

        y_pred, Pyy, sigma_meas = self.unscented_transform(
            sigma_pred, lambda sp: self.ecm.measurement_model(sp, I)
        )
        Pyy += self.R

        Pxy = sum(
            self.Wc[i] * np.outer(sigma_pred[i] - x_pred, sigma_meas[i] - y_pred)
            for i in range(len(self.Wm))
        )

        K = Pxy @ np.linalg.inv(Pyy)
        innovation = y_meas - y_pred

        self.x = x_pred + K @ innovation
        self.P = P_pred - K @ Pyy @ K.T

        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.x[3] = np.clip(self.x[3], 250.0, 350.0)

        nis = float(innovation @ np.linalg.inv(Pyy) @ innovation)

        return {
            "soc": self.x[0],
            "v1": self.x[1],
            "v2": self.x[2],
            "temp": self.x[3],
            "sigma_soc": np.sqrt(max(self.P[0, 0], 0.0)),
            "innovation_voltage": innovation[0] * 1000.0,
            "nis": nis,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PF
# ═══════════════════════════════════════════════════════════════════════════════

class ParticleFilter:
    def __init__(self, ecm: EquivalentCircuitModel, x0, P0, Q, R, n_particles=500):
        self.ecm = ecm
        self.Q = np.diag(Q).astype(float)
        self.R = np.diag(R).astype(float)
        self.n_particles = n_particles

        self.particles = np.random.multivariate_normal(x0, np.diag(P0), n_particles)
        self.weights = np.ones(n_particles) / n_particles
        self.x = np.average(self.particles, weights=self.weights, axis=0)

    def predict(self, I, dt):
        process_noise = np.random.multivariate_normal(np.zeros(4), self.Q, self.n_particles)

        for i in range(self.n_particles):
            self.particles[i] = self.ecm.state_transition(self.particles[i], I, dt)
            self.particles[i] += process_noise[i]
            self.particles[i, 0] = np.clip(self.particles[i, 0], 0.0, 1.0)
            self.particles[i, 3] = np.clip(self.particles[i, 3], 250.0, 350.0)

    def update(self, y_meas, I):
        invR = np.linalg.inv(self.R)
        for i in range(self.n_particles):
            y_pred = self.ecm.measurement_model(self.particles[i], I)
            innovation = y_meas - y_pred
            log_likelihood = -0.5 * innovation @ invR @ innovation
            self.weights[i] *= np.exp(log_likelihood)

        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.n_particles / 2:
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            u = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
            indices = np.searchsorted(cumsum, u)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

    def step(self, y_meas, I, dt):
        self.predict(I, dt)
        self.update(y_meas, I)
        self.resample()

        self.x = np.average(self.particles, weights=self.weights, axis=0)

        deviations = self.particles - self.x
        P = (
            self.weights[:, None, None]
            * deviations[:, :, None]
            * deviations[:, None, :]
        ).sum(axis=0)

        sigma_soc = np.sqrt(max(P[0, 0], 0.0))

        y_pred = self.ecm.measurement_model(self.x, I)
        innovation = y_meas - y_pred

        return {
            "soc": self.x[0],
            "v1": self.x[1],
            "v2": self.x[2],
            "temp": self.x[3],
            "sigma_soc": sigma_soc,
            "innovation_voltage": innovation[0] * 1000.0,
            "particles": self.particles.copy(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UQ METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class UQMetrics:
    @staticmethod
    def rmse(estimates, truth):
        return np.sqrt(np.mean((estimates - truth) ** 2))

    @staticmethod
    def mae(estimates, truth):
        return np.mean(np.abs(estimates - truth))

    @staticmethod
    def picp(truth, lower, upper):
        in_interval = (truth >= lower) & (truth <= upper)
        return 100.0 * np.mean(in_interval)

    @staticmethod
    def mpiw(lower, upper):
        return np.mean(upper - lower)

    @staticmethod
    def nis_consistency(nis_values, alpha=0.05):
        threshold = chi2.ppf(1 - alpha, df=2)
        within_bounds = np.mean(nis_values < threshold) * 100.0
        return within_bounds, threshold


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_digital_twin_system(asset_data, ecm_params, filter_params, enable_pf=True, dt_hint=1.0):
    time = np.asarray(asset_data["time"]).flatten()
    V_meas = np.asarray(asset_data["voltage_meas"]).flatten()
    T_meas = np.asarray(asset_data["temp_meas"]).flatten()
    I_meas = np.asarray(asset_data["current_meas"]).flatten()

    n_steps = min(len(time), len(V_meas), len(T_meas), len(I_meas))
    time = time[:n_steps]
    V_meas = V_meas[:n_steps]
    T_meas = T_meas[:n_steps]
    I_meas = I_meas[:n_steps]

    dt = float(np.mean(np.diff(time))) if n_steps > 1 else dt_hint

    ecm = EquivalentCircuitModel(
        Q_nom=asset_data["Q_nominal"],
        **ecm_params,
        config=BatteryConfig(),
    )

    x0 = [1.0, 0.0, 0.0, ecm_params["T_amb"]]

    aekf = AdaptiveEKF(ecm, x0, filter_params["P0"], filter_params["Q"], filter_params["R"])
    ukf = UnscentedKalmanFilter(ecm, x0, filter_params["P0"], filter_params["Q"], filter_params["R"])

    pf = None
    if enable_pf:
        pf = ParticleFilter(
            ecm,
            x0,
            filter_params["P0"],
            filter_params["Q"],
            filter_params["R"],
            n_particles=filter_params.get("n_particles", 500),
        )

    results = {
        "aekf": {"soc": [], "v1": [], "v2": [], "sigma": [], "temp": [], "innov": [], "nis": []},
        "ukf": {"soc": [], "v1": [], "v2": [], "sigma": [], "temp": [], "innov": [], "nis": []},
    }

    if enable_pf:
        results["pf"] = {"soc": [], "v1": [], "v2": [], "sigma": [], "temp": [], "innov": [], "particles": []}

    for k in range(n_steps):
        y = np.array([V_meas[k], T_meas[k]])
        I = -float(I_meas[k])

        aekf_out = aekf.step(y, I, dt)
        results["aekf"]["soc"].append(aekf_out["soc"])
        results["aekf"]["v1"].append(aekf_out["v1"])
        results["aekf"]["v2"].append(aekf_out["v2"])
        results["aekf"]["sigma"].append(aekf_out["sigma_soc"])
        results["aekf"]["temp"].append(aekf_out["temp"])
        results["aekf"]["innov"].append(aekf_out["innovation_voltage"])
        results["aekf"]["nis"].append(aekf_out["nis"])

        ukf_out = ukf.step(y, I, dt)
        results["ukf"]["soc"].append(ukf_out["soc"])
        results["ukf"]["v1"].append(ukf_out["v1"])
        results["ukf"]["v2"].append(ukf_out["v2"])
        results["ukf"]["sigma"].append(ukf_out["sigma_soc"])
        results["ukf"]["temp"].append(ukf_out["temp"])
        results["ukf"]["innov"].append(ukf_out["innovation_voltage"])
        results["ukf"]["nis"].append(ukf_out["nis"])

        if enable_pf:
            pf_out = pf.step(y, I, dt)
            results["pf"]["soc"].append(pf_out["soc"])
            results["pf"]["v1"].append(pf_out["v1"])
            results["pf"]["v2"].append(pf_out["v2"])
            results["pf"]["sigma"].append(pf_out["sigma_soc"])
            results["pf"]["temp"].append(pf_out["temp"])
            results["pf"]["innov"].append(pf_out["innovation_voltage"])
            if k % 50 == 0:
                results["pf"]["particles"].append(pf_out["particles"])

    for filter_name in results:
        for key in results[filter_name]:
            if key != "particles":
                results[filter_name][key] = np.array(results[filter_name][key])

    return results, ecm


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_voltage_from_states(ecm, soc, v1, v2, temp, current_meas):
    current_ecm = -np.asarray(current_meas)
    v_model = np.zeros_like(soc, dtype=float)
    for k in range(len(soc)):
        xk = np.array([soc[k], v1[k], v2[k], temp[k]])
        v_model[k] = ecm.measurement_model(xk, current_ecm[k])[0]
    return v_model


def compute_metrics(asset_data, results, ecm, enable_pf=True):
    soc_true = np.asarray(asset_data["soc_true"])
    voltage_true = np.asarray(asset_data["voltage_true"])
    current_meas = np.asarray(asset_data["current_meas"])

    cutoff = int(0.10 * len(soc_true))
    cutoff = min(cutoff, len(soc_true) - 1)

    metrics = {}
    filters_to_evaluate = ["aekf", "ukf"]
    if enable_pf and "pf" in results:
        filters_to_evaluate.append("pf")

    for filter_name in filters_to_evaluate:
        soc_est = results[filter_name]["soc"]
        sigma = results[filter_name]["sigma"]
        temp_est = results[filter_name]["temp"]
        v1_est = results[filter_name]["v1"]
        v2_est = results[filter_name]["v2"]
        innov_v = results[filter_name]["innov"]

        voltage_model = reconstruct_voltage_from_states(
            ecm, soc_est, v1_est, v2_est, temp_est, current_meas
        )

        metrics[filter_name] = {
            "rmse_soc": UQMetrics.rmse(soc_est[cutoff:], soc_true[cutoff:]) * 100.0,
            "mae_soc": UQMetrics.mae(soc_est[cutoff:], soc_true[cutoff:]) * 100.0,
            "rmse_volt": UQMetrics.rmse(voltage_model[cutoff:], voltage_true[cutoff:]) * 1000.0,
            "innov_rms": np.sqrt(np.mean(innov_v[cutoff:] ** 2)),
            "picp": UQMetrics.picp(
                soc_true[cutoff:],
                soc_est[cutoff:] - 2 * sigma[cutoff:],
                soc_est[cutoff:] + 2 * sigma[cutoff:],
            ),
            "mpiw": UQMetrics.mpiw(
                soc_est[cutoff:] - 2 * sigma[cutoff:],
                soc_est[cutoff:] + 2 * sigma[cutoff:],
            ) * 100.0,
        }

        if "nis" in results[filter_name]:
            nis_cons, nis_thr = UQMetrics.nis_consistency(results[filter_name]["nis"][cutoff:])
            metrics[filter_name]["nis_within"] = nis_cons
            metrics[filter_name]["nis_threshold"] = nis_thr

    return metrics, cutoff


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def create_comprehensive_plots(time, asset_data, results, enable_pf=True):
    soc_true = asset_data["soc_true"]
    T_true = asset_data["temp_true"]

    aekf_upper = results["aekf"]["soc"] + 2 * results["aekf"]["sigma"]
    aekf_lower = results["aekf"]["soc"] - 2 * results["aekf"]["sigma"]
    ukf_upper = results["ukf"]["soc"] + 2 * results["ukf"]["sigma"]
    ukf_lower = results["ukf"]["soc"] - 2 * results["ukf"]["sigma"]

    if enable_pf and "pf" in results:
        pf_upper = results["pf"]["soc"] + 2 * results["pf"]["sigma"]
        pf_lower = results["pf"]["soc"] - 2 * results["pf"]["sigma"]

    fig = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=(
            "SOC Estimation: DFN Truth vs Digital Twin Filters",
            "Uncertainty Propagation: σ(SOC) Evolution",
            "Core Temperature Tracking",
            "Innovation Sequence (Voltage Residuals)",
            "Normalized Innovation Squared (NIS)",
        ),
        vertical_spacing=0.06,
        row_heights=[0.25, 0.2, 0.2, 0.15, 0.2],
    )

    fig.add_trace(
        go.Scatter(x=time, y=soc_true, name="DFN Truth", line=dict(color="#2E86AB", width=3)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=time, y=results["aekf"]["soc"], name="AEKF", line=dict(color="#A23B72", dash="dash", width=2)),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=time, y=aekf_upper, fill=None, mode="lines", line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=time, y=aekf_lower, fill="tonexty", fillcolor="rgba(162, 59, 114, 0.15)", line=dict(width=0), name="AEKF 95% CI"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=time, y=results["ukf"]["soc"], name="UKF", line=dict(color="#F18F01", dash="dot", width=2)),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=time, y=ukf_upper, fill=None, mode="lines", line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=time, y=ukf_lower, fill="tonexty", fillcolor="rgba(241, 143, 1, 0.15)", line=dict(width=0), name="UKF 95% CI"),
        row=1, col=1
    )

    if enable_pf and "pf" in results:
        fig.add_trace(
            go.Scatter(x=time, y=results["pf"]["soc"], name="PF", line=dict(color="#06A77D", dash="dashdot", width=2)),
            row=1, col=1
        )
        fig.add_trace(go.Scatter(x=time, y=pf_upper, fill=None, mode="lines", line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=time, y=pf_lower, fill="tonexty", fillcolor="rgba(6, 167, 125, 0.15)", line=dict(width=0), name="PF 95% CI"),
            row=1, col=1
        )

    fig.add_trace(go.Scatter(x=time, y=results["aekf"]["sigma"], name="σ(AEKF)", line=dict(color="#A23B72", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=results["ukf"]["sigma"], name="σ(UKF)", line=dict(color="#F18F01", width=2)), row=2, col=1)
    if enable_pf and "pf" in results:
        fig.add_trace(go.Scatter(x=time, y=results["pf"]["sigma"], name="σ(PF)", line=dict(color="#06A77D", width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=time, y=T_true, name="T True", line=dict(color="#D62828", width=3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=time, y=results["aekf"]["temp"], name="T Estimated (AEKF)", line=dict(color="#A23B72", dash="dash", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=time, y=results["ukf"]["temp"], name="T Estimated (UKF)", line=dict(color="#F18F01", dash="dot", width=2)), row=3, col=1)
    if enable_pf and "pf" in results:
        fig.add_trace(go.Scatter(x=time, y=results["pf"]["temp"], name="T Estimated (PF)", line=dict(color="#06A77D", dash="dashdot", width=2)), row=3, col=1)

    fig.add_trace(go.Scatter(x=time, y=results["aekf"]["innov"], name="ν(AEKF)", line=dict(color="#A23B72", width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=time, y=results["ukf"]["innov"], name="ν(UKF)", line=dict(color="#F18F01", width=1.5)), row=4, col=1)
    if enable_pf and "pf" in results:
        fig.add_trace(go.Scatter(x=time, y=results["pf"]["innov"], name="ν(PF)", line=dict(color="#06A77D", width=1.5)), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)

    window = min(50, max(5, len(time) // 20))
    nis_aekf_smooth = np.convolve(results["aekf"]["nis"], np.ones(window) / window, mode="same")
    nis_ukf_smooth = np.convolve(results["ukf"]["nis"], np.ones(window) / window, mode="same")

    fig.add_trace(go.Scatter(x=time, y=nis_aekf_smooth, name="NIS(AEKF)", line=dict(color="#A23B72", width=2)), row=5, col=1)
    fig.add_trace(go.Scatter(x=time, y=nis_ukf_smooth, name="NIS(UKF)", line=dict(color="#F18F01", width=2)), row=5, col=1)

    chi2_95 = chi2.ppf(0.95, df=2)
    fig.add_hline(
        y=chi2_95,
        line_dash="dash",
        line_color="#D62828",
        annotation_text=f"χ²(0.95, df=2) = {chi2_95:.2f}",
        annotation_position="right",
        row=5, col=1
    )

    fig.update_xaxes(title_text="Time [s]", row=5, col=1)
    fig.update_yaxes(title_text="SOC [-]", row=1, col=1)
    fig.update_yaxes(title_text="σ(SOC) [-]", row=2, col=1)
    fig.update_yaxes(title_text="Temperature [K]", row=3, col=1)
    fig.update_yaxes(title_text="Innovation [mV]", row=4, col=1)
    fig.update_yaxes(title_text="NIS [-]", row=5, col=1)

    fig.update_layout(
        height=1400,
        template="plotly_white",
        font=dict(family="IBM Plex Sans, sans-serif", size=11),
        title=dict(text="Digital Twin Uncertainty Quantification — Multi-Filter Comparison", font=dict(size=18)),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="right", x=1.12),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Battery Digital Twin - UQ Analysis",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔋 NMC622 Digital Twin — Advanced UQ Framework")
    st.caption("PhD-Level Battery State Estimation with AEKF, UKF, PF and corrected UQ metrics")

    with st.sidebar:
        st.header("⚙️ Configuration")

        cycles = st.number_input("Cycles", min_value=1, max_value=20, value=3)
        c_rate = st.slider("Discharge C-rate", 0.5, 2.0, 1.0, 0.1)

        st.subheader("Sensor Noise")
        noise_voltage = st.number_input("Voltage noise σ [V]", min_value=0.0001, max_value=0.05, value=0.005, format="%.4f")
        noise_temp = st.number_input("Temperature noise σ [K]", min_value=0.001, max_value=5.0, value=0.2, format="%.3f")
        noise_current = st.number_input("Current noise σ [A]", min_value=0.0001, max_value=1.0, value=0.02, format="%.4f")

        st.subheader("ECM Parameters")
        R0 = st.number_input("R0 [Ω]", 0.001, 0.1, 0.015, 0.001, format="%.3f")
        R1 = st.number_input("R1 [Ω]", 0.001, 0.1, 0.010, 0.001, format="%.3f")
        C1 = st.number_input("C1 [F]", 10.0, 100000.0, 2000.0, 100.0, format="%.1f")
        R2 = st.number_input("R2 [Ω]", 0.001, 0.1, 0.005, 0.001, format="%.3f")
        C2 = st.number_input("C2 [F]", 10.0, 100000.0, 5000.0, 100.0, format="%.1f")
        R_th = st.number_input("R_th [K/W]", 0.1, 100.0, 15.0, 0.1, format="%.1f")
        C_th = st.number_input("C_th [J/K]", 10.0, 5000.0, 500.0, 10.0, format="%.1f")
        T_amb = st.number_input("Ambient temperature [K]", 250.0, 350.0, 298.15, 0.1, format="%.2f")

        st.subheader("Filter Parameters")
        P0_soc = st.number_input("P0 SOC", 1e-6, 0.5, 0.01, format="%.6f")
        P0_v1 = st.number_input("P0 V1", 1e-8, 0.1, 1e-4, format="%.6f")
        P0_v2 = st.number_input("P0 V2", 1e-8, 0.1, 1e-4, format="%.6f")
        P0_t = st.number_input("P0 T", 1e-6, 50.0, 1.0, format="%.6f")

        Q_soc = st.number_input("Q SOC", 1e-10, 1e-2, 1e-6, format="%.8f")
        Q_v1 = st.number_input("Q V1", 1e-10, 1e-2, 1e-5, format="%.8f")
        Q_v2 = st.number_input("Q V2", 1e-10, 1e-2, 1e-5, format="%.8f")
        Q_t = st.number_input("Q T", 1e-10, 1e-1, 1e-4, format="%.8f")

        R_v = st.number_input("R Voltage", 1e-10, 1e-1, noise_voltage**2, format="%.8f")
        R_t = st.number_input("R Temp", 1e-10, 10.0, noise_temp**2, format="%.8f")

        enable_pf = st.checkbox("Enable Particle Filter", value=True)
        n_particles = st.slider("PF particles", 100, 2000, 500, 50)

        run_btn = st.button("🚀 Run Digital Twin")

    if run_btn:
        status_text = st.empty()
        progress_bar = st.progress(0)

        status_text.text("🔬 Running DFN physical asset...")
        progress_bar.progress(15)

        asset = PhysicalAsset(BatteryConfig())
        asset_data = asset.simulate(cycles, c_rate, noise_voltage, noise_temp, noise_current)

        ecm_params = {
            "R0": R0,
            "R1": R1,
            "C1": C1,
            "R2": R2,
            "C2": C2,
            "R_th": R_th,
            "C_th": C_th,
            "T_amb": T_amb,
        }

        filter_params = {
            "P0": [P0_soc, P0_v1, P0_v2, P0_t],
            "Q": [Q_soc, Q_v1, Q_v2, Q_t],
            "R": [R_v, R_t],
            "n_particles": n_particles,
        }

        status_text.text("🧠 Running digital twin filters...")
        progress_bar.progress(45)

        results, ecm = run_digital_twin_system(
            asset_data, ecm_params, filter_params, enable_pf=enable_pf
        )

        status_text.text("📊 Computing corrected metrics...")
        progress_bar.progress(70)

        metrics, cutoff = compute_metrics(asset_data, results, ecm, enable_pf=enable_pf)

        status_text.text("🎨 Rendering plots...")
        progress_bar.progress(90)

        fig = create_comprehensive_plots(asset_data["time"], asset_data, results, enable_pf=enable_pf)

        progress_bar.progress(100)
        status_text.text(f"✅ Completed. Metrics exclude first {cutoff} samples (~10%) for steady-state evaluation.")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Metrics")

        cols = st.columns(3 if enable_pf else 2)

        with cols[0]:
            st.markdown("### 🎯 AEKF")
            st.metric("SOC RMSE", f"{metrics['aekf']['rmse_soc']:.4f} %")
            st.metric("Voltage RMSE", f"{metrics['aekf']['rmse_volt']:.2f} mV")
            st.metric("SOC MAE", f"{metrics['aekf']['mae_soc']:.4f} %")
            st.metric("PICP", f"{metrics['aekf']['picp']:.2f} %")
            st.metric("MPIW", f"{metrics['aekf']['mpiw']:.4f} %")
            if "nis_within" in metrics["aekf"]:
                st.metric("NIS within χ²", f"{metrics['aekf']['nis_within']:.2f} %")

        with cols[1]:
            st.markdown("### 🧠 UKF")
            st.metric("SOC RMSE", f"{metrics['ukf']['rmse_soc']:.4f} %")
            st.metric("Voltage RMSE", f"{metrics['ukf']['rmse_volt']:.2f} mV")
            st.metric("SOC MAE", f"{metrics['ukf']['mae_soc']:.4f} %")
            st.metric("PICP", f"{metrics['ukf']['picp']:.2f} %")
            st.metric("MPIW", f"{metrics['ukf']['mpiw']:.4f} %")
            if "nis_within" in metrics["ukf"]:
                st.metric("NIS within χ²", f"{metrics['ukf']['nis_within']:.2f} %")

        if enable_pf:
            with cols[2]:
                st.markdown("### 🌫️ PF")
                st.metric("SOC RMSE", f"{metrics['pf']['rmse_soc']:.4f} %")
                st.metric("Voltage RMSE", f"{metrics['pf']['rmse_volt']:.2f} mV")
                st.metric("SOC MAE", f"{metrics['pf']['mae_soc']:.4f} %")
                st.metric("PICP", f"{metrics['pf']['picp']:.2f} %")
                st.metric("MPIW", f"{metrics['pf']['mpiw']:.4f} %")

        st.subheader("Steady-State Notes")
        st.write(
            "- SOC metrics are computed after excluding the first 10% of samples to avoid initialization bias."
        )
        st.write(
            "- Voltage RMSE is computed against the DFN true voltage using reconstructed ECM voltage, not from noisy innovation only."
        )
        st.write(
            "- NIS is reported for AEKF and UKF because it depends on innovation covariance."
        )


if __name__ == "__main__":
    main()
