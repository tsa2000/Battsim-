# 🔋 BattSim-NMC622: Advanced Battery Digital Twin with Uncertainty Quantification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://battsim.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Professional-grade battery state estimation framework combining physics-based modeling (PyBaMM DFN) with advanced Kalman filtering for comprehensive uncertainty quantification.**

Designed by **Eng. Thaer Abushawer**

---

## 🎯 Overview

BattSim-NMC622 implements a **dual-machine digital twin architecture** for lithium-ion battery state estimation with rigorous uncertainty quantification (UQ). The system combines:

- **Machine 1 (Physical Asset):** High-fidelity PyBaMM Doyle-Fuller-Newman (DFN) electrochemical model
- **Machine 2 (Digital Twin):** Equivalent Circuit Model (ECM) with multiple estimation filters

### Key Features

✨ **Multiple State Estimation Algorithms**
- Adaptive Extended Kalman Filter (AEKF) with Arrhenius temperature correction
- Unscented Kalman Filter (UKF) with sigma-point propagation
- Dual EKF for online parameter estimation (R₀ tracking)
- Particle Filter (PF) support for nonlinear systems

📊 **Comprehensive UQ Metrics**
- RMSE/MAE for SOC and voltage accuracy
- PICP (Prediction Interval Coverage Probability)
- MPIW (Mean Prediction Interval Width)
- NIS (Normalized Innovation Squared) χ² consistency test

🔬 **Advanced Thermal Modeling**
- Ohmic, polarization, and entropic heat generation
- Lumped thermal dynamics with ambient exchange
- Temperature-dependent resistance correction

📈 **Professional Reporting**
- Interactive Plotly visualizations with 6 analysis tabs
- Cycle-by-cycle performance breakdown
- Automated PDF report generation with metrics tables

---

## 🚀 Quick Start

### Online Demo
Access the live application instantly:
**[https://battsim.streamlit.app/](https://battsim.streamlit.app/)**

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/battsim-nmc622.git
cd battsim-nmc622

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## 📦 Dependencies

### Core Libraries
```
streamlit          # Web application framework
pybamm            # Physics-based battery modeling
numpy             # Numerical computing
pandas            # Data manipulation
plotly            # Interactive visualizations
scipy             # Scientific computing
fpdf2             # PDF report generation
kaleido==0.2.1    # Static image export
```

### System Requirements (for PDF generation)
The following system libraries are required for Plotly's Kaleido engine:
```
libnss3, libnspr4, libatk-bridge2.0-0, libx11-xcb1
libxcomposite1, libxdamage1, libxrandr2, libgbm1, libasound2
```

These are automatically handled in `packages.txt` for Streamlit Cloud deployment.

---

## 🏗️ System Architecture

### Machine 1: Physical Asset (Ground Truth)

**PyBaMM DFN Model**
- **Chemistry:** NMC622 cathode / Graphite anode
- **Parameter Set:** Chen2020 validated parameters
- **Thermal Model:** Lumped thermal with 3 heat sources
  - Ohmic: Q_ohm = I² · R₀
  - Polarization: Q_pol = V₁²/R₁ + V₂²/R₂
  - Entropic: Q_ent = -I · T · (dU/dT)
- **Sensors:** Voltage, current, temperature with configurable noise

### Machine 2: Digital Twin (State Estimator)

**Equivalent Circuit Model (ECM)**
- **Topology:** 2-RC network (Thevenin model)
- **State Vector:** [SOC, V₁, V₂, T]
- **Parameters:** R₀, R₁, C₁, R₂, C₂, R_th, C_th
- **Adaptive Features:** Temperature-dependent resistance scaling

**Estimation Filters**
1. **AEKF:** Linearized Jacobians with Arrhenius correction
2. **UKF:** Unscented transform for nonlinear propagation
3. **Dual EKF:** Joint state-parameter estimation (R₀ adaptation)

---

## 📊 Application Interface

### Configuration Panel (Sidebar)

#### 🔋 Physical Asset Settings
- **Cycles:** 1-100 (default: 3)
- **Discharge C-rate:** 0.5-2.0C (default: 1.0C)
- **Sensor Noise:**
  - Voltage: σ_v = 5 mV
  - Temperature: σ_T = 0.2 K
  - Current: σ_I = 20 mA

#### ⚡ ECM Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| R₀ | 15 mΩ | 1-100 mΩ | Ohmic resistance |
| R₁ | 10 mΩ | 1-100 mΩ | Charge transfer resistance |
| C₁ | 2000 F | 10-100k F | Charge transfer capacitance |
| R₂ | 5 mΩ | 1-100 mΩ | Diffusion resistance |
| C₂ | 5000 F | 10-100k F | Diffusion capacitance |
| R_th | 15 K/W | 0.1-100 K/W | Thermal resistance |
| C_th | 500 J/K | 10-5000 J/K | Thermal capacitance |

#### 🧮 Filter Tuning Matrices
- **P₀:** Initial covariance (SOC, V₁, V₂, T)
- **Q:** Process noise covariance
- **R:** Measurement noise covariance (voltage, temperature)
- **Q_w:** Parameter process noise (for Dual EKF)

### Analysis Tabs

#### 🎯 Tab 1: AEKF Analysis
- SOC tracking with 95% confidence bounds
- Estimation error vs. filter uncertainty
- Voltage residuals (innovation sequence)
- NIS χ² consistency test

#### 🧠 Tab 2: UKF Analysis
- SOC tracking performance
- Uncertainty propagation comparison
- Core temperature tracking
- Sigma envelope evolution

#### ⚡ Tab 3: Dual EKF Analysis
- Online R₀ estimation trajectory
- Parameter uncertainty bounds
- SOC estimation with adaptive resistance
- Thermal tracking accuracy

#### 📊 Tab 4: Benchmark & Cycles
- Multi-filter absolute error comparison
- Uncertainty envelope benchmark
- Cycle-by-cycle metrics table
- Per-cycle RMSE/MAE breakdown

---

## 🔬 Mathematical Framework

### State-Space Formulation

**State Transition (Discrete-time):**
```
x[k+1] = f(x[k], u[k], Δt) + w[k]
y[k]   = h(x[k], u[k]) + v[k]
```

Where:
- `x = [SOC, V₁, V₂, T]ᵀ` (state vector)
- `u = I` (current input)
- `y = [V_terminal, T]ᵀ` (measurements)
- `w ~ N(0, Q)` (process noise)
- `v ~ N(0, R)` (measurement noise)

### ECM Equations

**SOC dynamics:**
```
SOC[k+1] = SOC[k] - (I · Δt) / (Q_nom · 3600)
```

**RC network dynamics:**
```
V₁[k+1] = exp(-Δt/τ₁) · V₁[k] + R₁ · (1 - exp(-Δt/τ₁)) · I
V₂[k+1] = exp(-Δt/τ₂) · V₂[k] + R₂ · (1 - exp(-Δt/τ₂)) · I
```

**Thermal dynamics:**
```
T[k+1] = T[k] + (Δt/C_th) · [Q_gen - (T[k] - T_amb)/R_th]
Q_gen = Q_ohmic + Q_polarization + Q_entropic
```

**Terminal voltage:**
```
V_terminal = OCV(SOC) - V₁ - V₂ - I · R₀(SOC, T)
```

### Uncertainty Quantification

**PICP (95% Confidence Interval):**
```
PICP = (1/N) · Σ 𝟙(SOC_true ∈ [μ - 2σ, μ + 2σ])
```

**NIS Test (χ² with df=2):**
```
NIS[k] = νᵀ[k] · S⁻¹[k] · ν[k]
ν[k] = y[k] - ŷ[k]  (innovation)
```

---

## 📄 PDF Report Generation

The system generates publication-ready PDF reports with:

### Section 1: Executive Summary
- Multi-filter performance metrics table
- SOC RMSE, Voltage RMSE, PICP comparison

### Section 2: System Configuration
- Operating conditions & sensor noise
- ECM parameter values
- Kalman filter tuning matrices (P₀, Q, R)

### Section 3-5: Filter-Specific Analysis (AEKF, UKF, Dual)
- 2 plots per page (strict layout)
- SOC tracking accuracy
- Estimation error vs. uncertainty bounds
- Temperature tracking (UKF/Dual)
- R₀ evolution (Dual EKF only)

### Section 6: Comparative Benchmark
- Multi-filter absolute error plot
- Uncertainty envelope comparison

### Section 7: Cycle-by-Cycle Table
- Per-cycle RMSE/MAE for SOC and voltage
- Tabulated metrics for all filters

**Download:** Click "📄 Download Official Engineering Report (PDF)" button

---

## 🎓 Scientific References

### Key Publications
1. **Plett, G.L. (2004)** - "Extended Kalman filtering for battery management systems of LiPB-based HEV battery packs"  
   *Journal of Power Sources*, dual EKF framework

2. **Chen et al. (2020)** - "Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models"  
   *Journal of The Electrochemical Society*, NMC622 parameter set

3. **Onori et al. (2024)** - Advanced battery management systems literature (web:44 reference)

### PyBaMM Framework
- **Sulzer et al. (2021)** - "Python Battery Mathematical Modelling (PyBaMM)"  
  *Journal of Open Research Software*

---

## 🛠️ Development & Deployment

### Project Structure
```
battsim-nmc622/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── packages.txt          # System dependencies (Kaleido)
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── .streamlit/
    └── config.toml      # Streamlit configuration
```

### Streamlit Cloud Deployment

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repository
   - Select `app.py` as main file
   - Deploy automatically handles `requirements.txt` and `packages.txt`

3. **Custom Domain (Optional):**
   - Add CNAME in Streamlit Cloud settings
   - Configure DNS records

---

## ⚙️ Advanced Configuration

### Custom ECM Models
Modify the `EquivalentCircuitModel` class to implement:
- Higher-order RC networks (3-RC, 4-RC)
- Hysteresis effects
- Nonlinear capacitance models

### Filter Extensions
Add new filters by inheriting base structure:
```python
class MyCustomFilter:
    def __init__(self, ecm, x0, P0, Q, R):
        # Initialize filter
        
    def step(self, y_meas, I, dt):
        # Prediction + update
        return {"soc": ..., "sigma_soc": ...}
```

### Custom UQ Metrics
Extend `UQMetrics` class:
```python
@staticmethod
def custom_metric(est, truth):
    return np.custom_function(est, truth)
```

---

## 🐛 Troubleshooting

### PDF Generation Fails
**Issue:** Kaleido rendering errors  
**Solution:** Ensure system packages are installed:
```bash
sudo apt-get update
sudo apt-get install -y libnss3 libnspr4 libatk-bridge2.0-0 \
  libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libasound2
```

### PyBaMM Simulation Timeout
**Issue:** DFN solver takes too long  
**Solution:** Reduce cycles or increase C-rate in sidebar

### Memory Issues
**Issue:** Large particle filter crashes  
**Solution:** Disable PF or reduce `n_particles` in code

---

## 📧 Contact & Support

**Developer:** Eng. Thaer Abushawer  
**Application:** [https://battsim.streamlit.app/](https://battsim.streamlit.app/)

For issues, feature requests, or contributions:
- Open an issue on GitHub
- Submit pull requests with detailed descriptions

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If you find this project useful, please consider starring the repository!**