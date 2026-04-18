---
title: BattSim
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: battsim.py
pinned: false
---

# 🔋 BattSim — DFN ↔ DEKF Co-Simulation

**Research-grade battery simulator** using PyBaMM (DFN model) coupled with a Dual Extended Kalman Filter (DEKF) for real-time State-of-Charge (SOC) and State-of-Health (SOH) estimation.

## Features
- 🧪 NMC / LFP / NCA battery chemistries
- ⚡ Configurable C-Rate and cycle count
- 📡 Sensor noise simulation
- 📊 Interactive Plotly dashboard
- 🔬 Uncertainty Quantification (UQ)

## Usage
1. Select battery chemistry from the sidebar
2. Set number of cycles, C-Rate, and sensor noise
3. Click **Run co-simulation**
4. Explore results in the dashboard

## Model
- **DFN**: Doyle-Fuller-Newman electrochemical model via PyBaMM
- **DEKF**: Dual Extended Kalman Filter for SOC + SOH estimation
