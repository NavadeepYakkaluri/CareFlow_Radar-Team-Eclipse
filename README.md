# CareFlow_Radar-Team-Eclipse
CareFlow Radar™ is an AI patient-flow engine built from scratch in NumPy. It predicts hospital bottlenecks, eliminates "The Forgotten Patient" 52-minute post-consultation test lag, and features an impact-ranked action queue with an interactive "What-If" staff simulator to reduce patient delays by 25–35%.

# 🩺 CareFlow Radar™ — Clinical Patient-Flow & Bottleneck Intelligence

> **Datathon 2026 — Track:** *Health: The Forgotten Patient*  
> **Mission:** Transforming reactive hospital dashboards into an actionable, sub-2ms early-warning engine that predicts patient bottlenecks and uncovers the "Forgotten Patient" lag.

[![Live Prototype](https://img.shields.io/badge/Prototype-Interactive%20SPA-06b6d4?style=for-the-badge&logo=googlechrome&logoColor=white)](#-quick-start)
[![ML Engine](https://img.shields.io/badge/ML%20Engine-From--Scratch%20NumPy-10b981?style=for-the-badge&logo=python&logoColor=white)](#-ml-performance--benchmarks)
[![Inference Latency](https://img.shields.io/badge/Inference-%3C2ms%20Latency-8b5cf6?style=for-the-badge)](#-system-architecture)
[![Evaluation Split](https://img.shields.io/badge/Validation-Time--Aware%20Split-f59e0b?style=for-the-badge)](#-ml-performance--benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Zero Build Setup](https://img.shields.io/badge/Frontend-Vanilla%20JS%20%7C%20CSS3-ec4899?style=for-the-badge)](#-quick-start)

---

## 📌 Executive Summary

Traditional hospital management systems are **retrospective**—they inform charge nurses of a crisis only after wait times exceed acceptable thresholds and wards become gridlocked. 

**CareFlow Radar™** introduces a proactive clinical intelligence layer. By fusing real-time departmental census metrics, triage severity indices, and diagnostic turn-around times (TAT), CareFlow Radar predicts patient wait times, flags bottlenecks before they cascade, and shines a spotlight on the **"Forgotten Patient"**—those patients who endure silent, post-consultation delays exceeding 50+ minutes waiting for imaging, lab clearance, or bed transfers.

---

## 🚀 Key Prototype Modules

CareFlow Radar is designed as a zero-dependency, ultra-responsive Single Page Application (SPA) offering five dedicated operational cockpits:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CareFlow Radar™ Engine                          │
└───────┬──────────────┬──────────────┬──────────────┬───────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
 1. Executive   2. Investigate  3. Diagnostic   4. ML Signal    5. What-If
     Pulse          First         TAT & Lag      Explainer       Sandbox
   (12 Wards)    (Queue Triage)  (Forgotten Pt)   (Ridge/GBDT)  (Simulation)
```

### 1. 🏥 Hospital Ward Overview & Stress Heatmap
* Real-time monitoring across **12 hospital departments** (Emergency, Pediatrics, Cardiology, Neurology, Radiology, Laboratory, Surgery, etc.).
* Dynamic **Capacity vs. Census Stress Ratios** with visual color-coded thresholds (Optimal, Elevated, Critical).
* Real-time KPI ribbons tracking total visits (8,000 monitored), model Mean Absolute Error (**13.97 min**), and severe delay prevalence.

### 2. 🎯 "Investigate First" Action Engine
* Automated alert queue isolating high-risk patients approaching the severe delay threshold (>60 mins).
* Risk-scoring that combines triage priority with current ward bed saturation.
* Instant recommended clinical mitigations (e.g., triage fast-tracking, dynamic bed pre-clearing).

### 3. 🧪 Diagnostic TAT & "The Forgotten Patient" Deep-Dive
* Quantifies the hidden **Post-Service Lag** (averaging **52.6 min** in Pediatrics and **51.2 min** in Laboratory).
* Pinpoints delays occurring after clinical consultation while awaiting lab/radiology clearance.
* Identifies system friction points causing patients to be lost in transition.

### 4. 🧠 ML Performance & Feature Signals
* Live explainability panel detailing model weights and feature importances.
* Compares baseline heuristics vs. custom-built models on identical time-aware test sets.
* Shows how factors like Low Triage (+25.8m wait weight), Diagnostics routing (+2.28m), and Bed Count (-1.69m) influence delays.

### 5. 🎛️ "What-If" Operational Sandbox
* Interactive clinical simulator enabling charge nurses and operations executives to simulate staffing and bed interventions.
* Adjust staff count (+1 to +5) or bed availability (+2 to +10) per ward to compute projected wait-time reductions in real time.
* Delivers immediate ROI calculations for resource reallocations.

---

## 📊 ML Performance & Benchmarks

The predictive engine was evaluated on **8,000 hospital encounters** across **2,000 patients** using a strict **Time-Aware Temporal Split** (`2025-10-18 13:38:00`), ensuring 0% future-data leakage.

| Model / Approach | MAE (mins) | RMSE (mins) | $R^2$ Score | Target / Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Global Mean Baseline** | 20.07 min | 24.74 min | 0.000 | Naive hospital-wide average |
| **Triage-Only Baseline** | 14.42 min | 18.08 min | 0.466 | Standard clinical heuristic |
| **Ridge Regressor (From Scratch)** 🏆 | **13.97 min** | **17.67 min** | **0.490** | Closed-form $L_2$ regularized solution |
| **GBDT Ensemble (From Scratch)** | 16.04 min | 20.17 min | 0.336 | Non-linear tree boosting |
| **Severe Delay Classifier** | **73.0% Acc** | **75.2% Prec** | **0.784 AUC** | Binary early-warning flag (>60m delay) |

### Key Clinical Feature Attributions
* **Triage Level Low (ESI-4/5):** `+25.8 min` (41.4% importance) — Non-urgent patients suffer highest delay.
* **Triage Level Medium (ESI-3):** `+15.5 min` (24.8% importance) — Vulnerable middle tier at highest risk of decompensation.
* **Service Type (Diagnostics/Outpatient):** `+2.1m to +2.3m` — Bottlenecks at test acquisition points.
* **Bed Availability:** `-1.69 min per bed` — Strongest negative coefficient alleviating congestion.

---

## 🛠️ Tech Stack & Design Choices

* **Frontend:** Vanilla HTML5, CSS3, Modern ES6+ JavaScript.
  * **Zero build step:** No `npm install`, Webpack, or Vite required.
  * **Design System:** Custom CSS tokens, glassmorphism card surfaces, CSS grid responsive layout.
  * **Themes:** Instant toggle between **Clinical Day Shift** (light, crisp readability) and **ICU Night Monitor** (dark mode for low-fatigue night stations).
* **Machine Learning Engine:** Pure Python & NumPy (`train_engine.py`).
  * Implemented from first mathematical principles (Normal equations with $L_2$ regularization, MSE loss minimization).
  * Generates zero-dependency serialized intelligence (`model_results.json`).
* **Interoperability Standard:** Designed to map seamlessly to **HL7 FHIR v4** resources (`Encounter`, `Patient`, `Observation`).

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/careflow-radar.git
cd careflow-radar
```

### 2. Launch the Prototype (Zero Installation)
Because the prototype uses vanilla web standards, you can run it with any static web server:

**Using Python (Recommended):**
```bash
python3 -m http.server 8080 --directory prototype
```

**Using Node.js:**
```bash
npx serve prototype
```

Then open your browser and navigate to:
👉 **`http://localhost:8080`**

### 3. (Optional) Retrain the Machine Learning Engine
To regenerate the model weights, metrics, and JSON data from raw CSV files:
```bash
python3 train_engine.py
```
This updates `prototype/model_results.json` directly.

---

## 📂 Repository Structure

```text
├── prototype/                      # Complete client-side web application
│   ├── index.html                  # Semantic SPA architecture & cockpit views
│   ├── style.css                   # Custom clinical UI design system & dark mode
│   ├── app.js                      # Reactive view controller & simulation engine
│   ├── model_results.json          # Pre-computed model inference & ward KPIs
│   └── README.md                   # Prototype documentation
├── train_engine.py                 # Pure NumPy ML pipeline (Ridge, GBDT, Time-Split)
├── departments.csv                 # 12 Hospital wards, staff, and capacity specs
├── patients.csv                    # 2,000 synthetic patient demographic records
├── encounters.csv                  # 8,000 visit logs, triage, timestamps, wait times
├── diagnostic_tests.csv            # 5,000 diagnostic records with TAT metrics
└── README.txt                      # Datathon dataset notice & educational disclaimer
```

---

## 🩺 Clinical Feasibility & Governance

* **No Overhead for Clinicians:** CareFlow Radar ingests operational telemetry passively from ADT (Admission, Discharge, Transfer) systems without requiring nurses to enter additional forms.
* **Deterministic & Auditable:** By prioritizing mathematically transparent models (Ridge & explicit decision splits), decisions are fully explainable to clinical review boards.
* **Synthetic Data Notice:** Built and evaluated on the Datathon 2026 synthetic educational dataset. Real-world hospital deployment requires validation on de-identified clinical repositories (e.g., MIMIC-IV) and institutional IRB clearance.

---

## 👥 Authors & Datathon Team

Developed for **Datathon 2026** — *Health: The Forgotten Patient Track*.
* **Lead Engineers & Researchers:** *YAKKALURI NAVADEEP AND BATTA TEJASRI*
* **Institution:** *VIT AP UNIVERSITY*

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — see the [LICENSE](LICENSE) file for details. You are free to use, modify, distribute, and build upon this prototype for academic, clinical research, and operational benchmarking purposes.

---
*CareFlow Radar™ — Transforming hospital data into life-saving, proactive clinical intelligence.*
