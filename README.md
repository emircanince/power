# ⚡ Power Market Modeling

**Causal Analysis of Renewable Energy Penetration and Electricity Price Forecasting with Transformers**  
*Master’s Thesis – Humboldt-Universität zu Berlin*  
*Author : Emircan Ince*  
*Supervisors : Prof. Dr. Stefan Lessmann · Prof. Dr. Jan Mendling*

---

## Experimental Highlights

| Task              | Best Model | MAE  | MSE  | Horizon |
|-------------------|------------|------|------|---------|
| Price forecasting | **TimeXer**| **0.208** | **0.114** | 24 h |
| Price forecasting | DLinear    | 0.263 | 0.186 | 168 h |
| Causal impact     | Wind ↗ 1 pp| −0.8 €/MWh | — | up to 10 % penetration |

<details>
<summary>Visual snippets (click to expand)</summary>

<div align="center">
  <img src="png/LWPR_solar.png" alt="Solar LWPR surface" width="45%"/>
  <img src="png/LWPR_wind.png"  alt="Wind LWPR surface"  width="45%"/>
  <br/>
  <img src="png/mean_vs_cate_solar.png" alt="Solar CATE vs mean" width="45%"/>
  <img src="png/mean_vs_cate_wind.png"  alt="Wind CATE vs mean"  width="45%"/>
</div>
</details>

---

## Table of Contents
- [Summary](#summary)
- [Working with the repo](#working-with-the-repo)
  - [Dependencies](#dependencies)
  - [Setup](#setup)
- [Reproducing results](#reproducing-results)
  - [Causal analysis](#causal-analysis)
  - [Forecasting experiments](#forecasting-experiments)
  - [Visualization code](#visualization-code)
  - [Pre-trained models](#pre-trained-models)
- [Results](#results)
- [Project structure](#project-structure)

## Summary

This thesis quantifies how **solar and wind penetration affect German day-ahead electricity prices** and benchmarks state-of-the-art deep-learning forecasters.

* **Causal side.** A locally weighted polynomial regression (LWPR) visualises the merit-order surface. A *locally-partial* double-machine-learning (DML) estimator then delivers **conditional average treatment effects (CATE)** that remain negative up to ≈ 60 % wind share, with marginal price cuts of about **€ 0.8 per MWh** for the first 10 % of penetration.

* **Predictive side.** The custom **TimeXer** Transformer integrates endogenous price patches with variate-wise renewable and load tokens, achieving the lowest one-day MAE of **0.208 €/MWh**. For week-ahead horizons the parsimonious **DLinear** baseline prevails, underscoring the value of seasonal structure.

Overall, the work shows that *identification* (via DML) and *prediction* (via Transformers) complement each other when analysing power markets under high renewable penetration.

**Keywords:** Electricity Price Forecasting · Causal Inference · Double Machine Learning · Time-Series Transformers · Renewable Integration  
**Full text:** [`power.pdf`](./power.pdf)

---

## Working with the repo

### Dependencies
Developed with **Python 3.9**, **PyTorch 2.2**, **LightGBM 4.3**, and HuggingFace **transformers**.  
All packages are pinned in `requirements.txt`.

### Setup
~~~bash
# Clone project
git clone https://github.com/emircanince/power.git
cd power

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # on Linux/macOS
pip install -r requirements.txt
~~~

---

## Reproducing results

### Causal analysis
Compute LWPR, the overall DML estimate, and the CATE-vs-mean plot:
~~~bash
python models/nonlinear_regression_models/LWPR_mean_wind.py
python models/double_machine_learning/DML_overall_wind.py
python models/double_machine_learning/dml_vs_mean_plot_wind.py
~~~
Each script writes a CSV of estimates and exports figures to `png/`.

### Forecasting experiments
Train and evaluate **TimeXer** (plus five baselines):
~~~bash
chmod +x scripts/TimeXer.sh      # run once if needed
bash ./scripts/TimeXer.sh --pred_len 24    # options: 24 / 48 / 96 / 168
~~~
The script handles data splits, checkpoints, and TensorBoard logging automatically.

### Visualization code
Open `Visualization.ipynb` to  
1. load bootstrapped CATE estimates and create dose-response plots,  
2. overlay LWPR mean surfaces with DML-based marginal effects, and  
3. animate TimeXer’s forecast trajectory over training iterations.

### Pre-trained models
Download and unpack checkpoints:
~~~bash
unzip ckpt.zip -d ckpt
~~~
Copy the extracted folders into `./checkpoints/` to skip training.

---

## Results

* **Causal:** Wind penetration consistently lowers prices; solar effects taper beyond ≈ 30 % noon share.  
* **Forecasting:** TimeXer leads for short horizons; DLinear excels week-ahead; Transformer variants show clear horizon-wise trade-offs.  
* **Robustness:** Bootstrap confidence intervals, residual diagnostics, and cross-validation confirm estimator stability.

---

## Project structure
~~~text
.
├── ckpt.zip                          # pre-trained weights
├── requirements.txt                  # dependency list
├── README.md                         # << you are here
├── models
│   ├── nonlinear_regression_models
│   │   └── LWPR_mean_wind.py        # non-parametric merit-order surface
│   └── double_machine_learning
│       ├── DML_overall_wind.py      # average treatment effect
│       └── dml_vs_mean_plot_wind.py # CATE vs LWPR comparison
├── scripts
│   └── TimeXer.sh                    # shell launcher for forecasting
├── data                              # hourly price & forecast data (git-ignored)
└── png                               # generated figures
~~~

Happy forecasting! 🚀