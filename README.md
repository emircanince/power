# ⚡ Power Market Modeling

**Causal Analysis of Renewable Energy Penetration and Electricity Price Forecasting with Transformers**  
*Master’s Thesis – Humboldt-Universität zu Berlin*  
*Author: Emircan Ince*  
*Supervisors: Prof. Dr. Stefan Lessmann · Prof. Dr. Jan Mendling*

---

## Quick Start

Follow these steps to get up and running:

1. **Clone the repo**
   ```bash
   git clone https://github.com/emircanince/power.git
   cd power

	2.	(Optional) Create Conda environment

echo "name: power
channels:
  - defaults
dependencies:
  - python=3.9
  - pip
  - pip:
    - -r requirements.txt" > environment.yml
conda env create -f environment.yml
conda activate power


	3.	Install Python requirements

pip install -r requirements.txt


	4.	Run experiments
	•	Causal effect at 10% wind share:

python models/double_machine_learning/DML_overall_wind.py --treatment-share 0.10


	•	Forecast 48-hour price trajectory:

bash scripts/TimeXer.sh --pred_len 48



⸻

Table of Contents
	•	Summary
	•	Installation
	•	Usage Examples
	•	Project Structure
	•	Reproducing Results
	•	Causal Analysis
	•	Forecasting Experiments
	•	Visualization Code
	•	Pre-trained Models
	•	Results
	•	Contributing
	•	Citation
	•	License

⸻

Summary

This thesis quantifies how solar and wind penetration affect German day-ahead electricity prices and benchmarks state-of-the-art deep-learning forecasters:
	•	Causal analysis
	•	LWPR (locally weighted polynomial regression) visualises the merit-order surface.
	•	Double Machine Learning (DML) estimates conditional average treatment effects (CATE), showing marginal price cuts of ~€0.8/MWh for the first 10% wind share and negative effects up to ~60%.
	•	Predictive modeling
	•	The custom TimeXer Transformer integrates price patches with renewable and load tokens, achieving a 24 h MAE of 0.208 €/MWh.
	•	The DLinear baseline excels at 168 h horizons, highlighting seasonal patterns.

Keywords: Electricity Price Forecasting · Causal Inference · Double Machine Learning · Time Series Transformers · Renewable Integration

Download full thesis (PDF)

⸻

Installation
	1.	Clone the repo

git clone https://github.com/emircanince/power.git


	2.	Create virtual environment

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows


	3.	Install dependencies

pip install -r requirements.txt



Alternatively, use the provided environment.yml for Conda environments.

⸻

Usage Examples

Causal Analysis

# Estimate overall wind penetration effect (10% share)
python models/double_machine_learning/DML_overall_wind.py --treatment-share 0.10

# Compare CATE vs. mean with plots
python models/double_machine_learning/dml_vs_mean_plot_wind.py

Price Forecasting

# One-day forecast (24 h)
bash scripts/TimeXer.sh --pred_len 24

# Week-ahead forecast (168 h)
bash scripts/TimeXer.sh --pred_len 168

Visualization Code

Open Visualization.ipynb to:
	1.	Plot bootstrapped CATE dose–response curves.
	2.	Overlay LWPR and DML marginal effects.
	3.	Animate TimeXer’s training forecasts.

⸻

Project Structure

Path	Description
ckpt.zip	Pre-trained model weights
requirements.txt	Pinned Python dependencies
environment.yml	Sample Conda environment spec (optional)
README.md	Project overview (this file)
models/	Core Python modules
models/nonlinear_regression_models/	LWPR merit-order surface scripts
models/double_machine_learning/	DML scripts for causal estimates
scripts/TimeXer.sh	Shell script to train/evaluate TimeXer
data/	(not tracked) Hourly price & renewable data
png/	Generated figures and visual snippets


⸻

Reproducing Results

Causal Analysis

python models/nonlinear_regression_models/LWPR_mean_wind.py
python models/double_machine_learning/DML_overall_wind.py
python models/double_machine_learning/dml_vs_mean_plot_wind.py

Forecasting Experiments

bash scripts/TimeXer.sh --pred_len {24,48,96,168}

Pre-trained Models

unzip ckpt.zip -d ckpt
mv ckpt/* checkpoints/


⸻

Results
	•	Causal: Wind penetration consistently reduces prices; solar effects taper beyond ~30% noon share.
	•	Forecasting: TimeXer leads at short horizons; DLinear at week-ahead.
	•	Robustness: Bootstrapped confidence intervals, residual diagnostics, and cross-validation confirm stability.

⸻

Contributing

Contributions are welcome! Please:
	1.	Fork the repository and create a new branch.
	2.	Add tests for new features or bug fixes.
	3.	Submit a pull request with a clear description of your changes.

For questions or bug reports, open an issue or email Emircan Ince at [your-email@domain.com].

⸻

Citation

If you use this work, please cite:

Ince, E. (2025). Causal Analysis of Renewable Energy Penetration and Electricity Price Forecasting with Transformers.
Master’s Thesis, Humboldt-Universität zu Berlin.

⸻

License

This project is licensed under the MIT License. See LICENSE for details.