# Power Market Modeling
**Causal Analysis of Renewable Energy Penetration and Electricity Price Forecasting with Transformers**

*Master’s Thesis – Humboldt-Universität zu Berlin*  
*Author : Emircan Ince*
*Supervisors : Prof. Dr. Stefan Lessmann · Prof. Dr. Jan Mendling*

## Causal Impact of Renewable Penetration

This section quantifies the short-run price effects of renewable generation in the German day-ahead market by applying causal machine learning techniques.

### Non-parametric mean surfaces (LWPR)

<p align="center">
  <img src="png/solar_2.png" alt="Solar LWPR" width="49%"/>
  <img src="png/wind_2.png" alt="Wind LWPR" width="45%"/>
</p>

- **Solar** *(left)*: Steep price drops up to ~25%; strongest around noon (e.g. −3.1 €/MWh at 15%, 12:00).
- **Wind** *(right)*: Consistent decline up to ~60% share. Smoother and persistent across 24h, especially overnight.

### Causal Heterogeneity (DML)

<p align="center">
  <img src="png/mean_vs_cate_solar.png" alt="Solar LWPR" width="45%"/>
  <img src="png/mean_vs_cate_wind.png" alt="Wind LWPR" width="45%"/>
</p>

- **Solar CATE** *(left)*: Strong at low shares (−0.95 €/MWh at 5%). Weakens but remains negative beyond 30%.
- **Wind CATE** *(right)*: Stable impact (−0.8 €/MWh up to 10%, −0.75 €/MWh around 30%). Persists up to 60%.

## Forecasting Results

The forecasting section benchmarks six models, including both deep learning architectures and linear baselines across multiple prediction horizons, focusing on accuracy and stability under a unified experimental protocol.

| Horizon | Metric | TimeXer       | iTransformer | PatchTST     | DLinear       | SCINet        | Autoformer    |
|:-------:|:------:|:-------------:|:------------:|:------------:|:-------------:|:-------------:|:-------------:|
| 24h     | MSE    | **0.208**     | 0.302        | 0.380        | 0.209         | 0.210         | 0.261         |
|         | MAE    | **0.114**     | 0.212        | 0.288        | 0.118         | 0.117         | 0.153         |
| 48h     | MSE    | 0.243         | 0.303        | 0.360        | **0.234**     | 0.247         | 0.268         |
|         | MAE    | 0.152         | 0.217        | 0.269        | **0.147**     | 0.154         | 0.175         |
| 96h     | MSE    | 0.274         | 0.318        | 0.330        | **0.254**     | 0.288         | 0.280         |
|         | MAE    | 0.187         | 0.236        | 0.251        | **0.172**     | 0.205         | 0.191         |
| 168h    | MSE    | 0.278         | 0.322        | 0.368        | **0.263**     | 0.289         | 0.285         |
|         | MAE    | 0.192         | 0.240        | 0.274        | **0.186**     | 0.204         | 0.201         |

*Best values per horizon are bolded.*

## Working with the Repository

### Dependencies

Developed with:

- **Python 3.9**
- **PyTorch 2.0**
- **LightGBM 4.3**

All dependencies are listed and pinned in `requirements.txt`.

### Setup

```bash
# Clone the repository
git clone https://github.com/emircanince/power.git
cd power

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # for macOS/Linux
# On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Reproducing Results

### Causal Analysis

To compute the LWPR surface, the overall DML estimate, and compare CATE to the observational mean, run:

```bash
python models/nonlinear_regression_models/LWPR_mean_wind.py
python models/double_machine_learning/DML_overall_wind.py
python models/double_machine_learning/dml_vs_mean_plot_wind.py
```
*Replace wind with solar in the filenames to run the analysis on the solar dataset.*

Outputs are written to CSV and saved as figures in the `png/` directory.

### Forecasting Experiments

To train and evaluate **TimeXer** and the benchmark models:

```bash
chmod +x scripts/TimeXer.sh              # run once
bash scripts/TimeXer.sh --pred_len 24    # options: 24 / 48 / 96 / 168
```

The script handles data loading, training, checkpoints, and TensorBoard logging automatically.

Models: TimeXer, iTransformer, PatchTST, DLinear, SCINet, Autoformer

*Change `TimeXer.sh` to the corresponding script (e.g., `iTransformer.sh`) to run a different model.*

## Project Structure

```text
.
├── checkpoints/                      # Model checkpoints
├── data/                             # Raw input data (thesis_data.csv)
├── data_provider/                    # Data loading and preprocessing logic
├── exp/                              # Experiment configurations
├── layers/                           # Custom neural network layers
├── models/                           # Causal analysis + forecasting models
│   ├── double_machine_learning/      # DML-based treatment effect estimation
│   ├── nonlinear_regression_models/  # LWPR models
│   ├── Autoformer.py
│   ├── DLinear.py
│   ├── Informer.py
│   ├── PatchTST.py
│   ├── SCINet.py
│   ├── TimeXer.py
│   └── iTransformer.py
├── png/                              # Output figures and plots
├── results/                          # Stored outputs
├── scripts/                          # Shell scripts for model training/evaluation
├── test_results/                     # Test logs
├── utils/                            # Utility functions and helpers
│
├── .gitignore                        # Git exclusion rules
├── README.md                         # **← You are here**
├── requirements.txt                  # Dependency specification
├── run.py                            # Main script to launch workflows
```