import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from DML_utils_wind import residualize_data, fit_residualized_model

# Load data
# df = pd.read_csv('/Users/emircanince/Desktop/power/data/causal_data.csv')
df = pd.read_csv('data/causal_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Scale the relevant columns
df['total_load'] = df['total_load'] / 1000
df['wind_forecast'] = df['wind_forecast'] / 1000
df['solar_forecast'] = df['solar_forecast'] / 1000

# Sort the DataFrame by wind penetration
df = df.sort_values(by='wind_penetration')

# Parameters for sliding window
window_size = 5000  # 10000
step_size = 500     # 1000
n_iterations = 100  # Number of bootstraps

# Store CATE and corresponding penetration levels for each window
results = []

# Total number of windows
total_windows = (len(df) - window_size) // step_size + 1

# Sliding window analysis
for start in range(0, len(df) - window_size + 1, step_size):
    window_data = df.iloc[start:start + window_size]

    # Calculate the mean solar penetration for the current window
    mean_wind_penetration = window_data['wind_penetration'].mean()

    # Use tqdm for bootstrap iterations
    with tqdm(total=n_iterations, desc=f'Window {start // step_size + 1}/{total_windows}', leave=False) as pbar:
        # Bootstrap for the current window
        for _ in range(n_iterations):
            random_subset = window_data.sample(n=min(len(window_data), window_size), replace=True)
            df_residualized = residualize_data(random_subset, y='electricity_price')
            res = fit_residualized_model(df_residualized)

            # Append the mean solar penetration and CATE to results
            results.append({
                'mean_wind_penetration': mean_wind_penetration,
                'cate': res[0]  # Extract the first (and only) coefficient
            })
            pbar.update(1)  # Update the progress bar

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save raw results
results_df.to_csv('data/results_wind.csv', index=False)
results_dataset = pd.read_csv('data/results_wind.csv')

# Group by mean solar penetration and calculate mean and quantiles for CATE
mean_cate_df = results_dataset.groupby('mean_wind_penetration')['cate'].agg(['mean']).reset_index()
mean_cate_df['lower_ci'] = results_dataset.groupby('mean_wind_penetration')['cate'].quantile(0.1).values
mean_cate_df['upper_ci'] = results_dataset.groupby('mean_wind_penetration')['cate'].quantile(0.9).values

# Apply Gaussian smoothing
sigma_smoothing = np.var(mean_cate_df['mean'])
mean_cate_df['smoothed_mean'] = gaussian_filter1d(mean_cate_df['mean'], sigma=sigma_smoothing)
mean_cate_df['smoothed_lower_ci'] = gaussian_filter1d(mean_cate_df['lower_ci'], sigma=sigma_smoothing)
mean_cate_df['smoothed_upper_ci'] = gaussian_filter1d(mean_cate_df['upper_ci'], sigma=sigma_smoothing)

# Plotting the smoothed mean CATE against mean wind penetration with 80% CI
plt.figure(figsize=(6, 6), dpi=100)
plt.plot(mean_cate_df['smoothed_mean'], mean_cate_df['mean_wind_penetration'], marker='', linestyle='-',
         color='b', label='Mean', lw=3)
plt.fill_betweenx(mean_cate_df['mean_wind_penetration'], mean_cate_df['smoothed_lower_ci'],
                  mean_cate_df['smoothed_upper_ci'],
                  color='c', alpha=.3, label='80% CI')
plt.ylabel('Forecasted wind penetration (%)', fontsize=16)  # Now this is on y-axis
plt.xlabel('CATE (EUR/MWh)', fontsize=16)  # Now this is on x-axis
plt.axvline(0, color='gray', linestyle='-', lw=.8)  # Add a vertical line at x=0
plt.grid(axis='x', linestyle='-', alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.xlim(-1.1, -0.3)
plt.savefig('png/smooth_cate_wind.png', format='png', dpi=600)
plt.show()

plt.figure(figsize=(6, 6), dpi=100)
plt.scatter(results_df['cate'], results_df['mean_wind_penetration'], alpha=0.2, color='c',
            label='Individual CATE estimates')
# plt.axvline(0, color='k', linestyle='-', lw=1, alpha=0.5)  # Add a vertical line at x=0
plt.xlabel('CATE (EUR/MWh)', fontsize=16)
plt.ylabel('Forecasted wind penetration (%)', fontsize=16)
plt.grid(axis='x', linestyle='-', alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.xlim(-1.1, -0.3)
plt.savefig('png/individual_cate_wind.png', format='png', dpi=600)
plt.show()