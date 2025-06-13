import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt


# Function to generate polynomial features up to a given order for two-dimensional input
def polynomial_features(x, order=1):
    n = x.size(0)
    features = [torch.ones(n)]
    for i in range(1, order + 1):
        for j in range(i + 1):
            features.append((x[:, 0]**(i-j)) * (x[:, 1]**j))
    return torch.cat([f.unsqueeze(1) for f in features], dim=1)


# Tri-cube weight function
def tri_cube(u):
    return torch.where(u < 1, (1 - u**3)**3, torch.zeros_like(u))


# Calculate weights for all points based on a fitting point and bandwidth
def calculate_weights(data_points, fitting_point, bandwidth):
    if fitting_point.dim() == 1:
        fitting_point = fitting_point.unsqueeze(0)
    distances = torch.sqrt(torch.sum((data_points - fitting_point)**2, axis=1))
    scaled_distances = distances / bandwidth
    weights = tri_cube(scaled_distances)
    return weights


# Perform weighted least squares regression (optimized)
def weighted_least_squares(X, y, weights):
    # Assume X is n x m and weights is n x 1
    weighted_X = X * weights.unsqueeze(1)  # element-wise multiplication to weight each row
    XTWX = weighted_X.T @ X  # This replaces the need for explicitly creating W
    XTWy = weighted_X.T @ y
    beta = torch.linalg.pinv(XTWX) @ XTWy
    return beta.squeeze()


# Calculate adaptive bandwidth based on the percentile of distances
def adaptive_bandwidth(data_points, fitting_point, percentile=30):
    if fitting_point.dim() == 1:
        fitting_point = fitting_point.unsqueeze(0)
    distances = torch.sqrt(torch.sum((data_points - fitting_point)**2, axis=1))
    return torch.quantile(distances, percentile / 100.0)


def process_data(data, covariates, response):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data[covariates] = scaler.fit_transform(data[covariates])
    
    features = torch.tensor(data[covariates].values, dtype=torch.float32)
    target = torch.tensor(data[response].values, dtype=torch.float32)
    
    return features, target, scaler


def fit_model(features, target, n_fitting_points=24):
    x = torch.linspace(-1, 1, n_fitting_points)
    y = torch.linspace(-1, 1, n_fitting_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    fitting_points = torch.column_stack([X.ravel(), Y.ravel()])
    
    X_poly_base = polynomial_features(features)
    
    coefficients = torch.zeros(len(fitting_points), X_poly_base.size(1), dtype=torch.float32)
    predicted_values = torch.zeros(len(fitting_points), dtype=torch.float32)
    
    for i, fp in enumerate(tqdm(fitting_points, desc="Fitting Points")):
        bandwidth = adaptive_bandwidth(features, fp, percentile=30)
        weights = calculate_weights(features, fp, bandwidth)
        X_poly = polynomial_features(fp.clone().detach().unsqueeze(0))
        beta = weighted_least_squares(X_poly_base, target, weights)
        coefficients[i] = beta
        predicted_values[i] = torch.matmul(X_poly, beta)
    
    return fitting_points, predicted_values


def plot_results(n_grid_points, fitting_points, predicted_values, scaler, filename, plane_zero=False):
    fitting_points = fitting_points.cpu()  # Transfer to CPU
    predicted_values = predicted_values.cpu()  # Transfer to CPU
    
    grid_y, grid_x = np.meshgrid(np.linspace(-1, 1, n_grid_points), np.linspace(-1, 1, n_grid_points))
    grid_z = griddata(fitting_points.numpy(), predicted_values.numpy(), (grid_x, grid_y), method='cubic')
    
    original_grid_x, original_grid_y = scaler.inverse_transform(np.column_stack([grid_x.ravel(), grid_y.ravel()])).T
    original_grid_x = original_grid_x.reshape(grid_x.shape)
    original_grid_y = original_grid_y.reshape(grid_y.shape)
    
    fig = plt.figure(figsize=(6, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(original_grid_x, original_grid_y, grid_z, cmap='plasma', edgecolor='k', linewidth=0.5)
    contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='z', offset=np.min(grid_z), cmap='plasma', alpha=0.7)
    contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='x', offset=np.min(original_grid_x), cmap='plasma', alpha=0.7)
    contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='y', offset=np.max(original_grid_y), cmap='plasma', alpha=0.7)
    
    # Add a plane for price zero
    if plane_zero:
        ax.plot_surface(original_grid_x, original_grid_y, np.zeros_like(original_grid_x), color='red', alpha=0.8, rstride=100, cstride=100)
    
    cbar = fig.colorbar(surface, orientation='horizontal', shrink=0.5, aspect=40, pad=0.05, anchor=(0.5, 1.2))
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('Forecasted WP penetration (%)')
    ax.set_ylabel('Hour of the day')
    ax.set_zlabel('Price (EUR/MWh)')
    ax.set_zlim(-80, 250)
    plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85)
    if filename is not None:
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()