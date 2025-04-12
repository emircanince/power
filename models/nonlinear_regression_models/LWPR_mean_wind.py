from utils_mean import *
import matplotlib as mpl
import matplotlib.colors as mcolors

# Set plot aesthetics
mpl.rcParams.update({
    'font.family': 'Helvetica',  # Use Helvetica for clarity
    'axes.labelsize': 16,        # Larger axis labels
    'axes.titlesize': 16,        # Title font size
    'xtick.labelsize': 14,       # X-axis tick label size
    'ytick.labelsize': 14,       # Y-axis tick label size
    'legend.fontsize': 14,       # Legend font size
    'axes.grid': True,           # Enable grid
    'grid.alpha': 0.1,           # Subtle grid lines
    'grid.linestyle': '--',      # Dashed grid lines for a softer look
    'figure.dpi': 300,           # High DPI for high-quality figure
    'savefig.dpi': 300           # High DPI for saving the figure
})

df = pd.read_csv('data/causal_data.csv')

# Define covariates and response
covariates = ['wind_penetration', 'Hour']
response = 'electricity_price'

# Process data
input_variables, target_variable, scaler = process_data(data=df, covariates=covariates, response=response)

# Fit models
fitting_points, predicted_values = fit_model(features=input_variables, target=target_variable, n_fitting_points=24)

# Plot results
n_grid_points = 24

fitting_points = fitting_points.cpu()  # Transfer to CPU
predicted_values = predicted_values.cpu()  # Transfer to CPU

grid_y, grid_x = np.meshgrid(np.linspace(-1, 1, n_grid_points), np.linspace(-1, 1, n_grid_points))
grid_z = griddata(fitting_points.numpy(), predicted_values.numpy(), (grid_x, grid_y), method='cubic')

original_grid_x, original_grid_y = scaler.inverse_transform(np.column_stack([grid_x.ravel(), grid_y.ravel()])).T
original_grid_x = original_grid_x.reshape(grid_x.shape)
original_grid_y = original_grid_y.reshape(grid_y.shape)

# colors_wind = ["#f0f0f0", "#1f77b4"]  # Light gray to blue
# colors_wind = ["#add8e6", "#1f77b4"]  # Light blue to dark blue
colors_wind = ["#e0ffff", "#1f77b4"]  # Light cyan to bright blue
n_bins = 100  # Number of bins for colormap
# cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind_cmap", colors_wind, N=n_bins)
cmap_wind = 'plasma'
# cmap_wind = 'Blues'

fig = plt.figure(figsize=(8, 8), dpi=100)
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(original_grid_x, original_grid_y, grid_z, cmap=cmap_wind, edgecolor='k', linewidth=.5)
contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='z', offset=np.min(grid_z), cmap=cmap_wind, alpha=0.7)
contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='x', offset=np.min(original_grid_x), cmap=cmap_wind, alpha=0.7)
contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='y', offset=np.max(original_grid_y), cmap=cmap_wind, alpha=0.7)

cbar = fig.colorbar(surface, orientation='horizontal', shrink=0.5, aspect=50, pad=0.05)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlabel('Forecasted penetration [%]', labelpad=10)
ax.set_ylabel('Hour of the day', labelpad=10)
ax.set_zlabel('Price [EUR/MWh]', labelpad=10)
# ax.set_zlim(-80, 200)
plt.tight_layout()
plt.savefig(f'png/LWPR_wind.png',format='png', dpi=600)
plt.show()
