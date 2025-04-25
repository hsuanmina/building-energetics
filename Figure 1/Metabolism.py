import numpy as np
from scripts import myTools
from plotnine import *
from sklearn.linear_model import LinearRegression

# === Functionality Overview ===
# This script:
# 1. Loads Time Series data and CY2023 building data.
# 2. Filters the main dataset to only include buildings present in the 2023 data.
# 3. Merges building volume and log-transformed building characteristics into the main dataset.
# 4. Selects buildings that started use in 2024.
# 5. Fits a log-log linear regression model of energy use (GJ) vs. building volume.
# 6. Visualizes the results using plotnine with a regression line and annotations.
# 7. Saves the resulting plot.

# === Load and Preprocess Data ===
# Load cleaned time series data and CY2023 data
data = myTools.load_data('cleaned_Time_Series.csv')
data_2023 = myTools.load_data('cleaned_CY2023.csv')

# Get unique building numbers from 2023 data
building_numbers = data_2023['BUILDING_NUMBER'].unique()

# Filter the main dataset to only include buildings from the 2023 data
data = data[data['BUILDING_NUMBER'].isin(building_numbers)]
data.reset_index(drop=True, inplace=True)

# Merge volume and log-transformed metadata from 2023 into the main dataset
data = data.merge(
    data_2023[['BUILDING_NUMBER', 'BUILDING_VOLUME', 'LOG_BUILDING_VOLUME',
               'LOG_BUILDING_HEIGHT', 'LOG_EXT_GROSS_AREA']].drop_duplicates(),
    on='BUILDING_NUMBER',
    how='left'
)

# === Filter to Only 2024 Buildings ===
# Keep only rows corresponding to buildings with use starting in 2024
data_24 = data[data['START_DATE_USE_YEAR'] == 2024]

# === Prepare Data for Plotting and Regression ===
# Rename columns for clarity in plot construction
volume_energy_data_24 = data_24.rename(columns={"BUILDING_VOLUME": "X", "GJ": "y"})

# Take log10 of X and y for log-log regression
X_energy = np.log10(volume_energy_data_24["X"].values).reshape(-1, 1)
y_energy = np.log10(volume_energy_data_24["y"].values)

# Fit a linear regression model to log-transformed data
model_energy = LinearRegression().fit(X_energy, y_energy)
slope_energy = model_energy.coef_[0]         # Power-law exponent (alpha)
intercept_energy = model_energy.intercept_   # Intercept of the linear model
r_squared_energy = model_energy.score(X_energy, y_energy)  # R² value

# === Plot: Energy vs Volume with Regression Line and Annotations ===
p = (
    ggplot(volume_energy_data_24, aes(x='X', y='y'))
    + geom_smooth(  # Add regression line with confidence interval
        method='lm',
        se=True,
        color='#b0c4b1',
        fill='#a4c3b2',
        alpha=0.3,
        size=1
    )
    + geom_point(size=3, color='#4a5759', alpha=0.6)  # Plot data points
    + labs(
        x="Volume (m³)",
        y="Energy Consumption (GJ)",
    )
    + annotate(  # Add power-law relationship label
        "text",
        x=min(volume_energy_data_24["X"]) + 0.01,
        y=max(volume_energy_data_24["y"]) - 10000,
        label=r"$E \propto V^{\alpha}$",
        size=15,
        color="black",
        ha="left",
    )
    + annotate(  # Add slope and R² annotation
        "text",
        x=min(volume_energy_data_24["X"]) + 0.01,
        y=max(volume_energy_data_24["y"]) - 75000,
        label=rf"$\alpha = {slope_energy:.2f}, R^2 = {r_squared_energy:.2f}$",
        size=15,
        color="black",
        ha="left",
    )
    + scale_x_log10(labels=lambda x: [f'$10^{{{int(np.log10(i))}}}$' for i in x])  # Log scale for X axis
    + scale_y_log10(labels=lambda y: [f'$10^{{{int(np.log10(i))}}}$' for i in y])  # Log scale for Y axis
    + theme_minimal()  # Minimalist theme
    + theme(  # Customize fonts and grid
        axis_text_x=element_text(size=15, rotation=45),
        axis_text_y=element_text(size=15),
        axis_title_x=element_text(size=17, weight='bold'),
        axis_title_y=element_text(size=17, weight='bold'),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color="black", fill=None, size=0.2),
        panel_background=element_blank()
    )
)

# === Save Plot ===
# Save the generated plot as a high-resolution PNG image
p.save("MIT_Metabolism.png", dpi=300)