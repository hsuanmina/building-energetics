import pandas as pd
from plotnine import *
import numpy as np
from scipy.stats import linregress
from scripts import myTools

# === Load Time Series and CY2023 Building Data ===
data = myTools.load_data('cleaned_Time_Series.csv')
data_energy = myTools.load_data('cleaned_CY2023.csv')

# Filter time series data to include only buildings that appear in the 2023 dataset
building_numbers = data_energy['BUILDING_NUMBER'].unique()
data = data[data['BUILDING_NUMBER'].isin(building_numbers)]

# Merge selected building metadata into the time series data
data = data.merge(
    data_energy[['BUILDING_NUMBER', 'CLUSTER_NUM', 'LOG_BUILDING_VOLUME',
                 'LOG_EXT_GROSS_AREA', 'BUILDING_HEIGHT', 'EXT_GROSS_AREA']].drop_duplicates(),
    on='BUILDING_NUMBER',
    how='left'
)

# === Function to Perform Yearly Log-Log Regression Analysis ===
def calculate_stats(data, x_col, y_col, standardize=False, normalize=False):
    """
    Perform log-log linear regression by year and calculate slope (α), intercept (β),
    confidence intervals, and residuals for each year.

    Parameters:
        data (DataFrame): Combined dataset with year, x, y columns.
        x_col (str): Column name to use as predictor (e.g., log volume).
        y_col (str): Column name to use as response (e.g., log energy).
        standardize (bool): Whether to standardize x and y (mean=0, std=1).
        normalize (bool): Whether to normalize x and y (scale 0–1).

    Returns:
        stats (DataFrame): Regression coefficients and stats per year.
        residuals (DataFrame): Residuals for each observation.
    """
    stats = []
    residuals = []

    for year in data['START_DATE_USE_YEAR'].unique():
        year_data = data[data['START_DATE_USE_YEAR'] == year]

        # Standardize or normalize if requested
        if standardize:
            x = (year_data[x_col] - year_data[x_col].mean()) / year_data[x_col].std()
            y = (year_data[y_col] - year_data[y_col].mean()) / year_data[y_col].std()
        elif normalize:
            x = (year_data[x_col] - year_data[x_col].min()) / (year_data[x_col].max() - year_data[x_col].min())
            y = (year_data[y_col] - year_data[y_col].min()) / (year_data[y_col].max() - year_data[y_col].min())
        else:
            x = year_data[x_col]
            y = year_data[y_col]

        # Perform regression
        model = linregress(x, y)
        predicted_y = model.intercept + model.slope * x
        residuals_year = y - predicted_y

        # Store residuals
        residuals.append(pd.DataFrame({
            'START_DATE_USE_YEAR': year,
            'BUILDING_NUMBER': year_data['BUILDING_NUMBER'],
            'residual': residuals_year,
            'observed': y
        }))

        # Calculate 98% confidence intervals (rough approximation)
        alpha_ci = 0.98 * model.stderr
        beta_ci = 0.98 * model.intercept_stderr

        # Store regression statistics
        stats.append({
            'year': year,
            'alpha_volume': model.slope,
            'alpha_std_err_volume': model.stderr,
            'alpha_ci_volume': alpha_ci,
            'beta_volume': model.intercept,
            'beta_std_err_volume': model.intercept_stderr,
            'beta_ci_volume': beta_ci,
            'rvalue_volume': model.rvalue,
            'pvalue_volume': model.pvalue
        })

    return pd.DataFrame(stats), pd.concat(residuals).reset_index(drop=True)


# === Run Regression and Extract Statistics ===
stats, residuals = calculate_stats(data, 'LOG_BUILDING_VOLUME', 'LOG_GJ')

# Add confidence interval bounds for slope (α)
stats['alpha_volume_lower'] = stats['alpha_volume'] - stats['alpha_ci_volume']
stats['alpha_volume_upper'] = stats['alpha_volume'] + stats['alpha_ci_volume']

# === Plot Trend of α (Scaling Exponent) over Time ===
label = r"$E \propto V^{\alpha}$"
p = (
    ggplot(stats, aes(x='year', y='alpha_volume'))
    + geom_rect(  # Highlight years after 2020
        aes(xmin=2020, xmax=float('inf'), ymin=-float('inf'), ymax=float('inf')),
        fill='#edf2fb', alpha=0.1
    )
    + geom_hline(yintercept=0.75, color='#e0e1dd', linetype='dashed', size=0.9)  # Reference line
    + geom_errorbar(  # Add error bars
        aes(ymin='alpha_volume_lower', ymax='alpha_volume_upper'),
        width=0.2,
        color='#778da9'
    )
    + annotate(  # Annotate power-law relationship
        "text",
        x=min(stats['year']),
        y=0.88,
        label=label,
        size=15,
        color="black",
        ha="left",
        va="top"
    )
    + geom_line(size=3, color='#778da9', alpha=0.45)
    + geom_point(size=4.5, color='#778da9', alpha=0.85)
    + geom_text(  # Label α values on points
        aes(label='round(alpha_volume, 2)'),
        nudge_y=0.01,
        size=10
    )
    + labs(x='Year', y='α (scaling exponent)')
    + theme_minimal()
    + scale_y_continuous(labels=lambda l: [f'{x:.2f}' for x in l], limits=(0.5, 0.9))
    + scale_x_continuous(breaks=np.arange(2009, 2025, 1))
    + theme(
        axis_text_x=element_text(rotation=45, size=15),
        axis_text_y=element_text(size=15),
        axis_title_x=element_text(size=17, weight='bold'),
        axis_title_y=element_text(size=17, weight='bold'),
        legend_position='none',
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color="black", fill=None, size=0.2),
        panel_background=element_blank()
    )
)
p.save('Alpha_trend.png', dpi=300)

# === Plot Trend of β (Intercept) over Time ===
stats['beta_volume_lower'] = stats['beta_volume'] - stats['beta_ci_volume']
stats['beta_volume_upper'] = stats['beta_volume'] + stats['beta_ci_volume']

label = r"$\log(E) = {\alpha}\log(V)+\beta$"
p = (
    ggplot(stats, aes(x='year', y='beta_volume'))
    + geom_rect(
        aes(xmin=2020, xmax=float('inf'), ymin=-float('inf'), ymax=float('inf')),
        fill='#dbe9ee', alpha=0.08
    )
    + geom_errorbar(
        aes(ymin='beta_volume_lower', ymax='beta_volume_upper'),
        width=0.2,
        color='#9ad1d4'
    )
    + annotate(
        "text",
        x=min(stats['year']),
        y=1.3,
        label=label,
        size=15,
        color="black",
        ha="left",
        va="bottom"
    )
    + geom_line(size=3, color='#9ad1d4', alpha=0.45)
    + geom_point(size=4.5, color='#9ad1d4', alpha=0.85)
    + geom_text(
        aes(label='round(beta_volume, 2)'),
        nudge_y=0.05,
        size=10
    )
    + labs(
        x='Year',
        y='β (intercept) (GJ per unit volume)'
    )
    + theme_minimal()
    + scale_y_continuous(labels=lambda l: [f'{x:.2f}' for x in l], limits=(-0.5, 1.5))
    + scale_x_continuous(breaks=np.arange(2009, 2025, 1))
    + theme(
        axis_text_x=element_text(rotation=45, size=15),
        axis_text_y=element_text(size=15),
        axis_title_x=element_text(size=17, weight='bold'),
        axis_title_y=element_text(size=17, weight='bold'),
        legend_position='none',
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color="black", fill=None, size=0.2),
        panel_background=element_blank()
    )
)
p.save('Beta_trend.png', dpi=300)