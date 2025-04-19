import numpy as np
import pandas as pd
from plotnine import *
from scipy.stats import pearsonr
from scripts import myTools

# === Overview ===
# This script analyzes the persistence of building-level residuals in energy scaling
# by computing the Pearson correlation between each year's residuals and the baseline year (2009).
# The output is a year-by-year trend plot showing how similar the residual structure is compared to 2009.

# === Load Residual Data ===
residuals = myTools.load_data('residuals.csv')

# === Step 1: Extract 2009 residuals as the baseline ===
baseline_residuals = (
    residuals[residuals['START_DATE_USE_YEAR'] == 2009][['BUILDING_NUMBER', 'residual']]
    .rename(columns={'residual': 'baseline_residual'})
)

# === Step 2: Merge baseline residuals into all years for matching buildings ===
merged_residuals = residuals.merge(baseline_residuals, on='BUILDING_NUMBER', how='left')

# === Step 3: Compute Pearson correlation for each year with 2009 residuals ===
pearson_corr_baseline = []

for year in merged_residuals['START_DATE_USE_YEAR'].unique():
    year_data = merged_residuals[merged_residuals['START_DATE_USE_YEAR'] == year]

    # Calculate Pearson correlation between residuals in current year and 2009
    corr, _ = pearsonr(year_data['residual'], year_data['baseline_residual'])
    pearson_corr_baseline.append({'START_DATE_USE_YEAR': year, 'pearson_corr': corr})

# Convert correlation results to DataFrame
pearson_corr_baseline = pd.DataFrame(pearson_corr_baseline)

# Optional math label (not used directly here but helpful for documentation)
y_axis_label = r'$  \frac{y_{residual}}{s_y} = \rho \cdot \frac{x_{residual}}{s_x}$'

# === Step 4: Create Plot ===
p = (
        ggplot(pearson_corr_baseline, aes(x='START_DATE_USE_YEAR', y='pearson_corr'))

        # Highlight post-2020 period for context (e.g., COVID effects)
        + geom_rect(
    aes(xmin=2020, xmax=float('inf'), ymin=-float('inf'), ymax=float('inf')),
    fill='#fff0f3', alpha=0.1, color=None
)

        # Line and point plot of Pearson correlations over time
        + geom_line(size=3, color='#b5838d', alpha=0.5)
        + geom_point(size=4.5, color='#b5838d')

        # Annotate points with correlation values
        + geom_text(
    aes(label="round(pearson_corr, 2)"),
    nudge_y=0.01,
    size=12
)

        # Axis labels
        + labs(
    x="Year",
    y="\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003Pearson Correlation\n\u2003\n(residual year$_{2009}$, residual year$_t$)"
)

        # Scale and ticks
        + scale_x_continuous(breaks=np.arange(2009, 2025, 1))

        # Clean theme settings
        + theme_minimal()
        + theme(
    axis_text_x=element_text(size=15, rotation=45),
    axis_text_y=element_text(size=15),
    axis_title_x=element_text(size=17, weight="bold"),
    axis_title_y=element_text(size=17, weight="bold", va='center'),
    panel_grid_major=element_blank(),
    panel_grid_minor=element_blank(),
    panel_border=element_rect(color="black", fill=None, size=0.2),
    panel_background=element_blank()
)
)

# === Step 5: Save Plot ===
p.save("Pearson_Correlation.png", dpi=300)