from scripts import myTools
from plotnine import *
import numpy as np
import pandas as pd

# === FUNCTIONAL OVERVIEW ===
# This script analyzes "regression to the mean" in building energy use by:
# 1. Standardizing energy (GJ) data by year.
# 2. Using 2009 data as a baseline (X1) to identify buildings above/below various thresholds.
# 3. Computing conditional expectations for each threshold and each subsequent year (X2).
# 4. Creating a faceted line plot comparing conditional expectations over time.

# === Load and Prepare Data ===
data = myTools.load_data('cleaned_Time_Series.csv')
data_energy = myTools.load_data('cleaned_CY2023.csv')

# Merge static building attributes
data = data.merge(
    data_energy[['BUILDING_NUMBER', 'CLUSTER_NUM', 'BUILDING_VOLUME', 'LOG_BUILDING_VOLUME',
                 'LOG_EXT_GROSS_AREA', 'NUM_OF_ROOMS', 'DATE_OCCUPIED']].drop_duplicates(),
    on='BUILDING_NUMBER', how='left'
)

# Filter data to only include buildings present in CY2023
building_numbers = data_energy['BUILDING_NUMBER'].unique()
data = data[data['BUILDING_NUMBER'].isin(building_numbers)]

# === Standardize Energy Use (GJ) Within Each Year ===
data['GJ_mean'] = data.groupby('START_DATE_USE_YEAR')['GJ'].transform('mean')
data['GJ_std'] = data.groupby('START_DATE_USE_YEAR')['GJ'].transform('std')
data['GJ_standardized'] = (data['GJ'] - data['GJ_mean']) / data['GJ_std']
data.drop(columns=['GJ_mean', 'GJ_std'], inplace=True)

# === Define Baseline and Thresholds ===
baseline_year = 2009
condition_years = range(2009, 2025)
thresholds = [0, 1, 2, 3]

# Use 2009 data as the reference group X1
X1 = data[data['START_DATE_USE_YEAR'] == baseline_year][['BUILDING_NUMBER', 'GJ_standardized']]

# === Compute Conditional Expectations Over Time ===
results = []

for year in condition_years:
    X2 = data[data['START_DATE_USE_YEAR'] == year][['BUILDING_NUMBER', 'GJ_standardized']]
    year_result = {'Year': year}

    for threshold in thresholds:
        # Buildings above or below the threshold in 2009
        buildings_X1_gt = X1[X1['GJ_standardized'] > threshold]['BUILDING_NUMBER']
        buildings_X1_lt = X1[X1['GJ_standardized'] < threshold]['BUILDING_NUMBER']

        # Conditional means
        year_result[f'E[X1 | X1 > {threshold}]'] = X1[X1['BUILDING_NUMBER'].isin(buildings_X1_gt)]['GJ_standardized'].mean()
        year_result[f'E[X1 | X1 < {threshold}]'] = X1[X1['BUILDING_NUMBER'].isin(buildings_X1_lt)]['GJ_standardized'].mean()
        year_result[f'E[X2 | X1 > {threshold}]'] = X2[X2['BUILDING_NUMBER'].isin(buildings_X1_gt)]['GJ_standardized'].mean()
        year_result[f'E[X2 | X1 < {threshold}]'] = X2[X2['BUILDING_NUMBER'].isin(buildings_X1_lt)]['GJ_standardized'].mean()

    results.append(year_result)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# === Reformat Data for Plotting ===
plot_data_combined = pd.DataFrame()
thresholds = [0, 1, 2]  # We will plot only these three

for threshold in thresholds:
    temp_df = pd.DataFrame({
        'Year': results_df['Year'],
        'E[X1|X1>h]': results_df[f"E[X1 | X1 > {threshold}]"],
        'E[X2|X1>h]': results_df[f"E[X2 | X1 > {threshold}]"],
        'E[X1|X1<h]': results_df[f"E[X1 | X1 < {threshold}]"],
        'E[X2|X1<h]': results_df[f"E[X2 | X1 < {threshold}]"],
        'Threshold': threshold
    }).melt(id_vars=['Year', 'Threshold'], var_name='Expectation', value_name='Value')
    plot_data_combined = pd.concat([plot_data_combined, temp_df])

# Format for plot aesthetics
plot_data_combined['Threshold'] = plot_data_combined['Threshold'].astype(str)
plot_data_combined['Group'] = plot_data_combined['Expectation'].apply(
    lambda x: 'Above h' if '>' in x else 'Below h'
)

# Rename threshold column
plot_data_combined = plot_data_combined.rename(columns={'Threshold': 'h'})

# Replace Expectation with LaTeX-style labels
plot_data_combined['Expectation'] = plot_data_combined['Expectation'].replace({
    'E[X1|X1>h]': r'$E[\text{year}_{2009} | \text{year}_{2009} > h]$',
    'E[X2|X1>h]': r'$E[\text{year}_{t} | \text{year}_{2009} > h]$',
    'E[X1|X1<h]': r'$E[\text{year}_{2009} | \text{year}_{2009} < h]$',
    'E[X2|X1<h]': r'$E[\text{year}_{t} | \text{year}_{2009} < h]$'
})

# Update grouping labels to LaTeX
plot_data_combined['Group'] = plot_data_combined['Expectation'].apply(
    lambda x: r'$\text{year}_{2009} > h$' if '>' in x else r'$\text{year}_{2009} < h$'
)

# Color mapping
combined_colors = {
    r'$E[\text{year}_{2009} | \text{year}_{2009} > h]$': '#e07a5f',
    r'$E[\text{year}_{t} | \text{year}_{2009} > h]$': '#772f1a',
    r'$E[\text{year}_{2009} | \text{year}_{2009} < h]$': '#00509d',
    r'$E[\text{year}_{t} | \text{year}_{2009} < h]$': '#81b29a'
}

# === Plot Conditional Expectations ===
p = (
    ggplot(plot_data_combined, aes(x='Year', y='Value', color='Expectation', shape='h')) +
    geom_rect(
        aes(xmin=2020, xmax=float('inf'), ymin=-float('inf'), ymax=float('inf')),
        fill='#e7ecef', alpha=0.1, color=None
    ) +
    geom_line(size=1.5, alpha=0.8) +
    geom_point(size=2.5, alpha=0.9) +
    scale_color_manual(values=combined_colors) +
    scale_shape_manual(values=['o', 's', 'D']) +
    facet_wrap('~Group', nrow=1, ncol=2, scales='free_y') +
    labs(x="Year", y="Conditional Expectation") +
    theme_minimal() +
    scale_x_continuous(breaks=np.arange(2010, 2025, 2)) +
    scale_y_continuous(breaks=lambda l: np.linspace(min(l), max(l), 6),
                        labels=lambda l: [f'{x:.2f}' for x in l]) +
    theme(
        axis_title_x=element_text(size=17),
        axis_title_y=element_text(size=17, va="center"),
        subplots_adjust={'wspace': 0.2, 'hspace': 0.3},
        legend_title=element_blank(),
        legend_text=element_text(size=10),
        legend_position='none',
        strip_background=element_rect(fill='#fffcf2', color='black', size=0.2),
        strip_text=element_text(size=14),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color="black", fill=None, size=0.2),
        axis_text_x=element_text(rotation=45, size=15),
        axis_text_y=element_text(size=15)
    ) +
    guides(color='none', shape=guide_legend(ncol=1))
)

# === Convert to Matplotlib and Annotate Titles ===
fig = p.draw()

# Add custom LaTeX-like annotations for above and below thresholds
fig.text(0.12, 1.07, r'$\mathbf{0 \geq}$', fontsize=15, ha='center')
fig.text(0.335, 1.07, r'$\mathbf{E[{year}_{t} |{year}_{2009} < h]}$', fontsize=15, color='#81b29a', ha='center')
fig.text(0.10, 1.02, r'$\mathbf{>}$', fontsize=15, color='black', ha='center')
fig.text(0.33, 1.02, r'$\mathbf{E[{year}_{2009} | {year}_{2009} < h]}$', fontsize=15, color='#00509d', ha='center')

fig.text(0.60, 1.07, r'$\mathbf{0 \leq}$', fontsize=15, ha='center')
fig.text(0.815, 1.07, r'$\mathbf{E[{year}_{t} | {year}_{2009} > h]}$', fontsize=15, color='#772f1a', ha='center')
fig.text(0.58, 1.02, r'$\mathbf{<}$', fontsize=15, color='black', ha='center')
fig.text(0.81, 1.02, r'$\mathbf{E[{year}_{2009} |{year}_{2009} > h]}$', fontsize=15, color='#e07a5f', ha='center')

# Move Y axis for the right subplot
count_ax = 1
for ax in fig.axes:
    if count_ax % 2 == 0:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y', which='both', left=False, right=False)
        ax.yaxis.set_tick_params(labelsize=15, labelcolor='#4e4e4e')
    count_ax += 1

# === Save Plot ===
fig.savefig("Conditional_Expectation.png", dpi=300, bbox_inches='tight')