# Building Energetics

## Overview
This repository contains the data processing workflows and visualization code used in the forthcoming paper **"The Impact of COVID-19 on Building Energetics."** The study investigates how the COVID-19 pandemic disrupted energy consumption patterns across MIT’s campus buildings, and how these patterns regressed or evolved over time.

Using a combination of statistical analysis and high-resolution visualizations, the repository provides tools to explore year-over-year trends between buildings and energy consumption.

All plots in the paper were generated from the scripts in this repository, except the MIT buildings map, which was made using Adobe Illustrator.

## Data Sources
The data used in this project originates from Project Energize_MIT, housed within the MIT Sustainability Datapool. After extensive preprocessing and quality checks, a cleaned version of the dataset was imported and used across all scripts in this repository.

## Features
📈 **Log-Log Regression Models**: Analyzes the scaling relationship between building size and energy use across years.  
🎨 **High-Quality Plotting**: Leverages plotnine (Python’s implementation of ggplot2) and custom theming for academic-quality figures.  
📂 **Modular Codebase**: Organized scripts for each figure, enabling reproducibility and transparency in all analytical outputs.

## Requirements
- Python 3.8 or higher
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `plotnine`
  - `matplotlib`
  - `scipy`

You can install the required packages via pip:
```bash
pip install -r requirements.txt
```

## Installation
Clone the repository:
```bash
git clone https://github.com/hsuanmina/building-energetics.git
cd building-energetics# building-energetics
```
Ensure that the data/ folder contains the cleaned datasets (cleaned_Time_Series.csv and cleaned_CY2023.csv), either by using the ones provided or reproducing them from the raw MIT Sustainability Datapool.

## Contact
For questions, suggestions, or collaboration opportunities, please contact:  
**Yu-Hsuan (Mina) Hsu**  
Email:
minahsu1116@gmail.com