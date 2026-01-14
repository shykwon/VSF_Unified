# Variable Subset Forecasting (VSF) Research Platform

## Project Goal
This project aims to establish a unified environment for researching **Variable Subset Forecasting (VSF)**. It integrates various state-of-the-art models including FDW, GinAR, GIMCC, SRDI, and imputation baselines (CSDI, SAITS) under a standardized interface.

## Project Structure
```
root/
├── data/               # Raw and processed datasets
├── src/                # Source code
│   ├── core/           # Abstract base classes (Dataset, Model, Trainer)
│   ├── data/           # Data loaders and transformations
│   ├── models/         # Model implementations & Wrappers
│   └── utils/          # Metrics and Logging utilities
├── notebooks/          # Exploratory Data Analysis (EDA)
└── experiments/        # Configuration files and execution scripts
```

## Features
- **Unified Interface**: Standardized input/output format `(Batch, Time, Node, Channel)` for all models.
- **Model Wrappers**: Easy integration of external repositories (FDW, GinAR, GIMCC, SRDI, CSDI, SAITS).
- **Flexible Data Loading**: Supports masking strategies for VSF and imputation tasks.

## Installation
```bash
pip install -r requirements.txt
```
