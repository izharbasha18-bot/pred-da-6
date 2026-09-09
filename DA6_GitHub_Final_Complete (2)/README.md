# DA-6 — Time-Series Analysis and Forecasting of Reported Crime Incidents

A reproducible time-series forecasting project for weekly reported crime incidents using two official public datasets: Chicago Crimes and NYPD Complaint Data Historic.

The project follows a common forecasting protocol across both datasets, including data validation, weekly time-series construction, chronological splitting, exploratory analysis, classical forecasting models, validation-based ARIMA selection, prediction intervals, residual diagnostics, rolling-origin validation, and reproducibility checks.

## Project layout

```text
pred-da-6/
│
├── DA6_Time_Series_Crime_Forecasting.ipynb
│                                      # Complete executed DA-6 notebook
│
├── data/
│   └── README.md                      # Dataset and data-handling notes
│
├── src/
│   └── README.md                      # Implementation notes
│
├── lab06_outputs/
│   ├── figures/                       # Generated analysis and forecast figures
│   ├── chicago_arima_candidates.csv
│   ├── chicago_model_comparison.csv
│   ├── chicago_rolling_validation.csv
│   ├── nypd_arima_candidates.csv
│   ├── nypd_model_comparison.csv
│   ├── nypd_rolling_validation.csv
│   ├── two_dataset_model_comparison.csv
│   └── manifest.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

## What is included

- Problem framing and target definition.
- Dataset provenance and responsible-use notes.
- Two official public datasets.
- Chicago crime analysis for police district `001`.
- NYPD complaint analysis for borough `MANHATTAN`.
- Regular weekly `W-MON` time-series construction.
- Chronological train/validation/test splitting.
- Leakage prevention through time-ordered evaluation.
- Weekly-series exploratory analysis.
- Rolling mean and rolling standard-deviation analysis.
- Augmented Dickey-Fuller (ADF) stationarity testing.
- ACF and PACF analysis.
- Naive forecasting baseline.
- Autoregressive AR(4) forecasting.
- Multiple ARIMA candidate models.
- Validation-based ARIMA order selection.
- Locked test-set evaluation using MAE and RMSE.
- 95% prediction intervals.
- Residual plots and residual ACF.
- Ljung-Box residual diagnostics.
- Rolling-origin validation.
- Cross-dataset model comparison.
- Final reproducibility assertions and experiment manifest.
- Saved CSV result tables and PNG figures.
- The complete executed Jupyter notebook with its outputs.

## Datasets

### 1. Chicago Crimes

The Chicago Crimes — 2001 to Present dataset is used as the first dataset.

For this experiment:

- Date field: `date`
- Geographic field: `district`
- Core location: police district `001`
- Time-series frequency: weekly (`W-MON`)

Official source:

https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2

### 2. NYPD Complaint Data Historic

The NYPD Complaint Data Historic dataset is used as the second dataset.

For this experiment:

- Date field: `cmplnt_fr_dt`
- Geographic field: `boro_nm`
- Core location: `MANHATTAN`
- Time-series frequency: weekly (`W-MON`)

Official source:

https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i

Raw source datasets are not committed to this repository. The notebook contains the data-loading workflow and the official API acquisition logic.

## Analysis workflow

The notebook follows the same forecasting protocol for both datasets:

1. Identify and document the official data source.
2. Load or acquire the required date and geographic fields.
3. Standardize the datasets into a common analysis structure.
4. Filter to the required geographic location.
5. Aggregate reported incidents into a regular weekly series.
6. Check the resulting time index.
7. Create chronological training, validation, pre-test and locked-test portions.
8. Establish a naive baseline.
9. Examine the series using plots and rolling statistics.
10. Perform the ADF stationarity test.
11. Inspect ACF and PACF.
12. Fit an AR(4) model.
13. Compare candidate ARIMA specifications on validation data.
14. Select the best ARIMA order using validation performance.
15. Fit the selected model on the pre-test data.
16. Evaluate the locked test set using MAE and RMSE.
17. Produce forecast and 95% prediction-interval plots.
18. Inspect residuals and residual autocorrelation.
19. Run the Ljung-Box diagnostic.
20. Perform rolling-origin validation.
21. Compare the results across Chicago and NYPD.
22. Run reproducibility checks and save an experiment manifest.

## Model specifications

The notebook evaluates:

- Naive baseline
- AR(4)
- ARIMA(1,0,0)
- ARIMA(2,0,0)
- ARIMA(1,1,1)
- ARIMA(2,1,1)

ARIMA order selection is performed using the validation portion rather than the locked test set.

## Current experiment configuration

The notebook configuration records:

```text
DATA_MODE = DEMO
Date window = 2020-01-01 to 2025-12-31
Frequency = W-MON
Validation periods = 12
Test periods = 12
AR lags = 4
```

`DEMO` mode is used for a reproducible notebook run without requiring the full source datasets to be downloaded. The notebook also contains `LIVE` API acquisition logic for the official Chicago and NYPD sources.

## Outputs

The `lab06_outputs/` directory contains the generated artifacts from the notebook run.

### Result tables

- `chicago_arima_candidates.csv`
- `chicago_model_comparison.csv`
- `chicago_rolling_validation.csv`
- `nypd_arima_candidates.csv`
- `nypd_model_comparison.csv`
- `nypd_rolling_validation.csv`
- `two_dataset_model_comparison.csv`

### Figures

The `figures/` directory contains the generated plots for:

- Weekly crime/complaint series
- Rolling statistics
- ACF/PACF
- Locked-test forecast comparison
- Prediction intervals
- Residual diagnostics
- Residual ACF

### Reproducibility manifest

`manifest.json` records the experiment mode, dataset sources, geographic filters, date window, frequency, model candidates, selected ARIMA orders, diagnostic statistics, and environment information from the notebook run.

## Running the notebook

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Then open:

```text
DA6_Time_Series_Crime_Forecasting.ipynb
```

The notebook already contains executed cells and saved outputs.

## Re-running with the official APIs

To use the official public APIs instead of the reproducible demo data, change:

```python
DATA_MODE = "DEMO"
```

to:

```python
DATA_MODE = "LIVE"
```

The notebook contains the API-loading functions and the required field mappings for both datasets.

## Evaluation

The primary forecasting metrics reported in the notebook are:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The notebook also evaluates:

- ADF stationarity
- ACF/PACF behaviour
- Residual autocorrelation
- Ljung-Box diagnostic
- Rolling-origin validation

## Reproducibility and leakage control

The analysis uses chronological splits rather than random train/test splitting. The locked test set is kept separate from model-order selection, and the notebook includes assertions checking the ordering and uniqueness of the time indices and the required holdout sizes.

## Project objective

The objective of this DA-6 experiment is to determine how classical time-series forecasting methods perform when predicting weekly reported crime incidents and to compare the forecasting behaviour across two different urban crime datasets using a consistent evaluation protocol.
