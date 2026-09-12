"""
End-to-End Modular Pipeline for DA-6 Time-Series Crime Forecasting.
Reproduces the complete sequential workflow from the notebook:
Data Ingestion -> Preprocessing -> EDA Plots -> Chronological Split -> Baseline ->
ADF & ACF/PACF -> AR(4) -> ARIMA Candidate Selection -> Locked Test Evaluation ->
Prediction Intervals -> Residual Diagnostics -> Rolling-Origin Validation ->
Replication on Dataset 2 -> Cross-Dataset Comparison -> Assertions -> Manifest.
"""

from pathlib import Path
import pandas as pd

from src.config import (
    OUT_DIR,
    FIG_DIR,
    CHICAGO_CORE_LOCATION,
    NYPD_CORE_LOCATION,
    AR_LAGS,
    DATA_MODE,
    TEST_PERIODS,
)
from src.data import load_raw_data
from src.preprocessing import standardize_datasets, weekly_series, split_series
from src.train_evaluate import (
    evaluate_naive,
    fit_autoreg,
    run_adf_test,
    fit_arima_candidates,
    fit_eval_locked_arima,
    run_ljung_box,
    rolling_validation,
    build_model_comparison_table,
    save_manifest,
)
from src.plots import (
    plot_weekly_series,
    plot_rolling_stats,
    plot_acf_pacf,
    plot_test_forecasts,
    plot_prediction_intervals,
    plot_residuals,
    plot_residual_acf,
)


def run_pipeline():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("DA-6 TIME-SERIES CRIME FORECASTING — MODULAR REPRODUCIBLE PIPELINE")
    print(f"Data Mode: {DATA_MODE}")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # 1. Ingestion & Preprocessing
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading and standardizing data...")
    chicago_raw, nypd_raw = load_raw_data()
    chicago_df, nypd_df = standardize_datasets(chicago_raw, nypd_raw)

    chi_y = weekly_series(chicago_df, "event_date", "location", CHICAGO_CORE_LOCATION)
    ny_y = weekly_series(nypd_df, "event_date", "location", NYPD_CORE_LOCATION)

    print(f"  Chicago District {CHICAGO_CORE_LOCATION}: {len(chi_y)} weekly periods")
    print(f"  NYPD Borough {NYPD_CORE_LOCATION}:       {len(ny_y)} weekly periods")

    # -------------------------------------------------------------------------
    # 2. Exploratory Data Analysis & Plots
    # -------------------------------------------------------------------------
    print("\n[2/7] Generating EDA visualizations...")
    plot_weekly_series(
        chi_y,
        title="Chicago — Weekly Reported Crime Incidents",
        xlabel="Week",
        ylabel="Incidents",
        save_path=FIG_DIR / "01_chicago_weekly_series.png",
    )
    plot_rolling_stats(
        chi_y,
        title="Chicago — Rolling Mean and Volatility",
        window=12,
        save_path=FIG_DIR / "02_chicago_rolling_stats.png",
    )
    plot_weekly_series(
        ny_y,
        title="NYPD — Weekly Reported Complaints",
        xlabel="Week",
        ylabel="Complaints",
        save_path=FIG_DIR / "08_nypd_weekly_series.png",
    )
    plot_rolling_stats(
        ny_y,
        title="NYPD — Rolling Mean and Volatility",
        window=12,
        save_path=FIG_DIR / "09_nypd_rolling_stats.png",
    )

    # -------------------------------------------------------------------------
    # 3. Chronological Splits & Stationarity / ACF-PACF
    # -------------------------------------------------------------------------
    print("\n[3/7] Chronological splitting & stationarity diagnostics...")
    chi_train, chi_val, chi_pretest, chi_test = split_series(chi_y)
    ny_train, ny_val, ny_pretest, ny_test = split_series(ny_y)

    chi_adf = run_adf_test(chi_train)
    ny_adf = run_adf_test(ny_train)
    print(f"  Chicago ADF Statistic: {chi_adf['statistic']:.4f}, p-value: {chi_adf['p_value']:.6f}")
    print(f"  NYPD ADF Statistic:    {ny_adf['statistic']:.4f}, p-value: {ny_adf['p_value']:.6f}")

    plot_acf_pacf(chi_train, title_prefix="Chicago Training", save_path=FIG_DIR / "03_chicago_acf_pacf.png")
    plot_acf_pacf(ny_train, title_prefix="NYPD Training", save_path=FIG_DIR / "10_nypd_acf_pacf.png")

    # -------------------------------------------------------------------------
    # 4. Modeling — Dataset 1: Chicago (Core)
    # -------------------------------------------------------------------------
    print("\n[4/7] Modeling Dataset 1: Chicago (Core)...")
    chi_naive = evaluate_naive(chi_train, chi_val, chi_pretest, chi_test)
    chi_ar = fit_autoreg(chi_train, chi_val, chi_pretest, chi_test, lags=AR_LAGS)
    chi_candidates, chi_best = fit_arima_candidates(chi_train, chi_val)
    chi_candidates.to_csv(OUT_DIR / "chicago_arima_candidates.csv", index=False)
    print(f"  Selected ARIMA order: {chi_best}")

    chi_arima = fit_eval_locked_arima(chi_pretest, chi_test, chi_best)
    chi_results = build_model_comparison_table(
        "Chicago",
        chi_naive["test_score"],
        chi_ar["test_score"],
        chi_arima["test_score"],
        chi_best,
        ar_lags=AR_LAGS,
    )
    chi_results.to_csv(OUT_DIR / "chicago_model_comparison.csv", index=False)

    plot_test_forecasts(
        chi_test,
        chi_naive["test_pred"],
        chi_ar["test_pred"],
        chi_arima["test_pred"],
        best_order=chi_best,
        ar_lags=AR_LAGS,
        title="Chicago — Locked Test Forecast Comparison",
        ylabel="Incidents",
        save_path=FIG_DIR / "04_chicago_test_forecasts.png",
    )
    plot_prediction_intervals(
        chi_test,
        chi_arima["pred_mean"],
        chi_arima["conf_int"],
        best_order=chi_best,
        title="Chicago — ARIMA Forecast and 95% Prediction Interval",
        save_path=FIG_DIR / "05_chicago_prediction_intervals.png",
    )
    plot_residuals(
        chi_arima["residuals"],
        title=f"Chicago — ARIMA{chi_best} Residuals",
        save_path=FIG_DIR / "06_chicago_residuals.png",
    )
    plot_residual_acf(
        chi_arima["residuals"],
        title="Chicago — Residual ACF",
        save_path=FIG_DIR / "07_chicago_residual_acf.png",
    )

    chi_lb, chi_lb_lag = run_ljung_box(chi_arima["residuals"])
    chi_roll = rolling_validation(chi_pretest, chi_best)
    chi_roll.to_csv(OUT_DIR / "chicago_rolling_validation.csv", index=False)

    # -------------------------------------------------------------------------
    # 5. Modeling — Dataset 2: NYPD (Replication)
    # -------------------------------------------------------------------------
    print("\n[5/7] Modeling Dataset 2: NYPD (Replication)...")
    ny_naive = evaluate_naive(ny_train, ny_val, ny_pretest, ny_test)
    ny_ar = fit_autoreg(ny_train, ny_val, ny_pretest, ny_test, lags=AR_LAGS)
    ny_candidates, ny_best = fit_arima_candidates(ny_train, ny_val)
    ny_candidates.to_csv(OUT_DIR / "nypd_arima_candidates.csv", index=False)
    print(f"  Selected ARIMA order: {ny_best}")

    ny_arima = fit_eval_locked_arima(ny_pretest, ny_test, ny_best)
    ny_results = build_model_comparison_table(
        "NYPD",
        ny_naive["test_score"],
        ny_ar["test_score"],
        ny_arima["test_score"],
        ny_best,
        ar_lags=AR_LAGS,
    )
    ny_results.to_csv(OUT_DIR / "nypd_model_comparison.csv", index=False)

    plot_test_forecasts(
        ny_test,
        ny_naive["test_pred"],
        ny_ar["test_pred"],
        ny_arima["test_pred"],
        best_order=ny_best,
        ar_lags=AR_LAGS,
        title="NYPD — Locked Test Forecast Comparison",
        ylabel="Complaints",
        save_path=FIG_DIR / "11_nypd_test_forecasts.png",
    )
    plot_prediction_intervals(
        ny_test,
        ny_arima["pred_mean"],
        ny_arima["conf_int"],
        best_order=ny_best,
        title="NYPD — ARIMA Forecast and 95% Prediction Interval",
        save_path=FIG_DIR / "12_nypd_prediction_intervals.png",
    )
    plot_residuals(
        ny_arima["residuals"],
        title=f"NYPD — ARIMA{ny_best} Residuals",
        save_path=FIG_DIR / "13_nypd_residuals.png",
    )

    ny_lb, ny_lb_lag = run_ljung_box(ny_arima["residuals"])
    ny_roll = rolling_validation(ny_pretest, ny_best)
    ny_roll.to_csv(OUT_DIR / "nypd_rolling_validation.csv", index=False)

    # -------------------------------------------------------------------------
    # 6. Combined Results & Verification
    # -------------------------------------------------------------------------
    print("\n[6/7] Combining results & verifying assertions...")
    all_results = pd.concat([chi_results, ny_results], ignore_index=True)
    all_results.to_csv(OUT_DIR / "two_dataset_model_comparison.csv", index=False)

    # Reproducibility assertions matching cell 36
    assert chi_y.index.is_monotonic_increasing and chi_y.index.is_unique
    assert ny_y.index.is_monotonic_increasing and ny_y.index.is_unique
    assert len(chi_test) == TEST_PERIODS and len(ny_test) == TEST_PERIODS
    assert chi_train.index.max() < chi_val.index.min() < chi_test.index.min()
    assert ny_train.index.max() < ny_val.index.min() < ny_test.index.min()
    assert len(chi_candidates) >= 3 and len(ny_candidates) >= 3
    assert (OUT_DIR / "two_dataset_model_comparison.csv").exists()

    # -------------------------------------------------------------------------
    # 7. Experiment Manifest
    # -------------------------------------------------------------------------
    print("\n[7/7] Generating experiment manifest...")
    save_manifest(
        chi_best,
        ny_best,
        chi_adf,
        ny_adf,
        chi_lb,
        ny_lb,
        chi_lb_lag,
        ny_lb_lag,
    )

    print("\n================================================================================")
    print("PIPELINE EXECUTION COMPLETE")
    print("================================================================================")
    print(all_results.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
