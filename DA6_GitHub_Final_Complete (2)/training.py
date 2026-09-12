"""
Modular Training & Evaluation Script.
Executes the core time-series modeling workflow across Chicago and NYPD datasets:
Naive, AR(4), ARIMA candidate evaluation, locked test evaluation, rolling validation,
and result manifest generation.
"""

from pathlib import Path
import pandas as pd

from src.config import (
    OUT_DIR,
    CHICAGO_CORE_LOCATION,
    NYPD_CORE_LOCATION,
    AR_LAGS,
    DATA_MODE,
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


def train_and_evaluate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STEP 1: Loading Datasets (Mode:", DATA_MODE, ")")
    print("==================================================")
    chicago_raw, nypd_raw = load_raw_data()
    print(f"Chicago raw records: {len(chicago_raw):,}")
    print(f"NYPD raw records:    {len(nypd_raw):,}")

    print("\n==================================================")
    print("STEP 2: Standardizing & Constructing Weekly Series")
    print("==================================================")
    chicago_df, nypd_df = standardize_datasets(chicago_raw, nypd_raw)

    chi_y = weekly_series(chicago_df, "event_date", "location", CHICAGO_CORE_LOCATION)
    ny_y = weekly_series(nypd_df, "event_date", "location", NYPD_CORE_LOCATION)

    print(f"Chicago district {CHICAGO_CORE_LOCATION} weekly periods: {len(chi_y)}")
    print(f"NYPD borough {NYPD_CORE_LOCATION} weekly periods: {len(ny_y)}")

    print("\n==================================================")
    print("STEP 3: Chronological Splitting")
    print("==================================================")
    chi_train, chi_val, chi_pretest, chi_test = split_series(chi_y)
    ny_train, ny_val, ny_pretest, ny_test = split_series(ny_y)

    print(f"Chicago - Train: {len(chi_train)}, Val: {len(chi_val)}, Test: {len(chi_test)}")
    print(f"NYPD    - Train: {len(ny_train)}, Val: {len(ny_val)}, Test: {len(ny_test)}")

    print("\n==================================================")
    print("STEP 4: Training & Evaluating - Chicago (Core)")
    print("==================================================")
    chi_adf = run_adf_test(chi_train)
    print(f"ADF Statistic: {chi_adf['statistic']:.4f}, p-value: {chi_adf['p_value']:.6f}")

    chi_naive = evaluate_naive(chi_train, chi_val, chi_pretest, chi_test)
    print(f"Naive Val:  {chi_naive['val_score']}")
    print(f"Naive Test: {chi_naive['test_score']}")

    chi_ar = fit_autoreg(chi_train, chi_val, chi_pretest, chi_test, lags=AR_LAGS)
    print(f"AR({AR_LAGS}) Val:  {chi_ar['val_score']}")
    print(f"AR({AR_LAGS}) Test: {chi_ar['test_score']}")

    chi_candidates, chi_best = fit_arima_candidates(chi_train, chi_val)
    print(f"Selected ARIMA order for Chicago: {chi_best}")
    chi_candidates.to_csv(OUT_DIR / "chicago_arima_candidates.csv", index=False)

    chi_arima = fit_eval_locked_arima(chi_pretest, chi_test, chi_best)
    print(f"ARIMA{chi_best} Test: {chi_arima['test_score']}")
    print(f"Empirical 95% Interval Coverage: {chi_arima['coverage']:.3f}")

    chi_lb, chi_lb_lag = run_ljung_box(chi_arima["residuals"])
    print(f"Ljung-Box (lag={chi_lb_lag}) p-value: {chi_lb['lb_pvalue'].iloc[0]:.6e}")

    chi_roll = rolling_validation(chi_pretest, chi_best)
    chi_roll.to_csv(OUT_DIR / "chicago_rolling_validation.csv", index=False)
    print(f"Rolling mean MAE: {chi_roll['MAE'].mean():.4f}, RMSE: {chi_roll['RMSE'].mean():.4f}")

    chi_results = build_model_comparison_table(
        "Chicago",
        chi_naive["test_score"],
        chi_ar["test_score"],
        chi_arima["test_score"],
        chi_best,
        ar_lags=AR_LAGS,
    )
    chi_results.to_csv(OUT_DIR / "chicago_model_comparison.csv", index=False)

    print("\n==================================================")
    print("STEP 5: Training & Evaluating - NYPD (Replication)")
    print("==================================================")
    ny_adf = run_adf_test(ny_train)
    print(f"ADF Statistic: {ny_adf['statistic']:.4f}, p-value: {ny_adf['p_value']:.6f}")

    ny_naive = evaluate_naive(ny_train, ny_val, ny_pretest, ny_test)
    print(f"Naive Val:  {ny_naive['val_score']}")
    print(f"Naive Test: {ny_naive['test_score']}")

    ny_ar = fit_autoreg(ny_train, ny_val, ny_pretest, ny_test, lags=AR_LAGS)
    print(f"AR({AR_LAGS}) Val:  {ny_ar['val_score']}")
    print(f"AR({AR_LAGS}) Test: {ny_ar['test_score']}")

    ny_candidates, ny_best = fit_arima_candidates(ny_train, ny_val)
    print(f"Selected ARIMA order for NYPD: {ny_best}")
    ny_candidates.to_csv(OUT_DIR / "nypd_arima_candidates.csv", index=False)

    ny_arima = fit_eval_locked_arima(ny_pretest, ny_test, ny_best)
    print(f"ARIMA{ny_best} Test: {ny_arima['test_score']}")
    print(f"Empirical 95% Interval Coverage: {ny_arima['coverage']:.3f}")

    ny_lb, ny_lb_lag = run_ljung_box(ny_arima["residuals"])
    print(f"Ljung-Box (lag={ny_lb_lag}) p-value: {ny_lb['lb_pvalue'].iloc[0]:.6e}")

    ny_roll = rolling_validation(ny_pretest, ny_best)
    ny_roll.to_csv(OUT_DIR / "nypd_rolling_validation.csv", index=False)
    print(f"Rolling mean MAE: {ny_roll['MAE'].mean():.4f}, RMSE: {ny_roll['RMSE'].mean():.4f}")

    ny_results = build_model_comparison_table(
        "NYPD",
        ny_naive["test_score"],
        ny_ar["test_score"],
        ny_arima["test_score"],
        ny_best,
        ar_lags=AR_LAGS,
    )
    ny_results.to_csv(OUT_DIR / "nypd_model_comparison.csv", index=False)

    print("\n==================================================")
    print("STEP 6: Combined Cross-Dataset Comparison")
    print("==================================================")
    all_results = pd.concat([chi_results, ny_results], ignore_index=True)
    all_results.to_csv(OUT_DIR / "two_dataset_model_comparison.csv", index=False)
    print(all_results.to_string(index=False))

    print("\n==================================================")
    print("STEP 7: Saving Manifest")
    print("==================================================")
    manifest = save_manifest(
        chi_best,
        ny_best,
        chi_adf,
        ny_adf,
        chi_lb,
        ny_lb,
        chi_lb_lag,
        ny_lb_lag,
    )
    print("Manifest written to:", OUT_DIR / "manifest.json")
    print("Training and evaluation completed successfully.")


if __name__ == "__main__":
    train_and_evaluate()
