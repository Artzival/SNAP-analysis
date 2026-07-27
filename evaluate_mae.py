import numpy as np
import pandas as pd
import xgboost as xgb

from project import (
    prepare_merged_dataset,
    prepare_modeling_dataset,
    get_feature_columns,
    build_feature_matrix,
    train_xgboost_regressor,
)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def run_mae_evaluation():
    df = prepare_merged_dataset()
    model_df = prepare_modeling_dataset(df)

    if model_df["year"].nunique() < 2:
        raise ValueError("MAE evaluation requires at least 2 years of historical data.")

    validation_year = model_df["year"].max()
    train_df = model_df[model_df["year"] < validation_year].copy()
    validation_df = model_df[model_df["year"] == validation_year].copy()

    if train_df.empty or validation_df.empty:
        raise ValueError("Unable to build non-empty train/validation split for MAE evaluation.")

    feature_cols = get_feature_columns(model_df)

    X_train = build_feature_matrix(train_df, feature_cols)
    y_train = train_df["cases_per_1000_eligible"]

    X_validation = build_feature_matrix(validation_df, feature_cols)
    y_validation = validation_df["cases_per_1000_eligible"]

    X_train_aligned, X_validation_aligned = X_train.align(
        X_validation,
        join="left",
        axis=1,
        fill_value=0,
    )

    model = train_xgboost_regressor(X_train_aligned, y_train)
    validation_pred = model.predict(xgb.DMatrix(X_validation_aligned))

    overall_mae = mean_absolute_error(y_validation, validation_pred)

    evaluation_results = validation_df[["county", "year", "month", "month_num"]].copy()
    evaluation_results["actual_cases_per_1000_eligible"] = y_validation.values
    evaluation_results["predicted_cases_per_1000_eligible"] = validation_pred
    evaluation_results["absolute_error"] = np.abs(
        evaluation_results["actual_cases_per_1000_eligible"]
        - evaluation_results["predicted_cases_per_1000_eligible"]
    )

    monthly_mae = (
        evaluation_results.groupby(["year", "month_num", "month"], as_index=False)["absolute_error"]
        .mean()
        .rename(columns={"absolute_error": "mae"})
        .sort_values(["year", "month_num"])
    )

    county_mae = (
        evaluation_results.groupby("county", as_index=False)["absolute_error"]
        .mean()
        .rename(columns={"absolute_error": "mae"})
        .sort_values("mae", ascending=False)
    )

    evaluation_results.to_csv("snap_mae_evaluation_rows.csv", index=False)
    monthly_mae.to_csv("snap_mae_by_month.csv", index=False)
    county_mae.to_csv("snap_mae_by_county.csv", index=False)

    print(f"Validation year: {validation_year}")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Overall MAE (cases_per_1000_eligible): {overall_mae:.4f}")
    print("Saved row-level errors to snap_mae_evaluation_rows.csv")
    print("Saved monthly MAE to snap_mae_by_month.csv")
    print("Saved county MAE to snap_mae_by_county.csv")


if __name__ == "__main__":
    run_mae_evaluation()
