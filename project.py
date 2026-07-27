import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


def prompt_forecast_target():
    """Prompt for forecast month/year with input validation."""
    while True:
        month_input = input("Enter forecast month (e.g., January): ").strip()
        try:
            target_month_num = pd.to_datetime(month_input, format="%B").month
            target_month = pd.Timestamp(2000, target_month_num, 1).strftime("%B")
            break
        except ValueError:
            print("Invalid month. Please enter a full month name like January.")

    while True:
        year_input = input("Enter forecast year (e.g., 2026): ").strip()
        if not year_input:
            print("Year is required. Please enter a 4-digit year.")
            continue

        try:
            target_year = int(year_input)
        except ValueError:
            print("Invalid year. Please enter digits only, like 2026.")
            continue
        break

    return target_month_num, target_month, target_year


def load_source_data():
    """Load and clean source SNAP and food insecurity data."""
    df_snap = pd.read_csv("foodbank_csvs/snap_race.csv")
    df_fi_2025 = pd.read_csv("foodbank_csvs/food_insecurity_2025.csv")
    df_fi_2025["year"] = 2025
    df_fi_2024 = pd.read_csv("foodbank_csvs/food_insecurity_2024.csv")
    df_fi_2024["year"] = 2024
    df_fi = pd.concat([df_fi_2024, df_fi_2025], ignore_index=True)

    df_fi["food_insecure_population_percent"] = pd.to_numeric(
        df_fi["food_insecure_population_percent"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("percent", "", case=False, regex=False)
        .str.replace(" ", "", regex=False),
        errors="coerce",
    )

    fi_population = pd.Series(np.nan, index=df_fi.index, dtype=float)
    if "population" in df_fi.columns:
        fi_population = pd.to_numeric(
            df_fi["population"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    if "population_size" in df_fi.columns:
        fi_population = fi_population.fillna(
            pd.to_numeric(
                df_fi["population_size"].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
        )

    if fi_population.isna().all():
        raise KeyError("Neither 'population' nor 'population_size' exists in the food insecurity CSVs.")

    df_fi["fi_population"] = fi_population
    df_fi = df_fi.drop(columns=[col for col in ["population", "population_size"] if col in df_fi.columns])
    df_fi["year"] = df_fi["year"].astype(int)

    return df_snap, df_fi


def fill_sparse_months(trend_frame):
    """Fill sparse August/October values from neighboring months when available."""
    trend_frame = trend_frame.sort_index().astype(float).copy()
    for month_num in [8, 10]:
        prev_month = month_num - 1
        next_month = month_num + 1
        if prev_month in trend_frame.index and next_month in trend_frame.index:
            trend_frame.loc[month_num] = (
                trend_frame.loc[prev_month] + trend_frame.loc[next_month]
            ) / 2
    return trend_frame


def prepare_merged_dataset():
    """Build merged county-month dataset with engineered time features."""
    df_snap, df_fi = load_source_data()

    df_snap = df_snap.rename(columns={"county_name": "county"})
    df_snap["county"] = df_snap["county"].str.lower().str.strip()
    df_fi["county"] = df_fi["county"].str.lower().str.strip()
    df_snap["year"] = df_snap["year"].astype(str).str.replace(r"[^\d]", "", regex=True).astype(int)
    df_snap = df_snap[df_snap["year"].isin([2024, 2025])]
    fi_counties = df_fi["county"].unique()
    df_snap = df_snap[df_snap["county"].isin(fi_counties)]

    df_fi_weighted = (
        df_fi.groupby(["county", "year"])
        .apply(
            lambda g: (g["food_insecure_population_percent"] * g["fi_population"]).sum()
            / g["fi_population"].sum()
        )
        .reset_index(name="food_insecurity_weighted")
    )
    df_fi_std = (
        df_fi.groupby(["county", "year"])["food_insecure_population_percent"]
        .std()
        .reset_index(name="FI_std")
    )
    df_fi_combined = df_fi_weighted.merge(df_fi_std, on=["county", "year"], how="left")

    df = df_snap.merge(df_fi_combined, on=["county", "year"], how="left")
    df["month_num"] = pd.to_datetime(df["month"].str.strip(), format="%B").dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    money_cols = ["total_snap_payments", "avg_snap_payment"]
    df[money_cols] = df[money_cols].apply(
        lambda col: col.str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    )

    return df


def generate_descriptive_outputs(df, top_counties_n=10, show_plots=True):
    """Generate charts and CSV outputs for descriptive trend analysis."""
    monthly_trend = (
        df.groupby(["year", "month_num"], as_index=False)["number_of_cases"]
        .sum()
        .sort_values(["year", "month_num"])
    )

    trend_pivot = monthly_trend.pivot(
        index="month_num",
        columns="year",
        values="number_of_cases",
    ).sort_index()

    trend_pivot = fill_sparse_months(trend_pivot)

    county_monthly_trend = (
        df.groupby(["county", "year", "month_num"], as_index=False)["number_of_cases"]
        .sum()
        .sort_values(["county", "year", "month_num"])
    )

    county_trend_pivot = county_monthly_trend.pivot(
        index=["county", "month_num"],
        columns="year",
        values="number_of_cases",
    ).sort_index()

    county_trend_pivot = pd.concat(
        {
            county: fill_sparse_months(county_frame.droplevel("county"))
            for county, county_frame in county_trend_pivot.groupby(level="county")
        }
    )
    county_trend_pivot.index = county_trend_pivot.index.set_names(["county", "month_num"])

    county_trend_output = county_trend_pivot.reset_index()
    county_trend_output["month"] = county_trend_output["month_num"].map(
        lambda month: pd.Timestamp(2000, month, 1).strftime("%B")
    )
    county_trend_output = county_trend_output[["county", "month_num", "month", 2024, 2025]]
    county_trend_output.to_csv("snap_participation_trend_by_county.csv", index=False)

    month_labels = [pd.Timestamp(2000, month, 1).strftime("%b") for month in range(1, 13)]

    top_counties = (
        county_monthly_trend.groupby("county")["number_of_cases"]
        .sum()
        .nlargest(top_counties_n)
        .index
    )

    top_county_trends = county_trend_output[
        county_trend_output["county"].isin(top_counties)
    ].copy()

    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, county in zip(axes, top_counties):
        county_slice = top_county_trends[top_county_trends["county"] == county].sort_values("month_num")
        for year in [2024, 2025]:
            ax.plot(
                county_slice["month_num"],
                county_slice[year],
                marker="o",
                linewidth=2,
                label=str(year),
            )
        ax.set_title(county.title())
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels, rotation=45)
        ax.grid(alpha=0.3)

    for ax in axes[len(top_counties):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Year", loc="upper center", ncol=2)
    fig.suptitle(f"Top {top_counties_n} Counties by Monthly SNAP Cases", y=0.995)
    fig.supxlabel("Month")
    fig.supylabel("SNAP Cases")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("snap_participation_top_counties_comparison.png", dpi=300)

    plt.figure(figsize=(11, 6))
    for year in sorted(trend_pivot.columns):
        plt.plot(
            trend_pivot.index,
            trend_pivot[year],
            marker="o",
            linewidth=2,
            label=f"{year}",
        )

    plt.title("SNAP Participation Trend by Month (2024 vs 2025)")
    plt.xlabel("Month")
    plt.ylabel("Total SNAP Cases")
    plt.xticks(range(1, 13), month_labels)
    plt.grid(alpha=0.3)
    plt.legend(title="Year")
    plt.tight_layout()
    plt.savefig("snap_participation_trend_2024_2025.png", dpi=300)

    heatmap_cols = [
        "number_of_cases",
        "eligible_individuals",
        "total_snap_payments",
        "avg_snap_payment",
        "food_insecurity_weighted",
        "FI_std",
        "month_num",
        "year",
    ]
    heatmap_data = df[heatmap_cols].copy()
    correlation_matrix = heatmap_data.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of SNAP and Food Insecurity Metrics")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)

    for row_idx in range(correlation_matrix.shape[0]):
        for col_idx in range(correlation_matrix.shape[1]):
            corr_value = correlation_matrix.iloc[row_idx, col_idx]
            plt.text(col_idx, row_idx, f"{corr_value:.2f}", ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    plt.savefig("snap_correlation_heatmap.png", dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    print("County-level monthly SNAP participation trend saved to snap_participation_trend_by_county.csv")
    print(f"Top county comparison chart saved to snap_participation_top_counties_comparison.png for {top_counties_n} counties")
    print("Correlation heatmap saved to snap_correlation_heatmap.png")


def prepare_modeling_dataset(df):
    """Create model-ready rows and target variable."""
    model_df = df.dropna(subset=["number_of_cases", "eligible_individuals"]).copy()
    model_df = model_df[model_df["eligible_individuals"] > 0].copy()
    model_df["cases_per_1000_eligible"] = (
        model_df["number_of_cases"] / model_df["eligible_individuals"]
    ) * 1000
    return model_df


def get_feature_columns(model_df):
    """Return feature columns used by the model."""
    return [
        col
        for col in model_df.columns
        if col
        not in [
            "number_of_cases",
            "cases_per_1000_eligible",
            "eligible_individuals",
            "month",
            "month_num",
        ]
    ]


def build_feature_matrix(frame, feature_cols):
    """One-hot encode county while preserving the configured feature set."""
    return pd.get_dummies(frame[feature_cols], columns=["county"], drop_first=False)


def train_xgboost_regressor(X_train, y_train, num_boost_round=300):
    """Train an XGBoost regressor for SNAP cases per 1,000 eligible."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)


def build_forecast_frame(model_df, target_month_num, target_month, target_year):
    """Build the feature proxy frame for the requested month/year forecast."""
    proxy_year = model_df["year"].max()
    forecast_df = model_df[
        (model_df["year"] == proxy_year) & (model_df["month_num"] == target_month_num)
    ].copy()

    if forecast_df.empty:
        raise ValueError(
            f"No data found for {target_month} in the latest observed year ({proxy_year})."
        )

    forecast_df["year"] = target_year
    forecast_df["month"] = target_month
    forecast_df["month_num"] = target_month_num
    forecast_df["month_sin"] = np.sin(2 * np.pi * forecast_df["month_num"] / 12)
    forecast_df["month_cos"] = np.cos(2 * np.pi * forecast_df["month_num"] / 12)

    return forecast_df, proxy_year


def forecast_counties(model, X_train, forecast_df, feature_cols):
    """Predict county-level SNAP cases for the provided forecast frame."""
    X_forecast = build_feature_matrix(forecast_df, feature_cols)
    X_train_aligned, X_forecast_aligned = X_train.align(
        X_forecast,
        join="left",
        axis=1,
        fill_value=0,
    )

    forecast_pred = model.predict(xgb.DMatrix(X_forecast_aligned))
    forecast_results = forecast_df[["county", "month", "year", "eligible_individuals"]].copy()
    forecast_results["predicted_cases_per_1000_eligible"] = forecast_pred
    forecast_results["predicted_cases"] = (
        forecast_results["predicted_cases_per_1000_eligible"]
        * forecast_results["eligible_individuals"]
        / 1000
    )

    county_forecast = (
        forecast_results.groupby("county", as_index=False)[
            ["predicted_cases_per_1000_eligible", "predicted_cases"]
        ]
        .sum()
        .sort_values("predicted_cases_per_1000_eligible", ascending=False)
    )

    return county_forecast, X_train_aligned


def run_forecast_cli():
    """Run full descriptive analysis and county forecast from user prompts."""
    target_month_num, target_month, target_year = prompt_forecast_target()

    df = prepare_merged_dataset()
    generate_descriptive_outputs(df)

    model_df = prepare_modeling_dataset(df)
    feature_cols = get_feature_columns(model_df)

    train_df = model_df.copy()
    X_train = build_feature_matrix(train_df, feature_cols)
    y_train = train_df["cases_per_1000_eligible"]
    model = train_xgboost_regressor(X_train, y_train)

    forecast_df, proxy_year = build_forecast_frame(
        model_df,
        target_month_num,
        target_month,
        target_year,
    )

    county_forecast, _ = forecast_counties(model, X_train, forecast_df, feature_cols)

    print(f"Forecast for {target_month} {target_year}")
    print(f"Using {target_month} {proxy_year} county features as the proxy feature set.")
    print("Top counties by predicted SNAP cases per 1,000 eligible individuals:")
    print(county_forecast.head(10).to_string(index=False))


if __name__ == "__main__":
    run_forecast_cli()
