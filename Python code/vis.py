import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"val_predictions_efficientnet_v2_s_384_finetune_111.csv"  # <-- change if needed

TEMP_BINS = [0, 50, 100, 200, 300, 500, 800, 1200, 2000, 5000]
TEMP_LABELS = [
    "0-50", "50-100", "100-200", "200-300", "300-500",
    "500-800", "800-1200", "1200-2000", "2000+"
]

TINT_BINS = [0, 1, 2, 4, 6, 8, 12, 20, 50]
TINT_LABELS = ["0-1", "1-2", "2-4", "4-6", "6-8", "8-12", "12-20", "20+"]

TEMP_THRESHOLDS = [100, 200, 300, 500, 800]
TINT_THRESHOLDS = [1, 2, 4, 6, 8, 12]


def load_predictions(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())

    required_cols = [
        "Temperature", "Tint",
        "pred_Temperature", "pred_Tint"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["Tint"] = pd.to_numeric(df["Tint"], errors="coerce")
    df["pred_Temperature"] = pd.to_numeric(df["pred_Temperature"], errors="coerce")
    df["pred_Tint"] = pd.to_numeric(df["pred_Tint"], errors="coerce")

    #  actual - predicted
    df["diff_temp"] = df["Temperature"] - df["pred_Temperature"]
    df["diff_tint"] = df["Tint"] - df["pred_Tint"]

    df["abs_diff_temp"] = df["diff_temp"].abs()
    df["abs_diff_tint"] = df["diff_tint"].abs()

    return df


def bin_errors(df: pd.DataFrame):

    df["temp_error_bin"] = pd.cut(
        df["abs_diff_temp"],
        bins=TEMP_BINS,
        labels=TEMP_LABELS,
        include_lowest=True,
        right=False,
    )
    temp_counts = df["temp_error_bin"].value_counts().sort_index()

    df["tint_error_bin"] = pd.cut(
        df["abs_diff_tint"],
        bins=TINT_BINS,
        labels=TINT_LABELS,
        include_lowest=True,
        right=False,
    )
    tint_counts = df["tint_error_bin"].value_counts().sort_index()

    return temp_counts, tint_counts


def print_bucket_counts(temp_counts, tint_counts):

    print("\n==============================")
    print(" TEMPERATURE ERROR BUCKETS")
    print("==============================")
    for label, count in zip(temp_counts.index.astype(str), temp_counts.values):
        print(f"{label:<12} : {count} images")

    print("\n==============================")
    print(" TINT ERROR BUCKETS")
    print("==============================")
    for label, count in zip(tint_counts.index.astype(str), tint_counts.values):
        print(f"{label:<12} : {count} images")


def plot_error_distributions(temp_counts, tint_counts) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))


    axes[0].bar(temp_counts.index.astype(str), temp_counts.values)
    axes[0].set_title("Temperature absolute error distribution")
    axes[0].set_xlabel("Absolute error range (K)")
    axes[0].set_ylabel("Number of images")
    axes[0].tick_params(axis="x", rotation=45)

    # Tint error distribution
    axes[1].bar(tint_counts.index.astype(str), tint_counts.values)
    axes[1].set_title("Tint absolute error distribution")
    axes[1].set_xlabel("Absolute error range")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def summarize_error(df: pd.DataFrame, col: str, name: str, thresholds) -> None:

    print(f"\n{name} error summary:")
    for t in thresholds:
        frac = (df[col] <= t).mean() * 100
        print(f"  <= {t}: {frac:.1f}% of samples")


def main():
    df = load_predictions(CSV_PATH)

    temp_counts, tint_counts = bin_errors(df)

    print_bucket_counts(temp_counts, tint_counts)

    plot_error_distributions(temp_counts, tint_counts)

    summarize_error(df, "abs_diff_temp", "Temperature", TEMP_THRESHOLDS)
    summarize_error(df, "abs_diff_tint", "Tint", TINT_THRESHOLDS)


if __name__ == "__main__":
    main()
