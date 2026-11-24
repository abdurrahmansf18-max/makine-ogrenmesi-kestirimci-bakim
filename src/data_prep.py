"""CRISP-DM Asama 2-3: Veri yukleme, EDA ozeti ve sizintisiz on isleme modulu."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    EDA_SUMMARY_PATH,
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TYPE_CATEGORIES,
)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Ham AI4I 2020 CSV dosyasini yukler ve beklenen sema sutunlarini dogrular."""
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {path}")

    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN] + DROP_COLUMNS)
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Eksik sutunlar tespit edildi: {missing_cols}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Kimlik ve sizinti sutunlarini cikararak yalniz 6 ham girdiyi ve hedefi dondurur."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sinif oranini koruyarak tek seferlik stratified egitim/test bolunmesi yapar."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Kategorik ve sayisal sutunlar icin veri sizintisiz ColumnTransformer kurar."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[TYPE_CATEGORIES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    num_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=num_steps)

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def generate_eda_artifacts(
    df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
    summary_path: Path = EDA_SUMMARY_PATH,
) -> dict:
    """Veri anlama (EDA) istatistiklerini ve temel grafikleri uretir."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    total_rows = int(len(df))
    normal_count = int(target_counts.get(0, 0))
    failure_count = int(target_counts.get(1, 0))

    rnf_with_zero_target = int(
        ((df[FAILURE_MODE_COLUMNS].max(axis=1) == 1) & (df[TARGET_COLUMN] == 0)).sum()
    )

    numeric_stats = (
        df[NUMERIC_FEATURES]
        .agg(["min", "mean", "std", "median", "max"])
        .round(3)
        .to_dict()
    )

    summary = {
        "rows": total_rows,
        "columns": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "input_features": FEATURE_COLUMNS,
        "dropped_columns": DROP_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_distribution": {
            "normal_0": normal_count,
            "failure_1": failure_count,
            "failure_rate_pct": round(100.0 * failure_count / total_rows, 2),
        },
        "type_distribution": {str(k): int(v) for k, v in df["Type"].value_counts().items()},
        "failure_modes": {col: int(df[col].sum()) for col in FAILURE_MODE_COLUMNS},
        "failure_mode_with_zero_target_rows": rnf_with_zero_target,
        "numeric_stats": numeric_stats,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 1. Sinif dagilimi grafigi
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = ["Normal (0)", "Arıza (1)"]
    counts = [normal_count, failure_count]
    colors = ["#0284c7", "#dc2626"]
    bars = ax.bar(labels, counts, color=colors, width=0.52)
    ax.set_title("Hedef Sınıf Dağılımı (Machine failure)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Kayıt Sayısı")
    ax.set_ylim(0, max(counts) * 1.15)
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total_rows
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total_rows * 0.015,
            f"{count:,} (%{pct:.2f})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(plots_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Sayisal ozellik korelasyon matrisi (yalniz 5 sayisal girdi + hedef)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    corr_cols = NUMERIC_FEATURES + [TARGET_COLUMN]
    corr = df[corr_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Sayısal Öznitelikler ve Hedef Korelasyonu", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "eda_correlation.png", dpi=150)
    plt.close(fig)

    return summary


if __name__ == "__main__":
    raw_df = load_raw_data()
    eda_summary = generate_eda_artifacts(raw_df)
    print(
        f"EDA tamamlandi: {eda_summary['rows']} satir, "
        f"{eda_summary['missing_values_total']} eksik deger, "
        f"ariza orani %{eda_summary['class_distribution']['failure_rate_pct']}"
    )
