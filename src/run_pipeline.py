"""CRISP-DM 6 asamali uctan uca egitim, degerlendirme ve paketleme akisi."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    EDA_SUMMARY_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODELS_DIR,
    PLOTS_DIR,
    PRIMARY_METRIC,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_SIZE,
)
from src.data_prep import (
    create_train_test_split,
    generate_eda_artifacts,
    load_raw_data,
    split_features_target,
)
from src.evaluate import evaluate_model_on_test, save_evaluation_plots, select_champion
from src.models import get_baseline_spec, get_model_specs
from src.trainer import build_cv_splitter, train_and_tune_model


def _build_demo_presets(X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, dict[str, Any]]:
    """Canli Streamlit demosu icin kilitli test setinden gercek normal ve ariza ornegi secer."""
    normal_row = X_test[y_test == 0].iloc[0]
    failure_row = X_test[y_test == 1].iloc[0]

    def to_clean_dict(row: pd.Series) -> dict[str, Any]:
        return {
            col: (str(row[col]) if col == "Type" else float(row[col]))
            for col in FEATURE_COLUMNS
        }

    return {
        "normal_sample": to_clean_dict(normal_row),
        "failure_sample": to_clean_dict(failure_row),
    }


def run_full_pipeline(
    data_path: Path = RAW_DATA_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    """Tum 6 asamayi calistirir, modelleri/grafikleri/metrikleri kaydeder."""
    plots_dir = artifacts_dir / "plots"
    models_dir = artifacts_dir / "models"
    metrics_path = artifacts_dir / "metrics.json"
    eda_summary_path = artifacts_dir / "eda_summary.json"

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Asama 2: Veriyi Anlama (EDA)
    raw_df = load_raw_data(data_path)
    eda_summary = generate_eda_artifacts(
        raw_df,
        plots_dir=plots_dir,
        summary_path=eda_summary_path,
    )

    # Asama 3: Veri Hazirlama (Once stratified bol, sizinti sutunlarini cikar)
    X, y = split_features_target(raw_df)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    cv = build_cv_splitter(cv_folds=CV_FOLDS, random_state=RANDOM_STATE)

    # Asama 4 & 5: Taban cizgisi + 4 bagimsiz Pipeline egitimi ve testi
    all_specs = [get_baseline_spec(), *get_model_specs()]
    eval_results: list[dict[str, Any]] = []

    for spec in all_specs:
        trained = train_and_tune_model(
            spec,
            X_train,
            y_train,
            cv=cv,
            scoring=PRIMARY_METRIC,
        )
        evaluation = evaluate_model_on_test(trained, X_test, y_test)
        eval_results.append(evaluation)

        if spec.key != "dummy_baseline":
            joblib.dump(trained.best_estimator, models_dir / f"{spec.key}.joblib")

    champion_key = select_champion(eval_results)
    save_evaluation_plots(eval_results, plots_dir=plots_dir)

    serializable_models = [
        {k: v for k, v in item.items() if k != "_curves"}
        for item in eval_results
    ]

    report_payload = {
        "dataset": {
            "rows": eda_summary["rows"],
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "primary_metric": PRIMARY_METRIC,
            "features": FEATURE_COLUMNS,
        },
        "champion_key": champion_key,
        "champion_selection_rule": "5-katli StratifiedKFold CV F1-Score ortalamasi en yuksek aday",
        "models": serializable_models,
        "demo_presets": _build_demo_presets(X_test, y_test),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    return report_payload


def _print_summary(payload: dict[str, Any]) -> None:
    """Egitim ve test sonuclarini terminale ozet tablo olarak basar."""
    champion_key = payload["champion_key"]
    print(f"\nSampiyon Model (CV F1): {champion_key}")
    print("-" * 92)
    print(
        f"{'Model':<26} {'Ekip':<14} {'CV F1':>8} {'Test F1':>8} "
        f"{'Recall':>8} {'Prec':>8} {'ROC-AUC':>8} {'PR-AUC':>8}"
    )
    print("-" * 92)
    for m in payload["models"]:
        tm = m["test_metrics"]
        mark = " *" if m["key"] == champion_key else ""
        print(
            f"{(m['display_name'] + mark):<26} {m['owner']:<14} "
            f"{m['cv_f1_mean']:>8.4f} {tm['f1']:>8.4f} "
            f"{tm['recall']:>8.4f} {tm['precision']:>8.4f} "
            f"{tm['roc_auc']:>8.4f} {tm['pr_auc']:>8.4f}"
        )


if __name__ == "__main__":
    summary = run_full_pipeline()
    _print_summary(summary)
