"""Veri anlama ve sizintisiz hazirlama modulu birim testleri."""

import numpy as np
import pandas as pd

from src.config import DROP_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.data_prep import (
    build_preprocessor,
    create_train_test_split,
    generate_eda_artifacts,
    load_raw_data,
    split_features_target,
)


def test_raw_data_schema_and_completeness() -> None:
    df = load_raw_data()
    assert df.shape == (10000, 14)
    assert int(df.isna().sum().sum()) == 0
    assert int(df.duplicated().sum()) == 0


def test_split_features_target_prevents_leakage() -> None:
    df = load_raw_data()
    X, y = split_features_target(df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X.columns) == 6
    for forbidden in DROP_COLUMNS + [TARGET_COLUMN]:
        assert forbidden not in X.columns

    assert int(y.sum()) == 339
    assert len(y) == 10000


def test_stratified_train_test_split() -> None:
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert set(X_train.index).isdisjoint(set(X_test.index))
    # Stratified split sinif oranini (%3.39) korur
    assert abs(y_train.mean() - y_test.mean()) < 0.002


def test_preprocessors_scaled_vs_unscaled() -> None:
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, _, _ = create_train_test_split(X, y)

    scaled_prep = build_preprocessor(scale_numeric=True)
    X_train_scaled = scaled_prep.fit_transform(X_train)
    X_test_scaled = scaled_prep.transform(X_test)
    # 3 one-hot Type (L, M, H) + 5 numeric = 8 columns
    assert X_train_scaled.shape == (8000, 8)
    assert X_test_scaled.shape == (2000, 8)
    # Sayisal sutunlar (son 5 sutun) egitim setinde ort ~ 0, std ~ 1 olmali
    assert np.allclose(X_train_scaled[:, 3:].mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_train_scaled[:, 3:].std(axis=0), 1.0, atol=1e-7)

    unscaled_prep = build_preprocessor(scale_numeric=False)
    X_train_unscaled = unscaled_prep.fit_transform(X_train)
    assert X_train_unscaled.shape == (8000, 8)
    # Olceklenmemis Rotational speed ortalamasi ~1538 civari kalmali
    assert X_train_unscaled[:, 5].mean() > 1400.0

    # Gorulmemis kategori geldiginde hata vermeden 0,0,0 kodlamali
    unknown_row = pd.DataFrame(
        [
            {
                "Type": "X",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100,
            }
        ]
    )
    transformed_unknown = scaled_prep.transform(unknown_row)
    assert np.allclose(transformed_unknown[0, :3], 0.0)


def test_eda_artifacts_generation(tmp_path) -> None:
    df = load_raw_data()
    summary = generate_eda_artifacts(
        df,
        plots_dir=tmp_path / "plots",
        summary_path=tmp_path / "eda_summary.json",
    )
    assert summary["rows"] == 10000
    assert summary["class_distribution"]["failure_1"] == 339
    assert (tmp_path / "eda_summary.json").exists()
    assert (tmp_path / "plots" / "class_distribution.png").exists()
    assert (tmp_path / "plots" / "eda_correlation.png").exists()
