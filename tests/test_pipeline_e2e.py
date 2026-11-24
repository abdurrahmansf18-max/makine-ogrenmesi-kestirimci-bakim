"""6 asamali egitim, degerlendirme ve paketleme hattinin uctan uca testi."""

import json

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.run_pipeline import run_full_pipeline


def test_full_pipeline_end_to_end(tmp_path) -> None:
    payload = run_full_pipeline(artifacts_dir=tmp_path)

    assert payload["dataset"]["train_rows"] == 8000
    assert payload["dataset"]["test_rows"] == 2000
    assert payload["dataset"]["features"] == FEATURE_COLUMNS

    champion_key = payload["champion_key"]
    model_keys = [m["key"] for m in payload["models"]]
    assert "dummy_baseline" in model_keys
    assert champion_key in model_keys
    assert champion_key != "dummy_baseline"

    metrics_file = tmp_path / "metrics.json"
    assert metrics_file.exists()
    loaded_json = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert loaded_json["champion_key"] == champion_key

    for expected_plot in [
        "class_distribution.png",
        "eda_correlation.png",
        "model_comparison.png",
        "confusion_matrices.png",
        "roc_pr_curves.png",
    ]:
        assert (tmp_path / "plots" / expected_plot).exists()

    # 4 egitilmis model dosyasi yuklenebilmeli ve 6 ham girdiyle tahmin vermeli
    dummy_model = next(m for m in payload["models"] if m["key"] == "dummy_baseline")
    candidates = [m for m in payload["models"] if m["key"] != "dummy_baseline"]
    assert len(candidates) == 4

    champion_model = next(m for m in candidates if m["key"] == champion_key)
    assert champion_model["test_metrics"]["f1"] > 0.70
    for m in candidates:
        assert m["cv_f1_mean"] <= champion_model["cv_f1_mean"]

    sample_df = pd.DataFrame([payload["demo_presets"]["failure_sample"]])
    for m in candidates:
        assert m["test_metrics"]["f1"] > dummy_model["test_metrics"]["f1"]
        assert m["test_metrics"]["roc_auc"] >= 0.85
        model_path = tmp_path / "models" / f"{m['key']}.joblib"
        assert model_path.exists()
        pipe = joblib.load(model_path)
        probs = pipe.predict_proba(sample_df)
        assert probs.shape == (1, 2)
        assert abs(float(probs.sum()) - 1.0) < 1e-6
