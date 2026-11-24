"""Arayuz yardimci modulu (src/inference.py) ve Streamlit uygulamasi (src/app.py) testleri."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from src.config import BASE_DIR, FEATURE_COLUMNS
from src.inference import (
    build_comparison_rows,
    classify_risk,
    load_metrics_report,
    load_trained_models,
    predict_all_models,
    validate_raw_input,
)


def test_inference_end_to_end_on_presets() -> None:
    report = load_metrics_report()
    keys = [m["key"] for m in report["models"] if m["key"] != "dummy_baseline"]
    models = load_trained_models(keys)

    for preset_name in ("normal_sample", "failure_sample"):
        sample_df = validate_raw_input(report["demo_presets"][preset_name])
        assert list(sample_df.columns) == FEATURE_COLUMNS
        preds = predict_all_models(sample_df, report, models)
        assert len(preds) == 4
        assert preds[0]["is_champion"] is True
        assert preds[0]["key"] == report["champion_key"]
        if preset_name == "normal_sample":
            assert preds[0]["failure_probability"] < 0.5
        else:
            assert preds[0]["failure_probability"] >= 0.5
        for p in preds:
            assert 0.0 <= p["failure_probability"] <= 1.0
            assert p["status_text"].startswith(("⚠️", "✅"))


def test_validation_and_missing_artifact_messages(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        load_metrics_report(tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        load_trained_models(["hist_gradient_boosting"], models_dir=tmp_path)

    with pytest.raises(ValueError, match="kalite tipi"):
        validate_raw_input(
            {
                "Type": "Z",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100,
            }
        )

    high_risk = classify_risk(0.82)
    low_risk = classify_risk(0.12)
    assert high_risk["icon"] == "⚠️" and high_risk["level"] == "high"
    assert low_risk["icon"] == "✅" and low_risk["level"] == "low"


def test_comparison_table_and_streamlit_app_renders_cleanly() -> None:
    report = load_metrics_report()
    rows = build_comparison_rows(report)
    assert len(rows) == 5
    assert any("Şampiyon" in row["Model"] for row in rows)

    app_path = BASE_DIR / "src" / "app.py"
    at = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not at.exception
    assert len(at.tabs) == 2

    # "Gerçek Arıza Kaydı Yükle" dugmesine tiklayinca hata vermeden ariza riskini gostermeli
    failure_btn = next(b for b in at.button if "Arıza Kaydı" in b.label)
    failure_btn.click().run(timeout=20)
    assert not at.exception
    assert len(at.dataframe) >= 2
    assert len(at.error) >= 1
