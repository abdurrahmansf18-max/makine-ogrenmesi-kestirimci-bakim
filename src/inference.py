"""CRISP-DM Asama 6 yardimci modulu: model yukleme, girdi dogrulama ve canli tahmin."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TYPE_CATEGORIES,
)


def load_metrics_report(metrics_path: Path = METRICS_PATH) -> dict[str, Any]:
    """Egitim/test metrik ozetini yukler; eksikse ne yapilacagini aciklar."""
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrik raporu bulunamadi: {metrics_path}. "
            "Neden: Egitim hatti henuz calistirilmamis. "
            "Cozum: Docker icinde 'python -m src.run_pipeline' komutunu calistirin."
        )
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_trained_models(
    model_keys: list[str],
    models_dir: Path = MODELS_DIR,
) -> dict[str, Pipeline]:
    """Paketlenen scikit-learn Pipeline dosyalarini (.joblib) bellege yukler."""
    loaded: dict[str, Pipeline] = {}
    for key in model_keys:
        model_path = models_dir / f"{key}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model paketi bulunamadi: {model_path}. "
                "Neden: Secilen algoritma icin .joblib dosyasi olusturulmamis. "
                "Cozum: Docker icinde 'python -m src.run_pipeline' komutunu calistirin."
            )
        loaded[key] = joblib.load(model_path)
    return loaded


def validate_raw_input(payload: dict[str, Any]) -> pd.DataFrame:
    """6 ham girdi sutununu dogrular ve Pipeline'in bekledigi sirada DataFrame uretir."""
    missing = [col for col in FEATURE_COLUMNS if col not in payload]
    if missing:
        raise ValueError(
            f"Eksik girdi alanlari: {missing}. "
            f"Beklenen 6 ham girdi: {FEATURE_COLUMNS}."
        )

    cleaned: dict[str, Any] = {}
    product_type = str(payload["Type"])
    if product_type not in TYPE_CATEGORIES:
        raise ValueError(
            f"Gecersiz urun kalite tipi: '{product_type}'. "
            f"Beklenen degerler: {TYPE_CATEGORIES}."
        )
    cleaned["Type"] = product_type

    for col in NUMERIC_FEATURES:
        try:
            cleaned[col] = float(payload[col])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sayisal girdi okunamadi ({col}): {payload[col]}") from exc

    return pd.DataFrame([cleaned], columns=FEATURE_COLUMNS)


def classify_risk(failure_prob: float, threshold: float = 0.5) -> dict[str, str]:
    """Renk disinda simge ve acik metinle erisilebilir ariza durum etiketi uretir."""
    if failure_prob >= threshold:
        return {
            "level": "high",
            "icon": "⚠️",
            "label": "Arıza Riski Yüksek",
            "decision": "Arıza Bekleniyor (1)",
            "action": "Üretimi durdurup sıcaklık, tork ve takım aşınmasını kontrol edin.",
        }
    return {
        "level": "low",
        "icon": "✅",
        "label": "Normal Çalışma",
        "decision": "Normal (0)",
        "action": "Mevcut sensör değerlerinde acil arıza sinyali gözlenmedi.",
    }


def predict_all_models(
    sample_df: pd.DataFrame,
    metrics_report: dict[str, Any],
    models: dict[str, Pipeline],
) -> list[dict[str, Any]]:
    """Ayni ham girdi icin tum egitilmis modelleri (sampiyon ilk sirada) tahmin ettirir."""
    champion_key = metrics_report["champion_key"]
    predictions: list[dict[str, Any]] = []

    for meta in metrics_report["models"]:
        key = meta["key"]
        if key == "dummy_baseline":
            continue
        if key not in models:
            raise KeyError(f"Yuklenen modeller icinde '{key}' bulunamadi.")

        pipe = models[key]
        prob = float(pipe.predict_proba(sample_df)[0, 1])
        pred_class = int(pipe.predict(sample_df)[0])
        risk = classify_risk(prob)

        predictions.append(
            {
                "key": key,
                "display_name": meta["display_name"],
                "owner": meta["owner"],
                "family": meta["family"],
                "is_champion": key == champion_key,
                "cv_f1_mean": float(meta["cv_f1_mean"]),
                "test_f1": float(meta["test_metrics"]["f1"]),
                "failure_probability": round(prob, 4),
                "failure_pct": round(prob * 100.0, 1),
                "predicted_class": pred_class,
                "status_text": f"{risk['icon']} {risk['label']}",
                "risk": risk,
            }
        )

    predictions.sort(key=lambda item: (not item["is_champion"], -item["cv_f1_mean"]))
    return predictions


def build_comparison_rows(metrics_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Model karsilastirma sekmesi icin duzenli tablo satirlari hazirlar."""
    champion_key = metrics_report["champion_key"]
    rows: list[dict[str, Any]] = []

    for m in metrics_report["models"]:
        tm = m["test_metrics"]
        is_champion = m["key"] == champion_key
        name = f"★ {m['display_name']} (Şampiyon)" if is_champion else m["display_name"]
        rows.append(
            {
                "Model": name,
                "Ekip Sorumlusu": m["owner"],
                "Öğrenme Ailesi": m["family"],
                "Ölçekleme": "Evet" if m["scale_numeric"] else "Hayır",
                "CV F1": round(float(m["cv_f1_mean"]), 4),
                "Test F1": round(float(tm["f1"]), 4),
                "Recall": round(float(tm["recall"]), 4),
                "Precision": round(float(tm["precision"]), 4),
                "ROC-AUC": round(float(tm["roc_auc"]), 4),
                "PR-AUC": round(float(tm["pr_auc"]), 4),
                "Eğitim (sn)": round(float(m["fit_seconds"]), 2),
            }
        )
    return rows
