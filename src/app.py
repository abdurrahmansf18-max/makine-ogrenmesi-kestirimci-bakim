"""CRISP-DM Asama 6: Sade 2 sekmeli Streamlit kestirimci bakim arayuzu."""

from pathlib import Path
import sys
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, INPUT_METADATA, PLOTS_DIR  # noqa: E402
from src.inference import (  # noqa: E402
    build_comparison_rows,
    load_metrics_report,
    load_trained_models,
    predict_all_models,
    validate_raw_input,
)

st.set_page_config(
    page_title="CNC Kestirimci Bakım — ML Dönem Projesi",
    layout="centered",
)


@st.cache_resource
def _load_runtime_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    """Metrik raporunu ve 4 egitilmis modeli bellek onbellegine alir."""
    metrics_report = load_metrics_report()
    supervised_keys = [
        m["key"] for m in metrics_report["models"] if m["key"] != "dummy_baseline"
    ]
    models = load_trained_models(supervised_keys)
    return metrics_report, models


def _cast_for_widget(col: str, raw_value: Any) -> Any:
    """Widget adim turuyle (int/float) birebir uyumlu deger dondurur."""
    if col == "Type":
        return str(raw_value)
    step = INPUT_METADATA[col]["step"]
    return int(round(float(raw_value))) if isinstance(step, int) else float(raw_value)


def _init_input_state(default_sample: dict[str, Any]) -> None:
    """Ilk acilista form alanlarini varsayilan bir ornekle doldurur."""
    for col in FEATURE_COLUMNS:
        key = f"input_{col}"
        if key not in st.session_state:
            fallback = INPUT_METADATA[col]["default"]
            st.session_state[key] = _cast_for_widget(col, default_sample.get(col, fallback))
    if "prediction_requested" not in st.session_state:
        st.session_state["prediction_requested"] = True


def _apply_preset(sample: dict[str, Any], label: str) -> None:
    """Hazir test kaydini widget durumuna yukler."""
    for col in FEATURE_COLUMNS:
        st.session_state[f"input_{col}"] = _cast_for_widget(col, sample[col])
    st.session_state["active_preset_note"] = label
    st.session_state["prediction_requested"] = True


def _render_sensor_input(col_name: str) -> None:
    """Tek bir ham girdi kontrolunu erisilebilir etiket ve aralik yardimiyla cizer."""
    meta = INPUT_METADATA[col_name]
    widget_key = f"input_{col_name}"
    if col_name == "Type":
        st.selectbox(
            label=meta["label"],
            options=meta["options"],
            help=meta["help"],
            key=widget_key,
        )
        return

    st.number_input(
        label=meta["label"],
        min_value=meta["min"],
        max_value=meta["max"],
        step=meta["step"],
        help=meta["help"],
        key=widget_key,
    )


def _render_live_prediction_tab(
    metrics_report: dict[str, Any],
    models: dict[str, Any],
) -> None:
    """Sekme 1: 6 ham sensor girdisi ve canli ariza riski tahmini."""
    presets = metrics_report["demo_presets"]
    _init_input_state(presets["normal_sample"])

    st.subheader("1. Makine Çalışma Parametreleri (6 Ham Girdi)")
    st.caption(
        "Eğitimde sızıntı sütunları çıkarılmıştır; tahmin yalnızca 6 ham üretim ve sensör verisiyle yapılır."
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button(
            "Gerçek Normal Kayıt Yükle",
            on_click=_apply_preset,
            args=(presets["normal_sample"], "Test kümesinden gerçek Normal (0) örnek yüklendi."),
            width="stretch",
        )
    with btn_col2:
        st.button(
            "Gerçek Arıza Kaydı Yükle",
            on_click=_apply_preset,
            args=(presets["failure_sample"], "Test kümesinden gerçek Arıza (1) örnek yüklendi."),
            width="stretch",
        )

    if st.session_state.get("active_preset_note"):
        st.caption(st.session_state["active_preset_note"])

    with st.form("live_prediction_form"):
        left_col, right_col = st.columns(2)
        with left_col:
            for col_name in FEATURE_COLUMNS[:3]:
                _render_sensor_input(col_name)
        with right_col:
            for col_name in FEATURE_COLUMNS[3:]:
                _render_sensor_input(col_name)

        submitted = st.form_submit_button(
            "Arıza Riskini Tahmin Et",
            type="primary",
            width="stretch",
        )

    if submitted:
        st.session_state["active_preset_note"] = ""
        st.session_state["prediction_requested"] = True

    if not st.session_state.get("prediction_requested"):
        return

    payload = {col: st.session_state[f"input_{col}"] for col in FEATURE_COLUMNS}
    try:
        sample_df = validate_raw_input(payload)
        results = predict_all_models(sample_df, metrics_report, models)
    except (ValueError, KeyError) as exc:
        st.error(f"Tahmin hesaplanamadı: {exc}")
        return

    champion = results[0]
    risk = champion["risk"]

    st.subheader("2. Canlı Tahmin Sonucu")
    headline = (
        f"**{risk['icon']} {risk['label']}** — "
        f"Şampiyon Model (**{champion['display_name']}**) arıza olasılığı: **%{champion['failure_pct']:.1f}**"
    )
    if risk["level"] == "high":
        st.error(f"{headline}\n\nÖneri: {risk['action']}")
    else:
        st.success(f"{headline}\n\nDurum: {risk['action']}")

    st.caption("Aynı girdi üzerinde 4 denetimli algoritmanın karşılaştırmalı canlı olasılıkları:")
    comparison_live = [
        {
            "Model": f"★ {r['display_name']} (Şampiyon)" if r["is_champion"] else r["display_name"],
            "Ekip": r["owner"],
            "Durum": r["status_text"],
            "Arıza Olasılığı (%)": r["failure_pct"],
            "Model CV F1": round(r["cv_f1_mean"], 4),
        }
        for r in results
    ]
    st.dataframe(comparison_live, hide_index=True, width="stretch")


def _render_comparison_tab(metrics_report: dict[str, Any]) -> None:
    """Sekme 2: 4 denetimli modelin test metrikleri ve gorsel degerlendirmesi."""
    champion_key = metrics_report["champion_key"]
    champion_meta = next(m for m in metrics_report["models"] if m["key"] == champion_key)

    st.subheader("Şampiyon Seçimi Özeti")
    st.info(
        f"Ana seçim kuralı: **{metrics_report['champion_selection_rule']}** "
        "(kilitli test kümesine seçime kadar bakılmamıştır)."
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Şampiyon Model", champion_meta["display_name"])
    k2.metric("5-Katlı CV F1", f"{champion_meta['cv_f1_mean']:.4f}")
    k3.metric("Kilitli Test F1", f"{champion_meta['test_metrics']['f1']:.4f}")

    st.subheader("4 Model + Taban Çizgisi Karşılaştırma Tablosu")
    st.dataframe(
        build_comparison_rows(metrics_report),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Değerlendirme Grafikleri (Kilitli %20 Test Kümesi — 2.000 Kayıt)")
    plot_specs = [
        ("model_comparison.png", "Test kümesinde 4 modelin F1, Recall, Precision, ROC-AUC ve PR-AUC karşılaştırması"),
        ("confusion_matrices.png", "4 model için karışıklık matrisleri (DN, YP, YN, DP)"),
        ("roc_pr_curves.png", "ROC ve Precision-Recall eğrileri (renk ve çizgi stili ayrımıyla)"),
    ]
    for filename, caption in plot_specs:
        path = PLOTS_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")

    with st.expander("Veri Anlama (EDA) Grafikleri"):
        for filename, caption in [
            ("class_distribution.png", "Ham veri kümesi sınıf dağılımı (9.661 Normal / 339 Arıza)"),
            ("eda_correlation.png", "5 sayısal sensör ve hedef sütun korelasyon matrisi"),
        ]:
            path = PLOTS_DIR / filename
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")


def main() -> None:
    """Ana Streamlit uygulama giris noktasi."""
    st.title("CNC Makine Arızası Kestirimci Bakım")
    st.caption("AI4I 2020 • 6 Ham Sensör Girdisi • 4 Denetimli Algoritma ve F1-Score Şampiyon Seçimi")

    try:
        metrics_report, models = _load_runtime_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab_live, tab_compare = st.tabs(["Canlı Tahmin", "Model Karşılaştırması"])
    with tab_live:
        _render_live_prediction_tab(metrics_report, models)
    with tab_compare:
        _render_comparison_tab(metrics_report)


if __name__ == "__main__":
    main()
