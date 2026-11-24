# Aşama ve Issue Bazlı Doğrulama Kanıtları

Bu belge, projedeki her CRISP-DM aşamasının ve GitHub Issue adımının Docker ortamında gerçekten çalıştığını kanıtlayan test sonuçlarını ve yapılandırılmış çıktıları toplu biçimde sunar.

> [!NOTE]
> Güncel tam doğrulama komutu (`docker compose run --rm ml-app pytest -v`): **12 passed in 23.18s** (sıfır hata, sıfır atlanan test).

---

## Özet Matris

| Issue | CRISP-DM Aşaması | İlgili Modüller / Dosyalar | Test Dosyası ve Sonuç | Kanıtlanan Yapılandırılmış Çıktı |
|---|---|---|---|---|
| **#1** | Aşama 1–3 (Veriyi Anlama & Hazırlama) | `src/config.py`, `src/data_prep.py` | `tests/test_data_prep.py` (**5/5 PASSED**) | `artifacts/eda_summary.json`, `class_distribution.png`, `eda_correlation.png` |
| **#2** | Aşama 4 (1. Kişi Modelleri) | `src/models.py`, `src/trainer.py` | `tests/test_person1_models.py` (**1/1 PASSED**) | `Dummy`, `LogisticRegression`, `SVC (RBF)` CV/Test skorları |
| **#3** | Aşama 4 (2. Kişi Modelleri) | `src/models.py`, `src/trainer.py` | `tests/test_person2_models.py` (**2/2 PASSED**) | `HistGradientBoostingClassifier`, `MLPClassifier` CV/Test skorları |
| **#4** | Aşama 5 (Değerlendirme & Şampiyon Seçimi) | `src/evaluate.py`, `src/run_pipeline.py` | `tests/test_pipeline_e2e.py` (**1/1 PASSED**) | `artifacts/metrics.json`, 4 `.joblib` paketi, 3 değerlendirme grafiği |
| **#5** | Aşama 6 (Streamlit Arayüzü & Mimari Dokümantasyon) | `src/inference.py`, `src/app.py`, `docs/mimari-c4.md`, `README.md` | `tests/test_app_helpers.py` (**3/3 PASSED**) + canlı `ml-ui` | `AppTest` 2 sekme/preset kontrolü + `http://127.0.0.1:8501/_stcore/health` (`ok`) |

---

## 1. Veri Bütünlüğü ve Sızıntısız Hazırlama Kanıtı (Issue #1)

**Doğrulanan testler (`tests/test_data_prep.py`):**
- `test_raw_data_schema_and_completeness` — Ham `ai4i2020.csv` boyutu `(10000, 14)`, eksik değer `0`, yinelenen satır `0`.
- `test_split_features_target_prevents_leakage` — Yalnızca 6 ham girdi (`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`) korunur; `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` ve `Machine failure` özellik matrisinden çıkarılır (`339` arıza / `9.661` normal, `%3.39`).
- `test_stratified_train_test_split` — `8000 / 2000` ayrık indeksli eğitim/test bölünmesi ve sınıf oranı korunumu (`|oran farkı| < 0.002`).
- `test_preprocessors_scaled_vs_unscaled` — `3` One-Hot + `5` sayısal (`8` sütun); ölçekli hatta ortalama `~0`, std `~1`; bilinmeyen kategori `Type="X"` hatasız `[0, 0, 0]`.
- `test_eda_artifacts_generation` — `eda_summary.json`, `class_distribution.png` ve `eda_correlation.png` üretimi.

---

## 2. 4 Algoritma, Ortak CV ve Değerlendirme Kanıtları (Issue #2, #3, #4)

**Komutlar:**
```bash
docker compose run --rm ml-app python -m src.run_pipeline
docker compose run --rm ml-app pytest -v
```

**Şampiyon Seçimi (`X_test`'e bakılmadan 5-katlı StratifiedKFold CV `F1-Score`):**
- Seçilen Şampiyon: **`hist_gradient_boosting`** (`CV F1 = 0.7428`, `Test F1 = 0.7869`)

| Model | Ekip | Ölçekleme | En İyi Hiperparametreler | CV F1 | Test F1 | Recall | Precision | ROC-AUC | PR-AUC | Karışıklık (TN / FP / FN / TP) |
|---|---|---|---|---|---|---|---|---|---|---|
| `dummy_baseline` | Ortak Referans | Hayır | `{}` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.5000` | `0.0340` | `1932 / 0 / 68 / 0` |
| `logistic_regression` | 1. Kişi | Evet | `C=10.0, class_weight=null` | `0.3304` | `0.1975` | `0.1176` | `0.6154` | `0.9000` | `0.4234` | `1927 / 5 / 60 / 8` |
| `svc_rbf` | 1. Kişi | Evet | `C=30.0, gamma=0.2, class_weight=null` | `0.6725` | `0.6549` | `0.5441` | `0.8222` | `0.9764` | `0.7619` | `1924 / 8 / 31 / 37` |
| `hist_gradient_boosting` ★ | 2. Kişi | Hayır | `lr=0.1, max_iter=150, leaves=31, weight=null` | `0.7428` | `0.7869` | `0.7059` | `0.8889` | `0.9653` | `0.8415` | `1926 / 6 / 20 / 48` |
| `mlp_classifier` | 2. Kişi | Evet | `hidden=[64, 32], alpha=0.01, lr=0.005` | `0.6656` | `0.5946` | `0.4853` | `0.7674` | `0.9683` | `0.6969` | `1922 / 10 / 35 / 33` |

---

## 3. Arayüz, Canlı Tahmin ve Docker Sağlık Kanıtı (Issue #5)

**Testler (`tests/test_app_helpers.py` — 3/3 PASSED):**
- `test_inference_end_to_end_on_presets` — 4 eğitilmiş `.joblib` modelin yüklenmesi; `normal_sample` için şampiyon olasılığının `< 0.5`, `failure_sample` (`Type="M", Air=302.4, Process=310.1, rpm=1379.0, Torque=48.9, Tool wear=107.0`) için `>= 0.5` vermesi.
- `test_validation_and_missing_artifact_messages` — Eksik dosya ve geçersiz kalite tipi (`Type="Z"`) durumunda açıklayıcı hata mesajları ile simge+metin (`⚠️` / `✅`) erişilebilir risk sınıflandırması.
- `test_comparison_table_and_streamlit_app_renders_cleanly` — Streamlit `AppTest` ile `src/app.py` uygulamasının hatasız açılması, `2` sekmeyi oluşturması ve `"Gerçek Arıza Kaydı Yükle"` tıklamasında şampiyon arıza uyarısı (`st.error`) ile tabloyu güncellemesi.

**Canlı Konteyner Doğrulaması (`ml-ui`):**
- `docker compose up -d ml-ui` ardından:
  - `GET http://127.0.0.1:8501/_stcore/health` → `ok`
  - `GET http://127.0.0.1:8501/` → `<title>Streamlit</title>` (loglarda sıfır istisna)
  - `docker compose down` ile temiz durdurma ve ağ kaldırma.
