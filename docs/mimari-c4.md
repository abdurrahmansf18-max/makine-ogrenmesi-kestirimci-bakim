# C4 Mimari Diyagramları (C1–C4)

Bu doküman, **AI4I 2020 Kestirimci Bakım** uygulamasının yazılım mimarisini Simon Brown'un **C4 Modeli** kapsamındaki dört seviyede (**C1 Sistem Bağlamı**, **C2 Konteyner**, **C3 Bileşen** ve **C4 Kod**) belgeler.

> [!NOTE]
> Bu projede kod tabanı 8 küçük Python modülünden oluştuğu için C1–C3 seviyelerine ek olarak çekirdek `Pipeline` ve veri sözleşmelerini özetleyen C4 (Kod) seviyesi de uygulanabilir bulunmuş ve eklenmiştir.

---

## C1 — Sistem Bağlamı (System Context)

Kullanıcı rolleri, kapsamdaki ML sistemi ve harici veri/sürüm kaynakları arasındaki ilişkiyi gösterir.

```mermaid
flowchart LR
    Operator["Makine Operatörü / Ders Değerlendiricisi"]
    Team["Öğrenci Ekibi (1. Kişi ve 2. Kişi)"]
    System["AI4I 2020 Kestirimci Bakım Sistemi (Docker ml-app + ml-ui)"]
    UCI["UCI Machine Learning Repository (ID 601: ai4i2020.csv)"]
    GitHub["GitHub Deposu (Issues, Dallar, PR ve v1.0)"]

    Operator -->|"6 ham sensör girdisi gönderir, tahmin ve metrik izler"| System
    Team -->|"Eğitim hattı ve pytest doğrulamalarını çalıştırır"| System
    UCI -->|"Ham CSV veri kümesi sağlar (10.000 satır)"| System
    Team -->|"Issue bazlı kısa dal ve PR geliştirir"| GitHub
```

---

## C2 — Konteyner Diyagramı (Container)

Tüm çalışma zamanı **Docker Compose** (`compose.yaml`) içindeki Python 3.12 imajı (`ml-ders-projesi:py3.12`) ve paylaşılan yerel birim (`ml_proje/:/app`) üzerinden yürür; yerel makineye Python paketi kurulmaz.

```mermaid
flowchart TD
    User["Tarayıcı / Terminal Kullanıcısı"]
    subgraph DockerHost["Docker Compose Ortamı (ml-ders-net)"]
        UI["ml-ui Konteyneri (Streamlit 1.67 · 127.0.0.1:8501)"]
        App["ml-app Konteyneri (Python 3.12 · scikit-learn 1.9.1 · pytest)"]
        Storage["Paylaşılan Çalışma Alanı (data/raw + artifacts)"]
    end

    User -->|"HTTP :8501 canlı tahmin ve kıyaslama"| UI
    User -->|"CLI: pytest / python -m src.run_pipeline"| App
    App -->|"Ham veri okur (ai4i2020.csv)"| Storage
    App -->|"4 .joblib modeli, JSON özetler ve PNG grafikler yazar"| Storage
    UI -->|"Eğitilmiş .joblib modelleri, metrics.json ve grafikleri okur"| Storage
```

---

## C3 — Bileşen Diyagramı (Component — `ml_proje/src/`)

`src/` altındaki modüllerin **CRISP-DM 6 Aşamalı Eğitim Süreci** ile birebir eşleşimini gösterir.

```mermaid
flowchart TD
    Config["src/config.py (Sabit yollar, 6 ham girdi şeması, CV/Test ayarları)"]
    DataPrep["src/data_prep.py (Aşama 2-3: EDA özeti, sızıntısız 80/20 stratified bölme, ColumnTransformer)"]
    Models["src/models.py (Aşama 4: Dummy + 1. Kişi LR/SVC + 2. Kişi HGB/MLP tanımları)"]
    Trainer["src/trainer.py (Aşama 4: Ortak StratifiedKFold CV + GridSearchCV F1 ayarı)"]
    Evaluate["src/evaluate.py (Aşama 5: Kilitli test ölçümü, CV F1 şampiyon seçimi, grafikler)"]
    Runner["src/run_pipeline.py (Orkestratör: Aşama 2-5 akışını uçtan uca çalıştırır)"]
    Inference["src/inference.py (Aşama 6 Yardımcı: Model/JSON yükleme, girdi doğrulama, risk sınıflama)"]
    AppUI["src/app.py (Aşama 6 UI: 2 sekmeli sade Streamlit arayüzü)"]

    Config --> DataPrep
    Config --> Models
    Config --> Trainer
    Config --> Evaluate
    Config --> Inference
    DataPrep --> Runner
    Models --> Trainer
    Trainer --> Runner
    Evaluate --> Runner
    Runner -->|"artifacts/models/*.joblib ve metrics.json"| Inference
    Inference --> AppUI
```

---

## C4 — Kod Düzeyi (Code — Çekirdek Veri Sözleşmeleri ve `Pipeline` Yapısı)

Model adaylarını tanımlayan `ModelSpec`, eğitim çıktısı `TrainedModelResult` ve her algoritmayı veri sızıntısından koruyan `Pipeline` iç kompozisyonunu özetler.

```mermaid
classDiagram
    class ModelSpec {
        +str key
        +str display_name
        +str owner
        +str family
        +bool scale_numeric
        +Any estimator
        +dict param_grid
    }

    class TrainedModelResult {
        +ModelSpec spec
        +Pipeline best_estimator
        +dict best_params
        +float cv_f1_mean
        +float cv_f1_std
        +float cv_train_f1_mean
        +float fit_seconds
    }

    class SklearnPipeline {
        +ColumnTransformer preprocessor
        +Estimator classifier
        +fit(X_train, y_train)
        +predict_proba(X)
    }

    class ColumnTransformer {
        +cat: SimpleImputer + OneHotEncoder(Type)
        +num: SimpleImputer + optional StandardScaler(5 sensör)
    }

    ModelSpec --> SklearnPipeline : "build_model_pipeline(spec)"
    SklearnPipeline *-- ColumnTransformer : "preprocessor"
    SklearnPipeline --> TrainedModelResult : "train_and_tune_model() üretir"
```
