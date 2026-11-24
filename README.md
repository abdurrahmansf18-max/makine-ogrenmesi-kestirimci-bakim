# AI4I 2020 Kestirimci Bakım — Makine Öğrenmesi Dönem Projesi

İki kişilik ekip tarafından **CRISP-DM 6 Aşamalı Eğitim Süreci** doğrultusunda geliştirilen, **6 ham üretim/sensör girdisiyle** CNC makine arızası tahmini yapan ve **4 denetimli öğrenme ailesini** aynı test kümesinde kıyaslayan Docker tabanlı uygulamadır.

## Ekip İş Bölümü ve Algoritmalar

- **Taban Çizgisi:** `DummyClassifier` (`most_frequent`)
- **1. Kişi:** `LogisticRegression` (Doğrusal) ve `SVC` (RBF Çekirdeği, `CalibratedClassifierCV`)
- **2. Kişi:** `HistGradientBoostingClassifier` (Ağaç Topluluğu — CV F1 Şampiyonu) ve `MLPClassifier` (Yapay Sinir Ağı)
- **Veri ve Ölçüt:** `AI4I 2020 Predictive Maintenance` (`10.000` satır, `0` eksik değer), sızıntı sütunları (`UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`) çıkarılmış `6` ham girdi, ortak `80/20` stratified bölünme ve **5-katlı `StratifiedKFold` CV `F1-Score`** şampiyon seçimi.

## Klasör Yapısı

```text
ml_proje/
├── Dockerfile                # Python 3.12-slim görüntü tanımı
├── compose.yaml              # ml-app (CLI/test) ve ml-ui (Streamlit :8501) servisleri
├── requirements.txt          # Sabitlenmiş bağımlılık sürümleri
├── README.md                 # Proje kullanım ve klasör rehberi
├── data/
│   └── raw/ai4i2020.csv      # UCI ID 601 ham veri kümesi (10.000 x 14)
├── artifacts/
│   ├── eda_summary.json      # Aşama 2 yapılandırılmış veri özeti
│   ├── metrics.json          # Aşama 4-5 CV/test metrikleri ve şampiyon seçimi
│   ├── models/*.joblib       # 4 eğitilmiş sızıntısız scikit-learn Pipeline paketi
│   └── plots/*.png           # EDA ve değerlendirme grafikleri (5 adet)
├── src/
│   ├── config.py             # Ortak yollar, 6 girdi şeması ve sabitler
│   ├── data_prep.py          # Aşama 2-3: EDA ve sızıntısız ColumnTransformer
│   ├── models.py             # Aşama 4: Taban çizgisi + 4 denetimli model tanımı
│   ├── trainer.py            # Aşama 4: StratifiedKFold + GridSearchCV eğitimi
│   ├── evaluate.py           # Aşama 5: Kilitli test ölçümü, F1 seçimi ve grafikler
│   ├── run_pipeline.py       # Aşama 2-5 uçtan uca orkestratör
│   ├── inference.py          # Aşama 6: Model yükleme, girdi doğrulama, tahmin
│   ├── app.py                # Aşama 6: Sade 2 sekmeli Streamlit web arayüzü
│   └── check_env.py          # Docker ortamı paket ve temel akış doğrulaması
├── tests/
│   ├── test_data_prep.py     # Veri şeması, sızıntı önleme, ölçekleme testleri
│   ├── test_person1_models.py# 1. Kişi (LR + SVC) eğitim testleri
│   ├── test_person2_models.py# 2. Kişi (HistGB + MLP) eğitim testleri
│   ├── test_pipeline_e2e.py  # Uçtan uca pipeline, şampiyon ve .joblib testleri
│   └── test_app_helpers.py   # Arayüz yardımcıları ve Streamlit AppTest doğrulaması
└── docs/
    ├── proje-degerlendirme-raporu.pdf # Ekip değerlendirme raporu
    ├── mimari-c4.md                   # C4 mimari diyagramları (C1-C4)
    └── dogrulama-kanitlari.md         # Issue #1-#5 yapılandırılmış doğrulama kanıtları
```

## Docker Komutları

Tüm komutlar yerel kurulum yapılmadan `ml_proje/` içinde (veya proje kökünden `-f ml_proje/compose.yaml` ile) çalıştırılır:

```bash
# 1. Docker imajını oluştur
docker compose build

# 2. Tüm birim ve uçtan uca testleri çalıştır
docker compose run --rm ml-app pytest -v

# 3. 6 aşamalı eğitim, değerlendirme ve paketleme hattını çalıştır
docker compose run --rm ml-app python -m src.run_pipeline

# 4. Sade Streamlit web arayüzünü başlat (http://127.0.0.1:8501)
docker compose up -d ml-ui

# 5. Arayüzü durdur ve konteyneri temizle
docker compose down
```

## Mimari ve Doğrulama Dokümanları

- [C4 Mimari Diyagramları (C1–C4)](docs/mimari-c4.md)
- [Aşama Bazlı Doğrulama Kanıtları (Issue #1–#5)](docs/dogrulama-kanitlari.md)
- [Proje Değerlendirme Raporu (PDF)](docs/proje-degerlendirme-raporu.pdf)
