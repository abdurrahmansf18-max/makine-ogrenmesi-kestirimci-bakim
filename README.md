# Makine Öğrenmesi Dönem Projesi (Docker Ortamı)

Bu klasör, ders kapsamında geliştirilecek makine öğrenmesi uygulamasını ve sade Streamlit arayüzünü içerir. Tüm kurulum ve çalıştırma adımları yerel kurulum yapılmadan Docker üzerinden yürütülür.

## Docker Komutları

`ml_proje/` dizini içinde çalıştırılır:

```bash
docker compose build
docker compose run --rm ml-app python src/check_env.py
docker compose up ml-ui
```

Streamlit arayüzü `http://127.0.0.1:8501` adresinde açılır.

## Proje Dokümanları

- `docs/proje-degerlendirme-raporu.pdf`: Kodlama öncesi konu, veri kümesi, 4 algoritma, 6 aşamalı yöntem, mimari ve araştırma doğrulamasını içeren ekip değerlendirme raporu.
