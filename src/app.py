"""Makine Ogrenmesi Donem Projesi - Sade Streamlit Arayuz Iskeleti."""

import streamlit as st

st.set_page_config(
    page_title="ML Donem Projesi",
    layout="centered",
)

st.title("Makine Öğrenmesi Dönem Projesi")
st.info(
    "Arayüz ortamı Docker içinde hazır. Konu, veri kümesi ve 4 algoritma "
    "kesinleştirilip modeller eğitildikten sonra canlı tahmin ve karşılaştırma "
    "ekranları buraya bağlanacaktır."
)
