"""Proje genelindeki sabit yollar, sutun tanimlari ve egitim ayarlari."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "ai4i2020.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
EDA_SUMMARY_PATH = ARTIFACTS_DIR / "eda_summary.json"

CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
DROP_COLUMNS = ["UDI", "Product ID", *FAILURE_MODE_COLUMNS]

TYPE_CATEGORIES = ["L", "M", "H"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PRIMARY_METRIC = "f1"

INPUT_METADATA = {
    "Type": {
        "label": "Ürün Kalite Tipi",
        "options": TYPE_CATEGORIES,
        "help": "L: Düşük (%60), M: Orta (%30), H: Yüksek (%10)",
        "default": "M",
    },
    "Air temperature [K]": {
        "label": "Hava Sıcaklığı (K)",
        "min": 290.0,
        "max": 315.0,
        "default": 300.0,
        "step": 0.1,
        "help": "Veri aralığı: 295,3 - 304,5 K (yaklaşık 22 - 31 °C)",
    },
    "Process temperature [K]": {
        "label": "İşlem Sıcaklığı (K)",
        "min": 300.0,
        "max": 325.0,
        "default": 310.0,
        "step": 0.1,
        "help": "Veri aralığı: 305,7 - 313,8 K",
    },
    "Rotational speed [rpm]": {
        "label": "Dönüş Hızı (rpm)",
        "min": 1000,
        "max": 3200,
        "default": 1500,
        "step": 10,
        "help": "Veri aralığı: 1168 - 2886 rpm",
    },
    "Torque [Nm]": {
        "label": "Tork (Nm)",
        "min": 0.0,
        "max": 90.0,
        "default": 40.0,
        "step": 0.5,
        "help": "Veri aralığı: 3,8 - 76,6 Nm",
    },
    "Tool wear [min]": {
        "label": "Takım Aşınma Süresi (dk)",
        "min": 0,
        "max": 300,
        "default": 108,
        "step": 1,
        "help": "Veri aralığı: 0 - 253 dk",
    },
}
