"""Denetimli ogrenme algoritmalari ve hiperparametre arama uzaylari."""

from dataclasses import dataclass
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config import RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    """Bir model adayinin Pipeline ve GridSearchCV yapilandirmasi."""

    key: str
    display_name: str
    owner: str
    family: str
    scale_numeric: bool
    estimator: Any
    param_grid: dict[str, list[Any]]


def get_baseline_spec() -> ModelSpec:
    """Kiyaslama alt siniri icin basit cogunluk sinifi taban cizgisini dondurur."""
    return ModelSpec(
        key="dummy_baseline",
        display_name="Dummy Baseline (Çoğunluk)",
        owner="Ortak Referans",
        family="Taban Çizgisi",
        scale_numeric=False,
        estimator=DummyClassifier(strategy="most_frequent"),
        param_grid={},
    )


def get_model_specs() -> list[ModelSpec]:
    """Projede kiyaslanacak aktif denetimli algoritma listesini dondurur."""
    return [
        ModelSpec(
            key="logistic_regression",
            display_name="Lojistik Regresyon",
            owner="1. Kişi",
            family="Doğrusal / Olasılıksal",
            scale_numeric=True,
            estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            param_grid={
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            key="svc_rbf",
            display_name="SVC (RBF Çekirdeği)",
            owner="1. Kişi",
            family="Çekirdek / Marjin",
            scale_numeric=True,
            estimator=CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=RANDOM_STATE),
                cv=3,
                ensemble=False,
            ),
            param_grid={
                "classifier__estimator__C": [1.0, 10.0, 30.0],
                "classifier__estimator__gamma": ["scale", 0.2],
                "classifier__estimator__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            key="hist_gradient_boosting",
            display_name="HistGradientBoosting",
            owner="2. Kişi",
            family="Ağaç Topluluğu (Boosting)",
            scale_numeric=False,
            estimator=HistGradientBoostingClassifier(
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
            param_grid={
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_iter": [150, 250],
                "classifier__max_leaf_nodes": [15, 31],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            key="mlp_classifier",
            display_name="Yapay Sinir Ağı (MLP)",
            owner="2. Kişi",
            family="Yapay Sinir Ağı (YSA)",
            scale_numeric=True,
            estimator=MLPClassifier(
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=RANDOM_STATE,
            ),
            param_grid={
                "classifier__hidden_layer_sizes": [(64, 32), (100, 50)],
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__learning_rate_init": [0.001, 0.005],
            },
        ),
    ]
