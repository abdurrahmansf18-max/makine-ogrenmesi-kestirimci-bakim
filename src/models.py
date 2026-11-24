"""Denetimli ogrenme algoritmalari ve hiperparametre arama uzaylari."""

from dataclasses import dataclass
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
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
    ]
