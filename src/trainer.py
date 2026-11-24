"""CRISP-DM Asama 4: Ortak StratifiedKFold CV uzerinde bagimsiz Pipeline egitimi."""

from dataclasses import dataclass
import time
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, PRIMARY_METRIC, RANDOM_STATE
from src.data_prep import build_preprocessor
from src.models import ModelSpec


@dataclass
class TrainedModelResult:
    """Bir model icin CV sonuclari ve en iyi egitilmis Pipeline ciktisi."""

    spec: ModelSpec
    best_estimator: Pipeline
    best_params: dict[str, Any]
    cv_f1_mean: float
    cv_f1_std: float
    cv_train_f1_mean: float
    fit_seconds: float


def build_cv_splitter(
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Tum modellerde ayni fold bolunmelerini ureten sabit tam sayili splitter kurar."""
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def build_model_pipeline(spec: ModelSpec) -> Pipeline:
    """Algoritmanin olcekleme ihtiyacina gore sizintisiz Pipeline olusturur."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=spec.scale_numeric)),
            ("classifier", clone(spec.estimator)),
        ]
    )


def train_and_tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | None = None,
    scoring: str = PRIMARY_METRIC,
) -> TrainedModelResult:
    """Ortak CV katmanlari uzerinde tek bir model Pipeline'ini egitir ve ayarlar."""
    cv_splitter = cv or build_cv_splitter()
    pipeline = build_model_pipeline(spec)

    start = time.perf_counter()
    if not spec.param_grid:
        cv_out = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        return TrainedModelResult(
            spec=spec,
            best_estimator=pipeline,
            best_params={},
            cv_f1_mean=float(cv_out["test_score"].mean()),
            cv_f1_std=float(cv_out["test_score"].std()),
            cv_train_f1_mean=float(cv_out["train_score"].mean()),
            fit_seconds=round(elapsed, 3),
        )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.param_grid,
        cv=cv_splitter,
        scoring=scoring,
        refit=True,
        return_train_score=True,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    best_idx = int(search.best_index_)
    cv_results = search.cv_results_
    return TrainedModelResult(
        spec=spec,
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        cv_f1_mean=float(cv_results["mean_test_score"][best_idx]),
        cv_f1_std=float(cv_results["std_test_score"][best_idx]),
        cv_train_f1_mean=float(cv_results["mean_train_score"][best_idx]),
        fit_seconds=round(elapsed, 3),
    )
