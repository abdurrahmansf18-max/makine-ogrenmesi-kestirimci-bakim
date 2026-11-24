"""1. Kisi modelleri (LogisticRegression ve SVC) egitim testleri."""

from src.data_prep import create_train_test_split, load_raw_data, split_features_target
from src.models import get_baseline_spec, get_model_specs
from src.trainer import build_cv_splitter, train_and_tune_model


def test_person1_models_pipeline_and_tuning() -> None:
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, _ = create_train_test_split(X, y)
    cv = build_cv_splitter()

    baseline = train_and_tune_model(get_baseline_spec(), X_train, y_train, cv=cv)
    assert baseline.cv_f1_mean == 0.0

    person1_specs = {spec.key: spec for spec in get_model_specs() if spec.owner == "1. Kişi"}
    assert set(person1_specs.keys()) == {"logistic_regression", "svc_rbf"}

    for spec in person1_specs.values():
        assert spec.scale_numeric is True
        result = train_and_tune_model(spec, X_train, y_train, cv=cv)
        assert result.cv_f1_mean > baseline.cv_f1_mean
        probs = result.best_estimator.predict_proba(X_test.iloc[:10])
        assert probs.shape == (10, 2)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
