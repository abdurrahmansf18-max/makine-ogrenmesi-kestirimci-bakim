"""2. Kisi modelleri (HistGradientBoostingClassifier ve MLPClassifier) testleri."""

from src.data_prep import create_train_test_split, load_raw_data, split_features_target
from src.models import get_model_specs
from src.trainer import build_cv_splitter, train_and_tune_model


def test_four_distinct_supervised_models_and_team_split() -> None:
    specs = get_model_specs()
    assert len(specs) == 4
    owners = [s.owner for s in specs]
    assert owners.count("1. Kişi") == 2
    assert owners.count("2. Kişi") == 2
    assert len({s.family for s in specs}) == 4


def test_person2_models_pipeline_and_tuning() -> None:
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, _ = create_train_test_split(X, y)
    cv = build_cv_splitter()

    person2_specs = {spec.key: spec for spec in get_model_specs() if spec.owner == "2. Kişi"}
    assert set(person2_specs.keys()) == {"hist_gradient_boosting", "mlp_classifier"}
    assert person2_specs["hist_gradient_boosting"].scale_numeric is False
    assert person2_specs["mlp_classifier"].scale_numeric is True

    for spec in person2_specs.values():
        result = train_and_tune_model(spec, X_train, y_train, cv=cv)
        assert result.cv_f1_mean > 0.35
        probs = result.best_estimator.predict_proba(X_test.iloc[:10])
        assert probs.shape == (10, 2)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
