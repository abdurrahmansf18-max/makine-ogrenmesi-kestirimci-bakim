"""Docker ortamindaki Python ve ML kutuphanelerini dogrulama betigi."""

import importlib
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("joblib", "joblib"),
    ("streamlit", "streamlit"),
    ("pytest", "pytest"),
]


def main() -> None:
    """Kurulu kutuphane surumlerini yazdirir ve temel bir pipeline testi calistirir."""
    print(f"Python: {sys.version.split()[0]}")
    for display_name, module_name in REQUIRED_PACKAGES:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "bilinmiyor")
        print(f"- {display_name}: {version}")

    X, y = make_classification(n_samples=120, n_features=6, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(f"Pipeline test dogrulugu: {score:.4f}")


if __name__ == "__main__":
    main()
