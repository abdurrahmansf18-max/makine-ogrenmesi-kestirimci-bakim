"""CRISP-DM Asama 5: Test seti degerlendirmesi, grafikler ve F1 sampiyon secimi."""

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import PLOTS_DIR
from src.trainer import TrainedModelResult


def evaluate_model_on_test(
    result: TrainedModelResult,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Egitilmis bir Pipeline'i kilitli test kumesi uzerinde tum siniflandirma olcutleriyle olcer."""
    y_pred = result.best_estimator.predict(X_test)
    y_prob = result.best_estimator.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel())
    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    loss = float(log_loss(y_test, y_prob))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)

    return {
        "key": result.spec.key,
        "display_name": result.spec.display_name,
        "owner": result.spec.owner,
        "family": result.spec.family,
        "scale_numeric": result.spec.scale_numeric,
        "best_params": result.best_params,
        "fit_seconds": result.fit_seconds,
        "cv_train_f1_mean": round(result.cv_train_f1_mean, 4),
        "cv_f1_mean": round(result.cv_f1_mean, 4),
        "cv_f1_std": round(result.cv_f1_std, 4),
        "test_metrics": {
            "f1": round(f1, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(acc, 4),
            "error_rate": round(1.0 - acc, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "log_loss": round(loss, 4),
        },
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "_curves": {
            "fpr": fpr,
            "tpr": tpr,
            "pr_rec": pr_rec,
            "pr_prec": pr_prec,
        },
    }


def select_champion(model_evals: list[dict[str, Any]]) -> str:
    """Test setine bakmadan 5-katli CV F1-Score ortalamasina gore sampiyonu secer."""
    candidates = [m for m in model_evals if m["key"] != "dummy_baseline"]
    if not candidates:
        raise ValueError("Sampiyon secimi icin aday model bulunamadi.")
    champion = max(
        candidates,
        key=lambda m: (m["cv_f1_mean"], -m["cv_f1_std"]),
    )
    return str(champion["key"])


def _plot_metric_comparison(
    candidates: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    """4 modelin temel test metriklerini gruplu cubuk grafik olarak kaydeder."""
    metric_keys = ["f1", "recall", "precision", "roc_auc", "pr_auc"]
    metric_labels = ["F1-Score", "Recall", "Precision", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(candidates))
    width = 0.15
    palette = ["#0284c7", "#059669", "#d97706", "#7c3aed", "#dc2626"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for idx, (m_key, m_label, color) in enumerate(zip(metric_keys, metric_labels, palette)):
        values = [m["test_metrics"][m_key] for m in candidates]
        offset = (idx - 2) * width
        bars = ax.bar(x + offset, values, width=width, label=m_label, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90 if val > 0 else 0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m["display_name"] for m in candidates], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Skor (0 - 1)")
    ax.set_title("Test Kümesi Üzerinde 4 Modelin Performans Karşılaştırması", fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(plots_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrices(
    candidates: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    """4 model icin 2x2 karisiklik matrisi gridini kaydeder."""
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.2))
    for ax, m in zip(axes.ravel(), candidates):
        cm = m["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        labels = np.array(
            [
                [f"DN (TN)\n{cm['tn']}", f"YP (FP)\n{cm['fp']}"],
                [f"YN (FN)\n{cm['fn']}", f"DP (TP)\n{cm['tp']}"],
            ]
        )
        sns.heatmap(
            matrix,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Tahmin: Normal (0)", "Tahmin: Arıza (1)"],
            yticklabels=["Gerçek: Normal (0)", "Gerçek: Arıza (1)"],
        )
        ax.set_title(
            f"{m['display_name']} (F1={m['test_metrics']['f1']:.3f})",
            fontsize=9.5,
            fontweight="bold",
        )
    fig.suptitle("Kilitli Test Kümesi Karışıklık Matrisleri", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def _plot_roc_pr_curves(
    candidates: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    """4 model icin erisilebilir renk + cizgi stili ikilisiyle ROC ve PR egrilerini kaydeder."""
    styles = [
        ("#0284c7", "-"),
        ("#d97706", "--"),
        ("#059669", "-."),
        ("#7c3aed", ":"),
    ]
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for m, (color, ls) in zip(candidates, styles):
        c = m["_curves"]
        ax_roc.plot(
            c["fpr"],
            c["tpr"],
            color=color,
            linestyle=ls,
            linewidth=2.0,
            label=f"{m['display_name']} (AUC={m['test_metrics']['roc_auc']:.3f})",
        )
        ax_pr.plot(
            c["pr_rec"],
            c["pr_prec"],
            color=color,
            linestyle=ls,
            linewidth=2.0,
            label=f"{m['display_name']} (AP={m['test_metrics']['pr_auc']:.3f})",
        )

    ax_roc.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.0, label="Rastgele (0.500)")
    ax_roc.set_title("ROC Eğrisi", fontsize=10.5, fontweight="bold")
    ax_roc.set_xlabel("Yanlış Pozitif Oranı (FPR)")
    ax_roc.set_ylabel("Doğru Pozitif Oranı (TPR / Recall)")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(alpha=0.3)

    ax_pr.set_title("Precision-Recall Eğrisi", fontsize=10.5, fontweight="bold")
    ax_pr.set_xlabel("Duyarlılık (Recall)")
    ax_pr.set_ylabel("Kesinlik (Precision)")
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def save_evaluation_plots(
    model_evals: list[dict[str, Any]],
    plots_dir: Path = PLOTS_DIR,
) -> None:
    """4 denetimli model icin karsilastirma, karisiklik matrisi ve ROC/PR grafiklerini uretir."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    candidates = [m for m in model_evals if m["key"] != "dummy_baseline"]
    _plot_metric_comparison(candidates, plots_dir)
    _plot_confusion_matrices(candidates, plots_dir)
    _plot_roc_pr_curves(candidates, plots_dir)
