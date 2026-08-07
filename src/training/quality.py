"""Advisory quality gates for MLflow candidates."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class QualityGate:
    accuracy: float | None
    f1: float | None
    min_accuracy: float
    min_f1: float
    eligible: bool
    warnings: tuple[str, ...]

    def tags(self) -> dict[str, str]:
        return {
            "quality.eligible": str(self.eligible).lower(),
            "quality.min_accuracy": str(self.min_accuracy),
            "quality.min_f1": str(self.min_f1),
            "quality.accuracy": "missing" if self.accuracy is None else str(self.accuracy),
            "quality.f1": "missing" if self.f1 is None else str(self.f1),
            "quality.warnings": " | ".join(self.warnings) or "none",
        }


def evaluate_quality(
    metrics: dict[str, float], min_accuracy: float, min_f1: float
) -> QualityGate:
    """Evaluate inclusive Accuracy/F1 thresholds without controlling MLflow aliases."""
    accuracy = metrics.get("Accuracy")
    f1 = metrics.get("F1")
    warnings: list[str] = []
    if not isinstance(accuracy, (int, float)) or not isfinite(accuracy):
        accuracy = None
        warnings.append("Accuracy evidence is missing or non-finite.")
    elif accuracy < min_accuracy:
        warnings.append(f"Accuracy {accuracy:.4f} is below the required {min_accuracy:.4f}.")
    if not isinstance(f1, (int, float)) or not isfinite(f1):
        f1 = None
        warnings.append("F1 evidence is missing or non-finite.")
    elif f1 < min_f1:
        warnings.append(f"F1 {f1:.4f} is below the required {min_f1:.4f}.")
    return QualityGate(
        accuracy=accuracy,
        f1=f1,
        min_accuracy=min_accuracy,
        min_f1=min_f1,
        eligible=not warnings,
        warnings=tuple(warnings),
    )
