from __future__ import annotations

import pandas as pd
import pytest

from src.training import predict
from src.training.registry import (
    LoadedProductionModel,
    ModelIdentity,
    ProductionModelUnavailableError,
)


class FakeBundle:
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert frame.columns.tolist() == ["texto"]
        return pd.DataFrame(
            {
                "prediccion": ["Productivo"],
                "label_id": [1],
                "probabilidad": [0.95],
                "texto_limpio": ["terminé tarea"],
            }
        )


def _loaded_bundle() -> LoadedProductionModel:
    return LoadedProductionModel(
        model=FakeBundle(),
        identity=ModelIdentity("productivity_ensemble", "production", "7"),
    )


def test_prediction_uses_pyfunc_bundle_and_includes_model_identity(monkeypatch) -> None:
    monkeypatch.setattr(predict, "get_production_model", _loaded_bundle)

    result = predict.predict_one("Terminé mi tarea")

    assert result["prediccion"] == "Productivo"
    assert result["model"] == {
        "name": "productivity_ensemble",
        "alias": "production",
        "version": "7",
    }


def test_blank_text_is_rejected_before_model_loading(monkeypatch) -> None:
    monkeypatch.setattr(
        predict,
        "get_production_model",
        lambda: pytest.fail("the model must not load for invalid input"),
    )

    with pytest.raises(ValueError, match="no vacío"):
        predict.predict_one("   ")


def test_unloadable_production_bundle_has_no_local_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        predict,
        "get_production_model",
        lambda: (_ for _ in ()).throw(ProductionModelUnavailableError("unloadable")),
    )

    with pytest.raises(ProductionModelUnavailableError, match="unloadable"):
        predict.predict_one("Terminé mi tarea")
