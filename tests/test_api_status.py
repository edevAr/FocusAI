from __future__ import annotations

from types import SimpleNamespace

from api import main
from src.training.registry import LoadedProductionModel, ModelIdentity


class FakeClient:
    def __init__(self, _uri: str) -> None:
        pass

    def get_model_version(self, name: str, version: str):
        assert (name, version) == ("productivity_ensemble", "7")
        return SimpleNamespace(run_id="run-7")

    def get_run(self, run_id: str):
        assert run_id == "run-7"
        return SimpleNamespace(
            data=SimpleNamespace(
                tags={
                    "quality.eligible": "false",
                    "quality.accuracy": "0.70",
                    "quality.min_accuracy": "0.75",
                    "quality.f1": "0.80",
                    "quality.min_f1": "0.75",
                    "quality.warnings": "Accuracy 0.7000 is below the required 0.7500.",
                }
            )
        )


def test_status_reports_observed_production_identity_and_advisory_warning(monkeypatch) -> None:
    loaded = LoadedProductionModel(
        model=object(),
        identity=ModelIdentity("productivity_ensemble", "production", "7"),
    )
    monkeypatch.setattr(main, "get_production_model", lambda: loaded)
    monkeypatch.setattr(main, "MlflowClient", FakeClient)
    monkeypatch.setattr(main, "_readiness_components", lambda: {})

    result = main.mlops_status()

    assert result["readiness"]["status"] == "ready"
    assert result["production"] == loaded.identity.as_dict()
    assert result["quality"]["eligible"] is False
    assert result["quality"]["warnings"] == [
        "Accuracy 0.7000 is below the required 0.7500."
    ]
    assert result["checklist"]["quality_gates"] is False
    assert "native MLflow" in result["authority"]


def test_status_is_degraded_when_production_cannot_be_resolved(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_production_model",
        lambda: (_ for _ in ()).throw(main.ProductionModelUnavailableError("alias missing")),
    )
    monkeypatch.setattr(main, "_readiness_components", lambda: {"model": "alias missing"})

    result = main.mlops_status()

    assert result["readiness"] == {"status": "degraded", "causes": {"model": "alias missing"}}
    assert result["production"] is None
    assert result["checklist"]["production_alias"] is False
