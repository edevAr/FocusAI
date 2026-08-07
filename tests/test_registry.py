from __future__ import annotations

import pytest

from src.training.registry import ProductionModelCache, ProductionModelUnavailableError


class FakeVersion:
    def __init__(self, version: str) -> None:
        self.version = version


class FakeClient:
    def __init__(self, versions: list[object]) -> None:
        self.versions = iter(versions)

    def get_model_version_by_alias(self, name: str, alias: str):
        assert name == "productivity_ensemble"
        assert alias == "production"
        version = next(self.versions)
        if isinstance(version, Exception):
            raise version
        return FakeVersion(str(version))


def test_missing_production_alias_never_uses_a_cached_or_local_fallback() -> None:
    loads: list[str] = []
    cache = ProductionModelCache(
        "http://mlflow:5000",
        "productivity_ensemble",
        "production",
        ttl_seconds=1,
        client_factory=lambda _: FakeClient([RuntimeError("alias missing")]),
        model_loader=lambda uri: loads.append(uri),
    )

    with pytest.raises(ProductionModelUnavailableError, match="production"):
        cache.get()

    assert loads == []


def test_alias_change_reloads_only_after_ttl_and_returns_new_version_identity() -> None:
    now = [0.0]
    loads: list[str] = []
    client = FakeClient(["1", "2"])
    cache = ProductionModelCache(
        "http://mlflow:5000",
        "productivity_ensemble",
        "production",
        ttl_seconds=10,
        client_factory=lambda _: client,
        model_loader=lambda uri: {"uri": uri},
        clock=lambda: now[0],
    )

    first = cache.get()
    cached = cache.get()
    now[0] = 10.0
    refreshed = cache.get()

    assert first.identity.version == "1"
    assert cached.identity.version == "1"
    assert refreshed.identity.version == "2"
    assert [first.model["uri"], refreshed.model["uri"]] == [
        "models:/productivity_ensemble/1",
        "models:/productivity_ensemble/2",
    ]
