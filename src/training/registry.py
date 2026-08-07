"""Production-alias resolution and version-aware MLflow model caching."""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Callable

import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from config.settings import (
    MLFLOW_MODEL_NAME,
    MLFLOW_PRODUCTION_ALIAS,
    MLFLOW_TRACKING_URI,
    MODEL_CACHE_TTL_SECONDS,
)


class ProductionModelUnavailableError(RuntimeError):
    """Raised when the configured Production model cannot be resolved or loaded."""


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    alias: str
    version: str

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "alias": self.alias, "version": self.version}


@dataclass(frozen=True)
class LoadedProductionModel:
    model: Any
    identity: ModelIdentity


class ProductionModelCache:
    """Resolve the native alias at a TTL and reload only when its version changes."""

    def __init__(
        self,
        tracking_uri: str | None,
        model_name: str,
        alias: str,
        ttl_seconds: int,
        client_factory: Callable[[str], Any] = MlflowClient,
        model_loader: Callable[[str], Any] = mlflow.pyfunc.load_model,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.alias = alias
        self.ttl_seconds = max(0, ttl_seconds)
        self.client_factory = client_factory
        self.model_loader = model_loader
        self.clock = clock
        self._loaded: LoadedProductionModel | None = None
        self._next_resolution = 0.0
        self._lock = RLock()

    def _resolve_identity(self) -> ModelIdentity:
        if not self.tracking_uri:
            raise ProductionModelUnavailableError("MLFLOW_TRACKING_URI is not configured.")
        try:
            version = self.client_factory(self.tracking_uri).get_model_version_by_alias(
                self.model_name, self.alias
            )
        except Exception as exc:  # MLflow wraps missing aliases and remote errors differently.
            raise ProductionModelUnavailableError(
                f"Production alias '{self.alias}' for model '{self.model_name}' is unavailable: {exc}"
            ) from exc
        return ModelIdentity(self.model_name, self.alias, str(version.version))

    def get(self) -> LoadedProductionModel:
        with self._lock:
            if self._loaded is not None and self.clock() < self._next_resolution:
                return self._loaded

            identity = self._resolve_identity()
            self._next_resolution = self.clock() + self.ttl_seconds
            if self._loaded is not None and self._loaded.identity.version == identity.version:
                return self._loaded
            try:
                model = self.model_loader(identity.uri)
            except Exception as exc:  # A resolved alias is not ready until its bundle loads.
                self._loaded = None
                raise ProductionModelUnavailableError(
                    f"Production model '{identity.uri}' could not be loaded: {exc}"
                ) from exc
            self._loaded = LoadedProductionModel(model=model, identity=identity)
            return self._loaded


_production_cache = ProductionModelCache(
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    MLFLOW_PRODUCTION_ALIAS,
    MODEL_CACHE_TTL_SECONDS,
)


def get_production_model() -> LoadedProductionModel:
    """Return only the configured Production bundle; never fall back to local files."""
    return _production_cache.get()
