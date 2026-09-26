"""Component registry: named factories for readers, transforms and sinks."""

from __future__ import annotations

from typing import Any, Callable

from pipeline.errors import ConfigurationError

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}


def register(kind: str, name: str, factory: Callable[..., Any]) -> None:
    """Register a factory under a kind (reader/transform/sink)."""
    if kind not in {"reader", "transform", "sink"}:
        raise ConfigurationError(f"unknown kind: {kind}")
    bucket = _REGISTRY.setdefault(kind, {})
    if name in bucket:
        raise ConfigurationError(f"{kind} {name!r} already registered")
    bucket[name] = factory


def create(kind: str, name: str, **kwargs) -> Any:
    """Instantiate a registered component."""
    factory = _REGISTRY.get(kind, {}).get(name)
    if factory is None:
        raise ConfigurationError(f"no {kind} named {name!r}")
    return factory(**kwargs)


def list_names(kind: str) -> list[str]:
    """Registered component names for a kind, sorted."""
    return sorted(_REGISTRY.get(kind, {}))


def reset_registry() -> None:
    """Test helper: drop all registrations."""
    _REGISTRY.clear()
