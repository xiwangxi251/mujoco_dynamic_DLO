"""Helpers for deprecated top-level module compatibility wrappers."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def reexport(module_name: str, namespace: dict[str, Any]) -> ModuleType:
    module = import_module(module_name)
    namespace.update({
        name: value
        for name, value in vars(module).items()
        if not name.startswith("__")
    })
    return module

