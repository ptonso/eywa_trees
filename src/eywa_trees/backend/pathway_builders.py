from __future__ import annotations

from typing import Any, Optional, Sequence

from eywa_trees.backend.adapters.base import PathwayAdapterResult
from eywa_trees.backend.adapters.sklearn_forest import SklearnForestAdapter
from eywa_trees.backend.adapters.sklearn_tree import SklearnTreeAdapter
from eywa_trees.backend.adapters.xgboost import XGBoostAdapter

_ADAPTERS = (
    XGBoostAdapter(),
    SklearnForestAdapter(),
    SklearnTreeAdapter(),
)


def build_pathway_inputs_from_model(
    model: Any,
    feature_names: Optional[Sequence[str]] = None,
) -> PathwayAdapterResult:
    for adapter in _ADAPTERS:
        if adapter.is_compatible(model):
            return adapter.build_pathway_inputs(model, feature_names=feature_names)
    raise ValueError("Model does not expose a tree_ attribute and is not a supported ensemble.")
