from __future__ import annotations

from typing import Any, List, Optional, Sequence

from eywa_trees.backend.adapters.base import ArrayLike, build_vis_tree_from_struct as _build_vis_tree_from_struct
from eywa_trees.backend.adapters.sklearn_forest import SklearnForestAdapter
from eywa_trees.backend.adapters.sklearn_tree import SklearnTreeAdapter
from eywa_trees.backend.adapters.xgboost import XGBoostAdapter
from eywa_trees.backend.vistree import VisTree

build_vis_tree_from_struct = _build_vis_tree_from_struct

_ADAPTERS = (
    XGBoostAdapter(),
    SklearnForestAdapter(),
    SklearnTreeAdapter(),
)


def build_vis_trees_from_model(
    model: Any,
    X: Optional[ArrayLike] = None,
    class_names: Optional[Sequence[str]] = None,
    log_coloring: bool = False,
) -> List[VisTree]:
    for adapter in _ADAPTERS:
        if adapter.is_compatible(model):
            return adapter.build_vis_trees(
                model,
                X=X,
                class_names=class_names,
                log_coloring=log_coloring,
            )
    raise ValueError("Model does not expose a tree_ attribute and is not a supported ensemble.")
