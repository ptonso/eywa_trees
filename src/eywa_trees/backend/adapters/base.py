from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import pandas as pd

from eywa_trees.backend.vistree import VisTree

ArrayLike = Union[pd.DataFrame, np.ndarray]


@runtime_checkable
class AdapterProtocol(Protocol):
    name: str

    def is_compatible(self, model: Any) -> bool: ...

    def build_pathway_inputs(
        self,
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "PathwayAdapterResult": ...

    def build_vis_trees(
        self,
        model: Any,
        X: Optional[ArrayLike] = None,
        class_names: Optional[Sequence[str]] = None,
        log_coloring: bool = False,
        colorscale: str = "Viridis",
    ) -> List[VisTree]: ...


@dataclass(frozen=True)
class PathwayAdapterResult:
    trees: List[Any]
    feature_names: Optional[List[str]]
    is_classifier: bool
    uses_scores: bool
    label_names: Optional[Sequence[Any]]
    n_trees: int


class BaseModelAdapter(ABC):
    name = "unknown"

    @abstractmethod
    def is_compatible(self, model: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_pathway_inputs(
        self,
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> PathwayAdapterResult:
        raise NotImplementedError

    @abstractmethod
    def build_vis_trees(
        self,
        model: Any,
        X: Optional[ArrayLike] = None,
        class_names: Optional[Sequence[str]] = None,
        log_coloring: bool = False,
        colorscale: str = "Viridis",
    ) -> List[VisTree]:
        raise NotImplementedError


def infer_feature_names(model: Any, X: Optional[ArrayLike]) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        try:
            names = getattr(model, "feature_names_in_")
            return names.tolist() if hasattr(names, "tolist") else list(names)
        except Exception:
            return None
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return None


def infer_class_names(model: Any, class_names: Optional[Sequence[str]]) -> Optional[List[str]]:
    if class_names is not None:
        return [str(c) for c in list(class_names)]
    if hasattr(model, "classes_"):
        try:
            return [str(c) for c in list(getattr(model, "classes_"))]
        except Exception:
            return None
    return None


def prepare_array(X: Optional[ArrayLike]) -> Optional[np.ndarray]:
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return X


def build_vis_tree_from_struct(
    model: Any,
    tree_struct: Any,
    X: Optional[ArrayLike],
    *,
    feature_names: Optional[List[str]],
    class_names: Optional[Sequence[str]],
    is_classifier: bool,
    uses_scores: bool = False,
    learning_rate: float = 1.0,
    base_score: float = 0.0,
    log_coloring: bool = False,
    colorscale: str = "Viridis",
) -> VisTree:
    feat_names_list = list(feature_names) if feature_names is not None else None
    class_names_list = [str(c) for c in list(class_names)] if class_names is not None else None

    vis_tree = VisTree(
        model=model,
        feature_names=feat_names_list,
        class_names=class_names_list,
        is_classifier=is_classifier,
        uses_scores=uses_scores,
        log_coloring=log_coloring,
        colorscale=colorscale,
        learning_rate=learning_rate,
        base_score=base_score,
    )
    vis_tree.ingest_tree_struct(
        tree_struct,
        feature_names=feat_names_list,
        class_names=class_names_list,
        is_classifier=is_classifier,
        uses_scores=uses_scores,
        learning_rate=learning_rate,
        base_score=base_score,
    )

    arr = prepare_array(X)
    if arr is not None:
        vis_tree.populate_ns(arr)
        vis_tree.propagate_values(consider_proba=True)
    return vis_tree
