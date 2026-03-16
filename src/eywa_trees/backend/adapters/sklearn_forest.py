from __future__ import annotations

from typing import Any, List, Optional, Sequence

from sklearn.base import ClassifierMixin

from eywa_trees.backend.adapters.base import (
    ArrayLike,
    BaseModelAdapter,
    PathwayAdapterResult,
    build_vis_tree_from_struct,
    infer_class_names,
    infer_feature_names,
)
from eywa_trees.backend.vistree import VisTree


class SklearnForestAdapter(BaseModelAdapter):
    name = "forest"

    def is_compatible(self, model: Any) -> bool:
        return hasattr(model, "estimators_")

    def build_pathway_inputs(
        self,
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> PathwayAdapterResult:
        trees = []
        for est in getattr(model, "estimators_", []):
            tree_struct = getattr(est, "tree_", None)
            if tree_struct is not None:
                trees.append(tree_struct)
        is_clf = isinstance(model, ClassifierMixin)
        label_names = getattr(model, "classes_", None) if is_clf else None
        if feature_names:
            feat_names = list(feature_names)
        else:
            feat_names = infer_feature_names(model, None)
        return PathwayAdapterResult(
            trees=trees,
            feature_names=feat_names,
            is_classifier=is_clf,
            uses_scores=False,
            label_names=label_names,
            n_trees=len(trees),
        )

    def build_vis_trees(
        self,
        model: Any,
        X: Optional[ArrayLike] = None,
        class_names: Optional[Sequence[str]] = None,
        log_coloring: bool = False,
    ) -> List[VisTree]:
        feat_names = infer_feature_names(model, X)
        cls_names = infer_class_names(model, class_names)
        vis_trees: List[VisTree] = []

        for est in getattr(model, "estimators_", []):
            tree_struct = getattr(est, "tree_", None)
            if tree_struct is None:
                continue
            is_clf = isinstance(est, ClassifierMixin)
            est_class_names = infer_class_names(est, class_names if cls_names is None else cls_names)
            vis_trees.append(
                build_vis_tree_from_struct(
                    est,
                    tree_struct,
                    X,
                    feature_names=feat_names,
                    class_names=est_class_names,
                    is_classifier=is_clf,
                    uses_scores=False,
                    log_coloring=log_coloring,
                )
            )
        return vis_trees
