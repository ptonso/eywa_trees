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
    colorscale: str = "Viridis",
) -> List[VisTree]:
    for adapter in _ADAPTERS:
        if adapter.is_compatible(model):
            return adapter.build_vis_trees(
                model,
                X=X,
                class_names=class_names,
                log_coloring=log_coloring,
                colorscale=colorscale,
            )
    raise ValueError("Model does not expose a tree_ attribute and is not a supported ensemble.")


def make_tree_figure(
    vis_tree: VisTree,
    kind: str = "sankey",
    show_text: bool = True,
    highlight_path: Optional[Sequence[int]] = None,
    sankey_dim_alpha: float = 0.7,
) -> Any:
    """
    Render a VisTree to a Plotly figure with the chosen renderer.

    `kind` is "sankey" (SankeyTreePlot) or "go" (GoTreePlot). Both support
    `highlight_path` for emphasizing a single root-to-leaf path.
    """
    from eywa_trees.backend.go_plot import GoTreePlot
    from eywa_trees.backend.sankey_plot import SankeyTreePlot

    if kind == "go":
        return GoTreePlot(vis_tree, show_text=show_text, highlight_path=highlight_path).fig
    return SankeyTreePlot(
        vis_tree,
        show_text=show_text,
        highlight_path=highlight_path,
        dim_alpha=sankey_dim_alpha,
    ).fig
