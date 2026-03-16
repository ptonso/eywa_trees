from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from eywa_trees.logger import setup_logger
from eywa_trees.backend.sankey_plot import SankeyTreePlot
from eywa_trees.backend.adapters.xgboost import is_xgboost_model
from eywa_trees.backend.vis_builders import build_vis_trees_from_model


def _to_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.copy()
    arr = np.asarray(values)
    return pd.Series(arr)


def _map_class_indices_to_labels(preds: Any, class_names: Optional[List[str]]) -> np.ndarray:
    arr = np.asarray(preds)
    if class_names is None or not class_names or arr.ndim != 1:
        return arr

    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        idx = arr.astype(int)
        labels = np.asarray(class_names, dtype=object)
        out = np.empty(idx.shape[0], dtype=object)
        valid = (idx >= 0) & (idx < len(labels))
        out[valid] = labels[idx[valid]]
        out[~valid] = idx[~valid].astype(str)
        return out

    return arr


def _safe_accuracy(y_true: Any, y_pred: Any) -> float:
    y_true_s = _to_series(y_true)
    y_pred_s = _to_series(y_pred)
    mask = (~y_true_s.isna()) & (~y_pred_s.isna())
    if not bool(mask.any()):
        return float("nan")
    return float(accuracy_score(y_true_s[mask].astype(str), y_pred_s[mask].astype(str)))


class SingleTreeTab:
    """Tab that renders a single tree with a Sankey plot and metric."""

    def __init__(
        self,
        model: Any,
        X_train: Any,
        X_val: Any,
        y_val: Any,
        class_names: Optional[List[str]] = None,
        show_text: bool = False,
    ) -> None:
        self.logger = setup_logger("api.log")
        trees = build_vis_trees_from_model(model, X_train, class_names=class_names)
        if not trees:
            raise ValueError("No trees available for SingleTreeTab.")
        self.vis_tree = trees[0]
        self.is_classifier = self.vis_tree.is_classifier
        self.X_val = X_val
        self.y_val = y_val
        self.show_text = show_text
        self.max_depth = self.vis_tree.max_depth
        self.class_names = (
            [str(c) for c in class_names]
            if class_names is not None
            else ([str(c) for c in self.vis_tree.class_names] if self.vis_tree.class_names else None)
        )
        self.initial_sankey = SankeyTreePlot(self.vis_tree, show_text=show_text)

    @property
    def layout(self) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(id="st-metric", style={"fontSize": 16, "fontWeight": "bold", "marginBottom": "10px"}),
                        html.Label("Depth", style={"fontWeight": "bold"}),
                        dcc.Slider(
                            id="st-depth",
                            min=1,
                            max=self.max_depth,
                            value=self.max_depth,
                            marks={str(i): str(i) for i in range(1, self.max_depth + 1)},
                            step=None,
                            tooltip={"always_visible": True},
                            vertical=True,
                            verticalHeight=320,
                        ),
                    ],
                    style={
                        "width": "12%",
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "gap": "12px",
                        "paddingTop": "40px",
                    },
                ),
                html.Div(
                    dcc.Graph(
                        id="st-sankey",
                        figure=self.initial_sankey.fig,
                        style={"height": "70vh", "minHeight": "360px", "width": "100%"},
                    ),
                    style={"flex": "1", "padding": "20px"},
                ),
            ],
            style={"display": "flex", "flexDirection": "row"},
            id="single-tree-tab",
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            [Output("st-sankey", "figure"), Output("st-metric", "children")],
            Input("st-depth", "value"),
        )
        def update_sankey(max_depth: int) -> Tuple[Any, str]:
            pruned = self.vis_tree.prune(max_depth)
            sankey_obj = SankeyTreePlot(pruned, show_text=(max_depth <= 3 or self.show_text))
            preds = pruned.predict(self.X_val)
            if self.is_classifier:
                pred_labels = _map_class_indices_to_labels(preds, self.class_names)
                metric = _safe_accuracy(self.y_val, pred_labels)
                text = f"Accuracy: {metric:.2f}" if np.isfinite(metric) else "Accuracy: n/a"
            else:
                metric = mean_absolute_error(self.y_val, preds)
                text = f"MAE: {metric:.2f}"
            return sankey_obj.fig, text


class RandomForestTab:
    """Tab that renders per-tree Sankey plots for a random forest."""

    def __init__(
        self,
        model: Any,
        X_train: Any,
        X_val: Any,
        y_val: Any,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.rf = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_val = y_val

        self.vis_trees = build_vis_trees_from_model(model, X_train, class_names=class_names)
        if not self.vis_trees:
            raise ValueError("No trees available for RandomForestTab.")
        self.is_xgboost = is_xgboost_model(model)
        self.is_classifier = self.vis_trees[0].is_classifier
        self.xgb_group_size = int(getattr(self.vis_trees[0], "xgb_group_size", 1)) if self.is_xgboost else 1
        self.xgb_num_rounds = int(getattr(self.vis_trees[0], "xgb_num_rounds", len(self.vis_trees))) if self.is_xgboost else len(self.vis_trees)
        self.class_names = (
            [str(c) for c in class_names]
            if class_names is not None
            else ([str(c) for c in self.vis_trees[0].class_names] if self.vis_trees[0].class_names else None)
        )

        self.initial_tree_id = 0
        self.initial_tree = self.vis_trees[self.initial_tree_id]
        self.initial_sankey = SankeyTreePlot(self.initial_tree)
        self.initial_max_depth = self.initial_tree.max_depth

    def _tree_info_text(self, tree_id: int) -> str:
        idx = int(np.clip(tree_id, 0, len(self.vis_trees) - 1))
        if not self.is_xgboost:
            return f"Tree {idx + 1}/{len(self.vis_trees)}"

        tree = self.vis_trees[idx]
        round_idx = int(getattr(tree, "xgb_round_index", idx))
        group_size = int(getattr(tree, "xgb_group_size", 1))
        if group_size > 1:
            class_idx = int(getattr(tree, "xgb_class_index", 0))
            class_label = getattr(tree, "xgb_class_label", None)
            class_txt = f"class {class_label}" if class_label is not None else f"class {class_idx + 1}"
            return (
                f"Booster tree {idx + 1}/{len(self.vis_trees)} | "
                f"round {round_idx + 1}/{self.xgb_num_rounds} | "
                f"{class_txt}"
            )
        return (
            f"Booster tree {idx + 1}/{len(self.vis_trees)} | "
            f"round {round_idx + 1}/{self.xgb_num_rounds}"
        )

    def _metric_text(self, tree_id: int, max_depth: int) -> str:
        if self.is_xgboost:
            if self.xgb_group_size > 1:
                return "Multiclass XGBoost uses one tree per class per boosting round."
            return "XGBoost trees are additive score contributors, not standalone estimators."

        selected = self.vis_trees[tree_id]
        pruned = selected.prune(max_depth)
        y_pred = pruned.predict(self.X_val)
        if self.is_classifier:
            pred_labels = _map_class_indices_to_labels(y_pred, self.class_names)
            metric_val = _safe_accuracy(self.y_val, pred_labels)
            return f"Accuracy: {metric_val:.2f}" if np.isfinite(metric_val) else "Accuracy: n/a"
        metric_val = mean_absolute_error(self.y_val, y_pred)
        return f"MAE: {metric_val:.2f}"

    def _tree_marks(self, max_marks: int = 15) -> Dict[str, str]:
        n = len(self.vis_trees)
        if self.is_xgboost and self.xgb_group_size > 1:
            rounds = self.xgb_num_rounds
            if rounds <= max_marks:
                round_idxs = list(range(rounds))
            else:
                step = max(1, int(round((rounds - 1) / float(max_marks - 1))))
                round_idxs = list(range(0, rounds, step))
                if round_idxs[-1] != rounds - 1:
                    round_idxs.append(rounds - 1)
            return {str(r * self.xgb_group_size): f"r{r}" for r in round_idxs}

        if n <= max_marks:
            idxs = list(range(n))
        else:
            step = max(1, int(round((n - 1) / float(max_marks - 1))))
            idxs = list(range(0, n, step))
            if idxs[-1] != n - 1:
                idxs.append(n - 1)
        return {str(i): str(i) for i in idxs}

    @property
    def layout(self) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            self._tree_info_text(self.initial_tree_id),
                            id="rf-tree-info",
                            style={"fontSize": 14, "marginBottom": "10px", "color": "#374151"},
                        ),
                        html.Div(
                            self._metric_text(self.initial_tree_id, self.initial_max_depth),
                            id="rf-metric",
                            style={"fontSize": 16, "fontWeight": "bold", "marginBottom": "10px"},
                        ),
                        html.Label("Depth", style={"fontWeight": "bold"}),
                        dcc.Slider(
                            id="rf-depth",
                            min=1,
                            max=self.initial_max_depth,
                            value=self.initial_max_depth,
                            marks={str(i): str(i) for i in range(1, self.initial_max_depth + 1)},
                            step=None,
                            tooltip={"always_visible": True},
                            vertical=True,
                            verticalHeight=320,
                        ),
                    ],
                    style={
                        "width": "12%",
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "gap": "12px",
                        "paddingTop": "40px",
                    },
                ),
                html.Div(
                    [
                        dcc.Slider(
                            id="rf-tree-id",
                            min=0,
                            max=len(self.vis_trees) - 1,
                            value=self.initial_tree_id,
                            marks=self._tree_marks(),
                            step=1,
                            tooltip={
                                "always_visible": not (self.is_xgboost and self.xgb_group_size > 1),
                                "placement": "bottom",
                            },
                        ),
                        dcc.Graph(
                            id="rf-sankey",
                            figure=self.initial_sankey.fig,
                            style={"height": "70vh", "minHeight": "360px", "width": "100%", "marginTop": "20px"},
                        ),
                    ],
                    style={"flex": "1", "padding": "20px"},
                ),
            ],
            style={"display": "flex", "flexDirection": "row"},
            id="random-forest-tab",
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            [Output("rf-sankey", "figure"), Output("rf-tree-info", "children"), Output("rf-metric", "children")],
            [Input("rf-depth", "value"), Input("rf-tree-id", "value")],
        )
        def update_rf(max_depth: int, tree_id: int) -> Tuple[Any, str, str]:
            selected = self.vis_trees[tree_id]
            pruned = selected.prune(max_depth)
            sankey_obj = SankeyTreePlot(pruned, show_text=max_depth <= 3)
            return sankey_obj.fig, self._tree_info_text(tree_id), self._metric_text(tree_id, max_depth)

        @app.callback(
            [Output("rf-depth", "max"), Output("rf-depth", "marks"), Output("rf-depth", "value")],
            Input("rf-tree-id", "value"),
        )
        def update_depth_slider(tree_id: int) -> Tuple[int, Dict[str, str], int]:
            selected = self.vis_trees[tree_id]
            max_depth = selected.max_depth
            marks = {str(i): str(i) for i in range(1, max_depth + 1)}
            return max_depth, marks, max_depth


def build_xgboost_tab(*args: Any, **kwargs: Any) -> None:
    """Placeholder for XGBoost/GBDT model tab."""
    raise NotImplementedError("XGBoost tab not implemented yet.")
