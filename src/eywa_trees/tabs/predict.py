from typing import Any, Dict, List, Optional, Sequence

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, ALL
import numpy as np
import pandas as pd

from eywa_trees.logger import setup_logger
from eywa_trees.backend.vis_builders import build_vis_trees_from_model, make_tree_figure
from eywa_trees.backend.adapters.xgboost import is_xgboost_model

from eywa_trees.backend.feature_bins import FeatureBinManager

class PredictTab:
    """
    Tab that lets users synthesize a single sample via feature sliders and
    run a forward pass through the original model. Includes a Go-based plot
    for the selected tree (if a forest).
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        class_names: Optional[List[str]] = None,
        show_text: bool = True,
        feature_mgr: Optional[FeatureBinManager] = None,
        tree_plot_kind: str = "go",
        colorscale: str = "Viridis",
        plot_height: str = "62vh",
        sankey_dim_alpha: float = 0.7,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.model = model
        self.X_train = X_train
        self.class_names = class_names
        self.show_text = show_text
        self.tree_plot_kind = tree_plot_kind
        self.plot_height = plot_height
        self.sankey_dim_alpha = float(sankey_dim_alpha)
        self.model_use_df = hasattr(model, "feature_names_in_")
        self.is_xgboost = is_xgboost_model(model)

        self.feature_mgr = feature_mgr or FeatureBinManager(X_train)
        self.feature_bins = self.feature_mgr.feature_bins
        self.active_feature_names = self.feature_mgr.active_feature_names
        self.all_feature_names = self.feature_mgr.all_feature_names
        self.default_indices = self.feature_mgr.default_indices
        self.vis_trees = build_vis_trees_from_model(model, X_train, class_names=class_names, colorscale=colorscale)
        if not self.vis_trees:
            raise ValueError("No trees available for PredictTab.")
        self.xgb_lr = getattr(self.vis_trees[0], "learning_rate", 1.0) if self.is_xgboost else 1.0
        self.xgb_base_score = getattr(self.vis_trees[0], "base_score", 0.0) if self.is_xgboost else 0.0
        self.xgb_group_size = int(getattr(self.vis_trees[0], "xgb_group_size", 1)) if self.is_xgboost else 1
        self.xgb_num_rounds = int(getattr(self.vis_trees[0], "xgb_num_rounds", len(self.vis_trees))) if self.is_xgboost else len(self.vis_trees)

        self.is_classifier = self.vis_trees[0].is_classifier
        self.tree_use_df = hasattr(self.vis_trees[0].model, "feature_names_in_")
        self._go_cache: Dict[int, Any] = {}

        self.initial_tree_id = 0
        default_sample = self.feature_mgr.sample_from_indices(
            [self.default_indices[f] for f in self.active_feature_names],
            self.active_feature_names,
        )
        initial_path = self._leaf_path_for_sample(
            default_sample, self.initial_tree_id
        )
        self.initial_fig = self._get_go_fig(
            self.initial_tree_id, highlight_path=initial_path
        )
        self.initial_prediction = self._predict_sample(
            default_sample, self.initial_tree_id
        )

    def _tree_info_text(self, tree_id: int) -> str:
        tid = int(np.clip(tree_id, 0, len(self.vis_trees) - 1))
        if not self.is_xgboost:
            return f"Tree {tid + 1}/{len(self.vis_trees)}"

        tree = self.vis_trees[tid]
        round_idx = int(getattr(tree, "xgb_round_index", tid))
        group_size = int(getattr(tree, "xgb_group_size", 1))
        if group_size > 1:
            class_idx = int(getattr(tree, "xgb_class_index", 0))
            class_label = getattr(tree, "xgb_class_label", None)
            class_txt = f"class {class_label}" if class_label is not None else f"class {class_idx + 1}"
            return (
                f"Booster tree {tid + 1}/{len(self.vis_trees)} | "
                f"round {round_idx + 1}/{self.xgb_num_rounds} | "
                f"{class_txt}"
            )
        return (
            f"Booster tree {tid + 1}/{len(self.vis_trees)} | "
            f"round {round_idx + 1}/{self.xgb_num_rounds}"
        )

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
        sliders: List[html.Div] = []
        for feat in self.active_feature_names:
            bins = self.feature_bins.get(feat, [None])
            marks = self.feature_mgr.marks_for_feature(feat, bins)
            default_display = self.feature_mgr.display_for_index(
                feat, self.default_indices.get(feat, 0)
            )

            slider = dcc.Slider(
                id={"type": "predict-feature", "feature": feat},
                min=0,
                max=max(0, len(bins) - 1),
                step=1,
                value=self.default_indices.get(feat, 0),
                marks=marks,
                updatemode="mouseup",
            )
            sliders.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    feat,
                                    style={
                                        "fontWeight": "bold",
                                        "marginRight": "8px",
                                    },
                                ),
                                html.Span(
                                    default_display,
                                    id={
                                        "type": "predict-feature-display",
                                        "feature": feat,
                                    },
                                    style={
                                        "fontSize": "13px",
                                        "color": "#1f2937",
                                        "backgroundColor": "#e5e7eb",
                                        "padding": "2px 6px",
                                        "borderRadius": "4px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                            },
                        ),
                        slider,
                    ],
                    style={"paddingBottom": "12px"},
                )
            )

        tree_marks = self._tree_marks()

        tree_slider = dcc.Slider(
            id="predict-tree-id",
            min=0,
            max=max(0, len(self.vis_trees) - 1),
            value=self.initial_tree_id,
            step=1,
            marks=tree_marks,
            tooltip={
                "always_visible": not (self.is_xgboost and self.xgb_group_size > 1),
                "placement": "bottom",
            },
            disabled=len(self.vis_trees) <= 1,
        )

        return html.Div(
            [
                html.Div(
                    [
                        html.Button(
                            "Return to median",
                            id="predict-reset",
                            n_clicks=0,
                            style={"marginBottom": "12px"},
                        ),
                        html.Div(
                            sliders,
                            style={
                                "overflowY": "auto",
                                "maxHeight": "70vh",
                                "paddingRight": "8px",
                            },
                        ),
                    ],
                    style={
                        "width": "28%",
                        "display": "flex",
                        "flexDirection": "column",
                        "padding": "16px",
                        "borderRight": "1px solid #e5e7eb",
                        "backgroundColor": "#fafafa",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            tree_slider,
                            style={"paddingBottom": "8px"},
                        ),
                        html.Div(
                            self._tree_info_text(self.initial_tree_id),
                            id="predict-tree-info",
                            style={"fontSize": "14px", "marginBottom": "8px", "color": "#374151"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="predict-go-plot",
                                    figure=self.initial_fig,
                                    style={
                                        "height": self.plot_height,
                                        "minHeight": "320px",
                                        "width": "100%",
                                    },
                                ),
                                html.Div(
                                    id="predict-output",
                                    children=self.initial_prediction,
                                    style={
                                        "position": "absolute",
                                        "top": "12px",
                                        "left": "12px",
                                        "padding": "10px 12px",
                                        "fontSize": "20px",
                                        "fontWeight": "bold",
                                        "border": "1px solid #d1d5db",
                                        "borderRadius": "6px",
                                        "backgroundColor": "rgba(243, 244, 246, 0.92)",
                                        "backdropFilter": "blur(2px)",
                                        "whiteSpace": "pre-line",
                                    },
                                ),
                            ],
                            style={"position": "relative", "marginTop": "4px"},
                        ),
                    ],
                    style={"flex": "1", "padding": "16px 20px"},
                ),
            ],
            style={"display": "flex", "flexDirection": "row"},
            id="predict-tab",
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            Output({"type": "predict-feature", "feature": ALL}, "value"),
            Input("predict-reset", "n_clicks"),
            State({"type": "predict-feature", "feature": ALL}, "id"),
        )
        def reset_sliders(
            n_clicks: int,
            slider_ids: List[Dict[str, str]],
        ) -> List[int]:
            return [
                self.default_indices.get(s["feature"], 0) for s in slider_ids
            ]

        @app.callback(
            [
                Output("predict-go-plot", "figure"),
                Output("predict-output", "children"),
                Output("predict-tree-info", "children"),
                Output(
                    {"type": "predict-feature-display", "feature": ALL},
                    "children",
                ),
            ],
            [
                Input({"type": "predict-feature", "feature": ALL}, "value"),
                Input("predict-tree-id", "value"),
            ],
            State({"type": "predict-feature", "feature": ALL}, "id"),
        )
        def update_prediction(
            indices: List[int],
            tree_id: int,
            slider_ids: List[Dict[str, str]],
        ) -> List[Any]:
            if not slider_ids:
                return [no_update, no_update, no_update, no_update]

            try:
                feature_order = [sid["feature"] for sid in slider_ids]
                safe_indices = [
                    int(i) if i is not None else 0 for i in indices
                ]
                sample = self.feature_mgr.sample_from_indices(
                    safe_indices, feature_order
                )
                tid = tree_id or 0
                highlight_path = self._leaf_path_for_sample(sample, tid)
                fig = self._get_go_fig(tid, highlight_path=highlight_path)
                prediction_text = self._predict_sample(sample, tid)

                display_values: List[str] = []
                for idx, feat in zip(safe_indices, feature_order):
                    display_values.append(
                        self.feature_mgr.display_for_index(feat, idx)
                    )

                return [fig, prediction_text, self._tree_info_text(tid), display_values]
            except Exception as exc:  # pragma: no cover - runtime safety
                self.logger.error("PredictTab callback failed: %s", exc)
                return [no_update, "Prediction unavailable.", no_update, no_update]

    def _leaf_path_for_sample(
        self,
        sample: pd.DataFrame,
        tree_id: int,
    ) -> List[int]:
        if not self.vis_trees:
            return []

        tid = int(np.clip(tree_id, 0, len(self.vis_trees) - 1))
        vis_tree = self.vis_trees[tid]

        try:
            X_input = (
                sample[self.all_feature_names]
                if self.tree_use_df
                else sample[self.all_feature_names].to_numpy()
            )
            if self.is_xgboost:
                leaves = self.model.apply(X_input)  # type: ignore[arg-type]
                leaf_ids = leaves[:, tid] if hasattr(leaves, "ndim") and getattr(leaves, "ndim", 1) > 1 else leaves
            else:
                leaf_ids = vis_tree.model.apply(X_input)  # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.error(
                "Model does not support apply() for leaf path: %s", exc
            )
            return []

        if len(leaf_ids) == 0:
            return []

        leaf_id = int(leaf_ids[0])
        if self.is_xgboost:
            mapped_leaf_id = vis_tree.get_internal_node_id(leaf_id)
            if mapped_leaf_id is not None:
                leaf_id = mapped_leaf_id

        if hasattr(vis_tree, "leaf_paths") and isinstance(
            getattr(vis_tree, "leaf_paths"), dict
        ):
            if leaf_id in vis_tree.leaf_paths:  # type: ignore[attr-defined]
                return list(
                    vis_tree.leaf_paths[leaf_id]  # type: ignore[attr-defined]
                )

        if leaf_id not in vis_tree.nodes:
            return []

        path: List[int] = []
        node = vis_tree.nodes[leaf_id]
        while True:
            path.append(node.id)
            if node.parent is None or node.parent not in vis_tree.nodes:
                break
            node = vis_tree.nodes[node.parent]
        path.reverse()
        return path

    def _get_go_fig(
        self,
        tree_id: int,
        highlight_path: Optional[Sequence[int]] = None,
    ) -> Any:
        tid = int(np.clip(tree_id, 0, len(self.vis_trees) - 1))
        if highlight_path is None and tid in self._go_cache:
            return self._go_cache[tid]
        show_text = self._should_show_text(tid)
        fig = make_tree_figure(
            self.vis_trees[tid],
            kind=self.tree_plot_kind,
            show_text=show_text,
            highlight_path=highlight_path,
            sankey_dim_alpha=self.sankey_dim_alpha,
        )
        if highlight_path is None:
            self._go_cache[tid] = fig
        return fig

    def _should_show_text(self, tree_id: int) -> bool:
        tid = int(np.clip(tree_id, 0, len(self.vis_trees) - 1))
        return self.vis_trees[tid].max_depth <= 3

    def _predict_sample(self, sample: pd.DataFrame, tree_id: Optional[int] = None) -> str:
        if self.is_xgboost:
            tid = int(tree_id or 0)
            return self._predict_xgb(sample, tid)
        try:
            X_input = (
                sample[self.all_feature_names]
                if self.model_use_df
                else sample[self.all_feature_names].to_numpy()
            )
            preds = self.model.predict(X_input)
        except Exception as exc:
            self.logger.error("Prediction failed: %s", exc)
            return "Prediction unavailable."

        if len(preds) == 0:
            return "Prediction unavailable."

        pred = preds[0]
        if self.is_classifier:
            label: Any = pred
            if self.class_names is not None:
                try:
                    idx = int(pred)
                    if 0 <= idx < len(self.class_names):
                        label = self.class_names[idx]
                except Exception:
                    label = pred
            return f"Ensemble Prediction: {label}"

        try:
            value = float(pred)
            return f"Ensemble Prediction: {value:.4f}"
        except Exception:
            return f"Ensemble Prediction: {pred}"

    def _predict_xgb(self, sample: pd.DataFrame, tree_id: int) -> str:
        tid = int(np.clip(tree_id, 0, len(self.vis_trees) - 1))
        vis_tree = self.vis_trees[tid]
        try:
            X_input = (
                sample[self.all_feature_names]
                if self.model_use_df
                else sample[self.all_feature_names].to_numpy()
            )
        except Exception as exc:
            self.logger.error("XGBoost input prep failed: %s", exc)
            return "Score unavailable."

        tree_score = float("nan")
        try:
            tree_score = float(vis_tree.predict(X_input)[0])
        except Exception as exc:
            self.logger.error("XGBoost tree score computation failed: %s", exc)

        try:
            final_pred = self.model.predict(X_input)[0]
        except Exception as exc:
            self.logger.error("XGBoost final prediction failed: %s", exc)
            final_pred = None

        lines = [self._tree_info_text(tid)]
        score_txt = f"{tree_score:+.4f}" if np.isfinite(tree_score) else "n/a"
        lines.append(f"Leaf score contribution {score_txt}")

        if self.xgb_group_size > 1 and self.class_names:
            try:
                probs = np.asarray(self.model.predict_proba(X_input)[0], dtype=float)
                prob_parts = []
                for idx, label in enumerate(self.class_names):
                    if idx < probs.shape[0]:
                        prob_parts.append(f"{label}: {probs[idx]:.3f}")
                if prob_parts:
                    lines.append("Full model probabilities " + " | ".join(prob_parts))
            except Exception as exc:
                self.logger.debug("XGBoost probability display failed: %s", exc)

        if final_pred is not None:
            if isinstance(final_pred, (float, np.floating)):
                lines.append(f"Full model prediction {float(final_pred):.4f}")
            else:
                lines.append(f"Full model prediction {final_pred}")

        return "\n".join(lines)
