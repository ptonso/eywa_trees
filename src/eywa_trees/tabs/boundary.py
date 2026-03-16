
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, ALL
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.base import ClassifierMixin
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from eywa_trees.logger import setup_logger
from eywa_trees.backend.feature_bins import FeatureBinManager


Array = np.ndarray


@dataclass
class LatentEmbeddingConfig:
    n_components: int = 2
    perplexity: float = 30.0
    learning_rate: float = 200.0
    max_iter: int = 1000  # TSNE renamed n_iter->max_iter in sklearn>=1.5
    random_state: int = 0
    max_entries: int = 2000
    grid_resolution: int = 180
    bounds_k: float = 2.5
    nn_k_encode: int = 5
    nn_k_decode: int = 1  # snap grid to nearest training point


class BoundaryTab:
    """
    Latent-space prediction landscape with the same feature sliders as PredictTab.
    Left: feature sliders + “Return to median”.
    Right: 2D latent map of the training data + prediction heatmap + pointer
           for the current synthesized sample.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_mgr: Optional[FeatureBinManager] = None,
        latent_cfg: Optional[LatentEmbeddingConfig] = None,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.model = model
        self.X_train = X_train

        self.feature_mgr = feature_mgr or FeatureBinManager(X_train)
        self.feature_bins = self.feature_mgr.feature_bins
        self.active_feature_names = self.feature_mgr.active_feature_names
        self.all_feature_names = self.feature_mgr.all_feature_names
        self.default_indices = self.feature_mgr.default_indices
        self.use_df_for_predict = hasattr(model, "feature_names_in_")
        self.is_classifier = self._is_classifier(model)
        self.class_names = self._infer_class_names(model)

        self.latent_cfg = latent_cfg or LatentEmbeddingConfig()
        self.slider_wrapper_base_style = {
            "padding": "8px 8px 12px 8px",
            "borderRadius": "8px",
            "marginBottom": "4px",
            "transition": "box-shadow 0.2s ease, background-color 0.2s ease",
        }

        self._latent_ok = False
        self.Z_train: Optional[Array] = None
        self.X_train_arr: Optional[Array] = None
        self.nn_X: Optional[NearestNeighbors] = None
        self.nn_Z: Optional[NearestNeighbors] = None
        self.grid_Z: Optional[Array] = None
        self.grid_y: Optional[Array] = None
        self.grid_extent: Optional[Tuple[float, float, float, float]] = None
        self._latent_df: Optional[pd.DataFrame] = None

        try:
            self._prepare_latent_embedding()
            self._latent_ok = True
        except Exception as exc:
            self.logger.error("BoundaryTab latent preparation failed: %s", exc)
            self._latent_ok = False

        default_sample = self.feature_mgr.sample_from_indices(
            [self.default_indices[f] for f in self.active_feature_names],
            self.active_feature_names,
        )
        self.initial_prediction = self._predict_sample(default_sample)
        self.initial_pointer = self._encode_sample(default_sample)
        self.initial_fig = self._build_figure(self.initial_pointer)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    @property
    def layout(self) -> html.Div:
        sliders: List[html.Div] = []
        for feat in self.active_feature_names:
            bins = self.feature_bins.get(feat, [None])
            marks = self.feature_mgr.marks_for_feature(feat, bins)
            default_display = self.feature_mgr.display_for_index(
                feat,
                self.default_indices.get(feat, 0),
            )

            slider = dcc.Slider(
                id={"type": "boundary-feature", "feature": feat},
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
                                        "type": "boundary-feature-display",
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
                    id={"type": "boundary-feature-wrapper", "feature": feat},
                    style=dict(self.slider_wrapper_base_style),
                )
            )

        left_panel = html.Div(
            [
                html.Button(
                    "Return to median",
                    id="boundary-reset",
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
        )

        right_panel = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            id="boundary-output",
                            children=self.initial_prediction,
                            style={
                                "padding": "10px 12px",
                                "fontSize": "20px",
                                "fontWeight": "bold",
                                "border": "1px solid #d1d5db",
                                "borderRadius": "6px",
                                "backgroundColor": "rgba(243, 244, 246, 0.92)",
                                "display": "inline-block",
                                "marginBottom": "8px",
                            },
                        ),
                    ],
                    style={"display": "flex", "justifyContent": "flex-start"},
                ),
                dcc.Graph(
                    id="boundary-map",
                    figure=self.initial_fig,
                    style={
                        "height": "70vh",
                        "minHeight": "360px",
                        "width": "100%",
                    },
                    config={
                        "displayModeBar": True,
                        "scrollZoom": True,
                    },
                ),
            ],
            style={"flex": "1", "padding": "16px 20px"},
        )

        return html.Div(
            [
                dcc.Store(id="boundary-pointer-store"),
                dcc.Store(id="boundary-active-feature-store"),
                left_panel,
                right_panel,
            ],
            style={"display": "flex", "flexDirection": "row"},
            id="boundary-tab",
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            [
                Output({"type": "boundary-feature", "feature": ALL}, "value"),
                Output("boundary-pointer-store", "data"),
            ],
            [Input("boundary-reset", "n_clicks"), Input("boundary-map", "clickData")],
            State({"type": "boundary-feature", "feature": ALL}, "id"),
        )
        def reset_sliders(
            n_clicks: int,
            click_data: Optional[Dict[str, Any]],
            slider_ids: List[Dict[str, str]],
        ) -> List[Any]:
            ctx = dash.callback_context
            if not ctx.triggered:
                return [[no_update] * len(slider_ids), no_update]

            trigger = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger == "boundary-reset":
                return [
                    [self.default_indices.get(s["feature"], 0) for s in slider_ids],
                    None,
                ]

            if trigger == "boundary-map" and click_data:
                point = click_data.get("points", [{}])[0]
                z = (point.get("x"), point.get("y"))
                sample = self._decode_latent(z)
                if sample is None:
                    return [[no_update] * len(slider_ids), no_update]
                feature_order = [sid["feature"] for sid in slider_ids]
                indices: List[int] = []
                for feat in feature_order:
                    val = sample.iloc[0].get(feat)
                    indices.append(
                        self.feature_mgr.nearest_index_for_value(feat, val)
                    )
                # store the raw clicked point so the marker sits exactly where the user clicked
                return [indices, z]

            return [[no_update] * len(slider_ids), no_update]

        @app.callback(
            [
                Output("boundary-map", "figure"),
                Output("boundary-output", "children"),
                Output(
                    {"type": "boundary-feature-display", "feature": ALL},
                    "children",
                ),
            ],
            [
                Input("boundary-pointer-store", "data"),
                Input({"type": "boundary-feature", "feature": ALL}, "value"),
                Input("boundary-active-feature-store", "data"),
            ],
            State({"type": "boundary-feature", "feature": ALL}, "id"),
        )
        def update_from_sliders(
            stored_pointer: Optional[Tuple[float, float]],
            indices: List[int],
            active_feature: Optional[str],
            slider_ids: List[Dict[str, str]],
        ) -> List[Any]:
            if not slider_ids:
                return [no_update, no_update, no_update]

            try:
                feature_order = [sid["feature"] for sid in slider_ids]
                safe_indices = [
                    int(i) if i is not None else 0 for i in indices
                ]
                sample = self.feature_mgr.sample_from_indices(
                    safe_indices,
                    feature_order,
                )

                prediction_text = self._predict_sample(sample)

                ctx = dash.callback_context
                triggered_ids = {t["prop_id"].split(".")[0] for t in ctx.triggered} if ctx.triggered else set()
                if "boundary-pointer-store" in triggered_ids and stored_pointer is not None:
                    pointer = tuple(stored_pointer)  # type: ignore[assignment]
                else:
                    pointer = self._encode_sample(sample)
                sweep_points = self._feature_sweep_points(
                    active_feature,
                    safe_indices,
                    feature_order,
                )
                fig = self._build_figure(pointer, sweep_points=sweep_points)

                display_values: List[str] = []
                for idx, feat in zip(safe_indices, feature_order):
                    display_values.append(
                        self.feature_mgr.display_for_index(feat, idx)
                    )

                return [fig, prediction_text, display_values]
            except Exception as exc:  # pragma: no cover - runtime safety
                self.logger.error("BoundaryTab callback failed: %s", exc)
                return [no_update, "Prediction unavailable.", no_update]

        @app.callback(
            Output("boundary-active-feature-store", "data"),
            [
                Input({"type": "boundary-feature", "feature": ALL}, "value"),
                Input("boundary-reset", "n_clicks"),
                Input("boundary-pointer-store", "data"),
            ],
        )
        def set_active_feature(
            _slider_values: List[int],
            _reset_clicks: int,
            _pointer: Optional[Tuple[float, float]],
        ) -> Optional[str]:
            ctx = dash.callback_context
            if not ctx.triggered:
                return None

            trigger_ids = {t["prop_id"].split(".")[0] for t in ctx.triggered}
            if "boundary-reset" in trigger_ids or "boundary-pointer-store" in trigger_ids:
                return None

            for trig in ctx.triggered:
                prop_id = trig["prop_id"].split(".")[0]
                if not prop_id:
                    continue
                if prop_id.startswith("{"):
                    try:
                        parsed = json.loads(prop_id)
                        return parsed.get("feature")
                    except Exception:
                        continue
            return no_update

        @app.callback(
            Output({"type": "boundary-feature-wrapper", "feature": ALL}, "style"),
            Input("boundary-active-feature-store", "data"),
            State({"type": "boundary-feature", "feature": ALL}, "id"),
        )
        def highlight_active_feature(
            active_feature: Optional[str],
            slider_ids: List[Dict[str, str]],
        ) -> List[Dict[str, Any]]:
            if not slider_ids:
                return []

            styles: List[Dict[str, Any]] = []
            for sid in slider_ids:
                feat = sid.get("feature")
                styles.append(
                    self._slider_wrapper_style(
                        bool(active_feature and feat == active_feature)
                    )
                )
            return styles

    def _prepare_latent_embedding(self) -> None:
        df_active = self.X_train[self.active_feature_names]
        df_full = self.X_train[self.all_feature_names]

        max_entries = getattr(self.latent_cfg, "max_entries", None)
        if max_entries and df_active.shape[0] > max_entries:
            rng = np.random.default_rng(self.latent_cfg.random_state)
            idxs = rng.choice(df_active.shape[0], size=max_entries, replace=False)
            idxs = np.sort(idxs)
            df_active = df_active.iloc[idxs]
            df_full = df_full.iloc[idxs]

        df_active = df_active.reset_index(drop=True)
        df_full = df_full.reset_index(drop=True)
        self._latent_df = df_full

        X = df_active.to_numpy()
        self.X_train_arr = X

        tsne_kwargs = dict(
            n_components=self.latent_cfg.n_components,
            perplexity=self.latent_cfg.perplexity,
            learning_rate=self.latent_cfg.learning_rate,
            random_state=self.latent_cfg.random_state,
            init="pca",
        )
        # TSNE switched from n_iter to max_iter in sklearn>=1.5.
        if "max_iter" in TSNE.__init__.__code__.co_varnames:
            tsne_kwargs["max_iter"] = self.latent_cfg.max_iter
        else:
            tsne_kwargs["n_iter"] = self.latent_cfg.max_iter

        reducer = TSNE(**tsne_kwargs)
        Z = reducer.fit_transform(X)
        self.Z_train = Z

        self.nn_X = NearestNeighbors(
            n_neighbors=self.latent_cfg.nn_k_encode
        )
        self.nn_X.fit(X)

        self.nn_Z = NearestNeighbors(
            n_neighbors=self.latent_cfg.nn_k_decode
        )
        self.nn_Z.fit(Z)

        z_mean = Z.mean(axis=0)
        z_std = Z.std(axis=0) + 1e-6
        k = self.latent_cfg.bounds_k
        lo = z_mean - k * z_std
        hi = z_mean + k * z_std

        xs = np.linspace(lo[0], hi[0], self.latent_cfg.grid_resolution)
        ys = np.linspace(lo[1], hi[1], self.latent_cfg.grid_resolution)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1)

        _, idx = self.nn_Z.kneighbors(grid_points, return_distance=True)
        idx = idx[:, 0]
        X_grid = df_full.to_numpy()[idx, :]

        try:
            if self.use_df_for_predict:
                df_grid = pd.DataFrame(X_grid, columns=self.all_feature_names)
                y_grid = self._predict_grid_values(df_grid)
            else:
                y_grid = self._predict_grid_values(X_grid)
        except Exception as exc:
            self.logger.error(
                "BoundaryTab grid prediction failed: %s", exc
            )
            y_grid = np.zeros(X_grid.shape[0], dtype=float)

        self.grid_Z = grid_points
        self.grid_y = y_grid.reshape(
            self.latent_cfg.grid_resolution,
            self.latent_cfg.grid_resolution,
        )
        self.grid_extent = (
            float(xs.min()),
            float(xs.max()),
            float(ys.min()),
            float(ys.max()),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_classifier(self, model: Any) -> bool:
        if isinstance(model, ClassifierMixin):
            return True
        if getattr(model, "_estimator_type", None) == "classifier":
            return True
        if hasattr(model, "estimator") and isinstance(getattr(model, "estimator"), ClassifierMixin):
            return True
        if hasattr(model, "estimators_") and getattr(model, "estimators_", []):
            first = model.estimators_[0]
            if isinstance(first, ClassifierMixin):
                return True
            if getattr(first, "_estimator_type", None) == "classifier":
                return True
        return False

    def _infer_class_names(self, model: Any) -> Optional[List[str]]:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return None
        try:
            return [str(c) for c in list(classes)]
        except Exception:
            return None

    def _encode_class_labels(self, preds: Any) -> np.ndarray:
        arr = np.asarray(preds)
        if arr.ndim == 0:
            arr = np.array([arr])
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(float)
        if self.class_names:
            mapping = {str(c): i for i, c in enumerate(self.class_names)}
            return np.array([mapping.get(str(v), 0) for v in arr], dtype=float)
        unique = {v: i for i, v in enumerate(pd.unique(arr))}
        return np.array([unique[v] for v in arr], dtype=float)

    def _predict_grid_values(self, X_input: Any) -> np.ndarray:
        if not self.is_classifier:
            return np.asarray(self.model.predict(X_input))
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(X_input)
                if isinstance(probs, list):
                    probs_arr = np.asarray(probs[0]) if probs else np.asarray([])
                else:
                    probs_arr = np.asarray(probs)
                if probs_arr.ndim == 2:
                    return np.argmax(probs_arr, axis=1).astype(float)
                if probs_arr.ndim == 1:
                    return probs_arr.astype(float)
            except Exception:
                pass
        preds = self.model.predict(X_input)
        return self._encode_class_labels(preds)

    def _classification_colorscale(self, n_classes: int) -> List[List[Any]]:
        palette = px.colors.qualitative.Plotly
        colors = [palette[i % len(palette)] for i in range(max(1, n_classes))]
        if n_classes <= 1:
            return [[0.0, colors[0]], [1.0, colors[0]]]
        scale: List[List[Any]] = []
        for idx, color in enumerate(colors):
            start = idx / float(n_classes)
            end = (idx + 1) / float(n_classes)
            scale.append([start, color])
            scale.append([end, color])
        return scale

    def _encode_sample(
        self,
        sample: pd.DataFrame,
    ) -> Optional[Tuple[float, float]]:
        if not self._latent_ok or self.nn_X is None or self.Z_train is None:
            return None

        try:
            x = sample[self.active_feature_names].to_numpy()
        except Exception as exc:
            self.logger.error(
                "BoundaryTab encode sample failed: %s", exc
            )
            return None

        if x.ndim == 1:
            x = x.reshape(1, -1)

        _, idx = self.nn_X.kneighbors(x, return_distance=True)
        idx0 = idx[0, 0]
        z = self.Z_train[idx0]
        return float(z[0]), float(z[1])

    def _build_figure(
        self,
        pointer: Optional[Tuple[float, float]],
        sweep_points: Optional[Sequence[Tuple[float, float, str]]] = None,
    ) -> go.Figure:
        fig = go.Figure()

        if self._latent_ok and self.grid_y is not None and self.grid_extent:
            xmin, xmax, ymin, ymax = self.grid_extent
            colorscale: Any = "Viridis"
            zmin = None
            zmax = None
            hovertemplate = "Prediction: %{z:.2f}<extra></extra>"
            colorbar: Dict[str, Any] = {
                "title": dict(text="Prediction", side="right"),
            }
            heatmap_kwargs: Dict[str, Any] = {}

            if self.is_classifier:
                grid = np.asarray(self.grid_y)
                if self.class_names:
                    n_classes = len(self.class_names)
                else:
                    n_classes = int(np.nanmax(grid)) + 1 if grid.size else 1
                colorscale = self._classification_colorscale(n_classes)
                zmin = 0
                zmax = max(0, n_classes - 1)
                hovertemplate = "Class: %{z}<extra></extra>"
                if self.class_names:
                    colorbar.update(
                        tickvals=list(range(n_classes)),
                        ticktext=self.class_names,
                    )
                    heatmap_kwargs["text"] = np.take(
                        np.asarray(self.class_names, dtype=object),
                        grid.astype(int),
                        mode="clip",
                    )
                    hovertemplate = "Class: %{text}<extra></extra>"

            fig.add_trace(
                go.Heatmap(
                    z=self.grid_y,
                    x=np.linspace(
                        xmin,
                        xmax,
                        self.latent_cfg.grid_resolution,
                    ),
                    y=np.linspace(
                        ymin,
                        ymax,
                        self.latent_cfg.grid_resolution,
                    ),
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=colorbar,
                    hovertemplate=hovertemplate,
                    **heatmap_kwargs,
                )
            )

        if sweep_points:
            xs, ys, texts = zip(*sweep_points)
            fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="rgba(99, 102, 241, 0.85)",
                        line=dict(width=1, color="white"),
                        symbol="circle",
                    ),
                    name="Feature sweep",
                    hovertemplate="%{text}<extra></extra>",
                    text=list(texts),
                    showlegend=False,
                )
            )

        if (
            self._latent_ok
            and self.Z_train is not None
            and pointer is not None
        ):
            fig.add_trace(
                go.Scatter(
                    x=[pointer[0]],
                    y=[pointer[1]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        symbol="x",
                        line=dict(width=2),
                    ),
                    name="Current sample",
                    showlegend=False,
                )
            )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Profile dimension 1",
            yaxis_title="Profile dimension 2",
            xaxis=dict(
                range=[self.grid_extent[0], self.grid_extent[1]] if self.grid_extent else None,
                fixedrange=True,
            ),
            yaxis=dict(
                range=[self.grid_extent[2], self.grid_extent[3]] if self.grid_extent else None,
                fixedrange=True,
            ),
        )

        return fig

    def _slider_wrapper_style(self, is_active: bool) -> Dict[str, Any]:
        style = dict(self.slider_wrapper_base_style)
        if is_active:
            style.update(
                backgroundColor="rgba(99, 102, 241, 0.08)",
                boxShadow="0 0 0 2px #6366f1",
            )
        return style

    def _predict_sample(self, sample: pd.DataFrame) -> str:
        try:
            X_input = (
                sample[self.all_feature_names]
                if self.use_df_for_predict
                else sample[self.all_feature_names].to_numpy()
            )
            preds = self.model.predict(X_input)
        except Exception as exc:
            self.logger.error(
                "BoundaryTab prediction failed: %s", exc
            )
            return "Prediction unavailable."

        if len(preds) == 0:
            return "Prediction unavailable."

        pred = preds[0]
        try:
            value = float(pred)
            return f"Prediction: {value:.4f}"
        except Exception:
            return f"Prediction: {pred}"

    def _predict_value(self, sample: pd.DataFrame) -> Any:
        try:
            X_input = (
                sample[self.all_feature_names]
                if self.use_df_for_predict
                else sample[self.all_feature_names].to_numpy()
            )
            preds = self.model.predict(X_input)
        except Exception as exc:
            self.logger.error("BoundaryTab prediction value failed: %s", exc)
            return None

        if len(preds) == 0:
            return None
        return preds[0]

    def _format_prediction_value(self, value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            v = float(value)
            if np.isfinite(v):
                return f"{v:.4f}"
        except Exception:
            pass
        return str(value)

    def _feature_sweep_points(
        self,
        active_feature: Optional[str],
        base_indices: List[int],
        feature_order: List[str],
    ) -> List[Tuple[float, float, str]]:
        if (
            not active_feature
            or not self._latent_ok
            or active_feature not in feature_order
        ):
            return []

        feat_idx = feature_order.index(active_feature)
        bins = self.feature_bins.get(active_feature, [None])
        points: List[Tuple[float, float, str]] = []

        for idx in range(len(bins)):
            varied_indices = list(base_indices)
            varied_indices[feat_idx] = idx
            sample = self.feature_mgr.sample_from_indices(
                varied_indices, feature_order
            )
            pointer = self._encode_sample(sample)
            if pointer is None:
                continue

            pred_value = self._predict_value(sample)
            pred_text = self._format_prediction_value(pred_value)
            value_display = self.feature_mgr.display_for_index(
                active_feature, idx
            )
            hover = f"{active_feature}: {value_display}"
            if pred_value is not None:
                hover += f"<br>Prediction: {pred_text}"
            points.append((pointer[0], pointer[1], hover))

        return points

    def _decode_latent(
        self,
        z: Tuple[Optional[float], Optional[float]],
    ) -> Optional[pd.DataFrame]:
        if (
            not self._latent_ok
            or self.nn_Z is None
            or self.Z_train is None
            or z[0] is None
            or z[1] is None
        ):
            return None

        try:
            zz = np.array([[float(z[0]), float(z[1])]])
        except Exception:
            return None

        try:
            _, idx = self.nn_Z.kneighbors(zz, return_distance=True)
            row_idx = int(idx[0, 0])
            if self._latent_df is None:
                return None
            row = self._latent_df.iloc[[row_idx]][self.all_feature_names]
            return row.reset_index(drop=True)
        except Exception as exc:
            self.logger.error("BoundaryTab decode failed: %s", exc)
            return None
