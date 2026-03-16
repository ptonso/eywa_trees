from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import math

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objects as go

from eywa_trees.backend.go_plot import GoTreePlot
from eywa_trees.backend.path_statistics import PathStatistics
from eywa_trees.backend.rule_engine import RuleEngine, RuleEngineConfig, SortMode
from eywa_trees.backend.subpath_engine import SubPathEngine, SubPathCombo


@dataclass
class SubPathTabConfig:
    top_k_combos: int = 4


class SubPathTab:
    def __init__(
        self,
        model: Any,
        X_train: Any,
        class_names: Optional[list[str]] = None,
        engine_config: Optional[RuleEngineConfig] = None,
        tab_config: Optional[SubPathTabConfig] = None,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.class_names = class_names
        self.tab_config = tab_config or SubPathTabConfig()
        self.slider_height = 320
        self.plot_height = 420

        stats = PathStatistics(model, X_train, class_names, ecdf_bin_config=None)
        rules_df = stats.rules_dataframe()
        if rules_df.empty:
            raise ValueError("Empty pathway statistics — cannot build sub-path tab")

        order = rules_df.index.to_numpy()
        path_order = None
        if (
            isinstance(stats.pset.features_upper, np.ndarray)
            and stats.pset.features_upper.size
            and order.size == stats.pset.features_upper.shape[0]
        ):
            path_order = order

        upper_depths = getattr(stats.pset, "features_upper_depth", None)
        if isinstance(upper_depths, np.ndarray) and upper_depths.ndim == 2:
            if path_order is not None and path_order.size == upper_depths.shape[0]:
                upper_depths = upper_depths[path_order]
            else:
                upper_depths = None
        else:
            upper_depths = None

        self.rule_engine = RuleEngine(
            pset_df=rules_df,
            feature_names=stats.feature_names,
            class_names=class_names,
            dataset_n=X_train.shape[0],
            ecdf_dict=getattr(stats.pset, "ecdf_dict", None),
            ecdf_config=stats.ecdf_bin_config,
            config=engine_config or RuleEngineConfig(),
            upper_depths=upper_depths,
        )
        self.prediction_label = self.rule_engine.prediction_column_label()
        self.hist_domain = None
        self.color_domain = None
        if not self.rule_engine.is_classification and self.rule_engine.pred_vector is not None:
            vals = np.asarray(self.rule_engine.pred_vector, dtype=float)
            finite = np.isfinite(vals)
            if np.any(finite):
                self.hist_domain = (float(vals[finite].min()), float(vals[finite].max()))
                self.color_domain = self.hist_domain
        self.subpath_engine = SubPathEngine(
            self.rule_engine,
            stats.pset,
            path_order=path_order,
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    @property
    def layout(self) -> html.Div:
        sort_toggle = dcc.RadioItems(
            id="subpath-sort-mode",
            options=[
                {"label": "count", "value": "paths"},
                {"label": "coverage", "value": "coverage"},
            ],
            value="paths",
            inline=True,
        )

        left = html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "Sort rules by:",
                            style={"fontWeight": "bold", "marginRight": "8px"},
                        ),
                        sort_toggle,
                    ],
                    style={"marginBottom": "12px"},
                ),
                html.Label("Length", style={"fontWeight": "bold"}),
                dcc.Slider(
                    id="subpath-length",
                    min=1,
                    max=self.subpath_engine.max_length,
                    value=1,
                    marks={str(i): str(i) for i in range(1, self.subpath_engine.max_length + 1)},
                    step=None,
                    tooltip={"always_visible": True},
                    vertical=True,
                    verticalHeight=self.slider_height,
                ),
                dcc.Store(id="subpath-page", data=0),
            ],
            style={
                "width": "16%",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "gap": "12px",
                "padding": "16px",
                "borderRight": "1px solid #e5e7eb",
                "backgroundColor": "#fafafa",
            },
        )

        nav_buttons = html.Div(
            [
                html.Button("◀", id="subpath-page-prev", n_clicks=0),
                html.Button("Top", id="subpath-page-top", n_clicks=0),
                html.Button("▶", id="subpath-page-next", n_clicks=0),
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "gap": "8px",
                "marginBottom": "8px",
            },
        )

        right = html.Div(
            [
                nav_buttons,
                html.Div(
                    [self._plot_card(i) for i in range(self.tab_config.top_k_combos)],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                        "gridTemplateRows": "repeat(2, 1fr)",
                        "height": f"{self.plot_height}px",
                        "gap": "16px",
                    },
                ),
            ],
            style={
                "flex": "1",
                "padding": "16px 20px",
            },
        )

        return html.Div(
            [left, right],
            style={"display": "flex", "flexDirection": "row"},
            id="sub-path-tab",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _empty_fig(self, text: str = "No sub-paths available") -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=text,
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    def _stat_text(self, combo: Optional[SubPathCombo]) -> str:
        if combo is None:
            return "coverage: n/a\ncount: 0"
        label = self.prediction_label.lower()
        return (
            f"{label}: {combo.pred_text}\n"
            f"coverage: {combo.coverage:.3f}\n"
            f"count: {combo.count}"
        )

    def _plot_card(self, idx: int) -> html.Div:
        return html.Div(
            [
                html.Div(
                    self._stat_text(None),
                    id=f"subpath-stat-{idx}",
                    style={
                        "fontSize": "12px",
                        "color": "#111827",
                        "backgroundColor": "#f3f4f6",
                        "borderRadius": "6px",
                        "padding": "6px 8px",
                        "whiteSpace": "pre-line",
                    },
                ),
                dcc.Graph(
                    id=f"subpath-plot-{idx}",
                    figure=self._empty_fig(),
                    style={"flex": "1", "height": "100%"},
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "height": "100%"},
        )

    def _build_combo_fig(self, combo) -> go.Figure:
        vis_tree = self.subpath_engine.build_combo_tree(combo)
        fig = GoTreePlot(
            vis_tree,
            show_text=True,
            show_leaf_hist=True,
            horizontal_spacing=0.05,
            show_edge_labels=False,
            label_hide_depth=4,
            hist_domain=self.hist_domain,
            label_mode="rule",
            color_domain=self.color_domain,
        ).fig
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            Output("subpath-page", "data"),
            [
                Input("subpath-page-prev", "n_clicks"),
                Input("subpath-page-top", "n_clicks"),
                Input("subpath-page-next", "n_clicks"),
                Input("subpath-length", "value"),
                Input("subpath-sort-mode", "value"),
            ],
            State("subpath-page", "data"),
        )
        def update_subpath_page(
            prev_clicks: int,
            top_clicks: int,
            next_clicks: int,
            length: int,
            sort_mode_val: str,
            page: Optional[int],
        ) -> int:
            page = int(page or 0)
            sort_mode: SortMode = (
                sort_mode_val if sort_mode_val in ("paths", "coverage") else "paths"
            )
            total = len(self.subpath_engine.sorted_by_length.get(int(length or 1), {}).get(sort_mode, []))
            page_size = self.tab_config.top_k_combos
            max_page = max(0, int(math.ceil(total / float(page_size))) - 1)

            ctx = dash.callback_context
            if not ctx.triggered:
                return 0
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger in ("subpath-length", "subpath-sort-mode"):
                return 0
            if trigger == "subpath-page-prev":
                page -= 1
            elif trigger == "subpath-page-next":
                page += 1
            elif trigger == "subpath-page-top":
                page = 0

            if page < 0:
                page = 0
            if page > max_page:
                page = max_page
            return int(page)

        @app.callback(
            [
                Output("subpath-plot-0", "figure"),
                Output("subpath-plot-1", "figure"),
                Output("subpath-plot-2", "figure"),
                Output("subpath-plot-3", "figure"),
                Output("subpath-stat-0", "children"),
                Output("subpath-stat-1", "children"),
                Output("subpath-stat-2", "children"),
                Output("subpath-stat-3", "children"),
            ],
            [
                Input("subpath-length", "value"),
                Input("subpath-sort-mode", "value"),
                Input("subpath-page", "data"),
            ],
        )
        def update_subpaths(length: int, sort_mode_val: str, page: Optional[int]):
            if length is None:
                figs = [self._empty_fig("Select a length")] * self.tab_config.top_k_combos
                stats = [self._stat_text(None)] * self.tab_config.top_k_combos
                return tuple(figs + stats)

            sort_mode: SortMode = (
                sort_mode_val if sort_mode_val in ("paths", "coverage") else "paths"
            )
            combos_all = self.subpath_engine.sorted_by_length.get(int(length), {}).get(sort_mode, [])
            page_size = self.tab_config.top_k_combos
            page_val = int(page or 0)
            max_page = max(0, int(math.ceil(len(combos_all) / float(page_size))) - 1)
            if page_val > max_page:
                page_val = max_page
            start = page_val * page_size
            end = start + page_size
            combos = combos_all[start:end]
            figs = []
            stats = []
            for combo in combos:
                figs.append(self._build_combo_fig(combo))
                stats.append(self._stat_text(combo))
            while len(figs) < self.tab_config.top_k_combos:
                figs.append(self._empty_fig())
                stats.append(self._stat_text(None))
            return tuple(figs + stats)
