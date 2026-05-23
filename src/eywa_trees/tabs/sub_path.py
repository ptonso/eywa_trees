from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import dash
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
import numpy as np
import plotly.graph_objects as go

from eywa_trees.backend.go_plot import GoTreePlot
from eywa_trees.backend.path_statistics import PathStatistics
from eywa_trees.backend.rule_engine import RuleEngine, RuleEngineConfig, SortMode
from eywa_trees.backend.subpath_engine import SubPathCandidate, SubPathEngine
from eywa_trees.backend.ecdf_rule_group import ECDFBinConfig


@dataclass
class SubPathTabConfig:
    top_k_combos: int = 4


class SubPathTab:
    _CONTENT_HEIGHT = "calc(100vh - 190px)"

    def __init__(
        self,
        model: Any,
        X_train: Any,
        class_names: Optional[list[str]] = None,
        engine_config: Optional[RuleEngineConfig] = None,
        tab_config: Optional[SubPathTabConfig] = None,
        ecdf_bin_config: Optional[ECDFBinConfig] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.class_names = class_names
        self.tab_config = tab_config or SubPathTabConfig()
        self.max_length = max_length
        self.slider_height = 300

        stats = PathStatistics(model, X_train, class_names, ecdf_bin_config=ecdf_bin_config)
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
        self.feature_names = list(self.rule_engine.feature_names or [])
        self.prediction_label = self.rule_engine.prediction_column_label()
        self.hist_domain = None
        self.color_domain = None
        if (
            not self.rule_engine.is_classification
            and self.rule_engine.pred_vector is not None
        ):
            vals = np.asarray(self.rule_engine.pred_vector, dtype=float)
            finite = np.isfinite(vals)
            if np.any(finite):
                self.hist_domain = (
                    float(vals[finite].min()),
                    float(vals[finite].max()),
                )
                self.color_domain = self.hist_domain
        self.subpath_engine = SubPathEngine(
            self.rule_engine,
            stats.pset,
            max_length=self.max_length,
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
                        html.Div(
                            [
                                html.Span(
                                    "Sort rules by:",
                                    style={"fontWeight": "bold", "marginRight": "8px"},
                                ),
                                sort_toggle,
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "8px",
                            },
                        ),
                    ],
                    style={"marginBottom": "12px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Length", style={"fontWeight": "bold"}),
                                dcc.Slider(
                                    id="subpath-length",
                                    min=1,
                                    max=self.subpath_engine.max_length,
                                    value=1,
                                    marks={
                                        str(i): str(i)
                                        for i in range(1, self.subpath_engine.max_length + 1)
                                    },
                                    step=None,
                                    tooltip={"always_visible": True},
                                    vertical=True,
                                    verticalHeight=self.slider_height,
                                ),
                            ],
                            style={
                                "width": "86px",
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "center",
                                "gap": "12px",
                                "paddingRight": "12px",
                                "borderRight": "1px solid #e5e7eb",
                                "boxSizing": "border-box",
                            },
                        ),
                        html.Div(
                            [
                                html.Div("Feature filter", style={"fontWeight": "bold"}),
                                html.Div(
                                    id="subpath-feature-help",
                                    children=self._feature_help_text(length=1, selected_features=[]),
                                    style={"fontSize": "12px", "color": "#6b7280"},
                                ),
                                html.Div(
                                    id="subpath-feature-chip-container",
                                    children=self._render_feature_buttons([], 1),
                                    style={
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "gap": "8px",
                                        "alignContent": "flex-start",
                                        "overflowY": "auto",
                                        "paddingRight": "4px",
                                        "flex": "1 1 auto",
                                        "minHeight": "0",
                                    },
                                ),
                            ],
                            style={
                                "flex": "1 1 auto",
                                "minWidth": "0",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "8px",
                                "minHeight": "0",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "gap": "14px",
                        "alignItems": "stretch",
                        "flex": "1 1 auto",
                        "minHeight": "0",
                    },
                ),
                dcc.Store(id="subpath-selected-features", data=[]),
                dcc.Store(id="subpath-rank-index", data=0),
            ],
            style={
                "width": "28%",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "stretch",
                "gap": "10px",
                "padding": "16px",
                "borderRight": "1px solid #e5e7eb",
                "backgroundColor": "#fafafa",
                "height": "100%",
                "minHeight": "0",
                "boxSizing": "border-box",
            },
        )

        nav_buttons = html.Div(
            [
                html.Button("◀", id="subpath-rank-prev", n_clicks=0),
                html.Button("Top", id="subpath-rank-top", n_clicks=0),
                html.Button("▶", id="subpath-rank-next", n_clicks=0),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "8px",
            },
        )

        right = html.Div(
            [
                html.Div(
                    [
                        nav_buttons,
                        html.Div(
                            id="subpath-rank-label",
                            children="Showing #1",
                            style={
                                "fontSize": "14px",
                                "fontWeight": "bold",
                                "color": "#374151",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "12px",
                        "marginBottom": "10px",
                        "flexShrink": "0",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            id="subpath-summary",
                            children=self._summary_children(None),
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "16px",
                                "fontSize": "13px",
                                "color": "#111827",
                                "marginBottom": "6px",
                            },
                        ),
                        html.Div(
                            id="subpath-rule-text",
                            style={
                                "fontSize": "12px",
                                "color": "#4b5563",
                                "whiteSpace": "normal",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": "#f3f4f6",
                        "borderRadius": "8px",
                        "padding": "10px 12px",
                        "marginBottom": "12px",
                        "flexShrink": "0",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="subpath-plot",
                                    figure=self._empty_fig("No sub-paths available"),
                                    style={
                                        "flex": "1 1 auto",
                                        "height": "100%",
                                        "minHeight": "0",
                                        "width": "100%",
                                    },
                                ),
                            ],
                            style={
                                "flex": "0 1 860px",
                                "minHeight": "0",
                                "minWidth": "0",
                                "position": "relative",
                            },
                        ),
                        html.Div(
                            id="subpath-depth-panel",
                            style={
                                "width": "220px",
                                "minWidth": "220px",
                                "display": "flex",
                                "flexDirection": "column",
                                "overflow": "hidden",
                                "paddingRight": "4px",
                                "minHeight": "0",
                            },
                        ),
                    ],
                    style={
                        "flex": "1 1 auto",
                        "minHeight": "0",
                        "display": "flex",
                        "flexDirection": "row",
                        "alignItems": "stretch",
                        "justifyContent": "center",
                        "gap": "18px",
                    },
                ),
            ],
            style={
                "flex": "1",
                "padding": "16px 20px",
                "display": "flex",
                "flexDirection": "column",
                "minWidth": "0",
                "minHeight": "0",
                "height": "100%",
                "boxSizing": "border-box",
            },
        )

        return html.Div(
            [left, right],
            style={
                "display": "flex",
                "flexDirection": "row",
                "height": self._CONTENT_HEIGHT,
                "minHeight": self._CONTENT_HEIGHT,
                "alignItems": "stretch",
            },
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

    def _summary_children(self, candidate: Optional[SubPathCandidate]) -> list[html.Span]:
        if candidate is None:
            return [
                html.Span(
                    f"{self.prediction_label}: n/a",
                    style={"fontWeight": "bold"},
                ),
                html.Span("coverage: n/a"),
                html.Span("count: 0"),
            ]
        return [
            html.Span(
                f"{self.prediction_label}: {candidate.pred_text}",
                style={"fontWeight": "bold"},
            ),
            html.Span(f"coverage: {candidate.coverage:.3f}"),
            html.Span(f"count: {candidate.count}"),
        ]

    def _feature_help_text(
        self,
        *,
        length: int,
        selected_features: list[str],
    ) -> str:
        return (
            f"Select up to {int(length)} feature"
            f"{'' if int(length) == 1 else 's'} "
            f"({len(selected_features)}/{int(length)} selected)"
        )

    def _depth_histogram_fig(
        self,
        hist: dict[str, object],
        *,
        color: str,
        tickfont_size: int = 10,
    ) -> go.Figure:
        depths = np.asarray(hist.get("depths", []), dtype=int)
        counts = np.asarray(hist.get("counts", []), dtype=float)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=counts,
                y=depths,
                orientation="h",
                marker=dict(color=color),
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            autosize=True,
            margin=dict(l=8, r=8, t=2, b=2),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                ticks="",
            ),
            yaxis=dict(
                tickmode="linear",
                dtick=1,
                autorange="reversed",
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=max(8, int(tickfont_size)), color="#475569"),
                title=dict(text=""),
                side="right",
            ),
        )
        return fig

    def _depth_panel_metrics(self, node_count: int) -> dict[str, object]:
        depth = max(1, int(node_count))
        compact_steps = max(0, depth - 3)
        return {
            "stack_gap": f"{max(4, 10 - (compact_steps * 2))}px",
            "card_padding": f"{max(4, 8 - compact_steps)}px {max(6, 10 - compact_steps)}px",
            "title_size": f"{max(10, 12 - compact_steps)}px",
            "meta_size": f"{max(9, 11 - compact_steps)}px",
            "tickfont_size": max(8, 10 - compact_steps),
        }

    def _depth_card(
        self,
        candidate: SubPathCandidate,
        node_index: int,
        hist: dict[str, object],
        *,
        color: str,
        metrics: dict[str, object],
    ) -> html.Div:
        title = self.subpath_engine.candidate_node_text(candidate, node_index)
        mean_depth = float(hist.get("mean_depth", 0.0))
        total = int(hist.get("total", 0))
        return html.Div(
            [
                html.Div(
                    title,
                    style={
                        "fontSize": str(metrics["title_size"]),
                        "fontWeight": "bold",
                        "color": "#111827",
                        "lineHeight": "1.15",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                    },
                ),
                html.Div(
                    f"mean depth: {mean_depth:.2f} | matches: {total}",
                    style={
                        "fontSize": str(metrics["meta_size"]),
                        "color": "#6b7280",
                        "lineHeight": "1.1",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                    },
                ),
                html.Div(
                    dcc.Graph(
                        figure=self._depth_histogram_fig(
                            hist,
                            color=color,
                            tickfont_size=int(metrics["tickfont_size"]),
                        ),
                        config={"displayModeBar": False, "staticPlot": True},
                        responsive=True,
                        style={
                            "height": "100%",
                            "width": "100%",
                            "minHeight": "0",
                        },
                    ),
                    style={
                        "flex": "1 1 auto",
                        "minHeight": "0",
                    },
                ),
            ],
            style={
                "backgroundColor": "#f8fafc",
                "border": "1px solid #e5e7eb",
                "borderLeft": f"4px solid {color}",
                "borderRadius": "10px",
                "padding": str(metrics["card_padding"]),
                "display": "flex",
                "flexDirection": "column",
                "gap": "4px",
                "height": "100%",
                "minHeight": "0",
                "overflow": "hidden",
                "boxSizing": "border-box",
            },
        )

    def _depth_panel_children(self, candidate: Optional[SubPathCandidate]) -> list[html.Div]:
        if candidate is None:
            return []
        node_count = len(candidate.group_ids)
        if node_count <= 0:
            return []
        metrics = self._depth_panel_metrics(node_count)
        vis_tree = self.subpath_engine.build_candidate_tree(candidate)
        cards: list[html.Div] = []
        for node_index in range(node_count):
            hist = self.subpath_engine.depth_histogram_for_node(candidate, node_index)
            if hist is None:
                continue
            cards.append(
                html.Div(
                    self._depth_card(
                        candidate,
                        node_index,
                        hist,
                        color=vis_tree.color_for_node(node_index),
                        metrics=metrics,
                    ),
                    style={
                        "gridRow": str(node_index + 1),
                        "minHeight": "0",
                    },
                )
            )
        if not cards:
            return []
        return [
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateRows": f"repeat({node_count + 1}, minmax(0, 1fr))",
                    "gap": str(metrics["stack_gap"]),
                    "flex": "1 1 auto",
                    "minHeight": "0",
                },
                children=cards,
            ),
        ]

    def _feature_button(
        self,
        feature: str,
        *,
        selected: bool,
        disabled: bool,
    ) -> html.Button:
        style = self._feature_button_style(selected=selected, disabled=disabled)
        return html.Button(
            feature,
            id={"type": "subpath-feature-chip", "feature": feature},
            n_clicks=0,
            disabled=disabled,
            style=style,
        )

    def _feature_button_style(
        self,
        *,
        selected: bool,
        disabled: bool,
    ) -> dict[str, str | float]:
        background = "#e5e7eb"
        color = "#111827"
        border = "1px solid #d1d5db"
        cursor = "pointer"
        opacity = 1.0
        if selected:
            background = "#4f46e5"
            color = "white"
            border = "1px solid #4338ca"
        elif disabled:
            background = "#f3f4f6"
            color = "#9ca3af"
            border = "1px solid #e5e7eb"
            cursor = "not-allowed"
            opacity = 0.7
        return {
            "padding": "6px 10px",
            "borderRadius": "999px",
            "border": border,
            "backgroundColor": background,
            "color": color,
            "fontSize": "12px",
            "cursor": cursor,
            "opacity": opacity,
            "whiteSpace": "nowrap",
        }

    def _render_feature_buttons(
        self,
        selected_features: list[str],
        length: int,
    ) -> list[html.Button]:
        selected_set = set(selected_features)
        limit_reached = len(selected_features) >= int(length)
        return [
            self._feature_button(
                feature,
                selected=(feature in selected_set),
                disabled=(limit_reached and feature not in selected_set),
            )
            for feature in self.feature_names
        ]

    def _build_candidate_fig(self, candidate: SubPathCandidate) -> go.Figure:
        vis_tree = self.subpath_engine.build_candidate_tree(candidate)
        fig = GoTreePlot(
            vis_tree,
            show_text=True,
            show_leaf_hist=True,
            horizontal_spacing=0.12,
            show_edge_labels=False,
            label_hide_depth=6,
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
            Output("subpath-selected-features", "data"),
            [
                Input("subpath-length", "value"),
                Input({"type": "subpath-feature-chip", "feature": ALL}, "n_clicks"),
            ],
            State("subpath-selected-features", "data"),
        )
        def update_selected_features(length: int, _clicks, selected_features):
            depth = int(length or 1)
            selected = [
                feature
                for feature in (selected_features or [])
                if feature in self.feature_names
            ]
            if len(selected) > depth:
                selected = selected[:depth]

            trigger = dash.callback_context.triggered_id
            if isinstance(trigger, dict) and trigger.get("type") == "subpath-feature-chip":
                feature = trigger.get("feature")
                if feature in selected:
                    return [item for item in selected if item != feature]
                if len(selected) >= depth:
                    return selected
                if feature in self.feature_names:
                    return selected + [feature]
            return selected

        @app.callback(
            [
                Output({"type": "subpath-feature-chip", "feature": ALL}, "style"),
                Output({"type": "subpath-feature-chip", "feature": ALL}, "disabled"),
                Output("subpath-feature-help", "children"),
            ],
            [
                Input("subpath-selected-features", "data"),
                Input("subpath-length", "value"),
            ],
        )
        def render_feature_buttons(selected_features, length: int):
            selected = [
                feature
                for feature in (selected_features or [])
                if feature in self.feature_names
            ]
            depth = int(length or 1)
            selected_set = set(selected)
            limit_reached = len(selected) >= depth
            styles = []
            disabled = []
            for feature in self.feature_names:
                is_selected = feature in selected_set
                is_disabled = limit_reached and not is_selected
                styles.append(
                    self._feature_button_style(
                        selected=is_selected,
                        disabled=is_disabled,
                    )
                )
                disabled.append(is_disabled)
            return styles, disabled, self._feature_help_text(
                length=depth,
                selected_features=selected,
            )

        @app.callback(
            Output("subpath-rank-index", "data"),
            [
                Input("subpath-rank-prev", "n_clicks"),
                Input("subpath-rank-top", "n_clicks"),
                Input("subpath-rank-next", "n_clicks"),
                Input("subpath-length", "value"),
                Input("subpath-sort-mode", "value"),
                Input("subpath-selected-features", "data"),
            ],
            State("subpath-rank-index", "data"),
        )
        def update_rank_index(
            prev_clicks: int,
            top_clicks: int,
            next_clicks: int,
            length: int,
            sort_mode_val: str,
            selected_features,
            rank_index: Optional[int],
        ) -> int:
            del prev_clicks, top_clicks, next_clicks
            sort_mode: SortMode = (
                sort_mode_val if sort_mode_val in ("paths", "coverage") else "paths"
            )
            selected = list(selected_features or [])
            current = int(rank_index or 0)
            query = self.subpath_engine.query(
                length=int(length or 1),
                selected_features=selected,
                sort_mode=sort_mode,
                rank_index=current,
            )
            trigger = dash.callback_context.triggered_id
            if trigger in ("subpath-length", "subpath-sort-mode", "subpath-selected-features"):
                return 0
            if query.total <= 0:
                return 0
            max_index = query.total - 1
            if trigger == "subpath-rank-prev":
                return max(0, current - 1)
            if trigger == "subpath-rank-next":
                return min(max_index, current + 1)
            if trigger == "subpath-rank-top":
                return 0
            return max(0, min(current, max_index))

        @app.callback(
            [
                Output("subpath-plot", "figure"),
                Output("subpath-summary", "children"),
                Output("subpath-rule-text", "children"),
                Output("subpath-depth-panel", "children"),
                Output("subpath-rank-label", "children"),
                Output("subpath-rank-prev", "disabled"),
                Output("subpath-rank-top", "disabled"),
                Output("subpath-rank-next", "disabled"),
            ],
            [
                Input("subpath-length", "value"),
                Input("subpath-sort-mode", "value"),
                Input("subpath-selected-features", "data"),
                Input("subpath-rank-index", "data"),
            ],
        )
        def render_subpath(
            length: int,
            sort_mode_val: str,
            selected_features,
            rank_index: Optional[int],
        ):
            sort_mode: SortMode = (
                sort_mode_val if sort_mode_val in ("paths", "coverage") else "paths"
            )
            query = self.subpath_engine.query(
                length=int(length or 1),
                selected_features=list(selected_features or []),
                sort_mode=sort_mode,
                rank_index=int(rank_index or 0),
            )
            if query.selected is None:
                message = query.empty_reason or "No matching path found."
                return (
                    self._empty_fig(message),
                    self._summary_children(None),
                    message,
                    [],
                    "No matching path",
                    True,
                    True,
                    True,
                )

            candidate = query.selected
            rank_text = f"Showing #{query.rank_index + 1} of {query.total}"
            return (
                self._build_candidate_fig(candidate),
                self._summary_children(candidate),
                self.subpath_engine.candidate_rule_text(candidate),
                self._depth_panel_children(candidate),
                rank_text,
                query.rank_index <= 0,
                query.rank_index <= 0,
                query.rank_index >= (query.total - 1),
            )
