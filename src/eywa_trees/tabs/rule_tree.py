from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import dash
from dash import dcc, html, no_update, dash_table
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from eywa_trees.backend.rule_engine import RuleEngine, RuleEngineConfig, SortMode
from eywa_trees.backend.go_plot import GoTreePlot
from eywa_trees.backend.path_statistics import PathStatistics


@dataclass
class RuleTreeTabConfig:
    top_k_rules: int = 20


class RuleTreeTab:
    _AVERAGE_PREDICTION_MAX_CHARS = 56

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        class_names: Optional[list[str]] = None,
        engine_config: Optional[RuleEngineConfig] = None,
        tab_config: Optional[RuleTreeTabConfig] = None,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.class_names = class_names
        self.tab_config = tab_config or RuleTreeTabConfig()

        stats = PathStatistics(model, X_train, class_names, ecdf_bin_config=None)
        rules_df = stats.rules_dataframe()
        if rules_df.empty:
            raise ValueError("Empty pathway statistics — cannot build rules tab")

        upper_depths = getattr(stats.pset, "features_upper_depth", None)
        if isinstance(upper_depths, np.ndarray) and upper_depths.ndim == 2:
            order = rules_df.index.to_numpy()
            if order.size == upper_depths.shape[0]:
                upper_depths = upper_depths[order]
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
            config=engine_config or RuleEngineConfig(
                top_k_rules=self.tab_config.top_k_rules,
            ),
            upper_depths=upper_depths,
        )
        self.prediction_column_label = self.rule_engine.prediction_column_label()
        self.average_prediction_label = (
            "Average expected score"
            if self.prediction_column_label == "Expected score"
            else "Average prediction"
        )
        (
            self.average_prediction_text,
            self.average_prediction_full_text,
        ) = self._average_prediction_text()

    def _average_prediction_text(self) -> tuple[str, str]:
        root_summary = self.rule_engine.node_summary(self.rule_engine.root_mask())
        pred = root_summary.get("pred")
        if self.rule_engine.is_classification:
            probs = np.asarray(pred, dtype=float)
            if probs.ndim != 1 or probs.size == 0:
                return "n/a", "n/a"
            labels = (
                self.class_names
                if self.class_names is not None and len(self.class_names) == probs.size
                else [str(i) for i in range(probs.size)]
            )
            parts = [
                f"{label}: {prob:.3f}"
                for label, prob in zip(labels, probs.tolist())
            ]
            full_text = ", ".join(parts)
            if len(full_text) <= self._AVERAGE_PREDICTION_MAX_CHARS:
                return full_text, full_text

            ranked_parts = [
                f"{label}: {prob:.3f}"
                for label, prob in sorted(
                    zip(labels, probs.tolist()),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            visible_parts: list[str] = []
            for idx, part in enumerate(ranked_parts):
                candidate = ", ".join(visible_parts + [part])
                hidden = len(ranked_parts) - (idx + 1)
                suffix = f", +{hidden} more" if hidden > 0 else ""
                if len(candidate + suffix) <= self._AVERAGE_PREDICTION_MAX_CHARS:
                    visible_parts.append(part)
                    continue
                if not visible_parts:
                    visible_parts.append(part)
                break

            display_text = ", ".join(visible_parts)
            hidden = len(ranked_parts) - len(visible_parts)
            if hidden > 0:
                suffix = f", +{hidden} more"
                if len(display_text + suffix) <= self._AVERAGE_PREDICTION_MAX_CHARS:
                    display_text += suffix
                else:
                    display_text += ", ..."
            return display_text, full_text
        text = str(root_summary.get("pred_text", "n/a"))
        return text, text

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    @property
    def layout(self) -> html.Div:
        sort_toggle = dcc.RadioItems(
            id="rule-table-sort-mode",
            options=[
                {"label": "count", "value": "paths"},
                {"label": "coverage", "value": "coverage"},
            ],
            value="paths",
            inline=True,
            style={"display": "flex", "alignItems": "center", "gap": "12px"},
        )

        left = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    f"{self.average_prediction_label}:",
                                    style={"fontWeight": "bold", "marginRight": "8px"},
                                ),
                                html.Span(
                                    self.average_prediction_text,
                                    title=self.average_prediction_full_text,
                                    style={
                                        "flex": "1 1 auto",
                                        "minWidth": "0",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "whiteSpace": "nowrap",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "baseline",
                                "minWidth": "0",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Sort rules by:",
                                    style={"fontWeight": "bold", "marginRight": "8px"},
                                ),
                                sort_toggle,
                            ],
                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                        "marginBottom": "10px",
                    },
                ),
                dash_table.DataTable(
                    id="rule-table",
                    columns=[
                        {"name": "Feature", "id": "feature"},
                        {"name": "Rule", "id": "rule"},
                        {"name": self.prediction_column_label, "id": "prediction"},
                    ],
                    data=[],
                    tooltip_data=[],
                    tooltip_duration=None,
                    page_action="none",
                    style_table={"height": "70vh", "overflowY": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    css=[
                        {
                            "selector": ".dash-table-tooltip",
                            "rule": "font-size: 12px; white-space: pre-line;",
                        },
                    ],
                    cell_selectable=True,
                ),
            ],
            style={
                "width": "40%",
                "display": "flex",
                "flexDirection": "column",
                "padding": "16px",
                "borderRight": "1px solid #e5e7eb",
                "backgroundColor": "#fafafa",
            },
        )

        right = html.Div(
            [
                dcc.Store(id="rule-selected-cluster", data=None),
                dcc.Graph(
                    id="rule-tree-plot",
                    style={
                        "height": "60vh",
                        "width": "80%",
                        "maxWidth": "900px",
                        "margin": "0 auto",
                    },
                ),
            ],
            style={"flex": "1", "padding": "16px 20px"},
        )

        return html.Div(
            [left, right],
            style={"display": "flex", "flexDirection": "row"},
            id="rule-tree-tab",
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            [
                Output("rule-table", "data"),
                Output("rule-table", "tooltip_data"),
            ],
            Input("rule-table-sort-mode", "value"),
        )
        def update_table(sort_mode_val: str):
            sort_mode: SortMode = (
                sort_mode_val if sort_mode_val in ("paths", "coverage") else "coverage"
            )

            df = self.rule_engine.candidate_rules(
                mask=self.rule_engine.root_mask(),
                sort_mode=sort_mode,
                top_k=self.tab_config.top_k_rules,
            )
            df2 = df.rename(
                columns={
                    "threshold_text": "rule",
                    "pred_text": "prediction",
                }
            )[
                [
                    "feature",
                    "rule",
                    "prediction",
                    "threshold_min",
                    "threshold_max",
                    "coverage",
                    "coverage_std",
                    "n_rules",
                    "cluster_id",
                    "n_train",
                    "n_train_std",
                ]
            ]
            df2["threshold_min"] = df2["threshold_min"].map(lambda x: f"{x:.3f}")
            df2["threshold_max"] = df2["threshold_max"].map(lambda x: f"{x:.3f}")
            df2["coverage"] = df2["coverage"].map(lambda x: f"{x:.3f}")
            df2["coverage_std"] = df2["coverage_std"].map(lambda x: f"{x:.3f}")
            df2["n_train"] = df2["n_train"].map(lambda x: int(round(x)))
            df2["n_train_std"] = df2["n_train_std"].map(lambda x: f"{x:.2f}")
            df2["n_rules"] = df2["n_rules"].map(int)
            df2["cluster_id"] = df2["cluster_id"].map(int)
            records = df2.to_dict("records")

            tooltip_data = []
            for row in records:
                tip = (
                    f"thr: ({row['threshold_min']}, {row['threshold_max']})\n"
                    f"group_counts: {row['n_rules']}\n"
                    f"coverage: {row['coverage']} ± {row['coverage_std']}"
                )
                tooltip_data.append(
                    {
                        "feature": {"value": tip, "type": "text"},
                        "rule": {"value": tip, "type": "text"},
                        "prediction": {"value": tip, "type": "text"},
                    }
                )

            return records, tooltip_data

        @app.callback(
            Output("rule-table", "style_data_conditional"),
            Input("rule-table", "active_cell"),
        )
        def highlight_row(active_cell):
            if not active_cell:
                return []
            row_idx = active_cell.get("row")
            if row_idx is None:
                return []
            try:
                row_int = int(row_idx)
            except Exception:
                return []
            return [
                {
                    "if": {"row_index": row_int},
                    "backgroundColor": "rgba(99, 102, 241, 0.12)",
                    "border": "1px solid rgba(99, 102, 241, 0.35)",
                }
            ]

        @app.callback(
            Output("rule-selected-cluster", "data"),
            Input("rule-table", "active_cell"),
            State("rule-table", "data"),
        )
        def select_cluster(active_cell, rows):
            if not rows:
                return None
            row = 0
            if active_cell and active_cell.get("row") is not None:
                row = active_cell.get("row")
            if row is None or row < 0 or row >= len(rows):
                row = 0
            try:
                return int(rows[row]["cluster_id"])
            except (KeyError, ValueError, TypeError):
                return None

        @app.callback(
            Output("rule-table", "active_cell"),
            Input("rule-table", "data"),
            State("rule-table", "active_cell"),
        )
        def default_active_cell(data, active_cell):
            if active_cell:
                return active_cell
            if data and len(data) > 0:
                return {"row": 0, "column": 0, "column_id": "feature"}
            return dash.no_update

        @app.callback(
            Output("rule-tree-plot", "figure"),
            Input("rule-selected-cluster", "data"),
        )
        def render_rule_tree(cluster_id: Optional[int]):
            root_mask = self.rule_engine.root_mask()
            root_summary = self.rule_engine.node_summary(root_mask)
            root_n = int(round(root_summary["n_train"]))
            root_pred = root_summary["pred_text"]
            root_cov = root_summary.get("coverage", 0.0)
            root_cov_std = root_summary.get("coverage_std", 0.0)

            if cluster_id is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=(
                        f"ROOT — coverage = {root_cov:.3f} ± {root_cov_std:.3f}, "
                        f"n_train ≈ {root_n}, prediction = {root_pred}"
                    ),
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

            try:
                vis_tree = self.rule_engine.build_cluster_tree(int(cluster_id))
            except Exception:
                return no_update

            go_obj = GoTreePlot(
                vis_tree,
                show_text=True,
                show_leaf_hist=True,
                label_mode="split",
            )
            fig = go_obj.fig
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            depth_hist = self.rule_engine.depth_histogram_for_cluster(int(cluster_id))
            if depth_hist:
                depths = np.asarray(depth_hist.get("depths", []), dtype=int)
                counts = np.asarray(depth_hist.get("counts", []), dtype=float)
                max_depth = depth_hist.get("max_depth", None)
                total = float(depth_hist.get("total", 0))
                if depths.size and counts.size and depths.size == counts.size:
                    probs = counts / total if total > 0 else counts
                    fig.add_trace(
                        go.Bar(
                            x=probs,
                            y=depths,
                            orientation="h",
                            xaxis="x2",
                            yaxis="y2",
                            marker=dict(color="rgba(55,90,160,0.65)"),
                            hovertemplate="Depth %{y}<br>Prob %{x:.3f}<extra></extra>",
                            showlegend=False,
                        )
                    )
                    yaxis2 = dict(
                        domain=[0.70, 0.98],
                        anchor="x2",
                        showgrid=False,
                        zeroline=False,
                        tickmode="linear",
                        dtick=1,
                        ticks="outside",
                        side="right",
                        title=dict(text="", font=dict(size=10)),
                        tickfont=dict(size=9),
                    )
                    if max_depth is not None:
                        yaxis2["range"] = [max_depth, 0]
                    fig.update_layout(
                        xaxis=dict(domain=[0.0, 0.82]),
                        yaxis=dict(domain=[0.0, 1.0]),
                        xaxis2=dict(
                            domain=[0.88, 0.98],
                            anchor="y2",
                            showgrid=False,
                            zeroline=False,
                            autorange="reversed",
                            showticklabels=False,
                            ticks="",
                            tickmode="linear",
                            dtick=0.2,
                            title=dict(text="", font=dict(size=10)),
                            tickfont=dict(size=9),
                        ),
                        yaxis2=yaxis2,
                    )
            return fig
