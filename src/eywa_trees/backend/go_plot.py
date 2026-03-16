from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import numpy as np

import plotly.graph_objects as go

from eywa_trees.logger import setup_logger
from eywa_trees.backend.vistree import VisTree
from eywa_trees.backend.vis_builders import build_vis_trees_from_model

__all__ = ["GoTreePlot", "build_go_tree_plot"]


class GoTreePlot:
    """Render a decision tree with basic node-edge geometry using Plotly."""

    LABEL_HIDE_DEPTH = 6
    LABEL_DOT_ONLY_DEPTH = 3

    def __init__(
        self,
        vis_tree: VisTree,
        show_text: bool = True,
        highlight_path: Optional[Sequence[int]] = None,
        show_leaf_hist: bool = False,
        horizontal_spacing: float = 1.0,
        show_edge_labels: bool = True,
        label_hide_depth: Optional[int] = None,
        hist_domain: Optional[Tuple[float, float]] = None,
        label_mode: str = "split",
        threshold_precision: int = 2,
        color_domain: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.tree: VisTree = vis_tree
        self.value_label = "Score" if getattr(self.tree, "uses_scores", False) else "Prediction"
        tree_depth = self.tree.max_depth
        depth_for_scale = max(1, tree_depth)
        # Downscale visual footprint as trees get deeper.
        self.size_scale = float(np.clip(8.0 / float(depth_for_scale + 2), 0.35, 1.6))
        self.label_scale = float(np.clip(6.0 / float(depth_for_scale + 3), 0.45, 1.0))
        self.node_label_font_size = int(round(np.clip(12 * self.label_scale, 6, 14)))
        self.edge_label_font_size = int(round(np.clip(11 * self.label_scale, 5, 12)))
        self.layout_font_size = int(round(np.clip(13 * self.label_scale, 8, 16)))
        self.leaf_hist_font_size = int(round(np.clip(9 * self.label_scale, 6, 12)))
        self.label_y_offset = 0.18 + 0.12 * self.label_scale
        # Hide labels for very deep trees to avoid clutter.
        self.show_text: bool = bool(show_text)
        hide_depth = (
            int(label_hide_depth)
            if label_hide_depth is not None
            else self.LABEL_HIDE_DEPTH
        )
        if hide_depth < 0:
            hide_depth = 0
        if tree_depth >= hide_depth:
            self.show_text = False
        self.show_leaf_hist: bool = bool(show_leaf_hist)
        self.horizontal_spacing = float(horizontal_spacing) if horizontal_spacing is not None else 1.0
        if self.horizontal_spacing <= 0:
            self.horizontal_spacing = 1.0
        self.show_edge_labels = bool(show_edge_labels)
        self.label_mode = str(label_mode or "split")
        self.threshold_precision = int(threshold_precision) if threshold_precision is not None else 2
        if self.threshold_precision < 0:
            self.threshold_precision = 0
        self.color_domain: Optional[Tuple[float, float]] = None
        if color_domain is not None:
            try:
                lo, hi = float(color_domain[0]), float(color_domain[1])
                if np.isfinite(lo) and np.isfinite(hi):
                    if hi < lo:
                        lo, hi = hi, lo
                    if np.isclose(lo, hi):
                        hi = lo + 1.0
                    self.color_domain = (lo, hi)
            except Exception:
                self.color_domain = None
        if self.color_domain is not None and not self.tree.is_classifier:
            self.tree.possible_values = {self.color_domain[0], self.color_domain[1]}
            self.tree._generate_color_struct()
        self.hist_domain: Optional[Tuple[float, float]] = None
        if hist_domain is not None:
            try:
                lo, hi = float(hist_domain[0]), float(hist_domain[1])
                if np.isfinite(lo) and np.isfinite(hi):
                    if hi < lo:
                        lo, hi = hi, lo
                    if np.isclose(lo, hi):
                        hi = lo + 1.0
                    self.hist_domain = (lo, hi)
            except Exception:
                self.hist_domain = None

        self.highlight_path: List[int] = list(highlight_path) if highlight_path is not None else []
        self.highlight_nodes = set(self.highlight_path)
        self.highlight_edges = {
            (self.highlight_path[i], self.highlight_path[i + 1])
            for i in range(len(self.highlight_path) - 1)
        }

        self._update_edge_widths()
        positions = self._compute_layout()
        self.fig = self._build_figure(positions)

    def show(self) -> None:
        """Display the plot in a browser window."""
        self.fig.show()

    def save(self, path: str) -> None:
        """Persist the plot to disk."""
        self.fig.write_image(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_edge_widths(self) -> None:
        for node_id, node in self.tree.nodes.items():
            if node.left is not None:
                self.tree.nodes[node.left].parent_edge_width = max(
                    1, int(round(5 * self.size_scale))
                )
            if node.right is not None:
                self.tree.nodes[node.right].parent_edge_width = max(
                    1, int(round(5 * self.size_scale))
                )

    def _compute_layout(self) -> Dict[int, Tuple[float, float]]:
        def _recur(
            node: Any,
            x: float = 0,
            y: float = 0,
            pos: Optional[Dict[int, Tuple[float, float]]] = None,
            level: int = 0,
        ) -> Dict[int, Tuple[float, float]]:
            if pos is None:
                pos = {}
            pos[node.id] = (x, y)
            if node.left is not None:
                pos = _recur(
                    self.tree.nodes[node.left],
                    x - (2 ** (5 - level)) * self.horizontal_spacing,
                    y - 1,
                    pos,
                    level + 1,
                )
            if node.right is not None:
                pos = _recur(
                    self.tree.nodes[node.right],
                    x + (2 ** (5 - level)) * self.horizontal_spacing,
                    y - 1,
                    pos,
                    level + 1,
                )
            return pos

        return _recur(self.tree.nodes[0])

    def _build_leaf_histograms(
        self, positions: Dict[int, Tuple[float, float]]
    ) -> List[go.Bar]:
        traces: List[go.Bar] = []
        width_total = 12.0 * self.size_scale
        height_max = 0.70 * self.size_scale
        gap = 0.10 * self.size_scale

        # Shared min/max across regression histograms for comparability.
        global_min: Optional[float] = None
        global_max: Optional[float] = None
        for node in self.tree.nodes.values():
            hist = getattr(node, "hist", None)
            if not hist or hist.get("type") != "regression":
                continue
            edges = hist.get("bin_edges") or []
            centers = hist.get("centers") or []
            vals: List[float] = []
            for val in edges:
                try:
                    vals.append(float(val))
                except Exception:
                    continue
            for val in centers:
                try:
                    vals.append(float(val))
                except Exception:
                    continue
            if not vals:
                continue
            mn_v, mx_v = min(vals), max(vals)
            global_min = mn_v if global_min is None else min(global_min, mn_v)
            global_max = mx_v if global_max is None else max(global_max, mx_v)

        if self.hist_domain is not None:
            global_min, global_max = self.hist_domain
        else:
            if global_min is None or global_max is None:
                global_min, global_max = 0.0, 1.0
            if not np.isfinite(global_min):
                global_min = 0.0
            if not np.isfinite(global_max):
                global_max = 1.0
            if np.isclose(global_min, global_max):
                global_max = global_min + 1.0

        def _map_val_to_x(x0: float, val: float) -> float:
            frac = (val - global_min) / (global_max - global_min)
            frac = float(np.clip(frac, 0.0, 1.0))
            return x0 - 0.5 * width_total + width_total * frac

        for nid, node in self.tree.nodes.items():
            if node.left is not None or node.right is not None:
                continue
            hist = getattr(node, "hist", None)
            if not (self.show_leaf_hist and hist):
                continue
            if nid not in positions:
                continue

            x0, y0 = positions[nid]
            htype = hist.get("type")
            total = float(hist.get("total", node.n_train))
            base_y = y0 - (height_max + gap)

            if htype == "classification":
                probs = [max(0.0, float(p)) for p in hist.get("probs", [])]
                if not probs:
                    continue
                labels = hist.get("labels") or [f"class {i}" for i in range(len(probs))]
                n_bins = len(probs)
                bin_width = width_total / max(1, n_bins)
                xs = [
                    x0 - 0.5 * width_total + (i + 0.5) * bin_width
                    for i in range(n_bins)
                ]
                heights = [height_max * min(1.0, p) for p in probs]
                colors = []
                if isinstance(self.tree.color_struct, dict):
                    for i in range(n_bins):
                        colors.append(
                            self.tree.color_struct.get(i, "rgba(55,90,160,0.85)")
                        )
                else:
                    colors = ["rgba(55,90,160,0.85)"] * n_bins
                hover = [
                    f"{lbl}: {float(prob):.3f} (n={total:.0f})"
                    for lbl, prob in zip(labels, probs)
                ]
                traces.append(
                    go.Bar(
                        x=xs,
                        y=heights,
                        base=base_y,
                        width=bin_width * 0.85,
                        marker=dict(color=colors),
                        opacity=0.85,
                        hoverinfo="text",
                        hovertext=hover,
                        showlegend=False,
                    )
                )
            elif htype == "regression":
                freqs = [max(0.0, float(f)) for f in hist.get("freq", [])]
                if not freqs:
                    continue
                edges = hist.get("bin_edges") or []
                centers = hist.get("centers") or []
                n_bins = len(freqs)
                heights = [height_max * min(1.0, f) for f in freqs]
                hover: List[str] = []
                xs: List[float] = []
                widths: List[float] = []
                default_width = width_total / max(1, n_bins)
                for i, freq in enumerate(freqs):
                    lo_val = edges[i] if i < len(edges) - 1 else None
                    hi_val = edges[i + 1] if i + 1 < len(edges) else None
                    if lo_val is None or hi_val is None:
                        if centers:
                            center_val = centers[i] if i < len(centers) else centers[-1]
                            lo_val = center_val - 0.5 * (centers[1] - centers[0] if len(centers) > 1 else 1.0)
                            hi_val = center_val + 0.5 * (centers[1] - centers[0] if len(centers) > 1 else 1.0)
                        else:
                            lo_val = global_min + i * (global_max - global_min) / max(1, n_bins)
                            hi_val = global_min + (i + 1) * (global_max - global_min) / max(1, n_bins)
                    lo_x = _map_val_to_x(x0, float(lo_val))
                    hi_x = _map_val_to_x(x0, float(hi_val))
                    cx = 0.5 * (lo_x + hi_x)
                    widths.append(max(0.0, abs(hi_x - lo_x) * 0.9))
                    xs.append(cx)
                    hover.append(
                        f"{float(lo_val):.3f} to {float(hi_val):.3f}: {float(freq):.3f} (n={total:.0f})"
                    )
                traces.append(
                    go.Bar(
                        x=xs,
                        y=heights,
                        base=base_y,
                        width=widths or default_width,
                        marker=dict(color="rgba(214,39,40,0.65)"),
                        opacity=0.75,
                        hoverinfo="text",
                        hovertext=hover,
                        showlegend=False,
                    )
                )
                # Add ticks/labels for shared scale reference.
                tick_vals = np.linspace(global_min, global_max, num=3)
                tick_xs: List[float] = []
                tick_ys: List[float] = []
                for tv in tick_vals:
                    tx = _map_val_to_x(x0, float(tv))
                    tick_xs.extend([tx, tx, None])
                    tick_ys.extend(
                        [
                            base_y - 0.08 * height_max,
                            base_y + 0.08 * height_max,
                            None,
                        ]
                    )
                traces.append(
                    go.Scatter(
                        x=tick_xs,
                        y=tick_ys,
                        mode="lines",
                        line=dict(color="rgba(80,80,80,0.4)", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                label_vals = [global_min, 0.5 * (global_min + global_max), global_max]
                traces.append(
                    go.Scatter(
                        x=[_map_val_to_x(x0, float(v)) for v in label_vals],
                        y=[base_y - 0.15 * height_max] * len(label_vals),
                        mode="text",
                        text=[f"{float(v):.2f}" for v in label_vals],
                        textposition="bottom center",
                        textfont=dict(size=self.leaf_hist_font_size, color="rgba(60,60,60,0.8)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        return traces

    def _build_figure(self, positions: Dict[int, Tuple[float, float]]) -> go.Figure:
        node_ids = list(self.tree.nodes.keys())

        def _threshold_fmt(val: float) -> str:
            return f"{val:.{self.threshold_precision}f}"

        def _node_sign(node_id: int) -> str:
            node = self.tree.nodes[node_id]
            if self.label_mode == "rule" and node.is_left is not None:
                return self.tree.rule_text(node, unicode=True)
            return self.tree.split_text(node, unicode=True)

        if self.show_text:
            labels: List[str] = []
            for node_id in node_ids:
                node = self.tree.nodes[node_id]
                if node.feature is not None and node.threshold is not None:
                    fname = (
                        self.tree.feature_names[node.feature]
                        if self.tree.feature_names
                        else f"Feature {node.feature}"
                    )
                    labels.append(f"{fname} {_node_sign(node_id)} {_threshold_fmt(node.threshold)}")
                else:
                    if self.tree.is_classifier:
                        class_name = (
                            self.tree.class_names[node.value.argmax()]
                            if self.tree.class_names
                            else f"Class {node.value.argmax()}"
                        )
                        labels.append(class_name)
                    else:
                        val = float(node.value) if node.value is not None else 0.0
                        labels.append(f"{val:.2f}")
        else:
            labels = [""] * len(node_ids)

        ys = [pos[1] for pos in positions.values()]
        min_y = min(ys) if ys else 0.0
        positions = {k: (v[0], v[1] - min_y) for k, v in positions.items()}

        x_nodes: List[float] = [positions[nid][0] for nid in node_ids]
        y_nodes: List[float] = [positions[nid][1] for nid in node_ids]
        node_colors: List[str] = []

        edges: List[Dict[str, Any]] = []

        edge_label_x: List[float] = []
        edge_label_y: List[float] = []
        edge_texts: List[str] = []
        node_hover: List[str] = []

        def _is_leaf(node_id: int) -> bool:
            node = self.tree.nodes[node_id]
            return node.left is None and node.right is None

        for nid in node_ids:
            node_colors.append(self.tree.color_for_node(nid))
            node_obj = self.tree.nodes[nid]
            if node_obj.feature is not None and node_obj.threshold is not None:
                fname = (
                    self.tree.feature_names[node_obj.feature]
                    if self.tree.feature_names
                    else f"Feature {node_obj.feature}"
                )
                thr_range = ""
                if hasattr(node_obj, "threshold_min") and hasattr(node_obj, "threshold_max"):
                    try:
                        thr_min = float(getattr(node_obj, "threshold_min"))
                        thr_max = float(getattr(node_obj, "threshold_max"))
                        thr_range = f"<br>thr: ({thr_min:.3f}, {thr_max:.3f})"
                    except Exception:
                        thr_range = ""
                node_hover.append(
                    f"{fname}<br>{_node_sign(nid)} {_threshold_fmt(node_obj.threshold)}"
                    f"{thr_range}"
                    f"<br>coverage: {node_obj.coverage:.3f} ± {node_obj.coverage_std:.3f}"
                    f"<br>n_train: {node_obj.n_train} ± {node_obj.n_train_std:.1f}"
                )
            else:
                if self.tree.is_classifier:
                    cls_idx = int(node_obj.value.argmax())
                    cls_name = (
                        self.tree.class_names[cls_idx]
                        if self.tree.class_names
                        else f"Class {cls_idx}"
                    )
                    node_hover.append(
                        f"{self.value_label}: {cls_name}"
                        f"<br>coverage: {node_obj.coverage:.3f} ± {node_obj.coverage_std:.3f}"
                        f"<br>n_train: {node_obj.n_train} ± {node_obj.n_train_std:.1f}"
                    )
                else:
                    val = float(node_obj.value) if node_obj.value is not None else 0.0
                    node_hover.append(
                        f"{self.value_label}: {val:.4f}"
                        f"<br>coverage: {node_obj.coverage:.3f} ± {node_obj.coverage_std:.3f}"
                        f"<br>n_train: {node_obj.n_train} ± {node_obj.n_train_std:.1f}"
                    )

        base_node_size = max(6, int(16 * self.size_scale))
        base_line_width = max(1, int(2 * self.size_scale))
        highlight_node_size = max(base_node_size + 2, int(round(base_node_size * 1.4)))
        highlight_line_width = max(base_line_width + 1, int(round(base_line_width * 1.6)))

        node_sizes: List[int] = []
        node_line_colors: List[str] = []
        node_line_widths: List[int] = []

        for nid in node_ids:
            has_hist = self.show_leaf_hist and _is_leaf(nid) and getattr(self.tree.nodes[nid], "hist", None)
            if nid in self.highlight_nodes:
                size = highlight_node_size
                line_color = "black"
                line_width = highlight_line_width
            else:
                size = base_node_size
                line_color = "white"
                line_width = base_line_width
            if has_hist:
                size = max(4, int(round(size * 0.7)))
            node_sizes.append(size)
            node_line_colors.append(line_color)
            node_line_widths.append(line_width)

        for node_id, node in self.tree.nodes.items():
            parent_pos = positions[node_id]

            if node.left is not None:
                child_pos = positions[node.left]
                dx = child_pos[0] - parent_pos[0]
                cs = 0.18 * max(1.0, abs(dx))
                mid_x = (parent_pos[0] + child_pos[0]) / 2.0 - cs
                mid_y = (parent_pos[1] + child_pos[1]) / 2.0 + 0.10 * self.size_scale
                edge_color = self.tree.color_for_node(node.left)
                base_w = self.tree.nodes[node.left].parent_edge_width
                if (node_id, node.left) in self.highlight_edges:
                    edge_w = max(base_w + 1, int(round(base_w * 1.6)))
                else:
                    edge_w = base_w
                edges.append(
                    {
                        "x": [parent_pos[0], mid_x, child_pos[0], None],
                        "y": [parent_pos[1], mid_y, child_pos[1], None],
                        "color": edge_color,
                        "width": edge_w,
                    }
                )

                if self.show_text and self.show_edge_labels and node.threshold is not None:
                    edge_label_x.append((parent_pos[0] + child_pos[0]) / 2.0)
                    edge_label_y.append((parent_pos[1] + child_pos[1]) / 2.0)
                    edge_texts.append(
                        f"{self.tree.left_branch_text(node, unicode=True)} {_threshold_fmt(node.threshold)}"
                    )

            if node.right is not None:
                child_pos = positions[node.right]
                dx = child_pos[0] - parent_pos[0]
                cs = 0.18 * max(1.0, abs(dx))
                mid_x = (parent_pos[0] + child_pos[0]) / 2.0 + cs
                mid_y = (parent_pos[1] + child_pos[1]) / 2.0 + 0.10 * self.size_scale
                edge_color = self.tree.color_for_node(node.right)
                base_w = self.tree.nodes[node.right].parent_edge_width
                if (node_id, node.right) in self.highlight_edges:
                    edge_w = max(base_w + 1, int(round(base_w * 1.6)))
                else:
                    edge_w = base_w
                edges.append(
                    {
                        "x": [parent_pos[0], mid_x, child_pos[0], None],
                        "y": [parent_pos[1], mid_y, child_pos[1], None],
                        "color": edge_color,
                        "width": edge_w,
                    }
                )

                if self.show_text and self.show_edge_labels and node.threshold is not None:
                    edge_label_x.append((parent_pos[0] + child_pos[0]) / 2.0)
                    edge_label_y.append((parent_pos[1] + child_pos[1]) / 2.0)
                    edge_texts.append(
                        f"{self.tree.right_branch_text(node, unicode=True)} {_threshold_fmt(node.threshold)}"
                    )

        fig = go.Figure()

        for edge in edges:
            fig.add_trace(
                go.Scatter(
                    x=edge["x"],
                    y=edge["y"],
                    mode="lines",
                    line=dict(color=edge["color"], width=edge["width"], shape="spline"),
                    hoverinfo="none",
                )
            )

        if self.show_leaf_hist:
            for trace in self._build_leaf_histograms(positions):
                fig.add_trace(trace)

        fig.add_trace(
            go.Scatter(
                x=x_nodes,
                y=y_nodes,
                mode="markers",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(color=node_line_colors, width=node_line_widths),
                    opacity=1.0,
                ),
                hoverinfo="text",
                hovertext=node_hover,
            )
        )

        annotations: List[Dict[str, Any]] = []

        if self.show_text:
            for x, y, text in zip(x_nodes, y_nodes, labels):
                annotations.append(
                    dict(
                        x=x,
                        y=y + self.label_y_offset,
                        xref="x",
                        yref="y",
                        text=text,
                        showarrow=False,
                        font=dict(color="black", size=self.node_label_font_size),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="rgba(0,0,0,0)",
                        borderpad=2,
                        opacity=1.0,
                    )
                )

            for x, y, text in zip(edge_label_x, edge_label_y, edge_texts):
                annotations.append(
                    dict(
                        x=x,
                        y=y,
                        xref="x",
                        yref="y",
                        text=text,
                        showarrow=False,
                        font=dict(color="black", size=self.edge_label_font_size),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="rgba(0,0,0,0)",
                        borderpad=1,
                        opacity=1.0,
                    )
                )

        fig.update_layout(
            font_size=self.layout_font_size,
            showlegend=False,
            xaxis=dict(
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                ticks="",
            ),
            yaxis=dict(
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                ticks="",
            ),
            margin=dict(l=40, r=40, b=85, t=100),
            hovermode="closest",
            plot_bgcolor="rgb(255,255,255)",
            annotations=annotations,
        )
        return fig


def build_go_tree_plot(
    model_or_vis_tree: Union[Any, VisTree],
    X: Optional[Any] = None,
    *,
    class_names: Optional[List[str]] = None,
    show_text: bool = True,
    show_leaf_hist: bool = False,
) -> go.Figure:
    """Build a GoTreePlot from a fitted model or an existing VisTree."""
    if isinstance(model_or_vis_tree, VisTree):
        vis_tree = model_or_vis_tree
    else:
        vis_trees = build_vis_trees_from_model(model_or_vis_tree, X, class_names=class_names)
        if not vis_trees:
            raise ValueError("Could not build a VisTree from the provided model.")
        vis_tree = vis_trees[0]
    plot = GoTreePlot(
        vis_tree,
        show_text=show_text,
        show_leaf_hist=show_leaf_hist,
    )
    return plot.fig
