"""
Sankey plot for decision tree visualization.
Position uses a simplified Sugiyama barycentric layout:

x-columns are equally spaced; y is the weighted barycenter of the children,
with weights equal to the number of samples flowing through the edges.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from eywa_trees.logger import setup_logger
from eywa_trees.backend.vistree import VisTree


LOGGER = setup_logger("api.log")


# DEBUG

def debug_sankey_from_fig(fig: go.Figure, logger: Optional[Any] = None) -> None:
    """Pretty-print Sankey node info as seen by Plotly (feat, thr, pred, flow, x, y, ...)."""
    log = logger or LOGGER
    if not log.isEnabledFor(10):
        return

    trace = fig.data[0]
    node = trace.node
    link = trace.link

    labels = list(node.label or [])
    xs = list(node.x or [])
    ys = list(node.y or [])
    customdata = list(node.customdata or [])

    sources = list(link.source or [])
    targets = list(link.target or [])
    link_values = list(link.value or [])

    def clean_label(raw) -> str:
        return str(raw).replace("<br>", " ").strip()

    lines: List[str] = []
    header = (
        f"{'idx':>3}  {'feat':<8}  {'thr':>8}  "
        f"{'pred':>10}  {'flow_in':>10}  {'flow_out':>10}  "
        f"{'x':>7}  {'y':>7}  {'raw_label':<24}  customdata"
    )
    lines.append("=== DEBUG: Sankey nodes as seen by Plotly ===")
    lines.append(header)

    for i, raw_label in enumerate(labels):
        raw_x = xs[i] if i < len(xs) else None
        raw_y = ys[i] if i < len(ys) else None
        cd = customdata[i] if i < len(customdata) else None

        label_clean = clean_label(raw_label)

        feat = label_clean
        thr_str = ""
        for sym in ("≤", "<=", "<", "≥", ">="):
            if sym in label_clean:
                left, right = label_clean.split(sym, 1)
                feat = left.strip()
                thr_str = right.strip()
                break

        pred_str = ""
        if isinstance(cd, (list, tuple)) and len(cd) >= 2:
            pred_str = str(cd[1])
        elif isinstance(cd, dict) and "value" in cd:
            pred_str = str(cd["value"])

        flow_in = sum(v for s, t, v in zip(sources, targets, link_values) if t == i)
        flow_out = sum(v for s, t, v in zip(sources, targets, link_values) if s == i)

        def fmt_float(x, width, prec):
            try:
                return f"{float(x):{width}.{prec}f}"
            except (TypeError, ValueError):
                return f"{str(x):>{width}}"

        x_str = fmt_float(raw_x, 7, 3)
        y_str = fmt_float(raw_y, 7, 3)
        thr_fmt = fmt_float(thr_str, 8, 3) if thr_str else f"{'':>8}"
        pred_fmt = fmt_float(pred_str, 10, 4) if pred_str else f"{'':>10}"
        flow_in_fmt = fmt_float(flow_in, 10, 4)
        flow_out_fmt = fmt_float(flow_out, 10, 4)

        cd_str = repr(cd)
        if len(cd_str) > 60:
            cd_str = cd_str[:57] + "..."

        lines.append(
            f"{i:3d}  {feat[:8]:<8}  {thr_fmt}  "
            f"{pred_fmt}  {flow_in_fmt}  {flow_out_fmt}  "
            f"{x_str}  {y_str}  {label_clean[:24]:<24}  {cd_str}"
        )

    log.debug("\n".join(lines))





class SankeyTreePlot:
    def __init__(
        self,
        vis_tree: VisTree,
        show_text: bool = True,
        show_label: bool = False,
        margin: float = 0.05,
        gap: float = 0.02,
        dev_budget: float = 0.10,
        dev_decay: float = 0.60,
        pad: float = 0.015,
        nudge: float = 0.01,
        nudge_decay: float = 0.60,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.vis_tree: VisTree = vis_tree
        self.show_text: bool = show_text
        self.show_label: bool = show_label
        self.max_depth: int = vis_tree.max_depth
        self.margin = float(np.clip(margin, 0.0, 0.2))
        self.gap = float(np.clip(gap, 0.0, 0.25))
        self.dev_budget = float(np.clip(dev_budget, 0.0, 0.25))
        self.dev_decay = float(np.clip(dev_decay, 0.0, 0.99))
        self.pad = float(np.clip(pad, 0.0, 0.1))
        self.nudge = float(np.clip(nudge, 0.0, 0.05))
        self.nudge_decay = float(np.clip(nudge_decay, 0.0, 0.99))

        (
            labels,
            colors,
            sources,
            targets,
            values,
            x,
            y,
            customdata,
            edge_labels,
        ) = self._tree_to_sankey_data()
        self.fig = self._create_figure(
            labels, colors, sources, targets, values, x, y, customdata, edge_labels
        )

    def show(self) -> None:
        if self.show_label:
            self._add_color_label()
        self.fig.show()

    def save(self, path: str) -> None:
        if self.show_label:
            self._add_color_label()
        self.fig.write_image(path)

    def _create_figure(
        self,
        labels: List[str],
        colors: List[str],
        sources: List[int],
        targets: List[int],
        values: List[int],
        x: List[float],
        y: List[float],
        customdata: List[List[str]],
        edge_labels: List[str],
    ) -> go.Figure:
        hovertemplate = (
            '<b style="font-size:14px;">%{customdata[0]}</b><br>'
            '<b style="font-size:14px;">n_train:</b> %{value:.0f}<br>'
            '<b style="font-size:14px;">Value:</b> %{customdata[1]}<extra></extra>'
        )
        node_labels = labels if self.show_text else [""] * len(labels)
        sankey = go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=colors,
                x=x,
                y=y,
                customdata=customdata,
                hovertemplate=hovertemplate,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                hovertemplate='<b style="font-size:14px;">n_train:</b> %{value:.0f}<extra></extra>',
                label=edge_labels,
            ),
        )

        fig = go.Figure(data=[sankey])

        # DEBUG: confirm Plotly sees the same x
        debug_sankey_from_fig(fig)

        fig.update_layout(font=dict(family="Courier New", size=14, color="black"))
        return fig

    def _is_zero_sample_leaf(self, nid: int) -> bool:
        n = self.vis_tree.nodes[nid]
        return (n.left is None and n.right is None and int(n.n_train) == 0)

    def _is_effective_leaf(self, nid: int) -> bool:
        n = self.vis_tree.nodes[nid]
        if n.left is None and n.right is None:
            return True
        def mass(cid: Optional[int]) -> int:
            if cid is None:
                return 0
            return int(self.vis_tree.nodes[cid].n_train)
        return (mass(n.left) == 0 and mass(n.right) == 0)

    def _tree_to_sankey_data(
        self,
    ) -> Tuple[
        List[str],
        List[str],
        List[int],
        List[int],
        List[int],
        List[float],
        List[float],
        List[List[str]],
        List[str],
    ]:
        nodes: List[int] = []
        labels: List[str] = []
        colors: List[str] = []
        sources: List[int] = []
        targets: List[int] = []
        values: List[int] = []
        xs: List[float] = []
        ys: List[float] = []
        customdata: List[List[str]] = []
        edge_labels: List[str] = []

        pos, max_depth = self._compute_node_positions()
        x_lo, x_hi = self.margin, 1.0 - self.margin
        x_span = max(1e-9, x_hi - x_lo)
        eps = 1e-6


        def _x_from_depth(d: int, nid: int) -> float:
            v = x_lo + x_span * (d / max(1, max_depth))
            return float(np.clip(v, x_lo, x_hi - eps))



        def _ensure_node(node_id: int, node_name: str) -> int:
            if node_id in nodes:
                return nodes.index(node_id)
            node = self.vis_tree.nodes[node_id]
            label, customlabel = self._get_node_labels(node_id)
            color = self._get_color_for_value(node.value)
            nodes.append(node_id)
            labels.append(label)
            colors.append(color)
            customdata.append(
                [
                    customlabel,
                    (f"{np.round(node.value, 2)}" if self.vis_tree.is_classifier else f"{node.value:.4f}"),
                ]
            )
            depth, yv = pos[node_id]
            xs.append(_x_from_depth(depth, node_id))
            ys.append(yv)
            return len(nodes) - 1

        def _traverse(node_id: int, node_name: str = "A") -> None:
            src_idx = _ensure_node(node_id, node_name)
            n = self.vis_tree.nodes[node_id]

            if n.left is not None and not self._is_zero_sample_leaf(n.left):
                _ensure_node(n.left, f"{node_name}L")
                v = int(self.vis_tree.nodes[n.left].n_train)
                sources.append(src_idx)
                targets.append(nodes.index(n.left))
                values.append(int(v))
                edge_labels.append(str(v))
                _traverse(n.left, f"{node_name}L")

            if n.right is not None and not self._is_zero_sample_leaf(n.right):
                _ensure_node(n.right, f"{node_name}R")
                v = int(self.vis_tree.nodes[n.right].n_train)
                sources.append(src_idx)
                targets.append(nodes.index(n.right))
                values.append(int(v))
                edge_labels.append(str(v))
                _traverse(n.right, f"{node_name}R")

        _traverse(0)

        if self.logger.isEnabledFor(10):
            lines = ["=== DEBUG: node positions for Sankey ==="]
            for node_id, (d, yy) in sorted(pos.items()):
                try:
                    idx = nodes.index(node_id)
                    xv = xs[idx]
                except ValueError:
                    xv = float("nan")
                n = self.vis_tree.nodes[node_id]
                lines.append(
                    f"nid={node_id:4d} depth_for_x={d:2d} "
                    f"x={xv:.3f} y={yy:.3f} "
                    f"is_leaf={n.left is None and n.right is None}"
                )
            self.logger.debug("\n".join(lines))


        if len(colors) < len(labels):
            colors += ["rgba(0, 0, 0, 0.8)"] * (len(labels) - len(colors))

        return labels, colors, sources, targets, values, xs, ys, customdata, edge_labels

    def _get_node_labels(self, node_id: int) -> Tuple[str, str]:
        node = self.vis_tree.nodes[node_id]
        label = ""
        customlabel = ""
        if node.feature is not None and node.threshold is not None:
            fname = (
                self.vis_tree.feature_names[node.feature]
                if self.vis_tree.feature_names is not None
                else f"Feature {node.feature}"
            )
            sign = self.vis_tree.split_text(node, unicode=True)
            label = f"{fname[:6]}<br>{sign} {node.threshold:.2f}"
            customlabel = f"{fname}<br>{sign} {node.threshold:.2f}"
        else:
            if self.vis_tree.is_classifier:
                idx = int(np.argmax(node.value))
                if self.vis_tree.class_names is not None and idx < len(self.vis_tree.class_names):
                    label = self.vis_tree.class_names[idx]
                else:
                    label = f"Class {idx}"
            else:
                label = f"{node.value:.4f}"
        return label, customlabel

    def _get_color_for_value(self, value: Any) -> str:
        return self.vis_tree.color_for_value(value)

    def _add_color_label(self) -> None:
        if self.vis_tree.is_classifier:
            self._add_classification_color_label()
        else:
            self._add_regression_hue_bar()

    def _add_classification_color_label(self) -> None:
        color_struct = self.vis_tree.color_struct
        class_names = self.vis_tree.class_names or []
        labels_with_colors: List[str] = []
        for idx, name in enumerate(class_names):
            color = color_struct[idx]
            labels_with_colors.append(f'{name}: <span style="color:{color};">&#9608;</span>')
        self.fig.add_annotation(text="<br>".join(labels_with_colors), xref="paper", yref="paper", x=1.05, y=1, showarrow=False, align="left")

    def _add_regression_hue_bar(self) -> None:
        color_struct = self.vis_tree.color_struct
        cs = [c[0] for c in color_struct]
        hue_bar = go.Heatmap(
            z=[[c] for c in cs],
            colorscale=[[i / (len(color_struct) - 1), color[1]] for i, color in enumerate(color_struct)],
            showscale=False,
            x=[0.98],
            y=cs,
            xaxis="x2",
            yaxis="y2",
        )
        self.fig.add_trace(hue_bar)
        self.fig.update_layout(
            xaxis2=dict(range=[0, 0.5], domain=[0.75, 1.0], anchor="free", overlaying="x", side="right", position=0.95, visible=False),
            yaxis2=dict(range=[min(cs), max(cs)], domain=[0.40, 0.90], anchor="free", overlaying="y", side="right", position=1, showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
        )


    def _compute_node_positions(self) -> Tuple[Dict[int, Tuple[int, float]], int]:
        tree = self.vis_tree
        layers_raw = tree.get_nodes_depth_list()

        depth_map: Dict[int, int] = {}
        for d, col in enumerate(layers_raw):
            for nid in col:
                depth_map[nid] = d


        if self.logger.isEnabledFor(10):
            lines = ["VisTree depth map:"]
            for nid in sorted(self.vis_tree.nodes):
                n = self.vis_tree.nodes[nid]
                lines.append(
                    f"idx={nid:2d} depth={depth_map.get(nid, -1)} "
                    f"parent={n.parent} left={n.left} right={n.right}"
                )
            self.logger.debug("\n".join(lines))


        max_depth = max(depth_map.values()) if depth_map else 0

        def keep_nid(nid: int) -> bool:
            return not self._is_zero_sample_leaf(nid)

        layers: List[List[int]] = []
        for col in layers_raw:
            pruned = [nid for nid in col if keep_nid(nid)]
            if pruned:
                layers.append(pruned)

        if not layers:
            layers = [col[:] for col in layers_raw if col]
            if not layers:
                layers = [[]]

        margin_lo, margin_hi = self.margin, 1.0 - self.margin

        leaves_all = [
            i
            for i in tree.nodes
            if tree.nodes[i].left is None and tree.nodes[i].right is None
        ]
        leaves = [i for i in leaves_all if not self._is_zero_sample_leaf(i)]
        if not leaves:
            leaves = leaves_all[:]

        def _leaf_order(nid: int, out: List[int]) -> None:
            n = tree.nodes[nid]
            if n.left is None and n.right is None:
                out.append(nid)
                return
            if n.right is not None:
                _leaf_order(n.right, out)
            if n.left is not None:
                _leaf_order(n.left, out)

        leaf_order: List[int] = []
        _leaf_order(0, leaf_order)
        leaf_order = [i for i in leaf_order if i in leaves] or leaves[:]

        W_total = sum(max(1, int(tree.nodes[i].n_train)) for i in leaf_order) or 1
        cum = 0.0
        y_leaf: Dict[int, float] = {}
        for i in leaf_order:
            w = max(1, int(tree.nodes[i].n_train))
            pos = (cum + 0.5 * w) / W_total
            y_leaf[i] = margin_lo + (margin_hi - margin_lo) * pos
            cum += w

        bary: Dict[int, float] = {}
        weight_sum: Dict[int, float] = {}

        def _bary(nid: int) -> Tuple[float, float]:
            if nid in bary:
                return weight_sum[nid], bary[nid]
            n = tree.nodes[nid]
            if self._is_effective_leaf(nid):
                w = max(1, int(n.n_train))
                weight_sum[nid] = float(w)
                bary[nid] = y_leaf.get(nid, 0.5)
                return weight_sum[nid], bary[nid]
            s, t = 0.0, 0.0
            for cid in (n.left, n.right):
                if cid is None or self._is_zero_sample_leaf(cid):
                    continue
                w_c, b_c = _bary(cid)
                s += w_c
                t += w_c * b_c
            weight_sum[nid] = max(1e-6, s)
            bary[nid] = (t / weight_sum[nid]) if s > 0 else 0.5
            return weight_sum[nid], bary[nid]

        _bary(0)

        y_pos: Dict[int, float] = {}
        for depth_idx, col in enumerate(layers):
            col_sorted = sorted(col, key=lambda i: bary.get(i, 0.5))
            desired = [bary.get(i, 0.5) for i in col_sorted]
            lo, hi = margin_lo, margin_hi
            m = len(col_sorted)
            if m == 0:
                continue
            pad_eff = min(self.pad, (hi - lo) / max(1, m) * 0.5)
            ys = desired[:]
            ys[0] = max(ys[0], lo)
            for r in range(1, m):
                ys[r] = max(ys[r], ys[r - 1] + pad_eff)
            ys[-1] = min(ys[-1], hi)
            for r in range(m - 2, -1, -1):
                ys[r] = min(ys[r], ys[r + 1] - pad_eff)
            shift = 0.0
            if ys[0] < lo:
                shift = lo - ys[0]
            if ys[-1] > hi:
                shift = ys[-1] - hi if shift == 0.0 else min(shift, ys[-1] - hi)
            if shift:
                ys = [yy + shift for yy in ys]
            for nid, yy in zip(col_sorted, ys):
                y_pos[nid] = float(np.clip(yy, lo, hi))

        for depth_idx, col in enumerate(layers[1:], start=1):
            dnud = self.nudge * (self.nudge_decay ** (depth_idx - 1))
            if dnud <= 0:
                continue
            col_sorted = sorted(col, key=lambda i: y_pos.get(i, 0.5))
            for nid in col_sorted:
                n = self.vis_tree.nodes[nid]
                if n.parent is None:
                    continue
                if n.is_left is True:
                    y_pos[nid] = min(margin_hi, y_pos[nid] + dnud)
                elif n.is_left is False:
                    y_pos[nid] = max(margin_lo, y_pos[nid] - dnud)
            pad_eff = min(
                self.pad,
                (margin_hi - margin_lo) / max(1, len(col_sorted)) * 0.5,
            )
            ys = [y_pos[i] for i in col_sorted]
            ys[0] = max(ys[0], margin_lo)
            for r in range(1, len(col_sorted)):
                ys[r] = max(ys[r], ys[r - 1] + pad_eff)
            ys[-1] = min(ys[-1], margin_hi)
            for r in range(len(col_sorted) - 2, -1, -1):
                ys[r] = min(ys[r], ys[r + 1] - pad_eff)
            for nid, yy in zip(col_sorted, ys):
                y_pos[nid] = float(np.clip(yy, margin_lo, margin_hi))

        positions: Dict[int, Tuple[int, float]] = {}
        for nid, yy in y_pos.items():
            d_true = depth_map.get(nid, 0)
            positions[nid] = (d_true, yy)

        if self.logger.isEnabledFor(10):
            lines = ["positions:"]
            for nid, (d, yy) in sorted(positions.items()):
                lines.append(f"nid={nid:2d} depth_for_x={d} y={yy:.3f}")
            self.logger.debug("\n".join(lines))

        return positions, max_depth


    # def _compute_node_positions(self) -> Tuple[Dict[int, Tuple[int, float]], int]:
    #     tree = self.vis_tree
    #     layers_raw = tree.get_nodes_depth_list()

    #     def keep_nid(nid: int) -> bool:
    #        return not self._is_zero_sample_leaf(nid)

    #     layers: List[List[int]] = []
    #     for col in layers_raw:
    #         pruned = [nid for nid in col if keep_nid(nid)]
    #         if pruned:
    #             layers.append(pruned)

    #     margin_lo, margin_hi = self.margin, 1.0 - self.margin
    #     D_present = max(1, len(layers) - 1)

    #     leaves_all = [i for i in tree.nodes if tree.nodes[i].left is None and tree.nodes[i].right is None]
    #     leaves = [i for i in leaves_all if not self._is_zero_sample_leaf(i)]
    #     if not leaves:
    #         leaves = leaves_all[:]

    #     def _leaf_order(nid: int, out: List[int]) -> None:
    #         n = tree.nodes[nid]
    #         if n.left is None and n.right is None:
    #             out.append(nid)
    #             return
    #         if n.right is not None:
    #             _leaf_order(n.right, out)
    #         if n.left is not None:
    #             _leaf_order(n.left, out)

    #     leaf_order: List[int] = []
    #     _leaf_order(0, leaf_order)
    #     leaf_order = [i for i in leaf_order if i in leaves] or leaves[:]

    #     W_total = sum(max(1, int(tree.nodes[i].n_train)) for i in leaf_order) or 1
    #     cum = 0.0
    #     y_leaf: Dict[int, float] = {}
    #     for i in leaf_order:
    #         w = max(1, int(tree.nodes[i].n_train))
    #         pos = (cum + 0.5 * w) / W_total
    #         y_leaf[i] = margin_lo + (margin_hi - margin_lo) * pos
    #         cum += w

    #     bary: Dict[int, float] = {}
    #     weight_sum: Dict[int, float] = {}

    #     def _bary(nid: int) -> Tuple[float, float]:
    #         if nid in bary:
    #             return weight_sum[nid], bary[nid]
    #         n = tree.nodes[nid]
    #         if self._is_effective_leaf(nid):
    #             w = max(1, int(n.n_train))
    #             weight_sum[nid] = float(w)
    #             bary[nid] = y_leaf.get(nid, 0.5)
    #             return weight_sum[nid], bary[nid]
    #         s, t = 0.0, 0.0
    #         for cid in (n.left, n.right):
    #             if cid is None or self._is_zero_sample_leaf(cid):
    #                 continue
    #             w_c, b_c = _bary(cid)
    #             s += w_c
    #             t += w_c * b_c
    #         weight_sum[nid] = max(1e-6, s)
    #         bary[nid] = (t / weight_sum[nid]) if s > 0 else 0.5

    #         return weight_sum[nid], bary[nid]

    #     _bary(0)

    #     y_pos: Dict[int, float] = {}
    #     for depth, col in enumerate(layers):
    #         col_sorted = sorted(col, key=lambda i: bary.get(i, 0.5))
    #         desired = [bary.get(i, 0.5) for i in col_sorted]
    #         lo, hi = margin_lo, margin_hi
    #         m = len(col_sorted)
    #         if m == 0:
    #             continue
    #         pad_eff = min(self.pad, (hi - lo) / max(1, m) * 0.5)
    #         ys = desired[:]
    #         ys[0] = max(ys[0], lo)
    #         for r in range(1, m):
    #             ys[r] = max(ys[r], ys[r - 1] + pad_eff)
    #         ys[-1] = min(ys[-1], hi)
    #         for r in range(m - 2, -1, -1):
    #             ys[r] = min(ys[r], ys[r + 1] - pad_eff)
    #         shift = 0.0
    #         if ys[0] < lo:
    #             shift = lo - ys[0]
    #         if ys[-1] > hi:
    #             shift = ys[-1] - hi if shift == 0.0 else min(shift, ys[-1] - hi)
    #         if shift:
    #             ys = [yy + shift for yy in ys]
    #         for nid, yy in zip(col_sorted, ys):
    #             y_pos[nid] = float(np.clip(yy, lo, hi))

    #     for depth, col in enumerate(layers[1:], start=1):
    #         dnud = self.nudge * (self.nudge_decay ** (depth - 1))
    #         if dnud <= 0:
    #             continue
    #         col_sorted = sorted(col, key=lambda i: y_pos.get(i, 0.5))
    #         for nid in col_sorted:
    #             n = self.vis_tree.nodes[nid]
    #             if n.parent is None:
    #                 continue
    #             if n.is_left is True:
    #                 y_pos[nid] = min(margin_hi, y_pos[nid] + dnud)
    #             elif n.is_left is False:
    #                 y_pos[nid] = max(margin_lo, y_pos[nid] - dnud)
    #         pad_eff = min(self.pad, (margin_hi - margin_lo) / max(1, len(col_sorted)) * 0.5)
    #         ys = [y_pos[i] for i in col_sorted]
    #         ys[0] = max(ys[0], margin_lo)
    #         for r in range(1, len(col_sorted)):
    #             ys[r] = max(ys[r], ys[r - 1] + pad_eff)
    #         ys[-1] = min(ys[-1], margin_hi)
    #         for r in range(len(col_sorted) - 2, -1, -1):
    #             ys[r] = min(ys[r], ys[r + 1] - pad_eff)
    #         for nid, yy in zip(col_sorted, ys):
    #             y_pos[nid] = float(np.clip(yy, margin_lo, margin_hi))

    #     positions: Dict[int, Tuple[int, float]] = {}
    #     for depth, col in enumerate(layers):
    #         for nid in col:
    #             positions[nid] = (depth, y_pos.get(nid, 0.5))

    #     for nid, (d, yy) in list(positions.items()):
    #         n = self.vis_tree.nodes[nid]
    #         if (n.left is not None) or (n.right is not None):
    #             if d >= D_present:
    #                 positions[nid] = (max(0, D_present - 1), yy)

    #     # Debug diagnostics for layout issues
    #     zero_sample_leaves = [i for i in leaves_all if self._is_zero_sample_leaf(i)]
    #     max_depth_raw = len(layers_raw) - 1
    #     max_depth_present = len(layers) - 1
    #     print(
    #         f"layout debug | raw_depth={max_depth_raw} present_depth={max_depth_present} "
    #         f"D_present={D_present} raw_layers={len(layers_raw)} filtered_layers={len(layers)} "
    #         f"zero_sample_leaves={len(zero_sample_leaves)}"
    #     )

    #     return positions, D_present


class GoTreePlot:
    def __init__(self, vis_tree: VisTree, show_text: bool = True) -> None:
        self.logger = setup_logger("api.log")
        self.tree: VisTree = vis_tree
        self.show_text: bool = show_text
        self._update_edge_widths()
        positions = self._compute_layout()
        self.fig = self._build_figure(positions)

    def show(self) -> None:
        self.fig.show()

    def save(self, path: str) -> None:
        self.fig.write_image(path)

    def _update_edge_widths(self) -> None:
        for node_id, node in self.tree.nodes.items():
            if node.left is not None:
                self.tree.nodes[node.left].parent_edge_width = 5
            if node.right is not None:
                self.tree.nodes[node.right].parent_edge_width = 5

    def _compute_layout(self) -> Dict[int, Tuple[float, float]]:
        def _recur(
            node: Any, x: float = 0, y: float = 0,
            pos: Optional[Dict[int, Tuple[float, float]]] = None,
            level: int = 0
        ) -> Dict[int, Tuple[float, float]]:
            if pos is None:
                pos = {}
            pos[node.id] = (x, y)
            if node.left is not None:
                pos = _recur(self.tree.nodes[node.left], x - (2 ** (5 - level)), y - 1, pos, level + 1)
            if node.right is not None:
                pos = _recur(self.tree.nodes[node.right], x + (2 ** (5 - level)), y - 1, pos, level + 1)
            return pos
        return _recur(self.tree.nodes[0])

    def _build_figure(self, positions: Dict[int, Tuple[float, float]]) -> go.Figure:
        node_ids = list(self.tree.nodes.keys())
        labels: List[str] = []
        if self.show_text:
            for node_id in node_ids:
                n = self.tree.nodes[node_id]
                if n.feature is not None and n.threshold is not None:
                    fname = self.tree.feature_names[n.feature] if self.tree.feature_names else f"Feature {n.feature}"
                    sign = self.tree.split_text(n, unicode=True)
                    labels.append(f"{fname}<br>{sign} {n.threshold:.2f}")
                else:
                    if self.tree.is_classifier:
                        cname = self.tree.class_names[n.value.argmax()] if self.tree.class_names else f"Class {n.value.argmax()}"
                        labels.append(cname)
                    else:
                        val = float(n.value) if n.value is not None else 0.0
                        labels.append(f"{val:.2f}")
        else:
            labels = [""] * len(node_ids)

        ys = [pos[1] for pos in positions.values()]
        min_y = min(ys) if ys else 0.0
        positions = {k: (v[0], v[1] - min_y) for k, v in positions.items()}

        xe: List[float] = []
        ye: List[float] = []
        edge_colors: List[str] = []
        edge_widths: List[int] = []

        def _edge_color(parent_id: int) -> str:
            parent = self.tree.nodes[parent_id]
            if self.tree.is_classifier:
                return self.tree.color_struct[int(parent.value.argmax())]
            val = float(parent.value) if parent.value is not None else 0.0
            cs = self.tree.color_struct
            if not cs:
                return "rgba(0,0,0,0.8)"
            for i in range(len(cs) - 1):
                if cs[i][0] <= val < cs[i + 1][0]:
                    return cs[i][1]
            return cs[-1][1]

        for nid, n in self.tree.nodes.items():
            if n.left is not None:
                p, c = positions[nid], positions[n.left]
                xe += [p[0], c[0], None]
                ye += [p[1], c[1], None]
                edge_colors.append(_edge_color(nid))
                edge_widths.append(self.tree.nodes[n.left].parent_edge_width)
            if n.right is not None:
                p, c = positions[nid], positions[n.right]
                xe += [p[0], c[0], None]
                ye += [p[1], c[1], None]
                edge_colors.append(_edge_color(nid))
                edge_widths.append(self.tree.nodes[n.right].parent_edge_width)

        def _make_annotations(
            pos: Dict[int, Tuple[float, float]],
            text: Optional[List[str]] = None,
            font_size: int = 10,
            font_color: str = "rgb(250,250,250)",
        ) -> List[Dict[str, Any]]:
            keys = list(pos.keys())
            T = text or []
            if len(T) < len(keys):
                T = T + [""] * (len(keys) - len(T))
            return [
                dict(
                    text=str(T[i]),
                    x=pos[k][0],
                    y=pos[k][1],
                    xref="x1",
                    yref="y1",
                    font=dict(color=font_color, size=font_size),
                    showarrow=False,
                    bgcolor="rgb(31, 119, 180)",
                    opacity=1,
                )
                for i, k in enumerate(keys)
            ]

        fig = go.Figure()

        for i_edge in range(len(xe) // 3):
            fig.add_trace(
                go.Scatter(
                    x=[xe[3 * i_edge], xe[3 * i_edge + 1], None],
                    y=[ye[3 * i_edge], ye[3 * i_edge + 1], None],
                    mode="lines",
                    line=dict(color=edge_colors[i_edge], width=edge_widths[i_edge]),
                    hoverinfo="none",
                )
            )

        if self.show_text:
            annotations = _make_annotations(positions, labels)
            fig.update_layout(annotations=annotations)

        fig.update_layout(
            font_size=12,
            showlegend=False,
            xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, ticks=""),
            yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, ticks=""),
            margin=dict(l=40, r=40, b=85, t=100),
            hovermode="closest",
            plot_bgcolor="rgb(255,255,255)",
        )
        return fig


if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from eywa_trees.backend.vis_builders import build_vis_trees_from_model

    np.random.seed(2)
    n = 300
    p = 6
    X = np.random.rand(n, p)
    y = np.random.randint(0, 3, size=n)
    class_names = ["Class 0", "Class 1", "Class 2"]

    rf = RandomForestClassifier(n_estimators=1, max_depth=6, random_state=0)
    rf.fit(X, y)

    vis_tree = build_vis_trees_from_model(rf, X, class_names=class_names)[0]

    sankey_plot = SankeyTreePlot(vis_tree, show_text=False, show_label=False)
    sankey_plot.show()

    go_plot = GoTreePlot(vis_tree, show_text=True)
    go_plot.show()
