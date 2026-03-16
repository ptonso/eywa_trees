from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union, Set

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.base import ClassifierMixin
from sklearn.tree import _tree

from eywa_trees.logger import setup_logger
from eywa_trees.utils import _fmt


ArrayLike = Union[pd.DataFrame, np.ndarray]


@dataclass
class VisNode:
    id: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    value: Optional[Union[np.ndarray, float]] = None
    cover: float = 0.0
    parent: Optional[int] = None
    is_left: Optional[bool] = None
    left: Optional[int] = None
    right: Optional[int] = None
    n_train: int = 0
    hist: Optional[Dict[str, Any]] = None
    coverage: float = 0.0
    coverage_std: float = 0.0
    n_train_std: float = 0.0
    split_operator: str = "<="
    missing_to: Optional[str] = None
    external_id: Optional[Any] = None


@dataclass
class VisTree:
    """
    Data container for a single decision tree used across plots/tabs.
    Building the structure is delegated to builder helpers (see vis_builders.py).
    """

    model: Any
    feature_names: Optional[List[str]] = None
    class_names: Optional[List[str]] = None
    is_classifier: bool = False
    uses_scores: bool = False
    log_coloring: bool = False
    learning_rate: float = 1.0
    base_score: float = 0.0
    nodes: Dict[int, VisNode] = field(default_factory=dict)
    n_classes: int = 0
    n_train: int = 0
    max_depth: int = 0
    possible_values: Set[float] = field(default_factory=set)
    leaf_paths: Dict[int, List[int]] = field(default_factory=dict)
    color_struct: Any = field(default_factory=dict)
    external_to_internal: Dict[Any, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def reset_struct(self) -> None:
        self.nodes = {}
        self.possible_values = set()
        self.leaf_paths = {}
        self.max_depth = 0
        self.external_to_internal = {}

    def ingest_tree_struct(
        self,
        tree: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        is_classifier: Optional[bool] = None,
        uses_scores: Optional[bool] = None,
        learning_rate: Optional[float] = None,
        base_score: Optional[float] = None,
    ) -> None:
        """
        Populate the VisTree from a tree-like struct exposing children_left/right,
        feature, threshold, value, and n_node_samples.
        """
        self.reset_struct()
        if feature_names is not None:
            self.feature_names = list(feature_names)
        if class_names is not None:
            self.class_names = list(class_names)
            self.n_classes = len(self.class_names)

        if is_classifier is not None:
            self.is_classifier = is_classifier
        if uses_scores is not None:
            self.uses_scores = uses_scores
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if base_score is not None:
            self.base_score = float(base_score)

        def walk(idx: int, parent: Optional[int], is_left: Optional[bool], depth: int) -> Optional[int]:
            if idx is None or idx < 0:
                return None

            val: Union[np.ndarray, float]
            if self.is_classifier:
                raw = getattr(tree, "value", None)
                arr = raw[idx] if raw is not None else None
                val = arr.flatten() if isinstance(arr, np.ndarray) else np.array([])
                if self.n_classes == 0 and isinstance(val, np.ndarray):
                    self.n_classes = val.shape[-1] if val.ndim > 0 else 0
            else:
                raw = getattr(tree, "value", None)
                if isinstance(raw, np.ndarray) and raw.size:
                    arr = raw[idx]
                    val = float(arr.flatten()[0])
                else:
                    val = 0.0
                self.possible_values.add(val)

            feat_idx_raw = tree.feature[idx] if getattr(tree, "feature", None) is not None else None
            feat = int(feat_idx_raw) if feat_idx_raw is not None and feat_idx_raw >= 0 else None
            thr_raw = tree.threshold[idx] if getattr(tree, "threshold", None) is not None else None
            thr = None
            if feat is not None and thr_raw is not None and not np.isnan(thr_raw):
                thr = float(thr_raw)

            split_operator = self._struct_attr_at(tree, "split_operator", idx, "<=")
            if split_operator not in {"<=", "<"}:
                split_operator = "<="
            missing_to = self._struct_attr_at(tree, "missing_child", idx, None)
            if missing_to not in {"left", "right", None}:
                missing_to = None
            external_id = self._struct_attr_at(tree, "raw_node_ids", idx, idx)
            cover_raw = self._struct_attr_at(tree, "booster_cover", idx, None)
            if cover_raw is None:
                cover_raw = self._struct_attr_at(tree, "cover", idx, None)
            if cover_raw is None:
                cover_raw = self._struct_attr_at(tree, "n_node_samples", idx, 0.0)
            try:
                cover = 0.0 if cover_raw is None or np.isnan(cover_raw) else float(cover_raw)
            except Exception:
                cover = 0.0

            self._add_node(
                idx,
                feat,
                thr,
                val,
                cover,
                parent,
                is_left,
                split_operator=split_operator,
                missing_to=missing_to,
                external_id=external_id,
            )
            self.max_depth = max(self.max_depth, depth)

            l = tree.children_left[idx] if getattr(tree, "children_left", None) is not None else _tree.TREE_LEAF
            r = tree.children_right[idx] if getattr(tree, "children_right", None) is not None else _tree.TREE_LEAF
            # If a node is missing split info, treat it as a leaf.
            if feat is None or thr is None:
                lnode = None
                rnode = None
            else:
                lnode = walk(int(l), idx, True, depth + 1) if l is not None and int(l) != _tree.TREE_LEAF and int(l) >= 0 else None
                rnode = walk(int(r), idx, False, depth + 1) if r is not None and int(r) != _tree.TREE_LEAF and int(r) >= 0 else None
            self.nodes[idx].left, self.nodes[idx].right = lnode, rnode
            return idx

        walk(0, None, None, 0)
        self._build_leaf_paths()
        self._rebuild_external_mapping()
        self._generate_color_struct()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_color_struct(self, n: int = 100, opacity: float = 0.8) -> None:
        if self.is_classifier:
            palette = px.colors.qualitative.Plotly
            count = max(1, int(getattr(self, "n_classes", 1)))
            self.color_struct = {i: palette[i % len(palette)] for i in range(count)}
        else:
            if not self.possible_values:
                self.color_struct = []
                return
            mn, mx = min(self.possible_values), max(self.possible_values)
            ticks = np.linspace(mn, mx, n)
            colors = px.colors.sample_colorscale("Viridis", n)
            self.color_struct = [
                (ticks[i], colors[i].replace("rgb", "rgba").replace(")", f", {opacity})"))
                for i in range(n)
            ]

    def color_for_value(self, value: Any) -> str:
        """
        Central color lookup so all plots share the same mapping.
        """
        if self.is_classifier:
            try:
                idx = int(np.argmax(value))
                return self.color_struct.get(idx, "rgba(0,0,0,0.8)")  # type: ignore[arg-type]
            except Exception:
                return "rgba(0,0,0,0.8)"
        cs = self.color_struct
        if not cs:
            return "rgba(0,0,0,0.8)"
        try:
            val_f = float(value)
        except Exception:
            return cs[-1][1]
        for i in range(len(cs) - 1):
            if cs[i][0] <= val_f < cs[i + 1][0]:
                return cs[i][1]
        return cs[-1][1]

    def color_for_node(self, node_id: int) -> str:
        node = self.nodes.get(node_id)
        if node is None:
            return "rgba(0,0,0,0.8)"
        return self.color_for_value(node.value)

    def _add_node(
        self,
        id: int,
        feature: Optional[int],
        threshold: Optional[float],
        value: Union[np.ndarray, float],
        cover: float,
        parent: Optional[int],
        is_left: Optional[bool],
        *,
        split_operator: str = "<=",
        missing_to: Optional[str] = None,
        external_id: Optional[Any] = None,
    ) -> None:
        node = VisNode(
            id=id,
            feature=feature,
            threshold=threshold,
            value=value,
            cover=cover,
            parent=parent,
            is_left=is_left,
            split_operator=split_operator,
            missing_to=missing_to,
            external_id=external_id if external_id is not None else id,
        )
        self.nodes[id] = node
        if node.external_id is not None:
            self.external_to_internal[node.external_id] = id

    @staticmethod
    def _struct_attr_at(tree: Any, attr_name: str, idx: int, default: Any) -> Any:
        raw = getattr(tree, attr_name, None)
        if raw is None:
            return default
        try:
            return raw[idx]
        except Exception:
            return default

    def get_internal_node_id(self, external_id: Any) -> Optional[int]:
        if external_id in self.external_to_internal:
            return int(self.external_to_internal[external_id])
        try:
            ext_int = int(external_id)
        except Exception:
            ext_int = None
        if ext_int is not None and ext_int in self.external_to_internal:
            return int(self.external_to_internal[ext_int])
        if isinstance(external_id, (int, np.integer)) and int(external_id) in self.nodes:
            return int(external_id)
        return None

    def split_text(self, node_or_id: Union[int, VisNode], *, unicode: bool = True) -> str:
        node = self.nodes[node_or_id] if isinstance(node_or_id, (int, np.integer)) else node_or_id
        if node.split_operator == "<":
            return "<"
        return "≤" if unicode else "<="

    def left_branch_text(self, node_or_id: Union[int, VisNode], *, unicode: bool = True) -> str:
        return self.split_text(node_or_id, unicode=unicode)

    def right_branch_text(self, node_or_id: Union[int, VisNode], *, unicode: bool = True) -> str:
        node = self.nodes[node_or_id] if isinstance(node_or_id, (int, np.integer)) else node_or_id
        if node.split_operator == "<":
            return ">="
        return ">" if not unicode else ">"

    def rule_text(self, node_or_id: Union[int, VisNode], *, unicode: bool = True) -> str:
        node = self.nodes[node_or_id] if isinstance(node_or_id, (int, np.integer)) else node_or_id
        if node.is_left is False:
            return self.right_branch_text(node, unicode=unicode)
        return self.left_branch_text(node, unicode=unicode)

    def _split_masks(
        self,
        node: VisNode,
        samples: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if node.feature is None or node.threshold is None:
            return mask.copy(), np.zeros_like(mask, dtype=bool)
        feature_vals = samples[:, node.feature]
        missing_mask = mask & pd.isna(feature_vals)
        active_mask = mask & ~missing_mask
        left_numeric = np.zeros_like(mask, dtype=bool)
        if np.any(active_mask):
            active_vals = feature_vals[active_mask]
            if node.split_operator == "<":
                left_numeric[active_mask] = active_vals < node.threshold
            else:
                left_numeric[active_mask] = active_vals <= node.threshold
        left_mask = active_mask & left_numeric
        right_mask = active_mask & ~left_numeric
        if node.missing_to == "left":
            left_mask |= missing_mask
        elif node.missing_to == "right":
            right_mask |= missing_mask
        return left_mask, right_mask

    def update_max_depth(self, nid: int = 0, depth: int = 0) -> None:
        node = self.nodes.get(nid)
        if node is None:
            return
        self.max_depth = max(self.max_depth, depth)
        if node.left is not None:
            self.update_max_depth(node.left, depth + 1)
        if node.right is not None:
            self.update_max_depth(node.right, depth + 1)

    def _build_leaf_paths(self) -> None:
        self.leaf_paths = {}
        for idx, node in self.nodes.items():
            if node.left is None and node.right is None:
                path: List[int] = []
                cur = node
                while True:
                    path.append(cur.id)
                    if cur.parent is None:
                        break
                    cur = self.nodes[cur.parent]
                path.reverse()
                self.leaf_paths[idx] = path

    def get_nodes_depth_list(self) -> List[List[int]]:
        depth = {0: 0}
        visited = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            n = self.nodes[u]
            for v in (n.left, n.right):
                if v is None or v in visited:
                    continue
                depth[v] = depth[u] + 1
                visited.add(v)
                stack.append(v)
        max_depth = max(depth.values()) if depth else 0
        layers: List[List[int]] = [[] for _ in range(max_depth + 1)]
        queue = [0]
        seen = set()
        while queue:
            u = queue.pop(0)
            if u in seen:
                continue
            seen.add(u)
            d = depth.get(u, 0)
            layers[d].append(u)
            n = self.nodes[u]
            if n.left is not None:
                queue.append(n.left)
            if n.right is not None:
                queue.append(n.right)
        self.max_depth = max_depth
        return layers

    # ------------------------------------------------------------------
    # Central pruning operator
    # ------------------------------------------------------------------
    def _cut_subtree(self, idx: int, keep_root: bool = False) -> None:
        """
        Delete the subtree rooted at *idx*.
        """
        if idx not in self.nodes:
            return

        root = self.nodes[idx]

        to_delete: List[int] = []
        stack = [idx]
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            for cid in (node.left, node.right):
                if cid is not None:
                    stack.append(cid)
            if not (keep_root and nid == idx):
                to_delete.append(nid)

        if keep_root:
            root.left = None
            root.right = None
        else:
            parent_id = root.parent
            if parent_id is not None and parent_id in self.nodes:
                p = self.nodes[parent_id]
                if p.left == idx:
                    p.left = None
                if p.right == idx:
                    p.right = None
                if p.left is None and p.right is None:
                    p.feature = None
                    p.threshold = None

        for nid in to_delete:
            if nid in self.nodes:
                del self.nodes[nid]

    # ------------------------------------------------------------------
    # Prediction / stats
    # ------------------------------------------------------------------
    def predict(self, X: ArrayLike, predict_probas: bool = False) -> np.ndarray:
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        n = arr.shape[0]
        if self.is_classifier:
            y = np.zeros((n, self.n_classes)) if predict_probas else np.zeros(n, int)
        else:
            y = np.zeros(n)

        def walk(idx: int, mask: np.ndarray) -> None:
            node = self.nodes[idx]
            if node.left is None and node.right is None or node.feature is None or node.threshold is None:
                if self.is_classifier:
                    if predict_probas:
                        denom = node.value.sum() if isinstance(node.value, np.ndarray) else 0.0
                        y[mask] = node.value / denom if denom else 0.0
                    else:
                        y[mask] = int(np.argmax(node.value))
                else:
                    y[mask] = float(node.value) if node.value is not None else 0.0
                return
            lm, rm = self._split_masks(node, arr, mask)
            if node.left is not None:
                walk(node.left, lm)
            if node.right is not None:
                walk(node.right, rm)

        walk(0, np.ones(n, bool))
        return y

    def populate_ns(self, samples: np.ndarray) -> None:
        self.n_train = samples.shape[0]

        def walk(idx: int, mask: np.ndarray) -> None:
            node = self.nodes[idx]
            if node.left is None and node.right is None or node.feature is None or node.threshold is None:
                node.n_train = int(mask.sum())
                return
            lm, rm = self._split_masks(node, samples, mask)
            if node.left is not None:
                walk(node.left, lm)
            if node.right is not None:
                walk(node.right, rm)
            node.n_train = int(mask.sum())

        walk(0, np.ones(self.n_train, bool))

    def prune(self, max_depth: int) -> "VisTree":
        """
        Return a deep-copy pruned to *max_depth* (keeps API identical to original).
        """
        from copy import deepcopy

        clone = deepcopy(self)
        layers = clone.get_nodes_depth_list()
        for d in range(max_depth + 1, len(layers)):
            for idx in layers[d]:
                clone._cut_subtree(idx, keep_root=False)
        clone.update_max_depth()
        clone._build_leaf_paths()
        clone._rebuild_external_mapping()
        return clone

    def propagate_values(self, consider_proba: bool = True) -> None:
        if self.is_classifier:
            self._propagate_class(consider_proba)
        else:
            self._propagate_reg()

    def _propagate_class(self, consider_proba: bool) -> None:
        layers = self.get_nodes_depth_list()
        for depth in range(self.max_depth, -1, -1):
            for idx in layers[depth]:
                node = self.nodes[idx]
                if node.left is None and node.right is None:
                    continue
                accum = np.zeros_like(node.value)
                for cid in (node.left, node.right):
                    if cid is None:
                        continue
                    child = self.nodes[cid]
                    if consider_proba:
                        accum += child.value * child.n_train
                    else:
                        accum[int(np.argmax(child.value))] += child.n_train
                node.value = accum / node.n_train if node.n_train else accum

    def _propagate_reg(self) -> None:
        layers = self.get_nodes_depth_list()
        for depth in range(self.max_depth, -1, -1):
            for idx in layers[depth]:
                node = self.nodes[idx]
                if node.left is None and node.right is None:
                    continue
                tot, cnt = 0.0, 0
                for cid in (node.left, node.right):
                    if cid is None:
                        continue
                    child = self.nodes[cid]
                    if child.value is None:
                        continue
                    tot += float(child.value) * child.n_train
                    cnt += child.n_train
                node.value = tot / cnt if cnt else tot
                self.possible_values.add(float(node.value) if node.value is not None else 0.0)
        self._generate_color_struct()

    def leaf_path(self, leaf_id: int) -> List[int]:
        return self.leaf_paths.get(leaf_id, [])

    def _rebuild_external_mapping(self) -> None:
        self.external_to_internal = {}
        for idx, node in self.nodes.items():
            if node.external_id is not None:
                self.external_to_internal[node.external_id] = idx

    def _build_path(self, leaf_id: int) -> Dict[str, Any]:
        conds: Dict[str, Any] = {}

        # init bounds
        if self.feature_names:
            for f in self.feature_names:
                conds[f"{f}_upper"] = np.inf
                conds[f"{f}_lower"] = -np.inf
                conds[f"{f}_upper_inclusive"] = False
                conds[f"{f}_lower_inclusive"] = False
        else:
            mx = max(
                (n.feature for n in self.nodes.values() if n.feature is not None),
                default=-1,
            )
            for i in range(mx + 1):
                conds[f"{i}_upper"] = np.inf
                conds[f"{i}_lower"] = -np.inf
                conds[f"{i}_upper_inclusive"] = False
                conds[f"{i}_lower_inclusive"] = False

        path = self.leaf_path(leaf_id)

        for nid in path[1:]:
            node = self.nodes[nid]
            if node.parent is None:
                continue
            parent = self.nodes[node.parent]
            if parent.threshold is None or parent.feature is None:
                continue
            fname = self.feature_names[parent.feature] if self.feature_names else str(parent.feature)
            if node.is_left:
                cur = float(conds[f"{fname}_upper"])
                new_val = float(parent.threshold)
                new_inclusive = parent.split_operator != "<"
                if new_val < cur:
                    conds[f"{fname}_upper"] = new_val
                    conds[f"{fname}_upper_inclusive"] = new_inclusive
                elif np.isclose(new_val, cur):
                    conds[f"{fname}_upper_inclusive"] = bool(conds[f"{fname}_upper_inclusive"]) and bool(new_inclusive)
            else:
                cur = float(conds[f"{fname}_lower"])
                new_val = float(parent.threshold)
                new_inclusive = parent.split_operator == "<"
                if new_val > cur:
                    conds[f"{fname}_lower"] = new_val
                    conds[f"{fname}_lower_inclusive"] = new_inclusive
                elif np.isclose(new_val, cur):
                    conds[f"{fname}_lower_inclusive"] = bool(conds[f"{fname}_lower_inclusive"]) and bool(new_inclusive)

        root_id = path[0] if path else 0
        root = self.nodes[root_id]
        importance = root.n_train / self.n_train if self.n_train else 0.0
        conds["path_importance"] = importance
        conds["value"] = self.nodes[leaf_id].value
        return conds

    def extract_df(self, tree_id: int = 0) -> pd.DataFrame:
        rows = [
            dict(path_id=i, **self._build_path(idx), tree_id=tree_id)
            for i, idx in enumerate(self.nodes)
            if self.nodes[idx].left is None and self.nodes[idx].right is None
        ]
        return pd.DataFrame(rows)

    def print_tree(self) -> None:
        def walk(idx: int, depth: int, tag: str, parent_id: int | None) -> None:
            ind = "  " * depth
            n = self.nodes[idx]

            is_leaf = n.left is None and n.right is None
            parent_str = "None" if parent_id is None else str(parent_id)
            val_str = _fmt(n.value)

            if is_leaf:
                print(
                    f"{ind}{tag} Leaf {idx} (parent={parent_str}) "
                    f"| value={val_str} | n_train={n.n_train}"
                )
            else:
                print(
                    f"{ind}{tag} Node {idx} (parent={parent_str}) "
                    f"| value={val_str} | n_train={n.n_train}"
                )

            if n.feature is not None:
                if self.feature_names and 0 <= n.feature < len(self.feature_names):
                    fname = self.feature_names[n.feature]
                else:
                    fname = str(n.feature)
                thr_str = _fmt(n.threshold)
                print(f"{ind}  Split: x[{fname}] {self.split_text(n, unicode=False)} {thr_str}")

            if n.left is not None:
                walk(n.left, depth + 1, "L", idx)
            if n.right is not None:
                walk(n.right, depth + 1, "R", idx)

        walk(0, 0, "Root", None)
