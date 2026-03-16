from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from eywa_trees.backend.ecdf_rule_group import ecdf_bin_indices
from eywa_trees.backend.pset import PathwaySet
from eywa_trees.backend.rule_engine import RuleEngine, SortMode
from eywa_trees.backend.vistree import VisTree, VisNode


Array = np.ndarray


@dataclass(frozen=True)
class SubPathGroup:
    group_id: int
    feature: str
    feature_idx: int
    bin_id: int
    threshold_mean: float
    threshold_min: float
    threshold_max: float
    direction: str  # "upper" (<=) or "lower" (>)
    inclusive: bool


@dataclass
class SubPathCombo:
    group_ids: Tuple[int, ...]
    count: int
    coverage: float
    coverage_std: float
    n_train: float
    n_train_std: float
    pred: Array | float
    pred_text: str
    path_indices: Array


class SubPathEngine:
    """Precompute grouped sub-path combinations across all paths."""

    def __init__(
        self,
        rule_engine: RuleEngine,
        pset: PathwaySet,
        max_length: Optional[int] = None,
        path_order: Optional[Array] = None,
    ) -> None:
        self.rule_engine = rule_engine
        self.ecdf_dict = rule_engine.ecdf_dict or {}
        self.ecdf_config = rule_engine.ecdf_config
        self.features_upper = pset.features_upper
        self.features_lower = pset.features_lower
        self.features_upper_inclusive = getattr(
            pset,
            "features_upper_inclusive",
            np.ones_like(self.features_upper, dtype=bool),
        )
        self.features_lower_inclusive = getattr(
            pset,
            "features_lower_inclusive",
            np.ones_like(self.features_lower, dtype=bool),
        )
        if (
            path_order is not None
            and self.features_upper.size
            and path_order.size == self.features_upper.shape[0]
        ):
            self.features_upper = self.features_upper[path_order]
            self.features_lower = self.features_lower[path_order]
            self.features_upper_inclusive = self.features_upper_inclusive[path_order]
            self.features_lower_inclusive = self.features_lower_inclusive[path_order]
        self.max_length = max_length
        self.n_paths = self.features_upper.shape[0] if self.features_upper.size else 0
        self.n_features = self.features_upper.shape[1] if self.features_upper.size else 0

        upper_bins, lower_bins = self._compute_feature_bins()
        self.groups, self._group_id_by_bin = self._build_groups(upper_bins, lower_bins)
        self.upper_group_ids, self.lower_group_ids = self._build_group_id_arrays(
            upper_bins,
            lower_bins,
            self._group_id_by_bin,
        )
        self._path_group_sets = self._build_path_group_sets()

        self.max_length = self._infer_max_length(self.max_length)
        self.combos_by_length: Dict[int, List[SubPathCombo]] = {}
        self.combo_lookup: Dict[Tuple[int, ...], SubPathCombo] = {}
        self.sorted_by_length: Dict[int, Dict[SortMode, List[SubPathCombo]]] = {}
        self._precompute_combos()

    def top_combos(
        self,
        length: int,
        sort_mode: SortMode,
        top_k: int,
    ) -> List[SubPathCombo]:
        length = int(max(1, min(length, self.max_length)))
        sorted_map = self.sorted_by_length.get(length, {})
        combos = sorted_map.get(sort_mode, [])
        return combos[:top_k]

    def ordered_group_ids(self, group_ids: Tuple[int, ...]) -> List[int]:
        def _sort_key(gid: int) -> Tuple[str, int, float]:
            info = self.groups[gid]
            dir_key = 0 if info.direction == "upper" else 1
            inclusive_key = 0 if info.inclusive else 1
            return (info.feature, dir_key, inclusive_key, float(info.bin_id))

        return sorted(group_ids, key=_sort_key)

    def combo_stats(self, group_ids: Tuple[int, ...]) -> Optional[SubPathCombo]:
        return self.combo_lookup.get(tuple(group_ids))

    def build_combo_tree(self, combo: SubPathCombo) -> VisTree:
        ordered = self.ordered_group_ids(combo.group_ids)
        vis_tree = VisTree(
            model=None,
            feature_names=self.rule_engine.feature_names,
            class_names=self.rule_engine.class_names,
            is_classifier=self.rule_engine.is_classification,
            uses_scores=bool(not self.rule_engine.is_classification and self.rule_engine.pred_vector is not None),
        )

        vis_tree.max_depth = max(0, len(ordered))
        vis_tree.n_train = int(round(combo.n_train))
        if self.rule_engine.is_classification and isinstance(combo.pred, np.ndarray):
            vis_tree.n_classes = combo.pred.shape[0]

        nodes: Dict[int, VisNode] = {}
        path_nodes: List[int] = []
        prev_id: Optional[int] = None
        prev_rule: Optional[SubPathGroup] = None

        for idx, gid in enumerate(ordered):
            subset = tuple(sorted(ordered[: idx + 1]))
            summary = self.combo_lookup.get(subset)
            if summary is None:
                summary = combo
            rule = self.groups[gid]
            node_is_left = True if rule.direction == "upper" else False
            split_operator = "<=" if rule.inclusive else "<"

            node = VisNode(
                id=idx,
                feature=rule.feature_idx,
                threshold=rule.threshold_mean,
                value=summary.pred,
                parent=prev_id,
                is_left=node_is_left,
                left=None,
                right=None,
                n_train=int(round(summary.n_train)),
                hist=None,
                coverage=float(summary.coverage),
                coverage_std=float(summary.coverage_std),
                n_train_std=float(summary.n_train_std),
                split_operator=split_operator,
            )
            node.threshold_min = float(rule.threshold_min)
            node.threshold_max = float(rule.threshold_max)
            nodes[idx] = node
            path_nodes.append(idx)

            if prev_id is not None:
                if node_is_left:
                    nodes[prev_id].left = idx
                else:
                    nodes[prev_id].right = idx

            prev_id = idx
            prev_rule = rule

        leaf_id = len(ordered)
        mask = np.zeros(self.rule_engine.n_paths, dtype=bool)
        mask[combo.path_indices] = True
        leaf_hist = self._histogram_for_mask(mask)
        leaf_node = VisNode(
            id=leaf_id,
            feature=None,
            threshold=None,
            value=combo.pred,
            parent=prev_id,
            is_left=None,
            left=None,
            right=None,
            n_train=int(round(combo.n_train)),
            hist=leaf_hist,
            coverage=float(combo.coverage),
            coverage_std=float(combo.coverage_std),
            n_train_std=float(combo.n_train_std),
        )
        if prev_id is not None and prev_rule is not None:
            if prev_rule.direction == "upper":
                nodes[prev_id].left = leaf_id
                leaf_node.is_left = True
            else:
                nodes[prev_id].right = leaf_id
                leaf_node.is_left = False

        nodes[leaf_id] = leaf_node
        path_nodes.append(leaf_id)

        vis_tree.nodes = nodes
        vis_tree.leaf_paths = {leaf_id: path_nodes}

        if not vis_tree.is_classifier:
            vals: List[float] = []
            for node in nodes.values():
                v = node.value
                if isinstance(v, (np.ndarray, list, tuple)):
                    arr = np.asarray(v, dtype=float).ravel()
                    if arr.size:
                        vals.append(float(arr[0]))
                elif isinstance(v, (float, int, np.floating, np.integer)):
                    vals.append(float(v))
            vis_tree.possible_values = set(vals)

        vis_tree._generate_color_struct()
        return vis_tree

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_feature_bins(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        upper_bins: List[np.ndarray] = []
        lower_bins: List[np.ndarray] = []
        for j in range(self.n_features):
            upper = np.full(self.n_paths, -1, dtype=int)
            lower = np.full(self.n_paths, -1, dtype=int)
            ecdf = self.ecdf_dict.get(j)
            if ecdf is not None and self.n_paths > 0:
                upper_vals = self.features_upper[:, j]
                finite = np.isfinite(upper_vals)
                if np.any(finite):
                    upper[finite] = ecdf_bin_indices(
                        upper_vals[finite],
                        ecdf,
                        self.ecdf_config,
                    )
                lower_vals = self.features_lower[:, j]
                finite = np.isfinite(lower_vals)
                if np.any(finite):
                    lower[finite] = ecdf_bin_indices(
                        lower_vals[finite],
                        ecdf,
                        self.ecdf_config,
                    )
            upper_bins.append(upper)
            lower_bins.append(lower)
        return upper_bins, lower_bins

    def _build_groups(
        self,
        upper_bins: List[np.ndarray],
        lower_bins: List[np.ndarray],
    ) -> Tuple[Dict[int, SubPathGroup], Dict[Tuple[int, str], Dict[Tuple[int, bool], int]]]:
        groups: Dict[int, SubPathGroup] = {}
        group_index: Dict[Tuple[int, int, str, bool], int] = {}
        group_id_by_bin: Dict[Tuple[int, str], Dict[Tuple[int, bool], int]] = {}
        feature_names = self.rule_engine.feature_names or []

        for j in range(self.n_features):
            feat_name = feature_names[j] if j < len(feature_names) else str(j)
            for direction, bins, values, inclusive_values in (
                ("upper", upper_bins[j], self.features_upper[:, j], self.features_upper_inclusive[:, j]),
                ("lower", lower_bins[j], self.features_lower[:, j], self.features_lower_inclusive[:, j]),
            ):
                mask = bins >= 0
                if not np.any(mask):
                    continue
                combo_keys = np.unique(
                    np.column_stack([bins[mask].astype(int), inclusive_values[mask].astype(int)]),
                    axis=0,
                )
                for bin_id_val, inclusive_val in combo_keys:
                    bin_id = int(bin_id_val)
                    inclusive = bool(inclusive_val)
                    key = (j, bin_id, direction, inclusive)
                    gid = group_index.setdefault(key, len(group_index))
                    group_id_by_bin.setdefault((j, direction), {})[(bin_id, inclusive)] = gid
                    thr_mask = (bins == bin_id) & (inclusive_values == inclusive)
                    thr_vals = values[thr_mask]
                    thr_vals = thr_vals[np.isfinite(thr_vals)]
                    if thr_vals.size == 0:
                        continue
                    groups[gid] = SubPathGroup(
                        group_id=gid,
                        feature=feat_name,
                        feature_idx=j,
                        bin_id=int(bin_id),
                        threshold_mean=float(np.mean(thr_vals)),
                        threshold_min=float(np.min(thr_vals)),
                        threshold_max=float(np.max(thr_vals)),
                        direction=direction,
                        inclusive=inclusive,
                    )
        return groups, group_id_by_bin

    def _build_group_id_arrays(
        self,
        upper_bins: List[np.ndarray],
        lower_bins: List[np.ndarray],
        group_id_by_bin: Dict[Tuple[int, str], Dict[Tuple[int, bool], int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        upper_ids = np.full((self.n_paths, self.n_features), -1, dtype=int)
        lower_ids = np.full((self.n_paths, self.n_features), -1, dtype=int)
        for j in range(self.n_features):
            mapping = group_id_by_bin.get((j, "upper"), {})
            bins = upper_bins[j]
            inclusive_vals = self.features_upper_inclusive[:, j]
            for (bin_id, inclusive), gid in mapping.items():
                mask = (bins == bin_id) & (inclusive_vals == inclusive)
                if np.any(mask):
                    upper_ids[mask, j] = gid
            mapping = group_id_by_bin.get((j, "lower"), {})
            bins = lower_bins[j]
            inclusive_vals = self.features_lower_inclusive[:, j]
            for (bin_id, inclusive), gid in mapping.items():
                mask = (bins == bin_id) & (inclusive_vals == inclusive)
                if np.any(mask):
                    lower_ids[mask, j] = gid
        return upper_ids, lower_ids

    def _build_path_group_sets(self) -> List[set[int]]:
        path_groups: List[set[int]] = []
        for i in range(self.n_paths):
            group_ids: set[int] = set()
            for j in range(self.n_features):
                upper_gid = int(self.upper_group_ids[i, j])
                if upper_gid >= 0:
                    group_ids.add(upper_gid)
                lower_gid = int(self.lower_group_ids[i, j])
                if lower_gid >= 0:
                    group_ids.add(lower_gid)
            path_groups.append(group_ids)
        return path_groups

    def _prediction_for_mask(self, mask: Array) -> Array | float:
        weights = self.rule_engine.expectation_weights[mask]
        total_w = float(weights.sum())
        if not np.any(mask) or total_w <= 0.0:
            if self.rule_engine.is_classification and self.rule_engine.pred_matrix is not None:
                return np.zeros(self.rule_engine.pred_matrix.shape[1], dtype=float)
            return float("nan")

        if self.rule_engine.is_classification and self.rule_engine.pred_matrix is not None:
            probs = self.rule_engine.pred_matrix[mask]
            weighted = (probs.T * weights).T
            return weighted.sum(axis=0) / total_w

        if self.rule_engine.pred_vector is None:
            return float("nan")
        vals = self.rule_engine.pred_vector[mask]
        return float((vals * weights).sum() / total_w)

    def _histogram_for_mask(
        self,
        mask: Array,
        max_bins: int = 12,
    ) -> Optional[Dict[str, object]]:
        weights = self.rule_engine.expectation_weights[mask]
        if not np.any(mask):
            return None
        total_w = float(weights.sum())
        if total_w <= 0.0:
            return None

        if self.rule_engine.is_classification and self.rule_engine.pred_matrix is not None:
            probs = self.rule_engine.pred_matrix[mask]
            weighted = (probs.T * weights).T
            sums = weighted.sum(axis=0)
            if sums.size == 0:
                return None
            dist = sums / total_w
            labels = (
                list(self.rule_engine.class_names)
                if self.rule_engine.class_names is not None
                else [str(i) for i in range(dist.shape[0])]
            )
            return {
                "type": "classification",
                "probs": dist.tolist(),
                "labels": labels,
                "total": total_w,
            }

        if self.rule_engine.pred_vector is None:
            return None

        vals = np.asarray(self.rule_engine.pred_vector[mask], dtype=float)
        finite_mask = np.isfinite(vals)
        vals = vals[finite_mask]
        weights = np.asarray(weights[finite_mask], dtype=float)
        total_w = float(weights.sum())
        if vals.size == 0 or total_w <= 0.0:
            return None

        unique_vals = np.unique(vals)
        n_bins = min(max_bins, max(1, unique_vals.size))
        if unique_vals.size == 1:
            v = float(unique_vals[0])
            eps = max(1e-6, abs(v) * 0.01)
            edges = np.array([v - eps, v + eps])
            counts = np.array([total_w], dtype=float)
        else:
            edges = np.linspace(float(vals.min()), float(vals.max()), n_bins + 1)
            counts, edges = np.histogram(vals, bins=edges, weights=weights, density=False)

        freq = counts / total_w if total_w > 0 else counts
        centers = 0.5 * (edges[:-1] + edges[1:]) if edges.size >= 2 else np.array([], dtype=float)
        return {
            "type": "regression",
            "bin_edges": edges.tolist(),
            "centers": centers.tolist(),
            "freq": freq.tolist(),
            "total": total_w,
        }

    def _infer_max_length(self, max_length: Optional[int]) -> int:
        if max_length is not None and max_length > 0:
            return int(max_length)
        max_len = 1
        for groups in self._path_group_sets:
            max_len = max(max_len, len(groups))
        return max_len

    def _precompute_combos(self) -> None:
        combos_by_len: Dict[int, Dict[Tuple[int, ...], List[int]]] = {}
        for idx, group_ids in enumerate(self._path_group_sets):
            if not group_ids:
                continue
            group_list = sorted(group_ids)
            limit = min(len(group_list), self.max_length)
            for length in range(1, limit + 1):
                combo_map = combos_by_len.setdefault(length, {})
                for combo in combinations(group_list, length):
                    combo_map.setdefault(combo, []).append(idx)

        n_paths = self.rule_engine.n_paths
        for length, combo_map in combos_by_len.items():
            combos: List[SubPathCombo] = []
            for combo, indices in combo_map.items():
                idx_arr = np.asarray(indices, dtype=int)
                mask = np.zeros(n_paths, dtype=bool)
                mask[idx_arr] = True
                tree_count = self.rule_engine._count_trees_in_mask(mask)
                summary = self.rule_engine.node_summary(mask)
                pred = self._prediction_for_mask(mask)
                combo_rec = SubPathCombo(
                    group_ids=combo,
                    count=int(tree_count),
                    coverage=float(summary.get("coverage", 0.0)),
                    coverage_std=float(summary.get("coverage_std", 0.0)),
                    n_train=float(summary.get("n_train", 0.0)),
                    n_train_std=float(summary.get("n_train_std", 0.0)),
                    pred=pred,
                    pred_text=self.rule_engine.prediction_text(pred),
                    path_indices=idx_arr,
                )
                combos.append(combo_rec)
                self.combo_lookup[combo] = combo_rec
            self.combos_by_length[length] = combos
            self.sorted_by_length[length] = {
                "paths": sorted(
                    combos,
                    key=lambda c: (c.count, c.coverage),
                    reverse=True,
                ),
                "coverage": sorted(
                    combos,
                    key=lambda c: (c.coverage, c.count),
                    reverse=True,
                ),
            }
