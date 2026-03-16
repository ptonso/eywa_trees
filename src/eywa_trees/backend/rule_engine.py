# eywa_trees/backend/rule_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Literal, Dict

import numpy as np
import pandas as pd

from eywa_trees.backend.vistree import VisTree, VisNode
from eywa_trees.backend.ecdf_rule_group import ECDFBinConfig, RuleCluster, build_clusters


Array = np.ndarray
SortMode = Literal["paths", "coverage"]


@dataclass
class RuleEngineConfig:
    """Configuration for rule extraction and clustering."""
    top_k_rules: int = 20


class RuleEngine:
    """Builds rule clusters from paths and supports rule statistics."""

    def __init__(
        self,
        pset_df: pd.DataFrame,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        config: Optional[RuleEngineConfig] = None,
        dataset_n: Optional[int] = None,
        ecdf_dict: Optional[Dict[int, Any]] = None,
        ecdf_config: Optional[ECDFBinConfig] = None,
        upper_depths: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config or RuleEngineConfig()
        self.feature_names = feature_names
        self.class_names = class_names
        self.dataset_n = int(dataset_n) if dataset_n is not None else None
        self.ecdf_dict = ecdf_dict
        self.ecdf_config = ecdf_config or ECDFBinConfig()

        self.pset_df = pset_df.reset_index(drop=True)
        self.n_paths = self.pset_df.shape[0]
        self.upper_depths = self._prepare_upper_depths(upper_depths)
        self.max_upper_depth = self._compute_max_upper_depth()

        self.n_train = self._extract_n_train()
        self.expectation_weights, self.expectation_mode = self._extract_expectation_weights()
        self.total_n_train = float(self.n_train.sum())
        self.tree_id_sets = self._extract_tree_id_sets()
        self.tree_root_counts = self._compute_tree_root_counts()
        self.total_trees = self._count_total_trees()
        self.root_n = self._root_sample_count()

        self.is_classification, self.pred_matrix, self.pred_vector = self._extract_predictions()

        self.clusters: List[RuleCluster] = []
        self._build_clusters()

    # ------------------------------------------------------------------
    # Masks / summaries
    # ------------------------------------------------------------------
    def root_mask(self) -> Array:
        return np.ones(self.n_paths, dtype=bool)

    def cluster_mask(self, cluster_id: int) -> Array:
        cluster = self.clusters[cluster_id]
        if cluster.path_mask is None:
            cluster.path_mask = self._build_cluster_mask(cluster.path_indices)
        return cluster.path_mask

    def aggregate_prediction(self, mask: Array) -> Array | float:
        w = self.expectation_weights[mask]
        if not np.any(mask) or float(w.sum()) <= 0.0:
            if self.is_classification and self.pred_matrix is not None:
                return np.zeros(self.pred_matrix.shape[1], dtype=float)
            if self.pred_vector is not None:
                return float("nan")
            return float("nan")

        if self.is_classification and self.pred_matrix is not None:
            probs = self.pred_matrix[mask]
            weighted = (probs.T * w).T
            mean = weighted.sum(axis=0) / float(w.sum())
            return mean
        if self.pred_vector is not None:
            vals = self.pred_vector[mask]
            return float((vals * w).sum() / float(w.sum()))
        return float("nan")

    def prediction_text(self, pred: Array | float) -> str:
        if self.is_classification:
            vec = np.asarray(pred, dtype=float)
            if vec.ndim != 1 or vec.size == 0:
                return "n/a"
            idx = int(np.argmax(vec))
            if self.class_names is not None and 0 <= idx < len(self.class_names):
                label = self.class_names[idx]
            else:
                label = str(idx)
            prob = float(vec[idx])
            return f"{label} ({prob:.2f})"
        if isinstance(pred, (float, int, np.floating, np.integer)):
            return f"{float(pred):.3f}"
        return "n/a"

    def prediction_column_label(self) -> str:
        if self.expectation_mode == "cover" and "scores" in self.pset_df.columns:
            return "Expected score"
        return "Prediction"

    def node_summary(self, mask: Array) -> Dict[str, object]:
        root_n = float(self.root_n) if self.root_n is not None else 1.0
        tree_count = self._count_trees_in_mask(mask)
        if tree_count <= 0:
            tree_count = 1
        n_train_vals = self.n_train[mask] if np.any(mask) else np.array([], dtype=float)
        n_train_val = float(n_train_vals.sum()) if n_train_vals.size else 0.0
        n_train_std = float(np.std(n_train_vals)) if n_train_vals.size else 0.0
        tree_ids = self._tree_ids_for_mask(mask)
        denom = self._root_total_for_tree_ids(tree_ids) if tree_ids else root_n * float(tree_count)
        coverage = n_train_val / denom if denom > 0 else 0.0
        coverage_std = n_train_std / denom if denom > 0 else 0.0
        pred = self.aggregate_prediction(mask)
        return {
            "coverage": coverage,
            "coverage_std": coverage_std,
            "n_train": n_train_val,
            "n_train_std": n_train_std,
            "pred": pred,
            "pred_text": self.prediction_text(pred),
        }

    def histogram_for_mask(
        self,
        mask: Array,
        max_bins: int = 12,
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate a histogram for the provided path mask, weighted by the
        active expectation weights.
        Classification returns class probabilities; regression returns binned frequencies.
        """
        weights = self.expectation_weights[mask]
        if not np.any(mask):
            return None
        total_w = float(weights.sum())
        if total_w <= 0.0:
            return None

        if self.is_classification and self.pred_matrix is not None:
            probs = self.pred_matrix[mask]
            weighted = (probs.T * weights).T
            sums = weighted.sum(axis=0)
            if sums.size == 0:
                return None
            dist = sums / total_w if total_w > 0 else sums
            labels = (
                list(self.class_names)
                if self.class_names is not None
                else [str(i) for i in range(dist.shape[0])]
            )
            return {
                "type": "classification",
                "probs": dist.tolist(),
                "labels": labels,
                "total": total_w,
            }

        if self.pred_vector is None:
            return None

        vals = np.asarray(self.pred_vector[mask], dtype=float)
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

    # ------------------------------------------------------------------
    # Public rule listing
    # ------------------------------------------------------------------
    def candidate_rules(
        self,
        mask: Array,
        sort_mode: SortMode,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        columns = [
            "cluster_id",
            "feature",
            "threshold_op",
            "threshold_text",
            "threshold_mean",
            "threshold_min",
            "threshold_max",
            "n_rules",
            "n_train",
            "n_train_std",
            "coverage",
            "coverage_std",
            "pred_text",
        ]
        if top_k is None:
            top_k = self.config.top_k_rules

        total_n_train = float(self.n_train[mask].sum()) if np.any(mask) else 0.0
        if total_n_train <= 0.0:
            return pd.DataFrame(columns=columns)

        records: List[Dict] = []
        for cluster in self.clusters:
            if cluster.path_mask is None:
                cluster.path_mask = self._build_cluster_mask(cluster.path_indices)
            local_mask = mask & cluster.path_mask
            if not np.any(local_mask):
                continue

            thresholds = self._thresholds_for_mask(
                cluster.feature,
                local_mask,
                upper_inclusive=cluster.upper_inclusive,
            )
            if thresholds.size == 0:
                continue
            node_mask = self._node_mask_for_thresholds(
                cluster.feature,
                thresholds,
                mask,
                upper_inclusive=cluster.upper_inclusive,
            )
            if not np.any(node_mask):
                continue

            n_train_vals = self.n_train[node_mask]
            n_train_display = float(n_train_vals.sum()) if n_train_vals.size > 0 else 0.0
            n_train_std = float(np.std(n_train_vals)) if n_train_vals.size > 0 else 0.0
            coverage, coverage_std, n_rules = self._node_coverage_stats(
                cluster.feature,
                thresholds,
                mask,
                upper_inclusive=cluster.upper_inclusive,
            )
            pred = self.aggregate_prediction(node_mask)
            op_text = "<=" if cluster.upper_inclusive else "<"
            rec = {
                "cluster_id": cluster.id,
                "feature": cluster.feature,
                "threshold_op": op_text,
                "threshold_text": f"{op_text} {cluster.threshold_mean:.3f}",
                "threshold_mean": cluster.threshold_mean,
                "threshold_min": cluster.threshold_min,
                "threshold_max": cluster.threshold_max,
                "n_rules": n_rules,
                "n_train": n_train_display,
                "n_train_std": n_train_std,
                "coverage": coverage,
                "coverage_std": coverage_std,
                "pred_text": self.prediction_text(pred),
            }
            records.append(rec)

        if not records:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame.from_records(records, columns=columns)
        if sort_mode == "paths":
            df.sort_values(["n_rules", "coverage"], ascending=[False, False], inplace=True)
        else:
            df.sort_values(["coverage", "n_rules"], ascending=[False, False], inplace=True)
        return df.head(top_k).reset_index(drop=True)

    def depth_histogram_for_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        if self.upper_depths is None:
            return None
        if cluster_id < 0 or cluster_id >= len(self.clusters):
            return None

        cluster = self.clusters[cluster_id]
        indices = cluster.path_indices
        if indices.size == 0:
            return None
        if int(indices.max()) >= self.upper_depths.shape[0]:
            return None

        depths = np.asarray(self.upper_depths[indices, cluster.feature_idx], dtype=int)
        mask = depths >= 0
        if not np.any(mask):
            return None
        depths = depths[mask]

        if depths.size == 0:
            return None

        max_depth = (
            int(self.max_upper_depth)
            if self.max_upper_depth is not None
            else int(depths.max())
        )
        counts = np.bincount(depths, minlength=max_depth + 1)
        return {
            "depths": list(range(max_depth + 1)),
            "counts": counts.astype(int).tolist(),
            "total": int(depths.size),
            "max_depth": max_depth,
        }

    # ------------------------------------------------------------------
    # Glue to VisTree / Sankey backend
    # ------------------------------------------------------------------
    def build_cluster_tree(self, cluster_id: int) -> VisTree:
        """
        Build a tiny VisTree with depth 1 representing the chosen rule:
        root = full ensemble, left/right = rule respected / violated.
        """
        if cluster_id < 0 or cluster_id >= len(self.clusters):
            raise ValueError(f"Invalid cluster_id {cluster_id}")

        cluster = self.clusters[cluster_id]

        root_mask = self.root_mask()
        root_summary = self.node_summary(root_mask)
        root_hist = self.histogram_for_mask(root_mask)

        cmask = self.cluster_mask(cluster.id)
        left_mask = root_mask & cmask
        right_mask = root_mask & ~cmask

        left_summary = self.node_summary(left_mask)
        right_summary = self.node_summary(right_mask)
        left_hist = self.histogram_for_mask(left_mask)
        right_hist = self.histogram_for_mask(right_mask)
        left_cov, left_cov_std, _ = self._rule_coverage_stats(cluster, left_mask)
        left_summary["coverage"] = left_cov
        left_summary["coverage_std"] = left_cov_std

        root_pred = root_summary["pred"]
        left_pred = left_summary["pred"]
        right_pred = right_summary["pred"]

        vis_tree = VisTree(
            model=None,
            feature_names=self.feature_names,
            class_names=self.class_names,
            is_classifier=self.is_classification,
            uses_scores=bool(not self.is_classification and self.pred_vector is not None),
        )
        vis_tree.n_train = int(root_summary["n_train"])
        vis_tree.max_depth = 1

        if self.is_classification and isinstance(root_pred, np.ndarray):
            vis_tree.n_classes = root_pred.shape[0]

        root_node = VisNode(
            id=0,
            feature=cluster.feature_idx,
            threshold=cluster.threshold_mean,
            value=root_pred,
            parent=None,
            is_left=None,
            left=1,
            right=2,
            n_train=int(round(root_summary["n_train"])),
            hist=root_hist,
            coverage=float(root_summary.get("coverage", 0.0)),
            coverage_std=float(root_summary.get("coverage_std", 0.0)),
            n_train_std=float(root_summary.get("n_train_std", 0.0)),
            split_operator="<=" if cluster.upper_inclusive else "<",
        )
        left_node = VisNode(
            id=1,
            feature=None,
            threshold=None,
            value=left_pred,
            parent=0,
            is_left=True,
            left=None,
            right=None,
            n_train=int(round(left_summary["n_train"])),
            hist=left_hist,
            coverage=float(left_summary.get("coverage", 0.0)),
            coverage_std=float(left_summary.get("coverage_std", 0.0)),
            n_train_std=float(left_summary.get("n_train_std", 0.0)),
        )
        right_node = VisNode(
            id=2,
            feature=None,
            threshold=None,
            value=right_pred,
            parent=0,
            is_left=False,
            left=None,
            right=None,
            n_train=int(round(right_summary["n_train"])),
            hist=right_hist,
            coverage=float(right_summary.get("coverage", 0.0)),
            coverage_std=float(right_summary.get("coverage_std", 0.0)),
            n_train_std=float(right_summary.get("n_train_std", 0.0)),
        )

        vis_tree.nodes = {0: root_node, 1: left_node, 2: right_node}
        vis_tree.leaf_paths = {1: [0, 1], 2: [0, 2]}

        if not vis_tree.is_classifier:
            vals: List[float] = []
            for v in (root_pred, left_pred, right_pred):
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
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_n_train(self) -> Array:
        if "n_samples" in self.pset_df.columns:
            return self.pset_df["n_samples"].to_numpy(dtype=float)
        if "n_train" in self.pset_df.columns:
            return self.pset_df["n_train"].to_numpy(dtype=float)
        return np.ones(self.pset_df.shape[0], dtype=float)

    def _extract_expectation_weights(self) -> tuple[Array, str]:
        if "leaf_cover" in self.pset_df.columns:
            w = self.pset_df["leaf_cover"].to_numpy(dtype=float)
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w = np.clip(w, 0.0, None)
            if float(w.sum()) > 0.0:
                return w, "cover"
        if "path_prob_forest" in self.pset_df.columns:
            w = self.pset_df["path_prob_forest"].to_numpy(dtype=float)
        elif self.dataset_n is not None and self.dataset_n > 0:
            w = self.n_train / float(self.dataset_n)
        else:
            total = float(self.n_train.sum())
            w = self.n_train / total if total > 0 else np.zeros_like(self.n_train, dtype=float)
        return w, "dataset"

    def _extract_predictions(self) -> tuple[bool, Optional[Array], Optional[Array]]:
        if "value_dist" in self.pset_df.columns:
            first = self.pset_df["value_dist"].iloc[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                mat = np.stack(
                    [
                        np.asarray(v, dtype=float)
                        if isinstance(v, (list, tuple, np.ndarray))
                        else np.asarray([], dtype=float)
                        for v in self.pset_df["value_dist"].tolist()
                    ],
                    axis=0,
                )
                return True, mat, None

        if "regressions" in self.pset_df.columns:
            vec = self.pset_df["regressions"].to_numpy(dtype=float)
            return False, None, vec

        if "scores" in self.pset_df.columns:
            vec = self.pset_df["scores"].to_numpy(dtype=float)
            return False, None, vec

        if "predicted_label" in self.pset_df.columns:
            vec = self.pset_df["predicted_label"].to_numpy(dtype=float)
            return False, None, vec

        return False, None, None

    def _prepare_upper_depths(self, upper_depths: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if upper_depths is None:
            return None
        arr = np.asarray(upper_depths)
        if arr.ndim != 2:
            return None
        if arr.shape[0] != self.n_paths:
            return None
        return arr

    def _compute_max_upper_depth(self) -> Optional[int]:
        if self.upper_depths is None:
            return None
        vals = self.upper_depths[self.upper_depths >= 0]
        if vals.size == 0:
            return None
        return int(vals.max())

    def _extract_tree_id_sets(self) -> Optional[List[set[int]]]:
        if "tree_id" not in self.pset_df.columns:
            return None
        tree_vals = self.pset_df["tree_id"].tolist()
        tree_sets: List[set[int]] = []
        for val in tree_vals:
            ids: set[int] = set()
            if val is None:
                tree_sets.append(ids)
                continue
            if isinstance(val, (int, np.integer)):
                ids.add(int(val))
            elif isinstance(val, (float, np.floating)):
                if not np.isnan(val):
                    ids.add(int(val))
            elif isinstance(val, str):
                for part in val.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        ids.add(int(part))
                    except ValueError:
                        try:
                            ids.add(int(float(part)))
                        except ValueError:
                            continue
            elif isinstance(val, (list, tuple, np.ndarray)):
                for item in val:
                    try:
                        ids.add(int(item))
                    except Exception:
                        continue
            tree_sets.append(ids)
        return tree_sets

    def _count_total_trees(self) -> int:
        if not self.tree_id_sets:
            return 0
        all_ids: set[int] = set()
        for ids in self.tree_id_sets:
            all_ids.update(ids)
        return int(len(all_ids))

    def _count_trees_in_mask(self, mask: Array) -> int:
        if not self.tree_id_sets or not np.any(mask):
            return 0
        ids: set[int] = set()
        for idx in np.flatnonzero(mask):
            ids.update(self.tree_id_sets[idx])
        return int(len(ids))

    def _tree_ids_for_mask(self, mask: Array) -> set[int]:
        if not self.tree_id_sets or not np.any(mask):
            return set()
        ids: set[int] = set()
        for idx in np.flatnonzero(mask):
            ids.update(self.tree_id_sets[idx])
        return ids

    def _compute_tree_root_counts(self) -> Dict[int, float]:
        if not self.tree_id_sets:
            return {}
        counts: Dict[int, float] = {}
        for idx, ids in enumerate(self.tree_id_sets):
            if not ids:
                continue
            weight = float(self.n_train[idx])
            share = weight / float(len(ids))
            for tid in ids:
                counts[tid] = counts.get(tid, 0.0) + share
        return counts

    def _root_total_for_tree_ids(self, tree_ids: set[int]) -> float:
        root_n = float(self.root_n) if self.root_n is not None else 1.0
        if not tree_ids:
            return root_n
        if not self.tree_root_counts:
            return root_n * float(len(tree_ids))
        total = 0.0
        for tid in tree_ids:
            total += float(self.tree_root_counts.get(tid, root_n))
        return total

    def _thresholds_for_mask(
        self,
        feature: str,
        mask: Array,
        upper_inclusive: Optional[bool] = None,
    ) -> np.ndarray:
        upper_col = f"{feature}_upper"
        if upper_col not in self.pset_df.columns or not np.any(mask):
            return np.array([], dtype=float)
        indices = np.flatnonzero(mask)
        values = self.pset_df.loc[indices, upper_col].to_numpy(dtype=float, copy=False)
        if upper_inclusive is not None and f"{feature}_upper_inclusive" in self.pset_df.columns:
            inclusive_vals = self.pset_df.loc[indices, f"{feature}_upper_inclusive"].to_numpy(
                dtype=bool,
                copy=False,
            )
            values = values[inclusive_vals == bool(upper_inclusive)]
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.array([], dtype=float)
        return np.unique(values[finite])

    def _upper_inclusive_values(self, feature: str) -> Optional[np.ndarray]:
        col = f"{feature}_upper_inclusive"
        if col not in self.pset_df.columns:
            return None
        return self.pset_df[col].to_numpy(dtype=bool, copy=False)

    def _node_mask_for_thresholds(
        self,
        feature: str,
        thresholds: np.ndarray,
        base_mask: Array,
        upper_inclusive: Optional[bool] = None,
    ) -> Array:
        if thresholds.size == 0:
            return np.zeros(self.n_paths, dtype=bool)
        upper_col = f"{feature}_upper"
        if upper_col not in self.pset_df.columns:
            return np.zeros(self.n_paths, dtype=bool)
        upper_vals = self.pset_df[upper_col].to_numpy(dtype=float, copy=False)
        upper_inclusive_vals = self._upper_inclusive_values(feature)
        node_mask = np.zeros(self.n_paths, dtype=bool)
        for thr in thresholds:
            thr_mask = upper_vals == thr
            if upper_inclusive is not None and upper_inclusive_vals is not None:
                thr_mask &= upper_inclusive_vals == bool(upper_inclusive)
            thr_mask &= base_mask
            node_mask |= thr_mask
        return node_mask

    def _node_coverage_stats(
        self,
        feature: str,
        thresholds: np.ndarray,
        base_mask: Array,
        upper_inclusive: Optional[bool] = None,
    ) -> tuple[float, float, int]:
        if thresholds.size == 0:
            return 0.0, 0.0, 0
        upper_col = f"{feature}_upper"
        if upper_col not in self.pset_df.columns:
            return 0.0, 0.0, 0
        upper_vals = self.pset_df[upper_col].to_numpy(dtype=float, copy=False)
        upper_inclusive_vals = self._upper_inclusive_values(feature)
        coverages: List[float] = []
        n_rules = 0
        for thr in thresholds:
            thr_mask = upper_vals == thr
            if upper_inclusive is not None and upper_inclusive_vals is not None:
                thr_mask &= upper_inclusive_vals == bool(upper_inclusive)
            thr_mask &= base_mask
            if not np.any(thr_mask):
                continue
            n_train_sum = float(self.n_train[thr_mask].sum())
            tree_ids = self._tree_ids_for_mask(thr_mask)
            if tree_ids:
                denom = self._root_total_for_tree_ids(tree_ids)
                n_rules += len(tree_ids)
            else:
                denom = float(self.root_n) if self.root_n is not None else 1.0
                n_rules += 1
            if denom > 0.0:
                coverages.append(n_train_sum / denom)
        if not coverages:
            return 0.0, 0.0, n_rules
        return float(np.mean(coverages)), float(np.std(coverages)), n_rules

    def _root_sample_count(self) -> float:
        if self.dataset_n is not None and self.dataset_n > 0:
            return float(self.dataset_n)
        total_n_train = float(self.n_train.sum())
        if self.total_trees > 0:
            if total_n_train > 0:
                return total_n_train / float(self.total_trees)
        if total_n_train > 0:
            return total_n_train
        return 1.0

    def _rule_coverage_stats(
        self,
        cluster: RuleCluster,
        mask: Array,
    ) -> tuple[float, float, int]:
        upper_col = f"{cluster.feature}_upper"
        if upper_col not in self.pset_df.columns or not np.any(mask):
            return 0.0, 0.0, 0

        indices = np.flatnonzero(mask)
        thresholds = self.pset_df.loc[indices, upper_col].to_numpy(dtype=float, copy=False)
        if f"{cluster.feature}_upper_inclusive" in self.pset_df.columns:
            inclusive_vals = self.pset_df.loc[indices, f"{cluster.feature}_upper_inclusive"].to_numpy(
                dtype=bool,
                copy=False,
            )
            keep = inclusive_vals == bool(cluster.upper_inclusive)
            thresholds = thresholds[keep]
            indices = indices[keep]
        if thresholds.size == 0:
            return 0.0, 0.0, 0

        unique_thr, inv = np.unique(thresholds, return_inverse=True)

        n_train_vals = self.n_train[indices]
        n_train_sums = np.bincount(inv, weights=n_train_vals, minlength=unique_thr.size).astype(float)

        if self.tree_id_sets:
            tree_sets_per_rule = [set() for _ in range(unique_thr.size)]
            for pos, row_idx in enumerate(indices):
                tree_sets_per_rule[inv[pos]].update(self.tree_id_sets[row_idx])
            n_trees = np.array([len(s) for s in tree_sets_per_rule], dtype=float)
            n_trees[n_trees <= 0] = 1.0
            denom = np.array(
                [self._root_total_for_tree_ids(s) for s in tree_sets_per_rule],
                dtype=float,
            )
        else:
            n_trees = np.ones(unique_thr.size, dtype=float)
            root_n = float(self.root_n) if self.root_n is not None else 1.0
            denom = np.full(unique_thr.size, root_n, dtype=float)

        n_rules = int(n_trees.sum())

        if not np.any(denom > 0.0):
            return 0.0, 0.0, n_rules

        valid = denom > 0.0
        coverages = n_train_sums[valid] / denom[valid]
        coverage = float(np.mean(coverages)) if coverages.size else 0.0
        coverage_std = float(np.std(coverages)) if coverages.size else 0.0
        return coverage, coverage_std, n_rules

    def _build_cluster_mask(self, path_indices: Array) -> Array:
        mask = np.zeros(self.n_paths, dtype=bool)
        mask[path_indices] = True
        return mask

    def _build_clusters(self) -> None:
        self.clusters = build_clusters(
            pset_df=self.pset_df,
            feature_names=self.feature_names,
            n_train=self.n_train,
            total_n_train=self.total_n_train,
            dataset_n=self.dataset_n,
            ecdf_dict=self.ecdf_dict,
            config=self.ecdf_config,
        )
