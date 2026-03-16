from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

from eywa_trees.logger import setup_logger
from eywa_trees.backend.pathway_builders import build_pathway_inputs_from_model


Array = np.ndarray


@dataclass(init=False)
class PathwaySet:
    """
    Extract decision-path statistics from a single tree or an ensemble.

    The stored arrays include per-feature bounds, empirical path coverage,
    leaf cover mass, and a tree_id column so downstream consumers can
    aggregate across trees.
    """

    feature_names: List[str]
    features_lower: Array
    features_upper: Array
    features_lower_inclusive: Array
    features_upper_inclusive: Array
    features_upper_depth: Array
    n_samples: Array
    leaf_cover: Array
    path_prob_mc: Array
    path_prob_forest: Array
    output: Array
    leaf_index: Array
    tree_id: Array
    path_prob: Array
    is_classifier: bool
    uses_scores: bool
    label_names: Optional[np.ndarray]
    ecdf_dict: Dict[int, ECDF]
    n_trees: int

    def __init__(
        self,
        feature_names: List[str],
        model: Any,
        X_train: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, "RandomState"]] = None,
        verbose: bool = True,
    ) -> None:
        logger = setup_logger("pathway_set.log")
        feature_names_list = list(feature_names)

        if isinstance(random_state, (int, np.integer)):
            np.random.seed(int(random_state))

        provided_names = feature_names_list if feature_names_list else None
        adapter_result = build_pathway_inputs_from_model(
            model,
            feature_names=provided_names,
        )
        trees = adapter_result.trees

        if not feature_names_list:
            if adapter_result.feature_names:
                feature_names_list = list(adapter_result.feature_names)
            else:
                inferred_count = self._infer_feature_count_from_trees(trees)
                feature_names_list = [str(i) for i in range(inferred_count)]

        is_classifier = adapter_result.is_classifier
        uses_scores = adapter_result.uses_scores
        label_names = adapter_result.label_names if is_classifier else None
        n_trees = adapter_result.n_trees or len(trees)

        ecdf_dict: Dict[int, ECDF] = {}
        if X_train is not None:
            ecdf_dict = {i: ECDF(X_train[:, i]) for i in range(len(feature_names_list))}

        n_features = len(feature_names_list)
        n_classes = len(label_names) if label_names is not None else 0
        (
            features_lower,
            features_upper,
            features_lower_inclusive,
            features_upper_inclusive,
            features_upper_depth,
            n_samples,
            leaf_cover,
            path_prob_mc,
            output,
            leaf_index,
            tree_id,
        ) = self._extract_paths(
            trees,
            n_features=n_features,
            is_classifier=is_classifier,
            n_classes=n_classes,
            X_train=X_train,
        )

        denom = float(n_trees) if n_trees else 1.0
        path_prob_forest = (
            path_prob_mc / denom if path_prob_mc.size else np.array([], dtype=float)
        )

        if ecdf_dict:
            path_prob = self._calculate_path_prob_by_ecdf(
                features_lower,
                features_upper,
                features_lower_inclusive,
                features_upper_inclusive,
                ecdf_dict,
            )
        else:
            path_prob = np.full(n_samples.shape, np.nan, dtype=float)

        self.feature_names = feature_names_list
        self.features_lower = features_lower
        self.features_upper = features_upper
        self.features_lower_inclusive = features_lower_inclusive
        self.features_upper_inclusive = features_upper_inclusive
        self.features_upper_depth = features_upper_depth
        self.n_samples = n_samples
        self.leaf_cover = leaf_cover
        self.path_prob_mc = path_prob_mc
        self.path_prob_forest = path_prob_forest
        self.output = output
        self.leaf_index = leaf_index
        self.tree_id = tree_id
        self.path_prob = path_prob
        self.is_classifier = is_classifier
        self.uses_scores = uses_scores
        self.label_names = label_names
        self.ecdf_dict = ecdf_dict
        self.n_trees = int(n_trees) if n_trees else 0

        if not verbose:
            return
        if n_samples.size == 0:
            logger.warning("PathwaySet produced an empty dataset.")
        else:
            total = float(path_prob_forest.sum()) if path_prob_forest.size else 0.0
            if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-6):
                logger.warning(
                    "path_prob_forest sums to %.6f (expected ~1.0)", total
                )

    def __len__(self) -> int:
        return int(self.n_samples.size)

    def get_pathway_set_df(self):
        return self.to_dataframe()

    def to_dataframe(self):
        import pandas as pd

        data: Dict[str, Any] = {}
        for idx, feat in enumerate(self.feature_names):
            data[f"{feat}_upper"] = (
                self.features_upper[:, idx] if self.features_upper.size else []
            )
            data[f"{feat}_lower"] = (
                self.features_lower[:, idx] if self.features_lower.size else []
            )
            data[f"{feat}_lower_inclusive"] = (
                self.features_lower_inclusive[:, idx]
                if self.features_lower_inclusive.size
                else []
            )
            data[f"{feat}_upper_inclusive"] = (
                self.features_upper_inclusive[:, idx]
                if self.features_upper_inclusive.size
                else []
            )

        data["n_samples"] = self.n_samples
        data["leaf_cover"] = self.leaf_cover
        data["path_prob_mc"] = self.path_prob_mc
        data["path_prob_forest"] = self.path_prob_forest
        data["leaf_index"] = self.leaf_index
        data["tree_id"] = self.tree_id
        data["path_prob"] = self.path_prob

        if self.is_classifier:
            if self.output.size:
                data["probas"] = [row.tolist() for row in self.output]
            else:
                data["probas"] = []
        else:
            value_col = "scores" if self.uses_scores else "regressions"
            data[value_col] = self.output.astype(float)

        return pd.DataFrame(data)

    def get_ecdf_dict(self) -> Dict[int, ECDF]:
        """
        Return the internal mapping {feature_index -> ECDF}.
        """
        return self.ecdf_dict

    def _infer_feature_count_from_trees(self, trees: List[Any]) -> int:
        for tree in trees:
            n_feat = getattr(tree, "n_features", None)
            if n_feat is not None:
                try:
                    return int(n_feat)
                except Exception:
                    pass
            feat_arr = getattr(tree, "feature", None)
            if isinstance(feat_arr, np.ndarray) and feat_arr.size:
                valid = feat_arr[feat_arr >= 0]
                if valid.size:
                    return int(valid.max()) + 1
        return 0

    def _extract_paths(
        self,
        trees: List[Any],
        n_features: int,
        is_classifier: bool,
        n_classes: int,
        X_train: Optional[np.ndarray] = None,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        if not trees:
            return self._empty_arrays(n_features, n_classes, is_classifier)

        features_lower_list: List[Array] = []
        features_upper_list: List[Array] = []
        features_lower_inclusive_list: List[Array] = []
        features_upper_inclusive_list: List[Array] = []
        features_upper_depth_list: List[Array] = []
        n_samples_list: List[float] = []
        leaf_cover_list: List[float] = []
        path_prob_mc_list: List[float] = []
        output_list: List[Any] = []
        leaf_index_list: List[str] = []
        tree_id_list: List[int] = []

        for tree_idx, tree_ in enumerate(trees):
            node_counts = self._node_counts_for_tree(tree_, X_train)
            leaf_idxs = [
                i
                for i in range(tree_.node_count)
                if tree_.children_left[i] == -1 and tree_.children_right[i] == -1
            ]
            root_samples = (
                float(node_counts[0]) if node_counts.size else 0.0
            )

            for local_leaf_idx, leaf_idx in enumerate(leaf_idxs):
                (
                    features_lower,
                    features_upper,
                    features_lower_inclusive,
                    features_upper_inclusive,
                    features_upper_depth,
                    n_leaf,
                    path_prob_mc,
                    output,
                ) = self._get_path_from_leaf(
                    tree_=tree_,
                    leaf_idx=leaf_idx,
                    n_features=n_features,
                    root_samples=root_samples,
                    node_counts=node_counts,
                    is_classifier=is_classifier,
                )
                leaf_cover = self._leaf_cover_for_leaf(tree_, leaf_idx, fallback=n_leaf)
                features_lower_list.append(features_lower)
                features_upper_list.append(features_upper)
                features_lower_inclusive_list.append(features_lower_inclusive)
                features_upper_inclusive_list.append(features_upper_inclusive)
                features_upper_depth_list.append(features_upper_depth)
                n_samples_list.append(float(n_leaf))
                leaf_cover_list.append(float(leaf_cover))
                path_prob_mc_list.append(float(path_prob_mc))
                output_list.append(output)
                leaf_index_list.append(f"{tree_idx}_{local_leaf_idx}")
                tree_id_list.append(tree_idx)

        if not n_samples_list:
            return self._empty_arrays(n_features, n_classes, is_classifier)

        features_lower = np.vstack(features_lower_list) if features_lower_list else np.empty((0, n_features), dtype=float)
        features_upper = np.vstack(features_upper_list) if features_upper_list else np.empty((0, n_features), dtype=float)
        features_lower_inclusive = (
            np.vstack(features_lower_inclusive_list)
            if features_lower_inclusive_list
            else np.empty((0, n_features), dtype=bool)
        )
        features_upper_inclusive = (
            np.vstack(features_upper_inclusive_list)
            if features_upper_inclusive_list
            else np.empty((0, n_features), dtype=bool)
        )
        features_upper_depth = (
            np.vstack(features_upper_depth_list)
            if features_upper_depth_list
            else np.empty((0, n_features), dtype=int)
        )
        n_samples = np.asarray(n_samples_list, dtype=float)
        leaf_cover = np.asarray(leaf_cover_list, dtype=float)
        path_prob_mc = np.asarray(path_prob_mc_list, dtype=float)
        leaf_index = np.asarray(leaf_index_list, dtype=str)
        tree_id = np.asarray(tree_id_list, dtype=int)

        if is_classifier:
            if output_list:
                output = np.vstack([np.asarray(v, dtype=float) for v in output_list])
            else:
                output = np.empty((0, n_classes), dtype=float)
        else:
            output = np.asarray(output_list, dtype=float)

        return (
            features_lower,
            features_upper,
            features_lower_inclusive,
            features_upper_inclusive,
            features_upper_depth,
            n_samples,
            leaf_cover,
            path_prob_mc,
            output,
            leaf_index,
            tree_id,
        )

    def _empty_arrays(
        self,
        n_features: int,
        n_classes: int,
        is_classifier: bool,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        features_lower = np.empty((0, n_features), dtype=float)
        features_upper = np.empty((0, n_features), dtype=float)
        features_lower_inclusive = np.empty((0, n_features), dtype=bool)
        features_upper_inclusive = np.empty((0, n_features), dtype=bool)
        features_upper_depth = np.empty((0, n_features), dtype=int)
        n_samples = np.array([], dtype=float)
        leaf_cover = np.array([], dtype=float)
        path_prob_mc = np.array([], dtype=float)
        leaf_index = np.array([], dtype=str)
        tree_id = np.array([], dtype=int)
        if is_classifier:
            output = np.empty((0, n_classes), dtype=float)
        else:
            output = np.array([], dtype=float)
        return (
            features_lower,
            features_upper,
            features_lower_inclusive,
            features_upper_inclusive,
            features_upper_depth,
            n_samples,
            leaf_cover,
            path_prob_mc,
            output,
            leaf_index,
            tree_id,
        )

    def _leaf_cover_for_leaf(
        self,
        tree_: Any,
        leaf_idx: int,
        fallback: float,
    ) -> float:
        for attr_name in ("booster_cover", "n_node_samples"):
            raw = getattr(tree_, attr_name, None)
            if raw is None:
                continue
            try:
                return float(raw[leaf_idx])
            except Exception:
                continue
        return float(fallback)

    def _needs_sample_routing(self, tree_: Any) -> bool:
        split_ops = getattr(tree_, "split_operator", None)
        if split_ops is not None:
            try:
                arr = np.asarray(split_ops, dtype=object)
                if np.any(arr == "<"):
                    return True
            except Exception:
                return True
        missing = getattr(tree_, "missing_child", None)
        if missing is not None:
            try:
                arr = np.asarray(missing, dtype=object)
                return bool(np.any(pd.notna(arr)))
            except Exception:
                return True
        return False

    def _split_operator_for_node(self, tree_: Any, node_idx: int) -> str:
        ops = getattr(tree_, "split_operator", None)
        if ops is not None:
            try:
                op = ops[node_idx]
                if op in ("<", "<="):
                    return str(op)
            except Exception:
                pass
        return "<="

    def _missing_child_for_node(self, tree_: Any, node_idx: int) -> Optional[str]:
        missing = getattr(tree_, "missing_child", None)
        if missing is None:
            return None
        try:
            child = missing[node_idx]
        except Exception:
            return None
        return child if child in ("left", "right") else None

    def _split_masks_for_tree(
        self,
        tree_: Any,
        node_idx: int,
        samples: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        feature_idx = int(tree_.feature[node_idx])
        threshold = float(tree_.threshold[node_idx])
        if feature_idx < 0 or np.isnan(threshold):
            return mask.copy(), np.zeros_like(mask, dtype=bool)

        feature_vals = samples[:, feature_idx]
        missing_mask = mask & pd.isna(feature_vals)
        active_mask = mask & ~missing_mask
        left_numeric = np.zeros_like(mask, dtype=bool)
        if np.any(active_mask):
            active_vals = feature_vals[active_mask]
            if self._split_operator_for_node(tree_, node_idx) == "<":
                left_numeric[active_mask] = active_vals < threshold
            else:
                left_numeric[active_mask] = active_vals <= threshold
        left_mask = active_mask & left_numeric
        right_mask = active_mask & ~left_numeric

        missing_child = self._missing_child_for_node(tree_, node_idx)
        if missing_child == "left":
            left_mask |= missing_mask
        elif missing_child == "right":
            right_mask |= missing_mask
        return left_mask, right_mask

    def _node_counts_for_tree(
        self,
        tree_: Any,
        X_train: Optional[np.ndarray],
    ) -> np.ndarray:
        fallback = getattr(tree_, "n_node_samples", None)
        if fallback is not None and not self._needs_sample_routing(tree_):
            return np.asarray(fallback, dtype=float)
        if X_train is None:
            if fallback is not None:
                return np.asarray(fallback, dtype=float)
            return np.zeros(int(getattr(tree_, "node_count", 0)), dtype=float)

        counts = np.zeros(int(tree_.node_count), dtype=float)

        def walk(node_idx: int, mask: np.ndarray) -> None:
            counts[node_idx] = float(mask.sum())
            left_idx = int(tree_.children_left[node_idx])
            right_idx = int(tree_.children_right[node_idx])
            feature_idx = int(tree_.feature[node_idx])
            threshold = float(tree_.threshold[node_idx])
            is_leaf = (
                left_idx == -1
                and right_idx == -1
            ) or feature_idx < 0 or np.isnan(threshold)
            if is_leaf:
                return
            left_mask, right_mask = self._split_masks_for_tree(tree_, node_idx, X_train, mask)
            if left_idx >= 0:
                walk(left_idx, left_mask)
            if right_idx >= 0:
                walk(right_idx, right_mask)

        walk(0, np.ones(X_train.shape[0], dtype=bool))
        return counts

    def _get_path_from_leaf(
        self,
        tree_: Any,
        leaf_idx: int,
        n_features: int,
        root_samples: int,
        node_counts: np.ndarray,
        is_classifier: bool,
    ) -> tuple[Array, Array, Array, Array, Array, float, float, Array | float]:
        if is_classifier:
            probas = tree_.value[leaf_idx][0]
            total = probas.sum()
            if total > 0:
                output = np.asarray(probas, dtype=float) / float(total)
            else:
                output = np.zeros_like(probas, dtype=float)
        else:
            output = float(tree_.value[leaf_idx][0][0])

        n_leaf = float(node_counts[leaf_idx]) if node_counts.size else 0.0
        path_prob_mc = float(n_leaf) / float(root_samples) if root_samples > 0 else 0.0

        features_upper = np.full(n_features, np.inf, dtype=float)
        features_lower = np.full(n_features, -np.inf, dtype=float)
        features_upper_inclusive = np.zeros(n_features, dtype=bool)
        features_lower_inclusive = np.zeros(n_features, dtype=bool)
        features_upper_depth = np.full(n_features, -1, dtype=int)

        node_id = leaf_idx
        depth_from_leaf = 0
        while node_id != 0:
            parent_idx = np.where(tree_.children_left == node_id)[0]
            bound = "upper"
            if parent_idx.size == 0:
                bound = "lower"
                parent_idx = np.where(tree_.children_right == node_id)[0]
            depth_from_leaf += 1
            pid = int(parent_idx[0])
            feature = tree_.feature[pid]
            threshold = float(tree_.threshold[pid])
            if np.isnan(threshold):
                node_id = pid
                continue

            if feature >= 0:
                split_operator = self._split_operator_for_node(tree_, pid)
                if bound == "lower":
                    is_inclusive = split_operator == "<"
                    if features_lower[feature] < threshold:
                        features_lower[feature] = threshold
                        features_lower_inclusive[feature] = is_inclusive
                    elif np.isclose(features_lower[feature], threshold):
                        features_lower_inclusive[feature] = bool(features_lower_inclusive[feature]) and bool(is_inclusive)
                else:
                    is_inclusive = split_operator != "<"
                    if features_upper[feature] > threshold:
                        features_upper[feature] = threshold
                        features_upper_inclusive[feature] = is_inclusive
                        features_upper_depth[feature] = depth_from_leaf
                    elif np.isclose(features_upper[feature], threshold):
                        features_upper_inclusive[feature] = bool(features_upper_inclusive[feature]) and bool(is_inclusive)

            node_id = pid

        if depth_from_leaf > 0:
            valid = features_upper_depth >= 0
            if np.any(valid):
                features_upper_depth[valid] = depth_from_leaf - features_upper_depth[valid]

        return (
            features_lower,
            features_upper,
            features_lower_inclusive,
            features_upper_inclusive,
            features_upper_depth,
            n_leaf,
            path_prob_mc,
            output,
        )

    def _calculate_path_prob_by_ecdf(
        self,
        features_lower: Array,
        features_upper: Array,
        features_lower_inclusive: Array,
        features_upper_inclusive: Array,
        ecdf_dict: Dict[int, ECDF],
    ) -> Array:
        if features_lower.size == 0:
            return np.array([], dtype=float)

        eps = 1e-9
        probs = np.ones(features_lower.shape[0], dtype=float)
        for idx, ecdf in ecdf_dict.items():
            low = features_lower[:, idx]
            up = features_upper[:, idx]
            low_adj = np.where(
                np.isfinite(low) & features_lower_inclusive[:, idx],
                np.nextafter(low, -np.inf),
                low,
            )
            up_adj = np.where(
                np.isfinite(up) & ~features_upper_inclusive[:, idx],
                np.nextafter(up, -np.inf),
                up,
            )
            p_low = np.asarray(ecdf(low_adj), dtype=float)
            p_up = np.asarray(ecdf(up_adj), dtype=float)
            q_low = np.clip(p_low, 0.0, 1.0)
            q_up = np.clip(p_up, 0.0, 1.0)
            mass_1d = np.maximum(q_up - q_low, 0.0) + eps
            probs *= mass_1d

        return probs


if __name__ == "__main__":
    import pandas as pd

    from eywa_trees.utils import setup_toy_classifier, setup_toy_regressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    seed = 42

    n_features = 3
    Xc, yc, class_names_c = setup_toy_classifier(
        n_samples=50, n_features=n_features, n_classes=2, random_state=seed
    )
    rf_c = RandomForestClassifier(random_state=seed)
    rf_c.fit(Xc, yc)
    Xc_df = pd.DataFrame(Xc, columns=[str(i + 1) for i in range(n_features)])
    ps_c = PathwaySet(
        feature_names=Xc_df.columns,
        X_train=Xc_df.values,
        model=rf_c,
        random_state=seed,
        verbose=True,
    )
    # df_c = ps_c.to_dataframe()
    # print(df_c.to_markdown())

    n_samples = 300
    n_features = 3
    Xr, yr = setup_toy_regressor(
        n_samples=n_samples, n_features=n_features, random_state=seed
    )
    rf_r = RandomForestRegressor(random_state=seed, n_estimators=1000)
    rf_r.fit(Xr, yr)
    Xr_df = pd.DataFrame(Xr, columns=[str(i + 1) for i in range(n_features)])

    import time

    start_time = time.time()
    ps_r = PathwaySet(
        feature_names=Xr_df.columns,
        X_train=Xr_df.values,
        model=rf_r,
        random_state=seed,
        verbose=True,
    )
    print(time.time() - start_time)

    # df_r = ps_r.to_dataframe()
    # print(df_r.to_markdown())
