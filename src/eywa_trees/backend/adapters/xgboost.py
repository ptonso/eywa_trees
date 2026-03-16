from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict
import json

import numpy as np
import pandas as pd

from eywa_trees.logger import setup_logger
from eywa_trees.backend.adapters.base import (
    ArrayLike,
    BaseModelAdapter,
    PathwayAdapterResult,
    build_vis_tree_from_struct,
    infer_class_names,
    infer_feature_names,
)
from eywa_trees.backend.vistree import VisTree

LOGGER = setup_logger("api.log")


def _safe_float(value: Any, default: float) -> float:
    try:
        return default if value is None else float(value)
    except Exception:
        return default


def _parse_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_feature_name(
    feat_raw: Any,
    feature_name_map: Optional[Dict[str, int]],
) -> Optional[int]:
    if feat_raw is None:
        return None
    if isinstance(feat_raw, str):
        feat_key = feat_raw.strip()
        if not feat_key or feat_key.lower() == "leaf":
            return None
        if feat_key.startswith("f") and feat_key[1:].isdigit():
            return int(feat_key[1:])
        if feature_name_map is not None and feat_key in feature_name_map:
            return feature_name_map[feat_key]
        try:
            return int(feat_key)
        except Exception:
            return None
    if isinstance(feat_raw, (int, np.integer)):
        return int(feat_raw)
    return None


def is_xgboost_model(model: Any) -> bool:
    cls = model.__class__
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__name__", "") or ""
    return ("xgboost" in module.lower() or name.lower().startswith("xgb")) and hasattr(model, "get_booster")


def get_xgb_tree_group_size(model: Any) -> int:
    params = model.get_params() if hasattr(model, "get_params") else {}
    try:
        num_parallel_tree = int(params.get("num_parallel_tree", 1) or 1)
    except Exception:
        num_parallel_tree = 1

    num_classes = 1
    classes = getattr(model, "classes_", None)
    try:
        if classes is not None:
            n = len(list(classes))
            if n > 2:
                num_classes = n
        elif params.get("num_class") is not None:
            n = int(params.get("num_class") or 1)
            if n > 2:
                num_classes = n
    except Exception:
        num_classes = 1

    return max(1, num_parallel_tree * num_classes)


def get_xgb_tree_metadata(
    model: Any,
    tree_index: int,
    total_trees: int,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    group_size = get_xgb_tree_group_size(model)
    idx = max(0, int(tree_index))
    num_rounds = max(1, int(np.ceil(float(total_trees) / float(group_size)))) if total_trees else 1
    round_index = idx // group_size if group_size > 1 else idx
    class_index = idx % group_size

    class_label: Optional[str] = None
    if class_names and 0 <= class_index < len(class_names):
        class_label = str(list(class_names)[class_index])

    return {
        "tree_index": idx,
        "group_size": group_size,
        "num_rounds": num_rounds,
        "round_index": round_index,
        "class_index": class_index,
        "class_label": class_label,
    }


def extract_xgb_tree_shims(model: Any) -> Tuple[List[SimpleNamespace], Optional[List[str]], float, float]:
    booster = model.get_booster()
    df = None
    try:
        df = booster.trees_to_dataframe()
    except Exception:
        df = None

    feature_names = getattr(booster, "feature_names", None)
    if feature_names is None and hasattr(model, "feature_names_in_"):
        try:
            feature_names = list(model.feature_names_in_)
        except Exception:
            feature_names = None
    feature_name_map: Optional[Dict[str, int]] = None
    if feature_names is not None:
        feature_name_map = {str(name): idx for idx, name in enumerate(feature_names)}

    learning_rate = _safe_float(getattr(model, "learning_rate", None), 1.0)
    base_score = _safe_float(getattr(model, "base_score", None), 0.0)

    trees: List[SimpleNamespace] = []
    dump: Optional[List[str]] = None
    try:
        dump = booster.get_dump(dump_format="json", with_stats=True)
    except Exception:
        try:
            dump = booster.get_dump(dump_format="json")
        except Exception:
            dump = None

    if dump is not None:
        try:
            for tree_json in dump:
                tree_obj = json.loads(tree_json)
                trees.append(_tree_json_to_shim(tree_obj, feature_name_map))
        except Exception:
            trees = []

    if not trees:
        if df is None:
            return (
                [],
                feature_names,
                learning_rate,
                base_score,
            )
        for _, tree_df in df.groupby("Tree"):
            trees.append(_tree_df_to_shim(tree_df, feature_name_map))

    try:
        lr_param = getattr(model, "learning_rate", None)
        if lr_param is None:
            lr_param = model.get_params().get("learning_rate", 1.0)
        learning_rate = float(lr_param) if lr_param is not None else 1.0
    except Exception:
        learning_rate = 1.0

    try:
        base_param = getattr(model, "base_score", None)
        if base_param is None:
            base_param = model.get_params().get("base_score", 0.0)
        base_score = float(base_param) if base_param is not None else 0.0
    except Exception:
        base_score = 0.0

    return trees, feature_names, learning_rate, base_score


class XGBoostAdapter(BaseModelAdapter):
    name = "xgboost"

    def is_compatible(self, model: Any) -> bool:
        return is_xgboost_model(model)

    def build_pathway_inputs(
        self,
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> PathwayAdapterResult:
        trees, xgb_feat_names, _, _ = extract_xgb_tree_shims(model)
        if feature_names:
            feat_names = list(feature_names)
        elif xgb_feat_names is not None:
            feat_names = list(xgb_feat_names)
        else:
            feat_names = infer_feature_names(model, None)
        return PathwayAdapterResult(
            trees=trees,
            feature_names=feat_names,
            is_classifier=False,
            uses_scores=True,
            label_names=None,
            n_trees=len(trees),
        )

    def build_vis_trees(
        self,
        model: Any,
        X: Optional[ArrayLike] = None,
        class_names: Optional[Sequence[str]] = None,
        log_coloring: bool = False,
    ) -> List[VisTree]:
        feat_names = infer_feature_names(model, X)
        cls_names = infer_class_names(model, class_names)
        trees, xgb_feat_names, lr, base_score = extract_xgb_tree_shims(model)
        feat_names = feat_names or (list(xgb_feat_names) if xgb_feat_names else None)

        if not trees:
            LOGGER.warning("No XGBoost trees extracted; falling back to model.tree_ if present.")
            if hasattr(model, "tree_"):
                trees = [getattr(model, "tree_")]

        vis_trees = [
            build_vis_tree_from_struct(
                model,
                tree,
                X,
                feature_names=feat_names,
                class_names=cls_names,
                is_classifier=False,
                uses_scores=True,
                learning_rate=lr,
                base_score=base_score,
                log_coloring=log_coloring,
            )
            for tree in trees
        ]
        total_trees = len(vis_trees)
        for tree_idx, vis_tree in enumerate(vis_trees):
            meta = get_xgb_tree_metadata(model, tree_idx, total_trees, class_names=cls_names)
            setattr(vis_tree, "xgb_tree_index", meta["tree_index"])
            setattr(vis_tree, "xgb_group_size", meta["group_size"])
            setattr(vis_tree, "xgb_num_rounds", meta["num_rounds"])
            setattr(vis_tree, "xgb_round_index", meta["round_index"])
            setattr(vis_tree, "xgb_class_index", meta["class_index"])
            setattr(vis_tree, "xgb_class_label", meta["class_label"])
        return vis_trees


class _XGBNode(TypedDict, total=False):
    raw_id: Optional[Any]
    feature: Optional[int]
    threshold: Optional[float]
    left: Optional[Any]
    right: Optional[Any]
    missing: Optional[Any]
    value: Optional[float]
    cover: float


def _build_xgb_shim(
    nodes: Dict[Any, _XGBNode],
    root_id: Any,
) -> SimpleNamespace:
    ordered_ids: List[Any] = []
    queue = [root_id]
    seen: set[Any] = set()
    while queue:
        current = queue.pop(0)
        if current in seen or current not in nodes:
            continue
        seen.add(current)
        ordered_ids.append(current)
        node = nodes[current]
        for child in (node.get("left"), node.get("right")):
            if child in nodes and child not in seen:
                queue.append(child)

    if not ordered_ids:
        return SimpleNamespace(
            node_count=0,
            children_left=np.array([], dtype=int),
            children_right=np.array([], dtype=int),
            feature=np.array([], dtype=int),
            threshold=np.array([], dtype=float),
            value=np.empty((0, 1, 1), dtype=float),
            n_node_samples=np.array([], dtype=float),
            raw_node_ids=np.array([], dtype=object),
            split_operator=np.array([], dtype=object),
            missing_child=np.array([], dtype=object),
            booster_cover=np.array([], dtype=float),
        )

    id_map = {nid: idx for idx, nid in enumerate(ordered_ids)}
    n_nodes = len(ordered_ids)

    children_left = np.full(n_nodes, -1, dtype=int)
    children_right = np.full(n_nodes, -1, dtype=int)
    feature = np.full(n_nodes, -2, dtype=int)
    threshold = np.full(n_nodes, np.nan, dtype=float)
    value = np.zeros((n_nodes, 1, 1), dtype=float)
    n_node_samples = np.zeros(n_nodes, dtype=float)
    raw_node_ids = np.empty(n_nodes, dtype=object)
    split_operator = np.full(n_nodes, "<=", dtype=object)
    missing_child = np.empty(n_nodes, dtype=object)
    missing_child[:] = None
    booster_cover = np.zeros(n_nodes, dtype=float)

    for node_id, idx in id_map.items():
        node = nodes.get(node_id, {})
        cover = float(node.get("cover", 0.0))
        n_node_samples[idx] = cover
        booster_cover[idx] = cover
        raw_node_ids[idx] = node.get("raw_id", node_id)

        node_value = node.get("value", None)
        if node_value is not None:
            value[idx, 0, 0] = float(node_value)
            continue

        split_operator[idx] = "<"
        feat_idx = node.get("feature", None)
        thr_val = node.get("threshold", None)
        if feat_idx is not None:
            feature[idx] = int(feat_idx)
        if thr_val is not None:
            threshold[idx] = float(thr_val)

        left_id = node.get("left")
        right_id = node.get("right")
        missing_id = node.get("missing")
        if left_id in id_map:
            children_left[idx] = id_map[left_id]
        if right_id in id_map:
            children_right[idx] = id_map[right_id]
        if missing_id == left_id:
            missing_child[idx] = "left"
        elif missing_id == right_id:
            missing_child[idx] = "right"

    return SimpleNamespace(
        node_count=n_nodes,
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        value=value,
        n_node_samples=n_node_samples,
        raw_node_ids=raw_node_ids,
        split_operator=split_operator,
        missing_child=missing_child,
        booster_cover=booster_cover,
    )


def _tree_json_to_shim(
    tree: Dict[str, Any],
    feature_name_map: Optional[Dict[str, int]],
) -> SimpleNamespace:
    nodes: Dict[int, _XGBNode] = {}

    def walk(node: Dict[str, Any]) -> None:
        node_id = int(node.get("nodeid", 0))
        cover_val = node.get("cover", node.get("sum_hess", 0.0))
        if "leaf" in node:
            nodes[node_id] = {
                "raw_id": node_id,
                "value": float(node["leaf"]),
                "cover": float(cover_val),
            }
            return

        feature_idx = _parse_feature_name(node.get("split"), feature_name_map)
        threshold_val = _parse_float(node.get("split_condition"))
        nodes[node_id] = {
            "raw_id": node_id,
            "feature": feature_idx,
            "threshold": threshold_val,
            "left": node.get("yes"),
            "right": node.get("no"),
            "missing": node.get("missing"),
            "cover": float(cover_val),
        }
        for child in node.get("children", []):
            if isinstance(child, dict):
                walk(child)

    walk(tree)
    root_id = int(tree.get("nodeid", 0))
    return _build_xgb_shim(nodes, root_id)


def _tree_df_to_shim(
    tree_df: pd.DataFrame,
    feature_name_map: Optional[Dict[str, int]],
) -> SimpleNamespace:
    tree_df = tree_df.copy()

    def _normalize_node_id(value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, float):
            if np.isnan(value):
                return None
            if value.is_integer():
                return str(int(value))
            return str(value)
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        try:
            return str(value)
        except Exception:
            return None

    def _parse_leaf_value(row: pd.Series) -> Optional[float]:
        leaf_val = row.get("Leaf", np.nan)
        if pd.notna(leaf_val):
            return float(leaf_val)
        feat_raw = row.get("Feature", None)
        if isinstance(feat_raw, str) and feat_raw.strip().lower() == "leaf":
            split_val = row.get("Split", np.nan)
            if pd.notna(split_val):
                return float(split_val)
        return None

    def _normalize_child_id(value: Any) -> Optional[str]:
        norm = _normalize_node_id(value)
        return norm if norm is not None else None

    def _raw_node_id(row: pd.Series, node_col_name: str) -> Optional[int]:
        node_val = row.get("Node", np.nan)
        if pd.notna(node_val):
            try:
                return int(node_val)
            except Exception:
                pass
        node_id = _normalize_node_id(row.get(node_col_name))
        if node_id is None:
            return None
        if "-" in node_id:
            suffix = node_id.rsplit("-", 1)[-1]
            try:
                return int(suffix)
            except Exception:
                return None
        try:
            return int(node_id)
        except Exception:
            return None

    node_col = "ID"
    if "ID" not in tree_df.columns and "Node" in tree_df.columns:
        node_col = "Node"
    elif "ID" in tree_df.columns and "Node" in tree_df.columns:
        sample_yes = tree_df["Yes"].dropna().head(20)
        if sample_yes.apply(lambda v: isinstance(v, (int, np.integer, float))).any():
            node_col = "Node"

    nodes: Dict[str, _XGBNode] = {}
    for _, row in tree_df.iterrows():
        node_id = _normalize_node_id(row.get(node_col))
        if node_id is None:
            continue
        raw_id = _raw_node_id(row, node_col)
        leaf_val = _parse_leaf_value(row)
        if leaf_val is not None:
            nodes[node_id] = {
                "raw_id": raw_id,
                "value": leaf_val,
                "cover": float(row.get("Cover", 0.0)),
            }
            continue

        feature_idx = _parse_feature_name(row.get("Feature", None), feature_name_map)
        threshold_val = _parse_float(row.get("Split", np.nan))
        left_id = _normalize_child_id(row.get("Yes", np.nan))
        right_id = _normalize_child_id(row.get("No", np.nan))
        missing_id = _normalize_child_id(row.get("Missing", np.nan))
        nodes[node_id] = {
            "raw_id": raw_id,
            "feature": feature_idx,
            "threshold": threshold_val,
            "left": left_id,
            "right": right_id,
            "missing": missing_id,
            "cover": float(row.get("Cover", 0.0)),
        }

    root_id: Optional[str] = None
    if "Node" in tree_df.columns:
        root_rows = tree_df[tree_df["Node"] == 0]
        if not root_rows.empty:
            root_id = _normalize_node_id(root_rows.iloc[0].get(node_col))

    if root_id is None:
        child_ids: set[str] = set()
        for col in ("Yes", "No", "Missing"):
            if col not in tree_df.columns:
                continue
            for raw in tree_df[col].tolist():
                child_id = _normalize_node_id(raw)
                if child_id is not None:
                    child_ids.add(child_id)
        for node_id in nodes.keys():
            if node_id not in child_ids:
                root_id = node_id
                break

    if root_id is None:
        root_id = next(iter(nodes.keys()), None)

    if root_id is None:
        return _build_xgb_shim({}, None)
    return _build_xgb_shim(nodes, root_id)
