from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from eywa_trees.backend.vistree import VisNode, VisTree


@dataclass(frozen=True)
class FocusedTreeStep:
    feature_idx: int
    threshold: float
    split_operator: str
    branch_is_left: bool
    summary: Dict[str, object]
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    hist: Optional[Dict[str, object]] = None
    extra_hover: str = ""


@dataclass(frozen=True)
class FocusedTreeLeaf:
    summary: Dict[str, object]
    hist: Optional[Dict[str, object]] = None
    extra_hover: str = ""


def build_linear_focused_tree(
    *,
    feature_names: Optional[List[str]],
    class_names: Optional[List[str]],
    is_classifier: bool,
    uses_scores: bool,
    steps: Sequence[FocusedTreeStep],
    leaf: FocusedTreeLeaf,
) -> VisTree:
    all_values = [step.summary.get("pred") for step in steps]
    all_values.append(leaf.summary.get("pred"))
    vis_tree = _base_vis_tree(
        feature_names=feature_names,
        class_names=class_names,
        is_classifier=is_classifier,
        uses_scores=uses_scores,
        values=all_values,
    )
    vis_tree.max_depth = max(0, len(steps))
    if steps:
        vis_tree.n_train = int(round(float(steps[0].summary.get("n_train", 0.0))))
    else:
        vis_tree.n_train = int(round(float(leaf.summary.get("n_train", 0.0))))

    nodes: Dict[int, VisNode] = {}
    leaf_paths: Dict[int, List[int]] = {}
    path_ids: List[int] = []

    for idx, step in enumerate(steps):
        node = _build_vis_node(
            node_id=idx,
            parent=(idx - 1) if idx > 0 else None,
            is_left=step.branch_is_left,
            feature=step.feature_idx,
            threshold=step.threshold,
            summary=step.summary,
            hist=step.hist,
            split_operator=step.split_operator,
            threshold_min=step.threshold_min,
            threshold_max=step.threshold_max,
            extra_hover=step.extra_hover,
        )
        nodes[idx] = node
        path_ids.append(idx)
        if idx > 0:
            prev = nodes[idx - 1]
            if steps[idx - 1].branch_is_left:
                prev.left = idx
            else:
                prev.right = idx

    leaf_id = len(steps)
    leaf_node = _build_vis_node(
        node_id=leaf_id,
        parent=(leaf_id - 1) if steps else None,
        is_left=steps[-1].branch_is_left if steps else None,
        feature=None,
        threshold=None,
        summary=leaf.summary,
        hist=leaf.hist,
        split_operator="<=",
        extra_hover=leaf.extra_hover,
    )
    if steps:
        last = nodes[leaf_id - 1]
        if steps[-1].branch_is_left:
            last.left = leaf_id
        else:
            last.right = leaf_id
    nodes[leaf_id] = leaf_node
    path_ids.append(leaf_id)
    leaf_paths[leaf_id] = path_ids

    vis_tree.nodes = nodes
    vis_tree.leaf_paths = leaf_paths
    vis_tree._generate_color_struct()
    return vis_tree


def build_binary_focused_tree(
    *,
    feature_names: Optional[List[str]],
    class_names: Optional[List[str]],
    is_classifier: bool,
    uses_scores: bool,
    root_step: FocusedTreeStep,
    left_leaf: FocusedTreeLeaf,
    right_leaf: FocusedTreeLeaf,
) -> VisTree:
    all_values = [
        root_step.summary.get("pred"),
        left_leaf.summary.get("pred"),
        right_leaf.summary.get("pred"),
    ]
    vis_tree = _base_vis_tree(
        feature_names=feature_names,
        class_names=class_names,
        is_classifier=is_classifier,
        uses_scores=uses_scores,
        values=all_values,
    )
    vis_tree.max_depth = 1
    vis_tree.n_train = int(round(float(root_step.summary.get("n_train", 0.0))))

    root_node = _build_vis_node(
        node_id=0,
        parent=None,
        is_left=None,
        feature=root_step.feature_idx,
        threshold=root_step.threshold,
        summary=root_step.summary,
        hist=root_step.hist,
        split_operator=root_step.split_operator,
        threshold_min=root_step.threshold_min,
        threshold_max=root_step.threshold_max,
        extra_hover=root_step.extra_hover,
    )
    root_node.left = 1
    root_node.right = 2

    left_node = _build_vis_node(
        node_id=1,
        parent=0,
        is_left=True,
        feature=None,
        threshold=None,
        summary=left_leaf.summary,
        hist=left_leaf.hist,
        split_operator="<=",
        extra_hover=left_leaf.extra_hover,
    )
    right_node = _build_vis_node(
        node_id=2,
        parent=0,
        is_left=False,
        feature=None,
        threshold=None,
        summary=right_leaf.summary,
        hist=right_leaf.hist,
        split_operator="<=",
        extra_hover=right_leaf.extra_hover,
    )

    vis_tree.nodes = {0: root_node, 1: left_node, 2: right_node}
    vis_tree.leaf_paths = {1: [0, 1], 2: [0, 2]}
    vis_tree._generate_color_struct()
    return vis_tree


def _base_vis_tree(
    *,
    feature_names: Optional[List[str]],
    class_names: Optional[List[str]],
    is_classifier: bool,
    uses_scores: bool,
    values: Iterable[Any],
) -> VisTree:
    vis_tree = VisTree(
        model=None,
        feature_names=feature_names,
        class_names=class_names,
        is_classifier=is_classifier,
        uses_scores=uses_scores,
    )
    collected = _collect_values(values)
    if is_classifier:
        for value in values:
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.size:
                vis_tree.n_classes = int(value.size)
                break
    else:
        vis_tree.possible_values = set(collected)
    return vis_tree


def _build_vis_node(
    *,
    node_id: int,
    parent: Optional[int],
    is_left: Optional[bool],
    feature: Optional[int],
    threshold: Optional[float],
    summary: Dict[str, object],
    hist: Optional[Dict[str, object]],
    split_operator: str,
    threshold_min: Optional[float] = None,
    threshold_max: Optional[float] = None,
    extra_hover: str = "",
) -> VisNode:
    node = VisNode(
        id=node_id,
        feature=feature,
        threshold=threshold,
        value=summary.get("pred"),
        parent=parent,
        is_left=is_left,
        left=None,
        right=None,
        n_train=int(round(float(summary.get("n_train", 0.0)))),
        hist=hist,
        coverage=float(summary.get("coverage", 0.0)),
        coverage_std=float(summary.get("coverage_std", 0.0)),
        n_train_std=float(summary.get("n_train_std", 0.0)),
        split_operator=split_operator,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        extra_hover=extra_hover,
    )
    return node


def _collect_values(values: Iterable[Any]) -> List[float]:
    collected: List[float] = []
    for value in values:
        if isinstance(value, (float, int, np.floating, np.integer)):
            collected.append(float(value))
            continue
        if isinstance(value, np.ndarray):
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size:
                collected.append(float(arr[0]))
    return collected
