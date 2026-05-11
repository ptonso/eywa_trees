from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from eywa_trees.backend.ecdf_rule_group import ecdf_bin_indices
from eywa_trees.backend.focused_tree import (
    FocusedTreeLeaf,
    FocusedTreeStep,
    build_linear_focused_tree,
)
from eywa_trees.backend.pset import PathStep, PathwaySet
from eywa_trees.backend.rule_engine import RuleEngine, SortMode
from eywa_trees.backend.vistree import VisTree


Array = np.ndarray


@dataclass(frozen=True)
class SubPathGroup:
    group_id: int
    feature: str
    feature_idx: int
    threshold_mean: float
    threshold_min: float
    threshold_max: float
    direction: str
    inclusive: bool


@dataclass(frozen=True)
class _GroupedPathStep:
    group_id: int
    tree_depth: int


@dataclass
class SubPathCandidate:
    group_ids: Tuple[int, ...]
    feature_indices: Tuple[int, ...]
    count: int
    coverage: float
    coverage_std: float
    n_train: float
    n_train_std: float
    pred: Array | float
    pred_text: str
    path_indices: Array
    depth_values: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class SubPathQuery:
    candidates: List[SubPathCandidate]
    selected: Optional[SubPathCandidate]
    rank_index: int
    total: int
    empty_reason: Optional[str]


class SubPathEngine:
    """Explore exact-length ordered path segments aggregated across all paths."""

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
        self.feature_names = list(rule_engine.feature_names or [])
        self.feature_name_to_index = {
            name: idx for idx, name in enumerate(self.feature_names)
        }
        self.path_steps = list(getattr(pset, "path_steps", []))
        if path_order is not None and path_order.size == len(self.path_steps):
            order = path_order.astype(int).tolist()
            self.path_steps = [self.path_steps[idx] for idx in order]

        self.n_paths = len(self.path_steps)
        self.groups: Dict[int, SubPathGroup] = {}
        self.grouped_paths: List[List[_GroupedPathStep]] = []
        self.max_length = int(max_length) if max_length is not None else 0
        self.candidates_by_length: Dict[int, List[SubPathCandidate]] = {}
        self.candidate_lookup: Dict[Tuple[int, ...], SubPathCandidate] = {}
        self.sorted_by_length: Dict[int, Dict[SortMode, List[SubPathCandidate]]] = {}

        self.groups, self.grouped_paths = self._build_groups_and_paths()
        self.max_length = self._infer_max_length(self.max_length)
        self._precompute_candidates()

    def query(
        self,
        *,
        length: int,
        selected_features: Sequence[str] | Sequence[int],
        sort_mode: SortMode,
        rank_index: int = 0,
    ) -> SubPathQuery:
        normalized_length = int(max(1, min(int(length or 1), max(1, self.max_length))))
        selected_feature_indices = self._normalize_selected_features(selected_features)
        if len(selected_feature_indices) > normalized_length:
            return SubPathQuery(
                candidates=[],
                selected=None,
                rank_index=0,
                total=0,
                empty_reason=(
                    f"Select at most {normalized_length} feature"
                    f"{'' if normalized_length == 1 else 's'} for path length {normalized_length}."
                ),
            )

        candidates = self.filtered_candidates(
            length=normalized_length,
            selected_features=selected_feature_indices,
            sort_mode=sort_mode,
        )
        if not candidates:
            return SubPathQuery(
                candidates=[],
                selected=None,
                rank_index=0,
                total=0,
                empty_reason="No matching path found for this length and feature selection.",
            )

        idx = int(rank_index or 0)
        idx = max(0, min(idx, len(candidates) - 1))
        return SubPathQuery(
            candidates=candidates,
            selected=candidates[idx],
            rank_index=idx,
            total=len(candidates),
            empty_reason=None,
        )

    def filtered_candidates(
        self,
        *,
        length: int,
        selected_features: Sequence[str] | Sequence[int],
        sort_mode: SortMode,
    ) -> List[SubPathCandidate]:
        normalized_length = int(max(1, min(int(length or 1), max(1, self.max_length))))
        selected_feature_indices = set(self._normalize_selected_features(selected_features))
        candidates = self.sorted_by_length.get(normalized_length, {}).get(sort_mode, [])
        if not selected_feature_indices:
            return candidates
        return [
            candidate
            for candidate in candidates
            if selected_feature_indices.issubset(set(candidate.feature_indices))
        ]

    def build_candidate_tree(self, candidate: SubPathCandidate) -> VisTree:
        mask = np.zeros(self.rule_engine.n_paths, dtype=bool)
        mask[candidate.path_indices] = True
        leaf_hist = self.rule_engine.histogram_for_mask(mask)
        step_specs: List[FocusedTreeStep] = []

        for idx, gid in enumerate(candidate.group_ids):
            prefix = candidate.group_ids[: idx + 1]
            prefix_candidate = self.candidate_lookup.get(prefix, candidate)
            group = self.groups[gid]
            step_specs.append(
                FocusedTreeStep(
                    feature_idx=group.feature_idx,
                    threshold=group.threshold_mean,
                    split_operator=self._split_operator_for_group(group),
                    branch_is_left=(group.direction == "upper"),
                    summary=self._candidate_summary(prefix_candidate),
                    threshold_min=group.threshold_min,
                    threshold_max=group.threshold_max,
                )
            )

        return build_linear_focused_tree(
            feature_names=self.feature_names,
            class_names=self.rule_engine.class_names,
            is_classifier=self.rule_engine.is_classification,
            uses_scores=bool(
                not self.rule_engine.is_classification
                and self.rule_engine.pred_vector is not None
            ),
            steps=step_specs,
            leaf=FocusedTreeLeaf(
                summary=self._candidate_summary(candidate),
                hist=leaf_hist,
            ),
        )

    def depth_histogram_for_node(
        self,
        candidate: SubPathCandidate,
        node_index: int,
    ) -> Optional[Dict[str, object]]:
        if node_index < 0 or node_index >= len(candidate.depth_values):
            return None
        depth_arr = np.asarray(candidate.depth_values[node_index], dtype=int)
        if depth_arr.size == 0:
            return None
        uniq, counts = np.unique(depth_arr, return_counts=True)
        probs = counts.astype(float) / float(counts.sum())
        return {
            "depths": uniq.astype(int).tolist(),
            "counts": counts.astype(int).tolist(),
            "probs": probs.tolist(),
            "mean_depth": float(depth_arr.mean()),
            "total": int(depth_arr.size),
        }

    def candidate_rule_text(self, candidate: SubPathCandidate) -> str:
        parts: List[str] = []
        for gid in candidate.group_ids:
            group = self.groups[gid]
            parts.append(
                f"{group.feature} {self._rule_operator_for_group(group)} {group.threshold_mean:.3f}"
            )
        return " -> ".join(parts)

    def candidate_node_text(self, candidate: SubPathCandidate, node_index: int) -> str:
        if node_index < 0 or node_index >= len(candidate.group_ids):
            return ""
        group = self.groups[candidate.group_ids[node_index]]
        return f"{group.feature} {self._rule_operator_for_group(group)} {group.threshold_mean:.3f}"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_groups_and_paths(
        self,
    ) -> tuple[Dict[int, SubPathGroup], List[List[_GroupedPathStep]]]:
        accumulators: Dict[int, Dict[str, object]] = {}
        group_ids: Dict[Tuple[int, str, bool, object], int] = {}
        grouped_paths: List[List[_GroupedPathStep]] = []

        for steps in self.path_steps:
            grouped_steps: List[_GroupedPathStep] = []
            for step in steps:
                key = self._group_key(step)
                gid = group_ids.get(key)
                if gid is None:
                    gid = len(group_ids)
                    group_ids[key] = gid
                    accumulators[gid] = {
                        "feature": step.feature_name,
                        "feature_idx": int(step.feature_idx),
                        "direction": step.direction,
                        "inclusive": bool(step.inclusive),
                        "thresholds": [],
                    }
                thresholds = accumulators[gid]["thresholds"]
                if isinstance(thresholds, list):
                    thresholds.append(float(step.threshold))
                grouped_steps.append(
                    _GroupedPathStep(group_id=gid, tree_depth=int(step.tree_depth))
                )
            grouped_paths.append(grouped_steps)

        groups: Dict[int, SubPathGroup] = {}
        for gid, acc in accumulators.items():
            thresholds = np.asarray(acc["thresholds"], dtype=float)
            groups[gid] = SubPathGroup(
                group_id=gid,
                feature=str(acc["feature"]),
                feature_idx=int(acc["feature_idx"]),
                threshold_mean=float(np.mean(thresholds)) if thresholds.size else 0.0,
                threshold_min=float(np.min(thresholds)) if thresholds.size else 0.0,
                threshold_max=float(np.max(thresholds)) if thresholds.size else 0.0,
                direction=str(acc["direction"]),
                inclusive=bool(acc["inclusive"]),
            )
        return groups, grouped_paths

    def _group_key(self, step: PathStep) -> Tuple[int, str, bool, object]:
        token: object
        ecdf = self.ecdf_dict.get(int(step.feature_idx))
        if ecdf is not None and np.isfinite(step.threshold):
            try:
                token = int(
                    ecdf_bin_indices(
                        np.asarray([float(step.threshold)], dtype=float),
                        ecdf,
                        self.ecdf_config,
                    )[0]
                )
            except Exception:
                token = round(float(step.threshold), 12)
        else:
            token = round(float(step.threshold), 12)
        return (
            int(step.feature_idx),
            str(step.direction),
            bool(step.inclusive),
            token,
        )

    def _infer_max_length(self, max_length: int) -> int:
        inferred = max((len(steps) for steps in self.grouped_paths), default=1)
        if max_length > 0:
            return int(min(max_length, max(1, inferred)))
        return int(max(1, inferred))

    def _precompute_candidates(self) -> None:
        sequences_by_length: Dict[int, Dict[Tuple[int, ...], Dict[str, object]]] = {}
        for path_idx, steps in enumerate(self.grouped_paths):
            if not steps:
                continue
            path_len = len(steps)
            limit = min(path_len, self.max_length)
            for length in range(1, limit + 1):
                seq_map = sequences_by_length.setdefault(length, {})
                for start in range(0, path_len - length + 1):
                    window = steps[start : start + length]
                    group_ids = tuple(step.group_id for step in window)
                    record = seq_map.setdefault(
                        group_ids,
                        {
                            "path_indices": set(),
                            "depth_occurrences": [],
                        },
                    )
                    path_indices = record["path_indices"]
                    depth_occurrences = record["depth_occurrences"]
                    if isinstance(path_indices, set):
                        path_indices.add(path_idx)
                    if isinstance(depth_occurrences, list):
                        depth_occurrences.append(
                            tuple(int(step.tree_depth) for step in window)
                        )

        for length, seq_map in sequences_by_length.items():
            candidates: List[SubPathCandidate] = []
            for group_ids, record in seq_map.items():
                path_index_set = record.get("path_indices", set())
                if not isinstance(path_index_set, set) or not path_index_set:
                    continue
                idx_arr = np.asarray(sorted(path_index_set), dtype=int)
                mask = np.zeros(self.rule_engine.n_paths, dtype=bool)
                mask[idx_arr] = True
                summary = self.rule_engine.node_summary(mask)
                depth_occurrences = record.get("depth_occurrences", [])
                if not isinstance(depth_occurrences, list):
                    depth_occurrences = []
                depth_values = tuple(
                    tuple(int(depths[pos]) for depths in depth_occurrences)
                    for pos in range(length)
                )
                feature_indices = tuple(
                    self.groups[gid].feature_idx for gid in group_ids
                )
                candidate = SubPathCandidate(
                    group_ids=group_ids,
                    feature_indices=feature_indices,
                    count=int(self.rule_engine._count_trees_in_mask(mask)),
                    coverage=float(summary.get("coverage", 0.0)),
                    coverage_std=float(summary.get("coverage_std", 0.0)),
                    n_train=float(summary.get("n_train", 0.0)),
                    n_train_std=float(summary.get("n_train_std", 0.0)),
                    pred=summary.get("pred"),
                    pred_text=str(summary.get("pred_text", "n/a")),
                    path_indices=idx_arr,
                    depth_values=depth_values,
                )
                candidates.append(candidate)
                self.candidate_lookup[group_ids] = candidate

            self.candidates_by_length[length] = candidates
            self.sorted_by_length[length] = {
                "paths": sorted(
                    candidates,
                    key=lambda candidate: (candidate.count, candidate.coverage),
                    reverse=True,
                ),
                "coverage": sorted(
                    candidates,
                    key=lambda candidate: (candidate.coverage, candidate.count),
                    reverse=True,
                ),
            }

    def _normalize_selected_features(
        self,
        selected_features: Sequence[str] | Sequence[int],
    ) -> List[int]:
        normalized: List[int] = []
        for value in selected_features:
            if isinstance(value, str):
                idx = self.feature_name_to_index.get(value)
                if idx is None:
                    continue
                normalized.append(idx)
            else:
                try:
                    idx = int(value)
                except Exception:
                    continue
                if 0 <= idx < len(self.feature_names):
                    normalized.append(idx)
        deduped: List[int] = []
        for idx in normalized:
            if idx not in deduped:
                deduped.append(idx)
        return deduped

    def _candidate_summary(self, candidate: SubPathCandidate) -> Dict[str, object]:
        return {
            "coverage": candidate.coverage,
            "coverage_std": candidate.coverage_std,
            "n_train": candidate.n_train,
            "n_train_std": candidate.n_train_std,
            "pred": candidate.pred,
            "pred_text": candidate.pred_text,
        }

    def _split_operator_for_group(self, group: SubPathGroup) -> str:
        if group.direction == "lower":
            return "<" if group.inclusive else "<="
        return "<=" if group.inclusive else "<"

    def _rule_operator_for_group(self, group: SubPathGroup) -> str:
        if group.direction == "lower":
            return ">=" if group.inclusive else ">"
        return "<=" if group.inclusive else "<"
