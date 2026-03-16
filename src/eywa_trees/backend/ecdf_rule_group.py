from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF


Array = np.ndarray


@dataclass
class ECDFBinConfig:
    """Configuration for ECDF-based binning."""
    bin_width: float = 0.05
    eps: float = 1e-9


@dataclass
class RuleCluster:
    """Single clustered rule group over a feature threshold."""
    id: int
    feature: str
    feature_idx: int
    upper_inclusive: bool
    threshold_mean: float
    threshold_min: float
    threshold_max: float
    path_indices: Array
    n_paths: int
    n_train: float
    coverage: float
    path_mask: Optional[Array] = None


def ecdf_bin_indices(
    values: np.ndarray,
    ecdf: ECDF,
    config: Optional[ECDFBinConfig] = None,
) -> np.ndarray:
    """
    Map values to ECDF bin indices using the configured bin width.
    """
    if values.size == 0:
        return np.array([], dtype=np.int16)

    cfg = config or ECDFBinConfig()
    if cfg.bin_width <= 0:
        raise ValueError("ECDF bin_width must be > 0.")

    q_vals = np.asarray(ecdf(values), dtype=float)
    q_vals = np.clip(q_vals, 0.0, 1.0 - cfg.eps)
    return np.floor(q_vals / cfg.bin_width).astype(np.int16)


def build_clusters(
    pset_df: pd.DataFrame,
    feature_names: List[str],
    n_train: Array,
    total_n_train: float,
    dataset_n: Optional[int],
    ecdf_dict: Optional[Dict[int, Any]],
    config: Optional[ECDFBinConfig] = None,
) -> List[RuleCluster]:
    """
    Group rules by ECDF bin for each feature and build RuleCluster entries.
    """
    if ecdf_dict is None:
        raise ValueError("ECDF rule grouping requires an ecdf_dict.")

    cfg = config or ECDFBinConfig()
    if cfg.bin_width <= 0:
        raise ValueError("ECDF rule grouping requires bin_width > 0.")

    clusters: List[RuleCluster] = []
    cluster_id = 0

    for j, feat in enumerate(feature_names):
        upper_col = f"{feat}_upper"
        upper_inclusive_col = f"{feat}_upper_inclusive"
        if upper_col not in pset_df.columns:
            continue

        upper = pset_df[upper_col].to_numpy(dtype=float)
        if upper_inclusive_col in pset_df.columns:
            upper_inclusive = pset_df[upper_inclusive_col].to_numpy(dtype=bool)
        else:
            upper_inclusive = np.ones(upper.shape, dtype=bool)
        mask_finite = np.isfinite(upper)
        if not np.any(mask_finite):
            continue

        thresholds = upper[mask_finite]
        threshold_inclusive = upper_inclusive[mask_finite]
        path_idx = np.nonzero(mask_finite)[0]
        if thresholds.size == 0:
            continue

        ecdf = ecdf_dict.get(j)
        if ecdf is None:
            continue

        bin_ids = ecdf_bin_indices(thresholds, ecdf, cfg)
        if bin_ids.size == 0:
            continue

        group_keys = np.unique(
            np.column_stack([bin_ids.astype(int), threshold_inclusive.astype(int)]),
            axis=0,
        )
        for bin_id_val, inclusive_val in group_keys:
            bin_mask = (bin_ids == int(bin_id_val)) & (threshold_inclusive == bool(inclusive_val))
            if not np.any(bin_mask):
                continue

            path_arr = path_idx[bin_mask]
            unique_paths = np.unique(path_arr)
            w = float(n_train[unique_paths].sum())
            if w <= 0.0:
                continue

            if dataset_n is not None and dataset_n > 0:
                coverage = w / float(dataset_n)
            else:
                coverage = w / total_n_train if total_n_train > 0 else 0.0

            thr_arr = thresholds[bin_mask]
            thr_mean = float(np.mean(thr_arr))
            thr_min = float(np.min(thr_arr))
            thr_max = float(np.max(thr_arr))

            cluster = RuleCluster(
                id=cluster_id,
                feature=feat,
                feature_idx=j,
                upper_inclusive=bool(inclusive_val),
                threshold_mean=thr_mean,
                threshold_min=thr_min,
                threshold_max=thr_max,
                path_indices=unique_paths,
                n_paths=int(unique_paths.size),
                n_train=w,
                coverage=coverage,
                path_mask=None,
            )
            clusters.append(cluster)
            cluster_id += 1

    return clusters


def cluster_paths_by_ecdf_bins(
    df: pd.DataFrame,
    feature_names: List[str],
    ecdf_dict: Dict[int, ECDF],
    config: Optional[ECDFBinConfig] = None,
) -> pd.DataFrame:
    """
    Group decision paths by ECDF bins across feature bounds to reduce duplicates.
    """
    if df.empty or not ecdf_dict:
        return df

    cfg = config or ECDFBinConfig()
    if cfg.bin_width <= 0:
        return df

    work = df.copy()
    bin_cols: List[str] = []

    for j, feat in enumerate(feature_names):
        lower_col = f"{feat}_lower"
        upper_col = f"{feat}_upper"
        lower_inclusive_col = f"{feat}_lower_inclusive"
        upper_inclusive_col = f"{feat}_upper_inclusive"
        if lower_col not in work.columns or upper_col not in work.columns:
            continue

        ecdf = ecdf_dict.get(j)
        if ecdf is None:
            continue

        for col, suffix in ((lower_col, "lower"), (upper_col, "upper")):
            values = work[col].to_numpy(dtype=float)
            bin_ids = np.full(values.shape, -1, dtype=np.int16)
            finite_mask = np.isfinite(values)
            if np.any(finite_mask):
                bin_ids[finite_mask] = ecdf_bin_indices(values[finite_mask], ecdf, cfg)
            bin_col = f"{feat}_{suffix}_bin"
            work[bin_col] = bin_ids
            bin_cols.append(bin_col)
        for col in (lower_inclusive_col, upper_inclusive_col):
            if col in work.columns:
                bin_cols.append(col)

    if not bin_cols:
        return df

    original_cols = list(df.columns)
    grouped = work.groupby(bin_cols, dropna=False, sort=False)
    records: List[Dict[str, Any]] = []

    for _, group in grouped:
        rec: Dict[str, Any] = {}
        weights = _weights_for_group(group)

        for feat in feature_names:
            lower_col = f"{feat}_lower"
            upper_col = f"{feat}_upper"
            if lower_col in group.columns:
                rec[lower_col] = _aggregate_bound(group[lower_col])
            if upper_col in group.columns:
                rec[upper_col] = _aggregate_bound(group[upper_col])
            if lower_inclusive_col in group.columns:
                rec[lower_inclusive_col] = bool(group[lower_inclusive_col].iloc[0])
            if upper_inclusive_col in group.columns:
                rec[upper_inclusive_col] = bool(group[upper_inclusive_col].iloc[0])

        for col in ("n_samples", "path_prob_mc", "path_prob_forest", "path_prob"):
            if col in group.columns:
                rec[col] = float(group[col].sum())

        if "probas" in group.columns:
            rec["probas"] = _weighted_vector(group["probas"], weights)
        if "regressions" in group.columns:
            rec["regressions"] = _weighted_mean(group["regressions"], weights)
        if "scores" in group.columns:
            rec["scores"] = _weighted_mean(group["scores"], weights)

        if "tree_id" in group.columns:
            rec["tree_id"] = _merge_tree_ids(group["tree_id"])
        if "leaf_index" in group.columns:
            rec["leaf_index"] = _merge_strings(group["leaf_index"])

        for col in original_cols:
            if col in rec:
                continue
            if col in group.columns:
                rec[col] = group[col].iloc[0]

        records.append(rec)

    return pd.DataFrame.from_records(records, columns=original_cols)


def _aggregate_bound(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if np.any(finite):
        return float(np.mean(values[finite]))
    if values.size:
        return float(values[0])
    return float("nan")


def _weights_for_group(group: pd.DataFrame) -> np.ndarray:
    if "n_samples" in group.columns:
        weights = group["n_samples"].to_numpy(dtype=float)
    elif "path_prob_forest" in group.columns:
        weights = group["path_prob_forest"].to_numpy(dtype=float)
    else:
        weights = np.ones(group.shape[0], dtype=float)

    total = float(weights.sum()) if weights.size else 0.0
    if total <= 0.0 and group.shape[0] > 0:
        weights = np.ones(group.shape[0], dtype=float)
    return weights


def _weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    vals = values.to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    wsum = float(weights.sum()) if weights.size else 0.0
    if wsum <= 0.0:
        return float(np.mean(vals))
    return float(np.dot(vals, weights) / wsum)


def _weighted_vector(values: pd.Series, weights: np.ndarray) -> List[float]:
    vecs = [np.asarray(v, dtype=float) for v in values]
    if not vecs:
        return []

    max_len = max(v.size for v in vecs)
    if max_len == 0:
        return []

    mat = np.zeros((len(vecs), max_len), dtype=float)
    for i, v in enumerate(vecs):
        if v.size:
            mat[i, : v.size] = v

    wsum = float(weights.sum()) if weights.size else 0.0
    if wsum <= 0.0:
        mean = mat.mean(axis=0)
    else:
        mean = (mat.T * weights).T.sum(axis=0) / wsum
    return mean.tolist()


def _merge_tree_ids(values: pd.Series) -> str:
    ids: set[int] = set()
    for val in values:
        if val is None:
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
    if not ids:
        return ""
    return ",".join(str(i) for i in sorted(ids))


def _merge_strings(values: pd.Series) -> str:
    seen: set[str] = set()
    ordered: List[str] = []
    for val in values:
        if val is None:
            continue
        text = str(val)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ",".join(ordered)
