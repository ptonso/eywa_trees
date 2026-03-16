from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd




@dataclass
class FeatureBinManager:
    X_train: pd.DataFrame
    active_features: Optional[List[str]] = None
    max_numeric_positions: int = 21
    max_categorical_positions: int = 15
    feature_bins: Dict[str, List[Any]] = field(init=False)
    feature_names: List[str] = field(init=False)
    all_feature_names: List[str] = field(init=False)
    active_feature_names: List[str] = field(init=False)
    default_indices: Dict[str, int] = field(init=False)
    default_values: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.all_feature_names = list(self.X_train.columns)
        if self.active_features:
            active = [f for f in self.active_features if f in self.all_feature_names]
        else:
            active = list(self.all_feature_names)
        if not active:
            active = list(self.all_feature_names)
        self.active_feature_names = active
        self.feature_bins = self._compute_feature_bins()
        self.feature_names = list(self.active_feature_names)
        self.default_values = self._compute_default_values()
        self.default_indices = {
            feat: self._default_index(feat) for feat in self.feature_names
        }

    def sample_from_indices(
        self,
        indices: List[int],
        feature_order: List[str],
    ) -> pd.DataFrame:
        data: Dict[str, Any] = dict(self.default_values)
        for idx, feat in zip(indices, feature_order):
            values = self.feature_bins.get(feat, [None])
            if not values:
                data[feat] = None
                continue
            safe_idx = int(
                np.clip(idx if idx is not None else 0, 0, len(values) - 1)
            )
            data[feat] = values[safe_idx]

        ordered = {feat: data.get(feat) for feat in self.all_feature_names}
        return pd.DataFrame([ordered])

    def marks_for_feature(
        self,
        feat: str,
        values: List[Any],
        max_marks: int = 7,
    ) -> Dict[str, str]:
        n = len(values)
        if n <= 0:
            return {"0": "None"}

        if n <= max_marks:
            indices = list(range(n))
        else:
            if max_marks <= 1:
                indices = [0]
            else:
                indices = sorted(
                    {
                        int(round(t * (n - 1)))
                        for t in np.linspace(0.0, 1.0, max_marks)
                    }
                )

        return {str(i): self._format_mark(values[i]) for i in indices}

    def display_for_index(self, feature: str, index: int) -> str:
        values = self.feature_bins.get(feature, [None])
        if not values:
            return "None"
        safe_idx = int(
            np.clip(index if index is not None else 0, 0, len(values) - 1)
        )
        return self._format_mark(values[safe_idx])

    def nearest_index_for_value(self, feature: str, value: Any) -> int:
        values = self.feature_bins.get(feature, [None])
        if not values:
            return 0
        best_idx = 0
        best_diff = np.inf
        for i, v in enumerate(values):
            if value == v:
                return i
            try:
                diff = abs(float(v) - float(value))
            except Exception:
                diff = np.inf
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def _compute_feature_bins(self) -> Dict[str, List[Any]]:
        bins: Dict[str, List[Any]] = {}

        if not self.active_feature_names:
            return bins

        df = self.X_train[self.active_feature_names]
        if df.empty:
            return {feat: [None] for feat in self.active_feature_names}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_set = set(numeric_cols)
        categorical_cols = [f for f in self.active_feature_names if f not in numeric_set]

        if numeric_cols:
            num_df = df[numeric_cols]
            nunique = num_df.nunique(dropna=True)
            low_unique_cols = [
                c for c in numeric_cols if nunique.get(c, 0) <= self.max_numeric_positions
            ]
            high_unique_cols = [c for c in numeric_cols if c not in low_unique_cols]

            for feat in low_unique_cols:
                col = num_df[feat].dropna()
                if col.empty:
                    bins[feat] = [None]
                    continue
                uniq = np.sort(pd.unique(col))
                values = [self._to_native(v) for v in uniq]
                bins[feat] = values or [None]

            if high_unique_cols:
                qs = np.linspace(0.0, 1.0, self.max_numeric_positions)
                quantiles = num_df[high_unique_cols].quantile(
                    qs, interpolation="linear"
                )
                for feat in high_unique_cols:
                    q_vals = quantiles[feat].to_numpy(dtype=float)
                    mapped: List[float] = []
                    last_val: Optional[float] = None
                    for qv in q_vals:
                        if last_val is not None and np.isclose(qv, last_val):
                            continue
                        mapped.append(float(qv))
                        last_val = float(qv)
                    values = [self._to_native(v) for v in mapped] if mapped else []
                    bins[feat] = values or [None]

        for feat in categorical_cols:
            col = df[feat].dropna()
            if col.empty:
                bins[feat] = [None]
                continue

            uniq = list(pd.unique(col))
            uniq_sorted = sorted(uniq, key=lambda x: str(x))
            if len(uniq_sorted) <= self.max_categorical_positions:
                values = [self._to_native(v) for v in uniq_sorted]
            else:
                idxs = np.linspace(
                    0,
                    len(uniq_sorted) - 1,
                    self.max_categorical_positions,
                )
                idxs_int = sorted({int(round(i)) for i in idxs})
                values = [self._to_native(uniq_sorted[i]) for i in idxs_int]

            bins[feat] = values or [None]

        return {feat: bins.get(feat, [None]) for feat in self.active_feature_names}

    def _compute_default_values(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        if not self.all_feature_names or self.X_train.empty:
            return {feat: None for feat in self.all_feature_names}

        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        numeric_set = set(numeric_cols)
        other_cols = [f for f in self.all_feature_names if f not in numeric_set]

        if numeric_cols:
            medians = self.X_train[numeric_cols].median()
            for feat in numeric_cols:
                val = medians.get(feat, np.nan)
                if pd.isna(val):
                    val = None
                defaults[feat] = self._to_native(val)

        if other_cols:
            modes = self.X_train[other_cols].mode(dropna=True)
            if not modes.empty:
                row = modes.iloc[0]
                for feat in other_cols:
                    val = row.get(feat)
                    if pd.isna(val):
                        val = None
                    defaults[feat] = self._to_native(val)
            else:
                for feat in other_cols:
                    defaults[feat] = None

        for feat in self.all_feature_names:
            defaults.setdefault(feat, None)
        return defaults

    def _default_index(self, feature: str) -> int:
        values = self.feature_bins.get(feature, [None])
        if not values:
            return 0

        col = self.X_train[feature].dropna()
        if pd.api.types.is_numeric_dtype(col):
            median = float(col.median())
            diffs: List[float] = []
            for v in values:
                try:
                    diffs.append(abs(float(v) - median))
                except Exception:
                    diffs.append(np.inf)
            return int(np.argmin(diffs)) if diffs else 0

        return len(values) // 2

    def _format_mark(self, value: Any) -> str:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            v = float(value)
            if abs(v) >= 1000:
                return f"{v:,.0f}".replace(",", "")
            if abs(v) >= 100:
                return f"{v:.0f}"
            return f"{v:.2g}"
        if value is None:
            return "None"
        text = str(value)
        return text if len(text) <= 12 else text[:12] + "…"

    def _to_native(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value
