from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from eywa_trees.backend.ecdf_rule_group import ECDFBinConfig
from eywa_trees.backend.pset import PathwaySet
from eywa_trees.logger import setup_logger


class PathStatistics:
    """Extract decision-path statistics used by downstream rule aggregation."""

    def __init__(
        self,
        model: Any,
        X: Any,
        class_names: Optional[List[str]] = None,
        ecdf_bin_config: Optional[ECDFBinConfig] = None,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.feature_names = X.columns.tolist()
        self.class_names = class_names
        self.ecdf_bin_config = ecdf_bin_config

        self.pset = PathwaySet(self.feature_names, model=model, X_train=X.values)
        self.uses_scores = getattr(self.pset, "uses_scores", False)

    def rules_dataframe(self):
        df = self.pset.to_dataframe()
        if df.empty:
            return df

        classes = self.class_names or getattr(self.pset, "label_names", None)

        if "probas" in df.columns:
            df["value_dist"] = df["probas"]
            if classes is not None:
                classes_list = list(classes)
                df["predicted_label"] = df["probas"].apply(
                    lambda v: classes_list[int(np.argmax(v))] if len(v) else None
                )
            else:
                df["predicted_label"] = df["probas"].apply(
                    lambda v: int(np.argmax(v)) if len(v) else None
                )
        elif "regressions" in df.columns:
            df["predicted_label"] = df["regressions"]
        elif "scores" in df.columns:
            df["predicted_label"] = df["scores"]
        else:
            df["predicted_label"] = None

        if "tree_id" in df.columns:
            df["num_trees"] = df["tree_id"].apply(
                lambda x: len(str(x).split(",")) if isinstance(x, str) else 1
            )
        else:
            df["num_trees"] = 1

        sort_col = "path_prob_forest"
        if sort_col not in df.columns:
            sort_col = (
                "branch_prob_forest"
                if "branch_prob_forest" in df.columns
                else "n_samples"
            )

        df.sort_values(
            [sort_col, "n_samples"],
            ascending=[False, False],
            inplace=True,
        )
        return df
