"""
Public entrypoint for the eywa trees dashboard.

`eywa_trees` keeps the surface area small: light validation up front, a
chained `config` method, and a single `run` that delegates to the internal
Dash app builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from eywa_trees.logger import setup_logger
from eywa_trees.backend.app import EywaTreesApp


ArrayLike = Union[np.ndarray, pd.DataFrame]
VectorLike = Union[np.ndarray, pd.Series, List[Any]]


@dataclass
class EywaTreesConfig:
    include_model_tab: bool = True
    include_predict_tab: bool = True
    include_boundary_tab: bool = True
    include_rule_tab: bool = True
    include_subpath_tab: bool = True
    show_text: bool = False

    # Visualization knobs
    # "auto" keeps each tab's native renderer (Model->sankey, Predict->go);
    # set to "sankey" or "go" to force both tabs to that renderer.
    tree_plot_kind: Literal["auto", "sankey", "go"] = "auto"
    colorscale: str = "Viridis"  # regression node coloring + boundary heatmap
    plot_height: str = "70vh"    # height of the main tree / heatmap plot area
    sankey_dim_alpha: float = 0.7  # non-traversed Sankey nodes/links use alpha * base_alpha

    # Inner-working knobs
    top_k_rules: int = 20    # rows shown in the Rules tab table
    subpath_max_length: Optional[int] = None  # max features combined per sub-path (None = auto)
    ecdf_bin_width: float = 0.05              # ECDF rule-grouping granularity (smaller = finer)


class EywaTreesDash:
    """
    Thin wrapper that wires user inputs to the internal Dash app.

    Parameters
    ----------
    model: fitted estimator (tree, random forest, etc.)
    X_train: training features (DataFrame or ndarray)
    X_val: validation features for metrics; defaults to X_train
    y_val: validation targets for metrics
    feature_names: optional list when arrays are passed
    class_names: optional class labels (required for classifiers without
                 discoverable labels)
    """

    def __init__(
        self,
        model: Any,
        X_train: ArrayLike,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[VectorLike] = None,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.model = model
        self.X_train = X_train
        self.X_val = X_val if X_val is not None else X_train
        self.y_val = y_val
        self.feature_names = feature_names
        self.class_names = class_names
        self.configs = EywaTreesConfig()
        self._app: Optional[EywaTreesApp] = None

        self.logger.info("Initialized eywa_trees wrapper")

    def config(self, **kwargs: Any) -> "EywaTreesDash":
        """
        Update configuration flags. Unknown keys raise a ValueError.
        """
        valid_keys = set(EywaTreesConfig.__dataclass_fields__.keys())
        unknown = set(kwargs) - valid_keys
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(sorted(unknown))}")
        if "tree_plot_kind" in kwargs and kwargs["tree_plot_kind"] not in ("auto", "sankey", "go"):
            raise ValueError(
                f"tree_plot_kind must be 'auto', 'sankey', or 'go', got {kwargs['tree_plot_kind']!r}"
            )
        if "sankey_dim_alpha" in kwargs:
            try:
                alpha = float(kwargs["sankey_dim_alpha"])
            except Exception as exc:
                raise ValueError("sankey_dim_alpha must be a number between 0 and 1.") from exc
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"sankey_dim_alpha must be between 0 and 1, got {kwargs['sankey_dim_alpha']!r}")
            kwargs["sankey_dim_alpha"] = alpha
        for key, value in kwargs.items():
            setattr(self.configs, key, value)
            self.logger.debug("Config set: %s=%r", key, value)
        return self

    def run(self, port: int = 8060, debug: bool = False, host: str = "127.0.0.1") -> None:
        """
        Build and launch the Dash application.
        """
        if self.y_val is None:
            raise ValueError("y_val is required for dashboard metrics.")

        self._app = EywaTreesApp(
            self.model,
            self.X_train,
            self.X_val,
            self.y_val,
            self.feature_names,
            self.class_names,
            self.configs,
        )
        self._app.run(port=port, debug=debug, host=host)
