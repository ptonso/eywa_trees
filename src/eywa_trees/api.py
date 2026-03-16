"""
Public entrypoint for the eywa trees dashboard.

`eywa_trees` keeps the surface area small: light validation up front, a
chained `config` method, and a single `run` that delegates to the internal
Dash app builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from eywa_trees.logger import setup_logger
from eywa_trees.backend.app import SplitApp


ArrayLike = Union[np.ndarray, pd.DataFrame]
VectorLike = Union[np.ndarray, pd.Series, List[Any]]


@dataclass
class SplitConfig:
    include_model_tab: bool = True
    include_predict_tab: bool = True
    include_boundary_tab: bool = True
    include_rule_tab: bool = True
    include_subpath_tab: bool = True
    show_text: bool = False
    # placeholder for future routing tweaks (e.g., combined forest view)
    prefer_combined_forest_view: bool = False


class SplitDash:
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
        self.configs = SplitConfig()
        self._app: Optional[SplitApp] = None

        self.logger.info("Initialized eywa_trees wrapper")

    def config(self, **kwargs: Any) -> "SplitDash":
        """
        Update configuration flags. Unknown keys raise a ValueError.
        """
        valid_keys = set(SplitConfig.__dataclass_fields__.keys())
        unknown = set(kwargs) - valid_keys
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(sorted(unknown))}")
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

        self._app = SplitApp(
            self.model,
            self.X_train,
            self.X_val,
            self.y_val,
            self.feature_names,
            self.class_names,
            self.configs,
        )
        self._app.run(port=port, debug=debug, host=host)
