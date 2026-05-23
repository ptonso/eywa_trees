
from typing import Any, Iterable, List, Optional, Sequence, TYPE_CHECKING, Union

import dash
from dash import dcc, html
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from eywa_trees.logger import setup_logger
from eywa_trees.backend.adapters.sklearn_forest import SklearnForestAdapter
from eywa_trees.backend.adapters.sklearn_tree import SklearnTreeAdapter
from eywa_trees.backend.adapters.xgboost import XGBoostAdapter, extract_xgb_tree_shims, is_xgboost_model
from eywa_trees.backend.feature_bins import FeatureBinManager
from eywa_trees.tabs.model import SingleTreeTab, RandomForestTab
from eywa_trees.tabs.predict import PredictTab
from eywa_trees.tabs.boundary import BoundaryTab
from eywa_trees.tabs.rule_tree import RuleTreeTab, RuleTreeTabConfig
from eywa_trees.tabs.sub_path import SubPathTab
from eywa_trees.backend.ecdf_rule_group import ECDFBinConfig

if TYPE_CHECKING:
    from eywa_trees.api import EywaTreesConfig


ArrayLike = Union[np.ndarray, pd.DataFrame]
VectorLike = Union[np.ndarray, pd.Series, List[Any]]

_ADAPTERS = (
    XGBoostAdapter(),
    SklearnForestAdapter(),
    SklearnTreeAdapter(),
)


class EywaTreesApp:
    """
    Internal Dash app builder. Responsible for routing model/analysis tabs based
    on the input model type and provided configuration.
    """

    def __init__(
        self,
        model: Any,
        X_train: ArrayLike,
        X_val: Optional[ArrayLike],
        y_val: Optional[VectorLike],
        feature_names: Optional[Sequence[str]],
        class_names: Optional[Sequence[str]],
        config: "EywaTreesConfig",
    ) -> None:
        self.logger = setup_logger("api.log")
        self.model = model
        self.model_kind = self._detect_model_kind(model)
        self.config = config

        inferred_feature_names = self._infer_feature_names(feature_names, X_train, model)
        self.X_train: pd.DataFrame = self._ensure_dataframe(X_train, inferred_feature_names, "X_train")
        self.X_val: pd.DataFrame = self._ensure_dataframe(X_val if X_val is not None else X_train, inferred_feature_names, "X_val")
        self.y_val: VectorLike = self._ensure_target_vector(y_val)
        self.feature_names: Optional[List[str]] = inferred_feature_names
        self.class_names: Optional[List[str]] = self._infer_class_names(class_names, model, y_val)

        self.app = dash.Dash(__name__)
        self.tabs: List[object] = []

        self._build_tabs()
        self._register_callbacks()

    def run(self, port: int = 8060, debug: bool = False, host: str = "0.0.0.0") -> None:
        # Disable the reloader when running in background threads to avoid signal errors.
        self.app.run(port=port, debug=debug, use_reloader=False, host=host)

    def _build_tabs(self) -> None:
        tab_components: List[html.Div] = []
        active_value: Optional[str] = None
        feature_mgr: Optional[FeatureBinManager] = None

        if getattr(self.config, "include_predict_tab", False) or getattr(self.config, "include_boundary_tab", False):
            active_features = self._infer_split_feature_names(self.model, self.feature_names)
            feature_mgr = FeatureBinManager(self.X_train, active_features=active_features)

        if self.config.include_model_tab:
            model_tab = self._build_model_tab()
            if model_tab is not None:
                self.tabs.append(model_tab)
                tab_components.append(dcc.Tab(label="Model", value="model", children=model_tab.layout))
                active_value = active_value or "model"

        ecdf_bin_config = ECDFBinConfig(bin_width=self.config.ecdf_bin_width)

        if getattr(self.config, "include_predict_tab", False):
            predict_tab = PredictTab(
                self.model,
                self.X_train,
                class_names=self.class_names,
                show_text=self.config.show_text,
                feature_mgr=feature_mgr,
                tree_plot_kind=self._resolve_plot_kind("go"),
                colorscale=self.config.colorscale,
                plot_height=self.config.plot_height,
                sankey_dim_alpha=self.config.sankey_dim_alpha,
            )
            self.tabs.append(predict_tab)
            tab_components.append(dcc.Tab(label="Predict", value="predict", children=predict_tab.layout))
            active_value = active_value or "predict"

        if getattr(self.config, "include_boundary_tab", False):
            boundary_tab = BoundaryTab(
                self.model,
                self.X_train,
                feature_mgr=feature_mgr,
                colorscale=self.config.colorscale,
                plot_height=self.config.plot_height,
            )
            self.tabs.append(boundary_tab)
            tab_components.append(dcc.Tab(label="Boundary", value="boundary", children=boundary_tab.layout))
            active_value = active_value or "boundary"

        if getattr(self.config, "include_rule_tab", False):
            rule_tab = RuleTreeTab(
                self.model,
                self.X_train,
                class_names=self.class_names,
                tab_config=RuleTreeTabConfig(top_k_rules=self.config.top_k_rules),
                ecdf_bin_config=ecdf_bin_config,
                plot_height=self.config.plot_height,
            )

            self.tabs.append(rule_tab)
            tab_components.append(dcc.Tab(label="Rules", value="rules", children=rule_tab.layout))
            active_value = active_value or "rules"

        if getattr(self.config, "include_subpath_tab", False):
            subpath_tab = SubPathTab(
                self.model,
                self.X_train,
                class_names=self.class_names,
                ecdf_bin_config=ecdf_bin_config,
                max_length=self.config.subpath_max_length,
            )
            self.tabs.append(subpath_tab)
            tab_components.append(dcc.Tab(label="Sub-Path", value="subpath", children=subpath_tab.layout))
            active_value = active_value or "subpath"

        if not tab_components:
            raise ValueError("No tabs enabled. Enable at least one via config().")


        title_kind = {
            "forest": "Random Forest",
            "tree": "Decision Tree",
            "xgboost": "XGBoost"
        }
        self.app.layout = html.Div(
            [
                html.H2(
                    f"EywaTrees Dashboard - {title_kind[self.model_kind]}", 
                    style={"margin": "16px"}
                ),
                dcc.Tabs(
                    id="eywatrees-tabs",
                    value=active_value,
                    children=tab_components,
                    vertical=False,
                    mobile_breakpoint=0,  # stay horizontal
                    style={"width": "100%"},
                    parent_style={"width": "100%"},
                ),
            ],
            style={"backgroundColor": "white"},
        )

    def _resolve_plot_kind(self, native: str) -> str:
        """Resolve tree_plot_kind: "auto" keeps the tab's native renderer."""
        kind = getattr(self.config, "tree_plot_kind", "auto")
        return native if kind == "auto" else kind

    def _build_model_tab(self) -> Optional[object]:
        if self.model_kind in ("forest", "xgboost"):
            label = "XGBoost" if self.model_kind == "xgboost" else "RandomForest"
            self.logger.info("Routing to %s tab", label)
            return RandomForestTab(
                self.model,
                self.X_train,
                self.X_val,
                self.y_val,
                class_names=self.class_names,
                tree_plot_kind=self._resolve_plot_kind("sankey"),
                colorscale=self.config.colorscale,
                plot_height=self.config.plot_height,
                sankey_dim_alpha=self.config.sankey_dim_alpha,
            )
        if self.model_kind == "tree":
            self.logger.info("Routing to SingleTree tab")
            return SingleTreeTab(
                self.model,
                self.X_train,
                self.X_val,
                self.y_val,
                class_names=self.class_names,
                show_text=self.config.show_text,
                tree_plot_kind=self._resolve_plot_kind("sankey"),
                colorscale=self.config.colorscale,
                plot_height=self.config.plot_height,
                sankey_dim_alpha=self.config.sankey_dim_alpha,
            )
        self.logger.error("Unsupported model type for model tab")
        return None

    def _register_callbacks(self) -> None:
        for tab in self.tabs:
            register = getattr(tab, "register_callbacks", None)
            if callable(register):
                register(self.app)

    def _detect_model_kind(self, model: Any) -> str:
        for adapter in _ADAPTERS:
            if adapter.is_compatible(model):
                return adapter.name
        name = model.__class__.__name__.lower()
        if "lightgbm" in name or "lgbm" in name:
            return "forest"
        return "unknown"

    def _infer_feature_names(
        self,
        provided: Optional[Sequence[str]],
        X: ArrayLike,
        model: Any,
    ) -> Optional[List[str]]:
        if provided is not None:
            return [str(c) for c in list(provided)]

        if hasattr(model, "feature_names_in_"):
            names = getattr(model, "feature_names_in_")
            return names.tolist() if hasattr(names, "tolist") else list(names)

        if isinstance(X, pd.DataFrame):
            return list(X.columns)

        if isinstance(X, np.ndarray):
            raise ValueError(
                "feature_names must be provided when X is a numpy array and the model "
                "does not expose feature_names_in_."
            )
        return None

    def _infer_split_feature_names(
        self,
        model: Any,
        feature_names: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        if not feature_names:
            return None

        names = list(feature_names)
        used: set[int] = set()

        def _add_features(feat_arr: Any) -> None:
            if feat_arr is None:
                return
            try:
                arr = np.asarray(feat_arr)
            except Exception:
                return
            for idx in np.unique(arr):
                try:
                    idx_int = int(idx)
                except Exception:
                    continue
                if idx_int >= 0:
                    used.add(idx_int)

        try:
            if is_xgboost_model(model):
                trees, _, _, _ = extract_xgb_tree_shims(model)
                for tree in trees:
                    _add_features(getattr(tree, "feature", None))

            if hasattr(model, "tree_"):
                _add_features(getattr(model.tree_, "feature", None))

            if hasattr(model, "estimators_") and getattr(model, "estimators_", []):
                for est in model.estimators_:
                    tree = getattr(est, "tree_", None)
                    if tree is not None:
                        _add_features(getattr(tree, "feature", None))

            if not used and hasattr(model, "estimator"):
                tree = getattr(model.estimator, "tree_", None)
                if tree is not None:
                    _add_features(getattr(tree, "feature", None))
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.debug("Split feature inference failed: %s", exc)
            return None

        if not used:
            return None

        return [names[i] for i in sorted(used) if 0 <= i < len(names)]

    def _infer_class_names(
        self,
        provided: Optional[Sequence[str]],
        model: Any,
        y_val: Optional[VectorLike],
    ) -> Optional[List[str]]:
        if provided is not None:
            return [str(c) for c in list(provided)]

        candidates: List[Iterable[Any]] = []
        if hasattr(model, "classes_"):
            candidates.append(getattr(model, "classes_"))
        if hasattr(model, "estimators_") and getattr(model, "estimators_", []):
            first = model.estimators_[0]
            if hasattr(first, "classes_"):
                candidates.append(getattr(first, "classes_"))

        for cand in candidates:
            try:
                return [str(c) for c in list(cand)]
            except Exception:
                continue

        if y_val is not None and self._is_classifier(model):
            try:
                unique_vals = pd.unique(y_val) if isinstance(y_val, pd.Series) else np.unique(y_val)
                return [str(v) for v in unique_vals]
            except Exception:
                pass

        if self._is_classifier(model):
            raise ValueError("class_names are required for classifier models.")
        return None

    def _ensure_dataframe(
        self,
        X: ArrayLike,
        feature_names: Optional[Sequence[str]],
        label: str,
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X

        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise ValueError(f"feature_names must be provided when {label} is a numpy array.")
            return pd.DataFrame(X, columns=feature_names)

        raise ValueError(f"{label} must be a pandas DataFrame or numpy array.")

    def _ensure_target_vector(self, y_val: Optional[VectorLike]) -> VectorLike:
        if y_val is None:
            raise ValueError("y_val is required for dashboard metrics.")
        if isinstance(y_val, (pd.Series, np.ndarray)):
            return y_val
        return pd.Series(y_val)

    def _is_classifier(self, model: Any) -> bool:
        if isinstance(model, ClassifierMixin):
            return True
        if getattr(model, "_estimator_type", None) == "classifier":
            return True
        if hasattr(model, "estimator") and isinstance(getattr(model, "estimator"), ClassifierMixin):
            return True
        if hasattr(model, "estimators_") and getattr(model, "estimators_", []):
            first = model.estimators_[0]
            if isinstance(first, ClassifierMixin):
                return True
            if getattr(first, "_estimator_type", None) == "classifier":
                return True
        return False
