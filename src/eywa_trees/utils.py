import numpy as np
import pandas as pd
from typing import *
from dataclasses import dataclass

# IPython is optional; guard imports for environments without it.
try:
    from IPython.display import display  # type: ignore
    from IPython import get_ipython  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    def display(obj: object) -> None:
        print(obj)
    def get_ipython() -> None:
        return None

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score



def _fmt(val: Union[float, np.ndarray, List[float]]) -> str:
    """
    Format scalars or 1-D arrays/lists.
    >>>_fmt([0.333333, 0.666])
    '[0.33, 0.67]'
    """
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val).flatten()
        return "[" + ", ".join(f"{x:.2f}" for x in arr) + "]"
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.2f}"
    if isinstance(val, (int, np.integer)):
        return f"{int(val)}"
    return str(val)


def show_df(df: pd.DataFrame, n: int = 5, max_col_width: int = 12) -> pd.DataFrame:
    """
    Display (and return) a formatted version of df.head(n):
      • Numeric columns are run through _fmt
      • Columns containing list/ndarray samples are run through _fmt
      • Column names longer than max_col_width are truncated to fit
    """
    head = df.head(n).copy()

    numeric_cols = head.select_dtypes(include=["number"]).columns.tolist()

    list_cols: List[str] = []
    for col in head.columns:
        if col in numeric_cols:
            continue
        nonnull = head[col].dropna()
        if not nonnull.empty and isinstance(nonnull.iloc[0], (list, np.ndarray)):
            list_cols.append(col)

    for col in numeric_cols + list_cols:
        head[col] = head[col].apply(_fmt)

    def _truncate(name: str) -> str:
        return name if len(name) <= max_col_width else name[: max_col_width - 3] + "..."

    truncated_map = {col: _truncate(col) for col in head.columns}
    head = head.rename(columns=truncated_map)

    print(head.to_string(index=True))
    return head
  

from dataclasses import dataclass
from typing import Union, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score


@dataclass
class Results:
    fbt_scores: list[float]
    rf_scores:  list[float]
    dt_scores:  list[float]
    fbt_mean:   float
    rf_mean:    float
    dt_mean:    float


@dataclass
class Results:
    X_train: pd.DataFrame
    y_train: Union[pd.Series, np.ndarray]
    fbt: "ForestBasedTree"
    rf: Union["RandomForestClassifier", "RandomForestRegressor"]
    dt: Union["DecisionTreeClassifier", "DecisionTreeRegressor"]
    rf_score: float
    fbt_score: float
    dt_score: float
    fbt_depth: float
    dt_depth: float


def test_fbt(
    X: pd.DataFrame,
    y: Union[np.ndarray, pd.Series],
    random_state: int = 42,
    n_splits: int = 3,
) -> Results:
    """
    Train RandomForest, DecisionTree, and ForestBasedTree on given dataset.
    Average train/test performance and depth over k-folds.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from eywa_trees import ForestBasedTree

    is_classification = len(np.unique(y)) < 20 and np.allclose(y, y.astype(int))
    kf = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if is_classification
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )

    rf_train, rf_test = [], []
    fbt_train, fbt_test = [], []
    dt_train, dt_test = [], []
    fbt_depths, dt_depths = [], []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]

        if is_classification:
            rf_cls, dt_cls = RandomForestClassifier, DecisionTreeClassifier
            metric = accuracy_score
        else:
            rf_cls, dt_cls = RandomForestRegressor, DecisionTreeRegressor
            metric = r2_score

        rf = rf_cls(random_state=random_state,)
        rf.fit(X_train, y_train)

        fbt = ForestBasedTree(random_state=random_state)
        fbt.fit(
            rf,
            X_train,
            y_train,
            X_train.dtypes,
            list(X_train.columns)
        )

        dt = dt_cls(random_state=random_state)
        dt.fit(X_train, y_train)

        rf_train.append(metric(y_train, rf.predict(X_train)))
        fbt_train.append(metric(y_train, fbt.predict(X_train)))
        dt_train.append(metric(y_train, dt.predict(X_train)))

        rf_test.append(metric(y_test, rf.predict(X_test)))
        fbt_test.append(metric(y_test, fbt.predict(X_test)))
        dt_test.append(metric(y_test, dt.predict(X_test)))

        if hasattr(fbt, "get_depth"):
            fbt_depths.append(fbt.get_depth())
        if hasattr(dt, "get_depth"):
            dt_depths.append(dt.get_depth())

    if is_classification:
        print("Classification task")
    else:
        print("Regression task")

    rf_score_train  = np.mean(rf_train)
    fbt_score_train = np.mean(fbt_train)
    dt_score_train  = np.mean(dt_train)
    rf_score_test   = np.mean(rf_test)
    fbt_score_test  = np.mean(fbt_test)
    dt_score_test   = np.mean(dt_test)
    fbt_depth_mean  = np.mean(fbt_depths) if fbt_depths else np.nan
    dt_depth_mean   = np.mean(dt_depths) if dt_depths else np.nan
    fbt_depth_std   = np.std(fbt_depths) if fbt_depths else np.nan
    dt_depth_std    = np.std(dt_depths) if dt_depths else np.nan

    print(f"RandomForest Score    : {rf_score_test:.2f} (train: {rf_score_train:.2f})")
    print(f"ForestBasedTree Score : {fbt_score_test:.2f} (train: {fbt_score_train:.2f})")
    print(f"Decision Tree Score   : {dt_score_test:.2f} (train: {dt_score_train:.2f})")
    print(f"fbt depth avg: {fbt_depth_mean:.2f} (±{fbt_depth_std:.2f})")
    print(f"dt depth avg:  {dt_depth_mean:.2f} (±{dt_depth_std:.2f})")

    return Results(
        X_train=X_train,
        y_train=y_train,
        fbt=fbt,
        rf=rf,
        dt=dt,
        rf_score=rf_score_test,
        fbt_score=fbt_score_test,
        dt_score=dt_score_test,
        fbt_depth=fbt_depth_mean,
        dt_depth=dt_depth_mean,
    )


def setup_toy_classifier(
    n_samples: int = 100,
    n_features: int = 4,
    n_classes: int = 2,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Generate a random classification dataset and train a RandomForestClassifier.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features*0.8),
        n_redundant=int(n_features*0.2),
        n_classes=n_classes,
        random_state=random_state,
    )
    class_names: List[str] = [f"Class {i}" for i in range(n_classes)]
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return X, y, class_names


def setup_toy_regressor(
    n_samples: int = 100,
    n_features: int = 4,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate a random regression dataset and train a RandomForestRegressor.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features*0.8),
        random_state=random_state,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return X, y


if __name__ == "__main__":
    # Example usage for testing utilities

    # Classifier example
    X_clf, y_clf, class_names, rf_clf = setup_toy_classifier(
        n_samples=50, n_features=3, n_classes=2, random_state=42
    )
    print("Classifier data shapes:", X_clf.shape, y_clf.shape)
    print("Classifier class names:", class_names)
    print("Classifier first 5 predictions:", rf_clf.predict(X_clf[:5]))

    # Regressor example
    X_reg, y_reg, rf_reg = setup_toy_regressor(
        n_samples=50, n_features=3, random_state=42
    )
    print("Regressor data shapes:", X_reg.shape, y_reg.shape)
    print("Regressor first 5 predictions:", rf_reg.predict(X_reg[:5]))




