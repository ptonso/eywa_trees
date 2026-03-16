# Eywa Trees

`eywa_trees` is an explainability toolkit for inspecting tree-based models through
an interactive Dash application.

The standardized package surface is:

- Install with `pip install eywa_trees`
- Import with `from eywa_trees import SplitDash, SplitConfig`

It wraps a fitted decision tree or forest and exposes focused analysis tabs:

- **Model tab**: Sankey view of a single tree with depth and tree selection controls.
- **Predict tab**: sample-level pathway inspection and interactive feature controls.
- **Boundary tab**: low-dimensional view of the model decision boundary.
- **Rules tab**: grouped decision rules with coverage and tree context.
- **Subpath tab**: repeated path segment analysis across trees.

## Installation

Install from PyPI:

```bash
pip install eywa_trees
```

Install with XGBoost support:

```bash
pip install "eywa_trees[xgboost]"
```

For local development:

```bash
git clone https://github.com/tonso/eywa_trees.git
cd eywa_trees
pip install -e .
```

## Quick start

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from eywa_trees import SplitDash

data = load_diabetes(as_frame=True)
X, y = data.data, data.target

model = RandomForestRegressor(n_estimators=50, random_state=0)
model.fit(X, y)

app = SplitDash(
    model,
    X_train=X,
    X_val=X,
    y_val=y,
    feature_names=X.columns,
)

app.config(show_text=True)
app.run(port=8060, debug=False)
```

For classifiers, also pass `class_names` when they are not discoverable from the
estimator.

## Public API

The top-level import is intentionally small:

- `SplitDash`: main application wrapper
- `SplitConfig`: configuration dataclass used by `SplitDash.config(...)`

Example:

```python
from eywa_trees import SplitConfig, SplitDash
```

## Notebooks

- [demo.ipynb](./demo.ipynb): English walkthrough of the dashboard on the Sleep Health and Lifestyle dataset.
- [demo_pt.ipynb](./demo_pt.ipynb): Portuguese walkthrough of the same flow.
- [demodengue.ipynb](./demodengue.ipynb): additional demo notebook.

## Screenshots

![Model tab](./reports/model_tab.png?raw=true)

![Predict tab](./reports/predict_tab.png?raw=true)

![Boundary tab](./reports/boundary_tab.png?raw=true)

## License

This project is released under the MIT License.
