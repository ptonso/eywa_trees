# Eywa Trees API overview

This project exposes a small set of high-level classes for visualization and analysis of decision trees and random forests.

Main classes
- TreeDashboard(tree, X=None, y=None, feature_names=None, target_names=None, **dash_kwargs)
  - Launch a Dash app focused on a single decision tree.
- RFDashboard(forest, X=None, y=None, feature_names=None, target_names=None, **dash_kwargs)
  - Dash app for exploring a random forest.
- CombinedDashboard(tree, forest, X=None, y=None, feature_names=None, target_names=None, **dash_kwargs)
  - Dashboard that shows both a decision tree and a forest together.
- SankeyTreePlot(tree_or_forest, X=None, feature_names=None, max_depth=None, **plot_kwargs)
  - Return a static/interactive sankey plot for a tree; internally constructs a VisTree.
- GoTreePlot(tree_or_forest, X=None, feature_names=None, go_options=None, **plot_kwargs)
  - Build a plot using the "Go" system; internally constructs a VisTree.

Design goals
- Each top-level API accepts the model(s) as the first positional argument. Optional data (X, y) is allowed to enable additional features like sample highlighting.
- Internals (VisTree) are created by plot classes; user code does not need to construct VisTree manually for normal plots.
