<center>
<h1><b>FADEx</b></h1>
Feature Attribution and Distortion-based Explanation of Dimensionality Reduction

[![PyPI version](https://img.shields.io/pypi/v/fadex-exp?color=blue&label=PyPI)](https://pypi.org/project/fadex-exp/)
[![Downloads](https://img.shields.io/pypi/dm/fadex-exp?color=brightgreen&label=Downloads)](https://pypi.org/project/fadex-exp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2607.27463-b31b1b.svg)](https://arxiv.org/abs/2607.27463)
</center>

---

**FADEx** is a local explainability method for dimensionality reduction (DR) algorithms. It approximates the DR mapping locally using a **weighted ridge least-squares** fit and derives per-feature importance scores through a **Singular Value Decomposition (SVD)** of the estimated Jacobian matrix.

When the data is mapped to a two-dimensional space by the DR algorithm, the importance of each high-dimensional feature $j$ is given by:

$$
\phi_j = \left|v_{1j}\, x_j\right| + \frac{\lambda_2}{\lambda_1} \left|v_{2j}\, x_j\right|
$$

where $v_{ij}$ denotes the $j$-th entry of the $i$-th right singular vector, $\lambda_i$ is the $i$-th singular value, and $x_j$ is the value of feature $j$.

The **spectral norm** of the Jacobian is used as a local distortion measure — how much the DR algorithm stretched or compressed the neighborhood around each point.

---

# Paper

FADEx will be published in **IEEE Transactions on Visualization and Computer Graphics (TVCG)**. A preprint is available on arXiv:

> Lucas Greff Meneses, Evandro S. Ortigossa, Claudio Silva, Luis Gustavo Nonato.
> **FADEx: Feature Attribution and Distortion-based Explanation of Dimensionality Reduction.**
> arXiv:2607.27463, 2026.

**[Read the preprint on arXiv »](https://arxiv.org/abs/2607.27463)**

---

# Installation

```bash
pip install fadex-exp
```

---

# Quick Start

```python
from fadex import FADEx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Fit your DR algorithm
X_scaled = StandardScaler().fit_transform(X)
X_embedded = TSNE(n_components=2).fit_transform(X_scaled)

# Create FADEx explainer
fadex = FADEx(
    high_dim_data=X_scaled,
    low_dim_data=X_embedded,
    n_neighbors=30,
    feature_names=feature_names,
    class_names=class_labels,
)

# Explain a single instance
result = fadex.fit(explain_index=0)

# Global feature importance plot
fadex.importance_plot()

# Interactive embedding plot
fadex.interactive_plot()
```

---

# FADEx Class

## Constructor

```python
FADEx(
    high_dim_data,
    low_dim_data,
    n_neighbors     = None,
    feature_names   = None,
    class_names     = None,
    remove_const_feat = False,
    use_pca         = False,
    ridge_lambda    = 0.1,
    verbose         = False,
    reference       = "mean",
)
```

### Required Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `high_dim_data` | `(n_samples, n_features)` | Dataset in the original high-dimensional space |
| `low_dim_data` | `(n_samples, d)` | Dataset in the reduced space produced by the DR algorithm |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | `None` | Number of neighbors for the local Jacobian estimation. `None` uses the entire dataset |
| `feature_names` | `None` | List of feature names. Auto-generated if not provided |
| `class_names` | `None` | Class label for each instance, used in plot tooltips |
| `remove_const_feat` | `False` | Drops near-constant features before Jacobian estimation to reduce effective dimensionality. Recommended, together with `use_pca`, for data with more than about 50 features |
| `use_pca` | `False` | Applies PCA (95% variance retained) before Jacobian estimation to reduce effective dimensionality. Recommended, together with `remove_const_feat`, for data with more than about 50 features |
| `ridge_lambda` | `0.1` | Regularization strength for the weighted ridge least-squares solver |
| `verbose` | `False` | Prints debug information during computation |
| `reference` | `"mean"` | Reference point for the importance scores: the score of feature $j$ uses the displacement $x_j - \text{reference}_j$. `"mean"` uses the dataset mean, so scores do not change when a constant is added to a feature. `"zero"` uses the raw values, exactly as in the formula above; the two coincide for centered (e.g. standardized) data. An array with one value per feature is also accepted. The Jacobian, the influence vectors and SND do not depend on it |

---

## Methods

### `fit`

```python
result = fadex.fit(
    explain_index,
    show     = True,
    width    = 10,
    height   = 8,
    n_top    = 10,
    save_fig = False,
)
```

Computes the FADEx explanation for a single instance. Returns a dict with keys `"phi"` (importance scores), `"SND"` (spectral norm of the Jacobian, the local distortion measure), `"jac"` (the estimated Jacobian), and `"feature_names_top"`. When `show=True`, displays a feature importance bar chart for that instance.

---

### `fit_quality`

```python
quality = fadex.fit_quality(
    indices      = None,
    n_points     = None,
    test_size    = 0.3,
    random_state = 0,
    baseline     = False,
)
```

Checks whether the local linear model can be trusted. For each instance, the Jacobian is estimated from 70% of its neighbors and used to predict where the other 30% lie in the embedding. Returns a dict with the per-instance `"cosine"` (direction agreement) and `"pearson"` (agreement of displacement lengths), their averages `"mean_cosine"` and `"mean_pearson"`, and the evaluated `"indices"`. Use `n_points` to evaluate a random sample.

An embedding unrelated to the data does **not** score 0 on the cosine measure. Pass `baseline=True` to also get `"baseline_cosine"` and `"baseline_pearson"`, the same scores for a randomly permuted embedding, and compare against those.

---

### `fit_cluster`

```python
result = fadex.fit_cluster(
    cluster_ids,
    n_top    = 10,
    show     = False,
    save_fig = False,
)
```

Computes a cluster-level importance by averaging `phi` over all instances in `cluster_ids`. Returns a dict with keys `"phi_cluster"` and `"feature_names_top"`. Useful for understanding which features drive a specific group of points in the embedding.

---

### `importance_plot`

```python
fadex.importance_plot(
    width    = 8,
    height   = 8,
    n_top    = 10,
    save_fig = False,
)
```

Runs FADEx on the entire dataset, sums `phi` across all instances, and plots a global feature importance ranking.

---

### `interactive_plot`

```python
fadex.interactive_plot(
    width         = 10,
    height        = 8,
    show_features = False,
)
```

Displays an interactive Plotly scatter plot of the 2D embedding. Points are colored by their spectral norm, and hovering over a point shows its class, index and spectral norm. With `show_features=True` the tooltip also lists the feature importances.

---

### `plot_distortion`

```python
fadex.plot_distortion(
    width    = 10,
    height   = 10,
    save_fig = False,
)
```

Plots the 2D embedding colored by the SND distortion measure, with the color scale centered at 1 (cold colors below 1, warm colors above). SND compares embedding units with data units, so its absolute value depends on how both are scaled.

---

### `plot_grid_feature_vectors`

```python
fadex.plot_grid_feature_vectors(
    feature,
    labels       = None,
    mask         = None,
    grid_bins    = 15,
    min_points   = 1,
    scale_factor = 1.0,
    width        = 12,
    height       = 10,
    save_fig     = False,
)
```

Overlays Jacobian-based feature vectors on the 2D embedding using a grid. The embedding plane is divided into a `grid_bins × grid_bins` grid; for each occupied cell, the mean Jacobian is computed and the selected feature is drawn as an arrow. Arrow color encodes magnitude — vectors with norm ≥ 1 are capped to unit length and colored by their original magnitude.

`feature` accepts a feature name (string) or a column index (integer).

---

### `plot_feature_heatmap`

```python
fadex.plot_feature_heatmap(
    feature,
    gridsize = 30,
    cmap     = 'magma',
    width    = 10,
    height   = 8,
    save     = False,
)
```

Renders a hexbin heatmap on the 2D embedding showing where a specific feature has the highest importance. The background is set to the darkest tone of the colormap for visual contrast, and real data points are shown as faint white dots for spatial context.

`feature` accepts a feature name (string) or column index (integer).

---
