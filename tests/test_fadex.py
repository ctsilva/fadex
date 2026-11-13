"""Tests for the FADEx package. Run with: pytest tests/"""
import numpy as np
import pytest

from fadex import FADEx


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 4))
    X = np.tanh(Z @ rng.normal(size=(4, 8))) + 0.05 * rng.normal(size=(300, 8))
    Y = np.c_[np.sin(Z[:, 0]) + Z[:, 1], Z[:, 2] ** 2 - Z[:, 3]]
    return X, Y


def test_recovers_a_linear_map():
    """On a linear projection the estimated Jacobian is the projection matrix, at every point."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6))
    W = rng.normal(size=(2, 6))
    Y = X @ W.T
    fx = FADEx(X, Y, n_neighbors=60, ridge_lambda=1e-8)
    for i in (0, 17, 399):
        out = fx.fit(i, show=False)
        assert np.allclose(out["jac"], W, atol=1e-3)
        assert out["SND"] == pytest.approx(np.linalg.norm(W, ord=2), rel=1e-3)


def test_integer_input_gives_the_same_scores_as_float_input():
    rng = np.random.default_rng(2)
    Xi = rng.integers(0, 17, size=(200, 10))
    Y = np.c_[Xi[:, 0] + 0.5 * Xi[:, 1], Xi[:, 2] - Xi[:, 3]] + rng.normal(0, 0.1, (200, 2))
    a = FADEx(Xi, Y, n_neighbors=40).fit(0, show=False)["phi"]
    b = FADEx(Xi.astype(float), Y, n_neighbors=40).fit(0, show=False)["phi"]
    assert a.dtype == np.float64
    assert np.allclose(a, b)
    assert not np.allclose(a, np.round(a))          # scores are not truncated to integers


def test_importance_formula_for_a_2d_layout(data):
    X, Y = data
    out = FADEx(X, Y, n_neighbors=40, reference="zero").fit(3, show=False)
    _, S, Vt = np.linalg.svd(out["jac"].astype(np.float64), full_matrices=False)
    expected = np.abs(Vt[0] * X[3]) + (S[1] / S[0]) * np.abs(Vt[1] * X[3])
    assert np.allclose(out["phi"], expected, atol=1e-6)
    assert (out["phi"] >= 0).all()


def test_fit_all_matches_single_fits(data):
    """The batched neighbour query used by _fit_all() gives the same result as fit()."""
    X, Y = data
    fx = FADEx(X, Y, n_neighbors=40)
    fx._fit_all()
    assert fx._nbr_table is None                    # the neighbour table is released afterwards
    single = FADEx(X, Y, n_neighbors=40)
    for i in (0, 5, 123, 299):
        ref = single.fit(i, show=False)
        assert np.allclose(fx.all_phis[i], ref["phi"])
        assert np.allclose(fx.all_jacs[i], ref["jac"])
        assert fx.all_norms[i] == pytest.approx(ref["SND"])


def test_fit_cluster_uses_cache_and_returns_a_dict(data):
    X, Y = data
    ids = [1, 4, 9, 20]
    fresh = FADEx(X, Y, n_neighbors=40).fit_cluster(ids)
    cached = FADEx(X, Y, n_neighbors=40)
    cached._fit_all()
    again = cached.fit_cluster(ids)
    assert set(fresh) == {"phi_cluster", "feature_names_top"}
    assert np.allclose(fresh["phi_cluster"], again["phi_cluster"])


@pytest.mark.parametrize("d", [1, 3])
def test_layouts_that_are_not_2d(data, d):
    X, Y = data
    Yd = np.c_[Y, Y[:, :1]][:, :d]
    out = FADEx(X, Yd, n_neighbors=40, reference="zero").fit(0, show=False)
    assert out["jac"].shape == (d, X.shape[1])
    _, S, Vt = np.linalg.svd(out["jac"].astype(np.float64), full_matrices=False)
    expected = sum((S[i] / S[0]) * np.abs(Vt[i] * X[0]) for i in range(len(S)))
    assert np.allclose(out["phi"], expected, atol=1e-6)


def test_high_dimensional_preprocessing_runs(data):
    X, Y = data
    Xh = np.c_[X, np.zeros((len(X), 60))]           # 60 constant features
    out = FADEx(Xh, Y, n_neighbors=40, remove_const_feat=True, use_pca=True).fit(0, show=False)
    assert out["jac"].shape == (2, Xh.shape[1])
    assert np.allclose(out["jac"][:, 8:], 0)        # constant features get a zero Jacobian column


@pytest.mark.parametrize("kwargs, match", [
    (dict(low=lambda Y: Y[:100]), "same number of rows"),
    (dict(n_neighbors=1), "n_neighbors"),
    (dict(n_neighbors=10_000), "n_neighbors"),
    (dict(feature_names=["a", "b"]), "feature_names"),
    (dict(class_names=["x"]), "class_names"),
    (dict(low=lambda Y: Y[:, 0]), "2-D"),
])
def test_input_validation(data, kwargs, match):
    X, Y = data
    kwargs = dict(kwargs)
    low = kwargs.pop("low", lambda Y: Y)(Y)
    kwargs.setdefault("n_neighbors", 40)
    with pytest.raises(ValueError, match=match):
        FADEx(X, low, **kwargs)


def test_nan_is_rejected(data):
    X, Y = data
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        FADEx(Xn, Y, n_neighbors=40)


def test_reference_mean_is_the_default_and_uses_the_displacement_from_the_mean(data):
    X, Y = data
    fx = FADEx(X, Y, n_neighbors=40)
    assert np.allclose(fx.reference_point, X.mean(0))
    out = fx.fit(3, show=False)
    _, S, Vt = np.linalg.svd(out["jac"].astype(np.float64), full_matrices=False)
    dx = X[3] - X.mean(0)
    assert np.allclose(out["phi"], np.abs(Vt[0] * dx) + (S[1] / S[0]) * np.abs(Vt[1] * dx), atol=1e-6)


def test_scores_do_not_change_when_a_constant_is_added_to_the_features(data):
    X, Y = data
    a = FADEx(X, Y, n_neighbors=40).fit(7, show=False)
    b = FADEx(X + 5.0, Y, n_neighbors=40).fit(7, show=False)
    assert np.allclose(a["phi"], b["phi"], atol=1e-5)
    assert np.allclose(a["jac"], b["jac"], atol=1e-5)
    # with the raw values as reference (Eq. 6 of the paper) the scores do depend on the shift
    c = FADEx(X, Y, n_neighbors=40, reference="zero").fit(7, show=False)
    d = FADEx(X + 5.0, Y, n_neighbors=40, reference="zero").fit(7, show=False)
    assert not np.allclose(c["phi"], d["phi"], atol=1e-3)


def test_reference_does_not_change_the_jacobian_or_snd(data):
    X, Y = data
    ref = np.full(X.shape[1], 0.3)
    a = FADEx(X, Y, n_neighbors=40, reference="zero").fit(2, show=False)
    b = FADEx(X, Y, n_neighbors=40, reference=ref).fit(2, show=False)
    assert np.array_equal(a["jac"], b["jac"]) and a["SND"] == b["SND"]
    assert not np.allclose(a["phi"], b["phi"])


@pytest.mark.parametrize("bad", ["median", [1.0, 2.0], np.full(8, np.nan)])
def test_reference_validation(data, bad):
    X, Y = data
    with pytest.raises(ValueError, match="reference"):
        FADEx(X, Y, n_neighbors=40, reference=bad)


def test_fit_quality_is_perfect_on_a_linear_map_and_poor_on_noise():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 5))
    good = FADEx(X, X @ rng.normal(size=(2, 5)).T, n_neighbors=50, ridge_lambda=1e-8).fit_quality(n_points=40, baseline=True)
    bad = FADEx(X, rng.normal(size=(300, 2)), n_neighbors=50).fit_quality(n_points=40, baseline=True)
    assert good["mean_cosine"] > 0.999 and good["mean_pearson"] > 0.999
    assert good["cosine"].shape == good["indices"].shape == (40,)
    # An unrelated embedding does not score 0 (neighbors lie towards the bulk of the data in
    # both spaces), which is why the chance level is reported: here the two are close.
    assert bad["mean_cosine"] < 0.7
    assert abs(bad["mean_cosine"] - bad["baseline_cosine"]) < 0.15
    assert good["mean_cosine"] - good["baseline_cosine"] > 0.3
    assert abs(bad["mean_pearson"]) < 0.3


def test_fit_quality_options(data):
    X, Y = data
    fx = FADEx(X, Y, n_neighbors=40)
    q = fx.fit_quality(indices=[0, 5, 9])
    assert list(q["indices"]) == [0, 5, 9] and np.isfinite(q["cosine"]).all()
    again = fx.fit_quality(indices=[0, 5, 9])
    assert np.array_equal(q["cosine"], again["cosine"])           # reproducible for a fixed seed
    whole = FADEx(X, Y).fit_quality(n_points=5)                  # n_neighbors=None: whole dataset
    assert np.isfinite(whole["mean_cosine"])
    hd = FADEx(np.c_[X, np.zeros((len(X), 60))], Y, n_neighbors=40, remove_const_feat=True, use_pca=True)
    assert np.isfinite(hd.fit_quality(n_points=5)["mean_cosine"])
    with pytest.raises(ValueError, match="too small"):
        FADEx(X, Y, n_neighbors=3).fit_quality()


# ---------------------------------------------------------------------------
# Distances other than Euclidean
# ---------------------------------------------------------------------------

def _pairwise(X, metric):
    from sklearn.metrics import pairwise_distances
    return pairwise_distances(X, metric=metric)


def test_euclidean_is_the_default_metric_and_aliases_are_accepted(data):
    X, Y = data
    a = FADEx(X, Y, n_neighbors=40).fit(3, show=False)
    b = FADEx(X, Y, n_neighbors=40, metric="L2").fit(3, show=False)
    assert np.array_equal(a["jac"], b["jac"]) and a["SND"] == b["SND"]
    assert FADEx(X, Y, n_neighbors=40, metric="angular").metric == "cosine"
    with pytest.raises(ValueError, match="metric"):
        FADEx(X, Y, n_neighbors=40, metric="chebyshev")


def test_manhattan_uses_l1_neighbors_and_the_largest_column_norm(data):
    X, Y = data
    fx = FADEx(X, Y, n_neighbors=40, metric="manhattan")
    expected = np.argsort(_pairwise(X, "manhattan")[5], kind="stable")[:40]
    assert set(fx._neighbor_table([5])[0]) == set(expected)
    out = fx.fit(5, show=False)
    assert out["SND"] == pytest.approx(np.linalg.norm(out["jac"], axis=0).max())
    # the stretch per unit of L1 distance can never exceed the stretch per unit of L2 distance
    assert out["SND"] <= np.linalg.norm(out["jac"], ord=2) + 1e-6


def test_manhattan_recovers_a_linear_map():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6))
    W = rng.normal(size=(2, 6))
    out = FADEx(X, X @ W.T, n_neighbors=60, ridge_lambda=1e-8, metric="manhattan").fit(0, show=False)
    assert np.allclose(out["jac"], W, atol=1e-3)
    assert out["SND"] == pytest.approx(np.linalg.norm(W, axis=0).max(), rel=1e-3)


def test_cosine_ignores_the_length_of_the_rows(data):
    X, Y = data
    scale = np.random.default_rng(4).uniform(0.5, 3.0, size=len(X))
    a = FADEx(X, Y, n_neighbors=40, metric="cosine")
    b = FADEx(X * scale[:, None], Y, n_neighbors=40, metric="cosine")
    assert np.array_equal(a._neighbor_table([7]), b._neighbor_table([7]))
    oa, ob = a.fit(7, show=False), b.fit(7, show=False)
    assert oa["SND"] == pytest.approx(ob["SND"], rel=1e-4)
    assert np.allclose(oa["jac"], ob["jac"] * scale[7], atol=1e-4)     # chain rule: J scales as 1 / ||x||
    assert np.allclose(oa["jac"] @ X[7], 0, atol=1e-4)                 # moving along x does nothing
    qa, qb = a.fit_quality(n_points=10), b.fit_quality(n_points=10)
    assert qa["mean_cosine"] == pytest.approx(qb["mean_cosine"], abs=1e-4)
    with pytest.raises(ValueError, match="zeros"):
        FADEx(np.r_[X[:-1], np.zeros((1, X.shape[1]))], Y, n_neighbors=40, metric="cosine")


def test_precomputed_recovers_the_stretch_of_a_linear_map():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 2))
    W = np.array([[3.0, 0.0], [0.0, 0.5]])
    fx = FADEx(_pairwise(X, "euclidean"), X @ W.T, n_neighbors=50, ridge_lambda=1e-8, metric="precomputed")
    out = fx.fit(0, show=False)
    assert out["phi"] is None and out["jac"] is None and out["feature_names_top"] == []
    assert np.allclose(out["singular_values"], [3.0, 0.5], rtol=1e-4)
    assert out["SND"] == pytest.approx(3.0, rel=1e-4)
    assert out["mds_components"] == 2 and out["negative_share"] < 1e-8
    q = fx.fit_quality(n_points=20, baseline=True)
    assert q["mean_cosine"] > 0.999 and q["mean_cosine"] - q["baseline_cosine"] > 0.3


def test_precomputed_euclidean_distances_agree_with_the_feature_based_fit(data):
    """With Euclidean distances the local MDS is the local PCA of use_pca, up to a rotation."""
    X, Y = data
    feat = FADEx(X, Y, n_neighbors=60, use_pca=True)
    pre = FADEx(_pairwise(X, "euclidean"), Y, n_neighbors=60, metric="precomputed")
    assert np.array_equal(np.sort(feat._neighbor_table([3, 9])), np.sort(pre._neighbor_table([3, 9])))
    a = np.array([feat.fit(i, show=False)["SND"] for i in range(30)])
    b = np.array([pre.fit(i, show=False)["SND"] for i in range(30)])
    assert np.allclose(a, b, rtol=0.05)


def test_precomputed_fit_all_and_whole_dataset_neighborhood(data):
    X, Y = data
    D = _pairwise(X[:80], "manhattan")
    fx = FADEx(D, Y[:80], n_neighbors=30, metric="precomputed")
    fx._fit_all()
    assert fx.all_norms.shape == (80,) and fx.all_phis is None and fx.all_jacs is None
    assert fx.all_norms[4] == pytest.approx(fx.fit(4, show=False)["SND"])
    assert fx.fit(4, show=False)["negative_share"] > 0            # L1 distances are not Euclidean
    whole = FADEx(D, Y[:80], metric="precomputed")              # one MDS shared by all instances
    assert np.isfinite(whole.fit(0, show=False)["SND"]) and np.isfinite(whole.fit_quality(n_points=5)["mean_cosine"])
    for call in (lambda: fx.fit_cluster([0, 1]), lambda: fx.importance_plot(),
                 lambda: fx.plot_feature_heatmap(0), lambda: fx.plot_grid_feature_vectors(0)):
        with pytest.raises(ValueError, match="precomputed"):
            call()


@pytest.mark.parametrize("bad, match", [
    (lambda D: D[:, :50], "square"),
    (lambda D: D + np.triu(np.ones_like(D), 1), "symmetric"),
    (lambda D: -D, "symmetric"),
])
def test_precomputed_validation(data, bad, match):
    X, Y = data
    D = _pairwise(X, "euclidean")
    with pytest.raises(ValueError, match=match):
        FADEx(bad(D), Y, n_neighbors=40, metric="precomputed")
    with pytest.raises(ValueError, match="feature_names"):
        FADEx(D, Y, n_neighbors=40, metric="precomputed", feature_names=list("abcdefgh"))


def test_cosine_measures_the_stretch_per_radian():
    """Points at angle t on rays of any length, placed at 3t: the stretch is 3 per radian."""
    rng = np.random.default_rng(6)
    t = rng.uniform(0.0, 2.0, size=300)
    X = rng.uniform(0.5, 4.0, size=300)[:, None] * np.c_[np.cos(t), np.sin(t)]
    Y = np.c_[3.0 * t, np.zeros(300)]
    fx = FADEx(X, Y, n_neighbors=100, ridge_lambda=1e-8, metric="cosine")
    for i in (0, 10, 200):
        out = fx.fit(i, show=False)
        assert out["SND"] == pytest.approx(3.0, rel=1e-3)
        tangent = np.array([-np.sin(t[i]), np.cos(t[i])])
        assert np.allclose(out["jac"][0] * np.linalg.norm(X[i]), 3.0 * tangent, atol=1e-3)
    assert fx.fit_quality(n_points=20)["mean_cosine"] > 0.999
