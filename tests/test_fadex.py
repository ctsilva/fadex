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
    out = FADEx(X, Y, n_neighbors=40).fit(3, show=False)
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
    out = FADEx(X, Yd, n_neighbors=40).fit(0, show=False)
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
