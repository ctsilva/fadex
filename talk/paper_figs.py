"""Regenerated paper figures for the VIS talk, in the same style as california_figs.py.

Only figures that can be produced in full from public data and the fadex package:
  jacobian-quality.png   (paper Fig. 2)   from reproduce/results/e2_fit_quality.json
  snd-mnist.png          (paper Fig. 4)   from reproduce/results/e4_mnist_arrays.npz
  breast-cancer-umap.png, how-vectors.png (paper Fig. 7)
  synthetic.png          (paper Fig. 3)
The paper's own versions are kept in figs/paper-original/.

Usage (from the repo root):  reproduce/.venv/bin/python talk/paper_figs.py
Run reproduce/e2_fit_quality.py and reproduce/e4_mnist.py first.
"""
import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(os.path.dirname(HERE), "reproduce")
sys.path.insert(0, REPRO)
from core import fit_all, knn, purity          # noqa: E402
from data import load                           # noqa: E402

OUT = os.path.join(HERE, "figs")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, AQUA, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"
SEQ = mc.LinearSegmentedColormap.from_list("seq", colormaps["Blues"](np.linspace(0.18, 1, 256)))
plt.rcParams.update({"font.size": 22, "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "savefig.facecolor": "white"})


def clean(a):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(name)


def grid_vectors(Y, J, j, G=14, minpts=3):
    xb = np.linspace(Y[:, 0].min(), Y[:, 0].max(), G + 1); yb = np.linspace(Y[:, 1].min(), Y[:, 1].max(), G + 1)
    xi = np.clip(np.digitize(Y[:, 0], xb) - 1, 0, G - 1); yi = np.clip(np.digitize(Y[:, 1], yb) - 1, 0, G - 1)
    o = []
    for p in range(G):
        for q in range(G):
            m = (xi == p) & (yi == q)
            if m.sum() >= minpts:
                g = J[m][:, :, j].mean(0)
                o.append(((xb[p] + xb[p + 1]) / 2, (yb[q] + yb[q + 1]) / 2, g[0], g[1]))
    return np.array(o).T


# ------------------------------------------------------------------ Fig. 2: fit quality on Musk v2
def jacobian_quality():
    r = json.load(open(os.path.join(REPRO, "results", "e2_fit_quality.json")))
    mk = r["curves"]["musk2"]
    sizes, cos, lay = np.array(mk["sizes"]), np.array(mk["cosine"]), mk["layouts"]
    params = [5, 10, 15, 30, 50, 100]
    SIZE = 200                                   # 10% of the 2,000 samples (the paper's Fig. 2B uses 247)
    ch = json.load(open(os.path.join(REPRO, "results", "e2_chance_musk200.json")))      # from reproduce/e2_chance200.py
    chance = float(np.mean([ch["chance_cosine"][ch["layouts"].index(f"umap:{p}")] for p in params]))
    fig, ax = plt.subplots(1, 2, figsize=(19, 8.0), gridspec_kw=dict(width_ratios=[1.25, 1]))

    a = ax[0]
    shades = colormaps["Blues"](np.linspace(0.35, 1.0, len(params)))
    for p, c in zip(params, shades):
        a.plot(sizes, cos[:, lay.index(f"umap:{p}")], color=c, lw=3, marker="o", ms=6, label=str(p))
    a.axvline(SIZE, color=MUTED, lw=1.5, ls="--"); a.text(SIZE + 10, 0.455, "200 (10% of the data)", color=MUTED, fontsize=19, va="bottom")
    a.axhline(chance, color=ORANGE, lw=2, ls="--")
    a.text(990, chance + 0.012, f"chance level ({chance:.2f})", color=ORANGE, fontsize=21, ha="right", va="bottom")
    a.set_ylim(0.44, 0.94); a.set_xlim(0, 1010)
    a.set_xlabel("Neighbourhood size used by FADEx"); a.set_ylabel("Mean cosine similarity")
    a.set_title("UMAP layouts of Musk v2", fontsize=26, pad=12)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.yaxis.grid(True, color=GRID); a.set_axisbelow(True)
    a.legend(title="UMAP n_neighbors", frameon=False, ncol=3, loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=21, title_fontsize=21)

    a = ax[1]
    cols = ["isomap", "lle", "tsne", "umap"]; names = ["Isomap", "LLE", "t-SNE", "UMAP"]
    rows = sorted(params, reverse=True)
    row = list(sizes).index(SIZE)
    M = np.array([[cos[row, lay.index(f"{m}:{p}")] for m in cols] for p in rows])
    paper = np.array([[r["paper_fig2b"][str(p)][cols.index(m)] for m in cols] for p in rows])
    a.imshow(M, cmap=SEQ, vmin=0.1, vmax=0.95, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=22, color="white" if M[i, j] > 0.6 else INK)
    a.set_xticks(range(4)); a.set_xticklabels(names, color=INK); a.set_yticks(range(len(rows))); a.set_yticklabels(rows, color=INK)
    a.tick_params(length=0); a.set_ylabel("DR parameter (neighbours or perplexity)")
    a.set_title(f"Four DR methods, neighbourhood size {SIZE}", fontsize=26, pad=12)
    for sp in a.spines.values():
        sp.set_visible(False)
    fig.tight_layout(w_pad=3); save(fig, "jacobian-quality.png")
    lle = [cos[row, lay.index(f"lle:{p}")] for p in (5, 10, 15)]
    lle_ch = [ch["chance_cosine"][ch["layouts"].index(f"lle:{p}")] for p in (5, 10, 15)]
    print(f"  size {SIZE}: LLE with 5/10/15 neighbours {np.round(lle, 2)} (chance {np.round(lle_ch, 2)}); UMAP chance level {chance:.2f}; "
          f"other cells {np.delete(M, 1, axis=1).min():.2f}-{np.delete(M, 1, axis=1).max():.2f}; "
          f"vs paper's Fig. 2B (at 247): mean |diff| {np.abs(M - paper).mean():.3f}, max {np.abs(M - paper).max():.3f}")


# ------------------------------------------------------------------ Fig. 4: MNIST distortion
def snd_mnist():
    z = np.load(os.path.join(REPRO, "results", "e4_mnist_arrays.npz"), allow_pickle=True)
    Y, y, snd = z["Y"], z["y"].astype(int), z["snd"]
    X0, _, _ = load("mnist")
    X = X0[np.random.default_rng(0).choice(len(X0), len(Y), replace=False)] / 255.0     # same sample as e4_mnist.py
    nb = knn(X, 31)[:, 1:]
    hi, lo = int(np.argmax(snd)), int(np.argmin(snd))
    fig = plt.figure(figsize=(13.2, 11.2))
    gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.05)

    a = fig.add_subplot(gs[0, 0]); tab = colormaps["tab10"]
    a.scatter(Y[:, 0], Y[:, 1], c=y, cmap=tab, vmin=0, vmax=9, s=7, alpha=0.75, linewidths=0)
    for d in range(10):
        cx, cy = np.median(Y[y == d], axis=0)
        a.text(cx, cy, str(d), fontsize=24, fontweight="bold", ha="center", va="center", color=INK,
               bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none", alpha=0.8))
    a.set_title("(A) t-SNE layout, by digit", fontsize=24, pad=8); clean(a)

    a = fig.add_subplot(gs[0, 1])
    order = np.argsort(snd)
    s = a.scatter(Y[order, 0], Y[order, 1], c=snd[order], cmap=SEQ, vmin=np.percentile(snd, 2), vmax=np.percentile(snd, 98), s=7, linewidths=0)
    a.set_title("(B) SND at every point", fontsize=24, pad=8); clean(a)
    cb = fig.colorbar(s, ax=a, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("SND (darker: more stretched)", fontsize=19); cb.ax.tick_params(labelsize=17)

    a = fig.add_subplot(gs[1, 0])
    a.scatter(Y[:, 0], Y[:, 1], s=5, color=OTHER, alpha=0.7, linewidths=0)
    for i, c in ((hi, ORANGE), (lo, BLUE)):
        for q in nb[i]:
            a.plot([Y[i, 0], Y[q, 0]], [Y[i, 1], Y[q, 1]], color=c, lw=1, alpha=0.6)
        a.scatter(Y[nb[i], 0], Y[nb[i], 1], s=26, color=c, linewidths=0, zorder=4)
        a.scatter(*Y[i], s=300, color=c, edgecolors=INK, linewidths=2, zorder=5)
    a.set_title("(C) Where true neighbours land", fontsize=24, pad=8); clean(a)

    a = fig.add_subplot(gs[1, 1]); a.axis("off")
    a.scatter([0.06], [0.72], s=300, color=ORANGE, edgecolors=INK, linewidths=2, transform=a.transAxes)
    a.text(0.14, 0.72, "Highest-SND point and its\n30 nearest neighbours\nin the original space", transform=a.transAxes, va="center", fontsize=21)
    a.scatter([0.06], [0.36], s=300, color=BLUE, edgecolors=INK, linewidths=2, transform=a.transAxes)
    a.text(0.14, 0.36, "Lowest-SND point and its\n30 nearest neighbours", transform=a.transAxes, va="center", fontsize=21)
    save(fig, "snd-mnist.png")
    spread = lambda i: np.linalg.norm(Y[nb[i]] - Y[i], axis=1).mean()
    print(f"  SND min {snd.min():.1f}, share above 1: {np.mean(snd > 1):.0%}; neighbour spread highest {spread(hi):.0f}, lowest {spread(lo):.0f} layout units")


# ------------------------------------------------------------------ Fig. 7: Breast Cancer
def breast_cancer(seed=0):
    import umap
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    Xr, y, names = load("breast_cancer")
    X = StandardScaler().fit_transform(Xr)
    mal = y == "malignant"
    Y = np.asarray(umap.UMAP(random_state=seed).fit_transform(X), dtype=np.float64)
    _, _, J = fit_all(X, Y, 57)
    ae = list(names).index("area error")

    fig, a = plt.subplots(figsize=(8.6, 7.8))
    a.scatter(Y[~mal, 0], Y[~mal, 1], s=42, color=ORANGE, alpha=0.85, linewidths=0, label="Benign")
    a.scatter(Y[mal, 0], Y[mal, 1], s=42, color=BLUE, alpha=0.85, linewidths=0, label="Malignant")
    a.legend(frameon=False, markerscale=1.6, loc="best"); clean(a)
    fig.tight_layout(); save(fig, "breast-cancer-umap.png")

    fig, a = plt.subplots(figsize=(8.6, 7.8))
    a.scatter(Y[~mal, 0], Y[~mal, 1], s=30, color=ORANGE, alpha=0.22, linewidths=0)
    a.scatter(Y[mal, 0], Y[mal, 1], s=30, color=BLUE, alpha=0.22, linewidths=0)
    cx, cy, u, v = grid_vectors(Y, J, ae, G=13)
    mag = np.hypot(u, v)
    q = a.quiver(cx, cy, u, v, mag, cmap=mc.LinearSegmentedColormap.from_list("ink", ["#9ec5ea", "#0b2a57"]), angles="xy", width=0.008,
                 norm=mc.Normalize(0, np.percentile(mag, 95)))
    cb = fig.colorbar(q, ax=a, orientation="horizontal", fraction=0.045, pad=0.03, shrink=0.7)
    cb.set_label("Length of the area error vector"); cb.outline.set_visible(False)
    clean(a); save(fig, "how-vectors.png")

    nb = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    bnd = (mal[nb] != mal[:, None]).sum(1) >= 4
    g = J[bnd][:, :, ae]; to = Y[mal].mean(0) - Y[bnd]
    cos = np.sum(g * to, 1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(to, axis=1) + 1e-12)
    print(f"  seed {seed}: {bnd.sum()} boundary points; area error vector points toward the malignant group for {np.mean(cos > 0):.0%}")


# ------------------------------------------------------------------ Fig. 3: synthetic clusters
def synthetic(seed=0, k=20):
    import umap
    from sklearn.cluster import KMeans
    from e_small import synthetic as make
    X, truth = make(seed)
    Y = umap.UMAP(n_neighbors=15, random_state=seed).fit_transform(X)
    phi, _, _ = fit_all(X, Y, k)
    lab = KMeans(3, n_init=10, random_state=seed).fit_predict(phi)
    pur = purity(lab, truth)
    order = sorted(pur, key=lambda c: int(pur[c][0]))               # explanation cluster for C0, C1, C2
    cols = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(1, 2, figsize=(17.5, 7.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = ax[0]
    for n, c in enumerate(order):
        m = lab == c
        a.scatter(Y[m, 0], Y[m, 1], s=16, color=cols[n], alpha=0.8, linewidths=0,
                  label=f"C{n} (features {4 * n}–{4 * n + 3}), purity {pur[c][1]:.2f}")
    a.legend(frameon=False, markerscale=3.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=21)
    a.set_title("UMAP layout, by cluster of FADEx scores", fontsize=26, pad=12); clean(a)
    a = ax[1]
    H = np.array([phi[lab == c].mean(0) for c in order]); H = H / H.max()
    im = a.imshow(H, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(12)); a.set_xticklabels(range(12), color=INK); a.set_yticks(range(3)); a.set_yticklabels([f"C{n}" for n in range(3)], color=INK)
    a.tick_params(length=0); a.set_xlabel("Feature")
    for b in (3.5, 7.5):
        a.axvline(b, color="white", lw=3)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_title("Mean score per feature, by cluster", fontsize=26, pad=12)
    cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.02); cb.outline.set_visible(False); cb.set_label("Mean score (largest = 1)")
    fig.tight_layout(w_pad=2); save(fig, "synthetic.png")
    print("  purity C0, C1, C2:", [round(pur[c][1], 2) for c in order],
          "| block diagonal correct:", all(set(np.argsort(-H[n])[:4]) == set(range(4 * n, 4 * n + 4)) for n in range(3)))


# ------------------------------------------------------------------ Fig. 8: Dry Bean, four DR methods
BEAN_COLORS = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6"]
METHODS = ["umap", "tsne", "isomap", "lle"]
MNAME = {"umap": "UMAP", "tsne": "t-SNE", "isomap": "Isomap", "lle": "LLE"}


def dry_bean(seed=1):
    import itertools
    import umap
    from scipy.stats import spearmanr
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    X0, y0, names = load("drybean")
    names = list(names)
    idx = np.random.default_rng(seed).choice(len(X0), 2000, replace=False)       # same sample as reproduce/e6_drybean.py
    X, y = StandardScaler().fit_transform(X0[idx]), y0[idx]
    classes = sorted(set(y)); col = {c: BEAN_COLORS[i] for i, c in enumerate(classes)}
    L = {"tsne": TSNE(2, random_state=seed).fit_transform(X), "umap": umap.UMAP(random_state=seed).fit_transform(X),
         "isomap": Isomap(n_components=2).fit_transform(X),
         # Default settings for every method, as in reproduce/e6_drybean.py. LLE's default (5 neighbours)
         # gives a thin, poorly separated layout here; a larger value looks better but changes its scores.
         "lle": LocallyLinearEmbedding(n_components=2, random_state=seed).fit_transform(X)}
    L = {m: np.asarray(v, dtype=np.float64) for m, v in L.items()}

    def frame(a, m):
        """Zoom on the bulk of the layout: a few far outliers otherwise squeeze everything else."""
        lo, hi = np.percentile(L[m], [0.5, 99.5], axis=0); pad = 0.08 * (hi - lo)
        a.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); a.set_ylim(lo[1] - pad[1], hi[1] + pad[1])

    fits = {m: fit_all(X, L[m], 200) for m in METHODS}
    phi = {m: fits[m][0] for m in METHODS}; J = {m: fits[m][2] for m in METHODS}
    Z = {m: MinMaxScaler().fit_transform(phi[m]) for m in METHODS}
    lab = KMeans(3, n_init=10, random_state=seed).fit_predict(np.vstack([Z[m] for m in METHODS])).reshape(4, -1)
    order = np.argsort(-np.bincount(lab.ravel(), minlength=3))                   # cluster 0 = largest
    member = {m: {n: lab[i] == c for n, c in enumerate(order)} for i, m in enumerate(METHODS)}
    top5 = {n: {m: np.argsort(-Z[m][member[m][n]].mean(0))[:5] for m in METHODS} for n in (0, 1)}

    def points(a, m, mask=None, s=14, alpha=0.8):
        for c in classes:
            q = (y == c) if mask is None else ((y == c) & mask)
            a.scatter(L[m][q, 0], L[m][q, 1], s=s, color=col[c], alpha=alpha, linewidths=0)

    # four layouts, one image each; the class legend goes in the first
    for m in METHODS:
        fig, a = plt.subplots(figsize=(7.2, 7.6))
        points(a, m); a.set_title(MNAME[m], fontsize=30, pad=10); clean(a); frame(a, m)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.2)
        if m == "umap":
            hs = [plt.Line2D([], [], marker="o", ls="", ms=11, color=col[c]) for c in classes]
            fig.legend(hs, [c.title() for c in classes], frameon=False, ncol=3, loc="lower center", fontsize=19,
                       handletextpad=0.1, columnspacing=0.8, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(os.path.join(OUT, f"bean-{m}.png"), dpi=110); plt.close(fig); print(f"bean-{m}.png")

    def bars(a, n, m):
        o = top5[n][m][::-1]
        a.barh([names[j] for j in o], Z[m][member[m][n]].mean(0)[o], color=BLUE, height=0.66)
        a.set_title(MNAME[m], fontsize=22, pad=6, loc="left"); a.set_xticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(axis="y", length=0, labelcolor=INK, labelsize=17)

    # cluster 0: top features and ShapeFactor1 vectors
    sf1 = names.index("ShapeFactor1")
    fig = plt.figure(figsize=(22, 8.8)); outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.08)
    gb = outer[0].subgridspec(2, 2, hspace=0.28, wspace=1.15); gs = outer[1].subgridspec(2, 2, hspace=0.2, wspace=0.05)
    for i, m in enumerate(METHODS):
        bars(fig.add_subplot(gb[i // 2, i % 2]), 0, m)
        a = fig.add_subplot(gs[i // 2, i % 2]); points(a, m, s=9, alpha=0.3); frame(a, m)
        cx, cy, u, v = grid_vectors(L[m], J[m], sf1, G=11, minpts=4)
        a.quiver(cx, cy, u, v, color=INK, angles="xy", width=0.007)
        a.set_title(MNAME[m], fontsize=22, pad=6); clean(a)
    fig.text(0.30, 0.95, "Cluster 0: top-5 features per method", fontsize=25, ha="center")
    fig.text(0.73, 0.95, "Influence vectors of ShapeFactor1", fontsize=25, ha="center")
    fig.savefig(os.path.join(OUT, "bean-cluster0.png"), dpi=110, bbox_inches="tight", pad_inches=0.12); plt.close(fig); print("bean-cluster0.png")

    # cluster 1: top features, which beans, and how their features differ
    two = sorted(classes, key=lambda c: -np.mean([np.mean(y[member[m][1]] == c) for m in METHODS]))[:2]
    feats = ["AspectRatio", "Eccentricity", "ShapeFactor2", "ShapeFactor3"]
    fig = plt.figure(figsize=(26, 8.4)); outer = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.1)
    gb = outer[0].subgridspec(2, 2, hspace=0.3, wspace=1.2); gs = outer[1].subgridspec(2, 2, hspace=0.22, wspace=0.05)
    gh = outer[2].subgridspec(2, 2, hspace=0.42, wspace=0.18)
    for i, m in enumerate(METHODS):
        bars(fig.add_subplot(gb[i // 2, i % 2]), 1, m)
        a = fig.add_subplot(gs[i // 2, i % 2])
        a.scatter(L[m][:, 0], L[m][:, 1], s=6, color=OTHER, alpha=0.5, linewidths=0); points(a, m, mask=member[m][1], s=12)
        a.set_title(MNAME[m], fontsize=22, pad=6); clean(a); frame(a, m)
    for i, f in enumerate(feats):
        a = fig.add_subplot(gh[i // 2, i % 2]); j = names.index(f)
        bins = np.linspace(X[np.isin(y, two), j].min(), X[np.isin(y, two), j].max(), 28)
        for c in two:
            a.hist(X[y == c, j], bins=bins, color=col[c], alpha=0.75, label=c.title())
        a.set_title(f, fontsize=20, pad=6); a.set_yticks([]); a.tick_params(labelsize=15)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        if i == 0:
            a.legend(frameon=False, fontsize=17, loc="upper center")
    fig.text(0.27, 0.95, "Cluster 1: top-5 features per method", fontsize=25, ha="center")
    fig.text(0.575, 0.95, "Beans in cluster 1", fontsize=25, ha="center")
    fig.text(0.80, 0.95, f"{two[0].title()} and {two[1].title()}: feature values", fontsize=25, ha="center")
    fig.savefig(os.path.join(OUT, "bean-cluster1.png"), dpi=110, bbox_inches="tight", pad_inches=0.12); plt.close(fig); print("bean-cluster1.png")

    # numbers to quote in the notes
    sp = {f"{a}-{b}": float(np.mean([spearmanr(phi[a][i], phi[b][i])[0] for i in range(len(X))])) for a, b in itertools.combinations(METHODS, 2)}
    print("  spearman:", {k: round(v, 2) for k, v in sp.items()})
    for n in (0, 1):
        print(f"  cluster {n}: sizes", {m: int(member[m][n].sum()) for m in METHODS})
        for m in METHODS:
            print(f"     {MNAME[m]:<7}", [names[j] for j in top5[n][m]])
    print("  cluster 1 classes:", {c: round(float(np.mean([np.mean(y[member[m][1]] == c) for m in METHODS])), 2) for c in two})
    print("  common to all cluster-1 top-5:", set.intersection(*[set(names[j] for j in top5[1][m]) for m in METHODS]))
    for m in METHODS:                       # how the ShapeFactor1 vectors behave, per method
        g = J[m][:, :, sf1]; mag = np.linalg.norm(g, axis=1); unit = g / (mag[:, None] + 1e-12)
        span = np.linalg.norm(L[m].max(0) - L[m].min(0))
        per = {c: round(float(np.linalg.norm(unit[y == c].mean(0))), 2) for c in classes}
        ang = {c: int(np.degrees(np.arctan2(*g[y == c].mean(0)[::-1]))) for c in classes}
        big = {c: round(float(mag[y == c].mean() / mag.mean()), 1) for c in classes}
        print(f"  {MNAME[m]:<7} overall alignment {np.linalg.norm(unit.mean(0)):.2f} | median length / layout diagonal {np.median(mag) / span:.3f}"
              f"\n      alignment within class {per}\n      mean direction (deg) {ang}\n      relative length {big}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["jacobian_quality", "snd_mnist", "breast_cancer", "synthetic", "dry_bean"]
    for name in which:
        globals()[name]()
