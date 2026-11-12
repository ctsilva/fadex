"""California Housing panels for the VIS talk (slides 2 and 3).

Usage: python california_figs.py path/to/all.npz
all.npz is written by talk-intro/compute.py (2,000 districts, t-SNE perplexity 50,
FADEx with 200 neighbours). Same data and styling as the talk-intro figures.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from sklearn.cluster import KMeans

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
r = np.load(sys.argv[1])
Xr, T, Jt, phi, snd = r["Xr"], r["Y_tSNE"], r["jac_tSNE"], r["phi_tSNE"], r["snd_tSNE"]
names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
LAT, LON = 6, 7
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, OTHER = "#2a78d6", "#eb6834", "#c9c8c2"
SEQ = mc.LinearSegmentedColormap.from_list("seq", colormaps["Blues"](np.linspace(0.18, 1, 256)))
DIV = mc.LinearSegmentedColormap.from_list("div", [BLUE, "#d9d8d3", ORANGE])
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


def cbar(fig, s, a, label):
    cb = fig.colorbar(s, ax=a, orientation="horizontal", fraction=0.045, pad=0.03, shrink=0.7)
    cb.set_label(label); cb.outline.set_visible(False); cb.solids.set_alpha(1)
    return cb


def grid_vectors(j, G=14, minpts=4):
    xb = np.linspace(T[:, 0].min(), T[:, 0].max(), G + 1); yb = np.linspace(T[:, 1].min(), T[:, 1].max(), G + 1)
    xi = np.clip(np.digitize(T[:, 0], xb) - 1, 0, G - 1); yi = np.clip(np.digitize(T[:, 1], yb) - 1, 0, G - 1)
    o = []
    for p in range(G):
        for q in range(G):
            m = (xi == p) & (yi == q)
            if m.sum() >= minpts:
                g = Jt[m][:, :, j].mean(0)
                o.append(((xb[p] + xb[p + 1]) / 2, (yb[q] + yb[q + 1]) / 2, g[0], g[1]))
    return np.array(o).T


lobe = KMeans(2, n_init=10, random_state=0).fit_predict(T)
south = lobe == (0 if Xr[lobe == 0, LAT].mean() < Xr[lobe == 1, LAT].mean() else 1)
agree = np.mean((Xr[:, LAT] < 36) == south)
print(f"latitude < 36 reproduces the two groups for {agree:.1%} of districts")

# Slide 2, left: the layout with its two groups
fig, a = plt.subplots(figsize=(9.6, 7.6))
a.scatter(T[south, 0], T[south, 1], s=18, color=ORANGE, alpha=0.8, linewidths=0, label="Left group")
a.scatter(T[~south, 0], T[~south, 1], s=18, color=BLUE, alpha=0.8, linewidths=0, label="Right group")
a.set_title("t-SNE layout", fontsize=26, pad=12); clean(a)
a.legend(frameon=False, markerscale=3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.03))
fig.tight_layout(); save(fig, "ca-layout.png")

# Slide 2, right: the same districts on the map
fig, a = plt.subplots(figsize=(7.4, 7.6))
a.scatter(Xr[~south, LON], Xr[~south, LAT], s=18, color=BLUE, alpha=0.8, linewidths=0)
a.scatter(Xr[south, LON], Xr[south, LAT], s=18, color=ORANGE, alpha=0.8, linewidths=0)
for name, lon, lat, tx, ty in [("San Francisco", -122.42, 37.77, -123.3, 36.3), ("Los Angeles", -118.24, 34.05, -120.3, 33.0)]:
    a.annotate(name, (lon, lat), (tx, ty), ha="right", va="center", fontsize=21, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))
a.set_xlim(-127.6, -113.8); a.set_ylim(32.0, 42.3); a.set_aspect(1.25)
a.set_title("The same districts on the map", fontsize=26, pad=12); clean(a)
fig.tight_layout(); save(fig, "ca-map.png")

# Slide 3, which: scores for one district (same choice of district A as talk-intro/figs.py)
share = phi / phi.sum(1, keepdims=True)
A = int(np.argmax(np.where(south, share[:, 1], -1)))
units = ["$%.0fk income", "%.0f-year-old houses", "%.1f rooms", "%.1f bedrooms", "%.0f people",
         "%.1f per household", "%.1f° N", "%.1f° W"]


def val(i, j):
    v = Xr[i, j]
    v = v * 10 if j == 0 else (-v if j == LON else v)
    return units[j] % v


fig, ax = plt.subplots(2, 1, figsize=(7.4, 8.4), gridspec_kw=dict(height_ratios=[1, 1.15]))
a = ax[0]; a.scatter(T[:, 0], T[:, 1], s=8, color=OTHER, alpha=0.7, linewidths=0); clean(a)
a.scatter(*T[A], s=300, color=ORANGE, edgecolors="white", linewidths=2, zorder=5)
a.text(T[A, 0] + 5, T[A, 1] + 5, "A", fontsize=30, fontweight="bold", color=INK, zorder=6)
a = ax[1]; o = np.argsort(phi[A])[::-1][:4][::-1]
a.barh([f"{names[j]}\n{val(A, j)}" for j in o], phi[A][o], color=BLUE, height=0.62)
a.set_xlabel("Importance score for district A")
for sp in ("top", "right", "left"):
    a.spines[sp].set_visible(False)
a.tick_params(axis="y", length=0, labelcolor=INK, labelsize=19); a.xaxis.grid(True, color=GRID); a.set_axisbelow(True)
fig.tight_layout(); save(fig, "ca-which.png")
print("district A", dict(zip(names, Xr[A].round(2))), "scores", dict(zip(names, phi[A].round(2))))

# Slide 3, how: Latitude influence vectors
fig, a = plt.subplots(figsize=(7.4, 8.4))
c = Xr[:, LAT]; lo, hi = np.percentile(c, [2, 98])
s = a.scatter(T[:, 0], T[:, 1], c=c, cmap=SEQ, vmin=lo, vmax=hi, s=18, alpha=0.75, linewidths=0)
cx, cy, u, v = grid_vectors(LAT)
a.quiver(cx, cy, u, v, color=INK, angles="xy", width=0.006)
clean(a); cbar(fig, s, a, "Latitude (° N)")
save(fig, "ca-how.png")

# Slide 3, where: stretch relative to the median district
rel = np.log2(snd / np.median(snd))
fig, a = plt.subplots(figsize=(7.4, 8.4))
s = a.scatter(T[:, 0], T[:, 1], c=rel, cmap=DIV, norm=mc.TwoSlopeNorm(0, -1.5, 1.5), s=20, linewidths=0)
cb = cbar(fig, s, a, "Stretch relative to the median")
cb.set_ticks([-1.5, -1, 0, 1, 1.5]); cb.set_ticklabels(["⅓×", "½×", "median", "2×", "3×"])
clean(a)
save(fig, "ca-where.png")
