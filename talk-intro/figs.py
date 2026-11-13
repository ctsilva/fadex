import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mc
from matplotlib import colormaps
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
OUT = "/Users/csilva/src/fadex/talk-intro/figs/"
r = np.load("all.npz"); X, Xr, price = r["X"], r["Xr"], r["price"]
names = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
Y = {k: r[f"Y_{k}"] for k in ("PCA","tSNE","UMAP")}; J = {k: r[f"jac_{k}"] for k in Y}; PHI = {k: r[f"phi_{k}"] for k in Y}
LAB = {"PCA":"PCA","tSNE":"t-SNE","UMAP":"UMAP"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, AQUA, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"
SEQ = mc.LinearSegmentedColormap.from_list("seq", colormaps["Blues"](np.linspace(0.18, 1, 256)))
DIV = mc.LinearSegmentedColormap.from_list("div", [BLUE, "#d9d8d3", ORANGE])
plt.rcParams.update({"font.size": 20, "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "savefig.facecolor": "white"})
def clean(a):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
def save(fig, name): fig.savefig(OUT+name, dpi=110, bbox_inches="tight", pad_inches=0.12); plt.close(fig); print(name)
def cbar(fig, s, a, label):
    cb = fig.colorbar(s, ax=a, fraction=0.035, pad=0.02); cb.set_label(label); cb.outline.set_visible(False); cb.solids.set_alpha(1); return cb
def grid_vectors(Yk, Jk, j, G=14, minpts=4):
    xb = np.linspace(Yk[:,0].min(), Yk[:,0].max(), G+1); yb = np.linspace(Yk[:,1].min(), Yk[:,1].max(), G+1)
    xi = np.clip(np.digitize(Yk[:,0], xb)-1, 0, G-1); yi = np.clip(np.digitize(Yk[:,1], yb)-1, 0, G-1)
    o = []
    for p in range(G):
        for q in range(G):
            m = (xi==p)&(yi==q)
            if m.sum() >= minpts:
                g = Jk[m][:,:,j].mean(0); o.append(((xb[p]+xb[p+1])/2, (yb[q]+yb[q+1])/2, g[0], g[1]))
    return np.array(o).T
def vec_panel(fig, a, k, f, color_by=None, clabel=None, title=None):
    j = names.index(f); c = Xr[:, names.index(color_by or f)]
    lo, hi = np.percentile(c, [2, 98])
    s = a.scatter(Y[k][:,0], Y[k][:,1], c=c, cmap=SEQ, vmin=lo, vmax=hi, s=16, alpha=0.75, linewidths=0)
    cx, cy, u, v = grid_vectors(Y[k], J[k], j)
    a.quiver(cx, cy, u, v, color=INK, angles="xy", width=0.005)
    a.set_title(title or f"{LAB[k]}: arrows for {f}", fontsize=24, pad=12); clean(a)
    cbar(fig, s, a, clabel or (color_by or f))

price_k = price*100; lobe = KMeans(2, n_init=10, random_state=0).fit_predict(Y["tSNE"])
south = lobe == (0 if Xr[lobe==0,6].mean() < Xr[lobe==1,6].mean() else 1)

# F1 map of California
fig, a = plt.subplots(figsize=(8.6, 8.2))
s = a.scatter(Xr[:,7], Xr[:,6], c=price_k, cmap=SEQ, s=26, alpha=0.85, linewidths=0, vmax=500)
a.set_aspect(1.25); a.set_xlabel("Longitude"); a.set_ylabel("Latitude")
for sp in ("top","right"): a.spines[sp].set_visible(False)
for name, lon, lat, tx, ty, ha in [("San Francisco",-122.42,37.77,-123.6,36.6,"right"),("Los Angeles",-118.24,34.05,-119.9,33.2,"right"),("San Diego",-117.16,32.72,-118.3,32.3,"right"),("Sacramento",-121.49,38.58,-120.2,39.6,"left")]:
    a.annotate(name, (lon, lat), (tx, ty), ha=ha, va="center", fontsize=19, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))
a.set_xlim(-127.2, -113.8); a.set_ylim(32.0, 42.3)
cbar(fig, s, a, "Median house value ($1,000)"); fig.tight_layout(); save(fig, "data-map.png")

# F3 three projections
fig, ax = plt.subplots(1, 3, figsize=(19, 7.4))
for a, k in zip(ax, Y):
    s = a.scatter(Y[k][:,0], Y[k][:,1], c=price_k, cmap=SEQ, vmax=500, s=14, alpha=0.8, linewidths=0)
    a.set_title(LAB[k], fontsize=26, pad=10); clean(a)
    if k == "PCA":
        q = np.percentile(Y[k], [0.5, 99.5], axis=0); a.set_xlim(q[0,0]-0.3, q[1,0]+0.3); a.set_ylim(q[0,1]-0.3, q[1,1]+0.3)
cbar(fig, s, ax, "Median house value ($1,000)"); save(fig, "three-projections.png")

# F4 lobes
fig, ax = plt.subplots(1, 2, figsize=(19, 8.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
for a, P, t in [(ax[0], Y["tSNE"], "t-SNE layout"), (ax[1], Xr[:, [7, 6]], "Map of California")]:
    a.scatter(P[~south,0], P[~south,1], s=16, color=BLUE, alpha=0.8, linewidths=0, label="Right group")
    a.scatter(P[south,0], P[south,1], s=16, color=ORANGE, alpha=0.8, linewidths=0, label="Left group")
    a.set_title(t, fontsize=24, pad=12); clean(a)
ax[1].set_aspect(1.25); ax[0].legend(frameon=False, markerscale=3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.02))
fig.tight_layout(); save(fig, "lobes-map.png")

# F5 colour by each feature
fig, ax = plt.subplots(2, 4, figsize=(19, 8.6))
for a, f in zip(ax.flat, names):
    c = Xr[:, names.index(f)]; lo, hi = np.percentile(c, [2, 98])
    a.scatter(Y["tSNE"][:,0], Y["tSNE"][:,1], c=c, cmap=SEQ, vmin=lo, vmax=hi, s=7, alpha=0.8, linewidths=0)
    a.set_title(f, fontsize=22, pad=6); clean(a)
fig.tight_layout(); save(fig, "color-by-feature.png")

# F6 PCA biplot
fig, a = plt.subplots(figsize=(7.4, 7.1))
P = Y["PCA"]; lim = np.percentile(np.abs(P), 99)
a.scatter(P[:,0], P[:,1], s=12, color=OTHER, alpha=0.7, linewidths=0)
C = r["pca_components"]; sc = lim*1.15
offs = {"AveRooms":(1.0,0.1),"AveBedrms":(-1.15,0.2),"Latitude":(0.1,0.3),"Longitude":(-0.1,0.3),"MedInc":(-0.85,0.1),"HouseAge":(0.75,-0.15),"Population":(-0.95,0.4),"AveOccup":(-0.6,-0.3)}
for j, f in enumerate(names):
    a.annotate("", (C[0,j]*sc, C[1,j]*sc), (0,0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3, mutation_scale=22))
    a.text(C[0,j]*sc*1.08+offs[f][0], C[1,j]*sc*1.08+offs[f][1], f, fontsize=19, color=INK, ha="center", va="center")
a.set_xlim(-lim*1.1, lim*1.1); a.set_ylim(-lim*1.1, lim*1.1); a.set_xlabel("PC 1: mostly location (north-west vs south-east)", fontsize=18); a.set_ylabel("PC 2: mostly house size", fontsize=18); clean(a)
fig.tight_layout(); save(fig, "pca-biplot.png")

# F8 latitude arrows PCA vs t-SNE
fig, ax = plt.subplots(1, 2, figsize=(19, 8.2))
vec_panel(fig, ax[0], "PCA", "Latitude", title="PCA: the same Latitude arrow at every district")
ax[0].set_xlim(-lim*1.1, lim*1.1); ax[0].set_ylim(-lim*1.1, lim*1.1)
vec_panel(fig, ax[1], "tSNE", "Latitude", title="t-SNE: a different Latitude arrow at each district")
fig.tight_layout(); save(fig, "lat-pca-vs-tsne.png")

# F7 local arrows at three points
T = Y["tSNE"]; nn = NearestNeighbors(n_neighbors=200).fit(X)
picks = [int(np.argmin(np.linalg.norm(T - np.array(q), axis=1))) for q in [(-25, 5), (-15, -35), (30, -5)]]
fig, a = plt.subplots(figsize=(7.6, 7.0))
a.scatter(T[:,0], T[:,1], s=12, color=OTHER, alpha=0.7, linewidths=0)
for i in picks:
    g = J["tSNE"][i]; L = np.linalg.norm(g, axis=0); top = np.argsort(-L)[:2]; sc = 14/L.max()
    for j in range(8):
        a.annotate("", (T[i,0]+g[0,j]*sc, T[i,1]+g[1,j]*sc), T[i], arrowprops=dict(arrowstyle="-|>", color=INK if j in top else MUTED, lw=3 if j in top else 1.2, mutation_scale=20 if j in top else 10))
    for j in top:
        dx, dy = g[0,j]*sc, g[1,j]*sc; n_ = np.hypot(dx, dy)
        a.text(T[i,0]+dx+2.2*dx/n_, T[i,1]+dy+2.2*dy/n_, names[j], fontsize=18, ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top", color=INK,
               bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    a.scatter(*T[i], s=120, color=ORANGE, edgecolors="white", linewidths=2, zorder=5)
clean(a); a.margins(0.08); fig.tight_layout(); save(fig, "local-arrows.png")

# F9/F10 vectors
fig, ax = plt.subplots(1, 2, figsize=(19, 8.2))
vec_panel(fig, ax[0], "tSNE", "Latitude", title="If Latitude increases")
vec_panel(fig, ax[1], "tSNE", "Longitude", title="If Longitude increases")
fig.tight_layout(); save(fig, "vectors-latlon.png")
fig, ax = plt.subplots(1, 2, figsize=(19, 8.2))
vec_panel(fig, ax[0], "tSNE", "MedInc", clabel="Median income ($10,000)", title="If median income increases")
vec_panel(fig, ax[1], "tSNE", "HouseAge", clabel="House age (years)", title="If house age increases")
fig.tight_layout(); save(fig, "vectors-income-age.png")

# F11 two districts
phi = PHI["tSNE"]; share = phi / phi.sum(1, keepdims=True)
A = int(np.argmax(np.where(south, share[:,1], -1))); B = int(np.argmax(np.where(~south, share[:,0], -1)))
units = ["$%.0fk income", "%.0f-year-old houses", "%.1f rooms", "%.1f bedrooms", "%.0f people", "%.1f per household", "%.1f° N", "%.1f° W"]
def val(i, j):
    v = Xr[i, j]; v = v*10 if j == 0 else (-v if j == 7 else v); return units[j] % v
fig, ax = plt.subplots(1, 3, figsize=(19, 7.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
a = ax[0]; a.scatter(T[:,0], T[:,1], s=12, color=OTHER, alpha=0.7, linewidths=0); clean(a)
for i, lab in [(A, "A"), (B, "B")]:
    a.scatter(*T[i], s=260, color=ORANGE, edgecolors="white", linewidths=2, zorder=5)
    a.text(T[i,0]+4, T[i,1]+4, lab, fontsize=30, fontweight="bold", color=INK, zorder=6)
for a, i, lab in [(ax[1], A, "A"), (ax[2], B, "B")]:
    o = np.argsort(phi[i])[::-1][:5][::-1]
    a.barh([f"{names[j]}\n{val(i,j)}" for j in o], phi[i][o], color=BLUE, height=0.62)
    a.set_title(f"District {lab}", fontsize=24, pad=10); a.set_xlabel("Importance score")
    for sp in ("top","right","left"): a.spines[sp].set_visible(False)
    a.tick_params(axis="y", length=0, labelcolor=INK, labelsize=17); a.xaxis.grid(True, color=GRID); a.set_axisbelow(True)
fig.tight_layout(); save(fig, "two-districts.png")
print("A", A, dict(zip(names, Xr[A].round(2))), "price", price[A]); print("B", B, dict(zip(names, Xr[B].round(2))), "price", price[B])

# F11b top feature map
gl = phi.mean(0); top3 = list(np.argsort(-np.bincount(phi.argmax(1), minlength=8))[:3]); t1 = phi.argmax(1)
fig, a = plt.subplots(figsize=(7.6, 7.0))
m = ~np.isin(t1, top3); a.scatter(T[m,0], T[m,1], s=18, color=OTHER, linewidths=0, label=f"Other ({m.mean():.0%})")
for j, c in zip(top3, [BLUE, ORANGE, AQUA]):
    m = t1 == j; a.scatter(T[m,0], T[m,1], s=18, color=c, linewidths=0, label=f"{names[j]} ({m.mean():.0%})")
h, l = a.get_legend_handles_labels(); a.legend(h[1:]+h[:1], l[1:]+l[:1], frameon=False, markerscale=3, loc="upper right", fontsize=18)
clean(a); fig.tight_layout(); save(fig, "top-feature.png")

# F12 distortion
snd = r["snd_tSNE"]; rel = np.log2(snd/np.median(snd))
fig, ax = plt.subplots(1, 2, figsize=(19, 8.2))
a = ax[0]; s = a.scatter(T[:,0], T[:,1], c=rel, cmap=DIV, norm=mc.TwoSlopeNorm(0, -1.5, 1.5), s=18, linewidths=0)
cb = cbar(fig, s, a, "Local stretch relative to the median"); cb.set_ticks([-1.5,-1,0,1,1.5]); cb.set_ticklabels(["⅓×","½×","median","2×","3×"])
a.set_title("Stretch at every district", fontsize=24, pad=12); clean(a)
a = ax[1]; a.scatter(T[:,0], T[:,1], s=12, color=OTHER, alpha=0.7, linewidths=0); clean(a)
nn30 = NearestNeighbors(n_neighbors=31).fit(X)
hi = int(np.argsort(snd)[-1]); lo = int(np.argsort(snd)[0]); spread = {}
for i, c, lab in [(hi, ORANGE, "most stretched"), (lo, BLUE, "least stretched")]:
    nb = nn30.kneighbors(X[i:i+1])[1][0][1:]
    for q in nb: a.plot([T[i,0], T[q,0]], [T[i,1], T[q,1]], color=c, lw=1, alpha=0.6)
    a.scatter(T[nb,0], T[nb,1], s=30, color=c, linewidths=0, zorder=4)
    a.scatter(*T[i], s=260, color=c, edgecolors=INK, linewidths=2, zorder=5, label=f"{lab} district and its 30 true neighbours")
    spread[lab] = np.linalg.norm(T[nb]-T[i], axis=1).mean()
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.04), fontsize=17); a.set_title("Where do a district's true neighbours land?", fontsize=24, pad=12)
fig.tight_layout(); save(fig, "distortion.png"); print("snd hi/lo", snd[hi].round(1), snd[lo].round(1), "mean layout dist to true nbrs", spread)
from scipy.stats import spearmanr
nb_all = nn30.kneighbors(X)[1][:,1:]; dl = np.array([np.linalg.norm(T[nb_all[i]]-T[i],axis=1).mean() for i in range(len(T))]); dh = np.array([np.linalg.norm(X[nb_all[i]]-X[i],axis=1).mean() for i in range(len(T))])
print("spearman SND vs layout/data distance ratio:", spearmanr(snd, dl/dh)[0].round(3))

# F13 methods compared
fig, a = plt.subplots(figsize=(10.5, 7.4))
o = np.argsort(PHI["tSNE"].mean(0)); yy = np.arange(8); h = 0.26
for n, (k, c) in enumerate(zip(("tSNE","UMAP","PCA"), (BLUE, ORANGE, AQUA))):
    a.barh(yy + (1-n)*h, PHI[k].mean(0)[o], height=h-0.03, color=c, label=LAB[k])
a.set_yticks(yy); a.set_yticklabels([names[j] for j in o], color=INK); a.tick_params(axis="y", length=0)
for sp in ("top","right","left"): a.spines[sp].set_visible(False)
a.xaxis.grid(True, color=GRID); a.set_axisbelow(True); a.set_xlabel("Mean importance score"); a.legend(frameon=False, loc="lower right", fontsize=20)
fig.tight_layout(); save(fig, "methods-compared.png")
print("medians", dict(zip(names, np.median(Xr,0).round(2))), "price median", np.median(price))

# F14 ball -> ellipse for district A (true singular values and orientation)
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
Ja = J["tSNE"][A].astype(float); Ua, Sa, Vta = np.linalg.svd(Ja, full_matrices=False)
if Ua[0, 0] < 0: Ua[:, 0] *= -1
if Ua[1, 1] < 0: Ua[:, 1] *= -1
fig = plt.figure(figsize=(7.4, 9.6))
a1 = fig.add_axes([0.05, 0.60, 0.9, 0.34]); a2 = fig.add_axes([0.05, 0.03, 0.9, 0.44])
def arr(a, v, c): a.annotate("", v, (0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=4, mutation_scale=26, shrinkA=0, shrinkB=0))
a1.add_patch(Circle((0, 0), 1, fc="#e9f1fb", ec=MUTED, lw=2)); arr(a1, (1, 0), BLUE); arr(a1, (0, 1), ORANGE)
a1.text(1.12, 0, "$\\mathbf{v}_1$", fontsize=26, va="center"); a1.text(0, 1.14, "$\\mathbf{v}_2$", fontsize=26, ha="center")
a1.text(0.5, -0.2, "length 1", fontsize=17, color=MUTED, ha="center")
a1.set_xlim(-2.6, 2.6); a1.set_ylim(-1.9, 1.5); a1.set_aspect("equal"); clean(a1)
a1.set_title("Feature space: a small ball around district A", fontsize=21, pad=8)
a1.text(0, -1.22, "6 other combinations of features\nleave the point where it is", fontsize=16, color=MUTED, va="top", ha="center")
ang = np.degrees(np.arctan2(Ua[1, 0], Ua[0, 0]))
a2.add_patch(Ellipse((0, 0), 2*Sa[0], 2*Sa[1], angle=ang, fc="#e9f1fb", ec=MUTED, lw=2))
p1, p2 = Ua[:, 0]*Sa[0], Ua[:, 1]*Sa[1]; arr(a2, p1, BLUE); arr(a2, p2, ORANGE)
a2.text(*(p1*1.08), f"$\\lambda_1$ = {Sa[0]:.1f}", fontsize=22, ha="left" if p1[0] > 0 else "right", va="center")
a2.text(*(p2*1.12), f"$\\lambda_2$ = {Sa[1]:.1f}", fontsize=22, ha="center", va="bottom" if p2[1] > 0 else "top")
L_ = Sa[0]*1.55; a2.set_xlim(-L_, L_); a2.set_ylim(-Sa[0]*1.12, Sa[0]*1.12); a2.set_aspect("equal"); clean(a2)
a2.set_title("Layout: the ball becomes an ellipse", fontsize=21, pad=8)
fig.patches.append(FancyArrowPatch((0.5, 0.585), (0.5, 0.515), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=30, lw=3, color=INK))
fig.text(0.53, 0.55, "apply $J$", fontsize=22, va="center")
save(fig, "ball-to-ellipse.png")
