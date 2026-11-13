import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import umap
sys.path.insert(0, "/Users/csilva/src/fadex")
from fadex import FADEx

SEED, N, K = 0, 2000, 200
d = fetch_california_housing(); names = list(d.feature_names)
rng = np.random.default_rng(SEED); idx = rng.choice(len(d.data), N, replace=False)
Xr, price = d.data[idx], d.target[idx]
X = StandardScaler().fit_transform(Xr)
pca = PCA(2, random_state=SEED).fit(X)
L = {"PCA": pca.transform(X),
     "t-SNE": TSNE(2, perplexity=50, init="pca", random_state=SEED).fit_transform(X),
     "UMAP": umap.UMAP(n_neighbors=30, random_state=SEED).fit_transform(X)}
out = dict(X=X, Xr=Xr, price=price, idx=idx, pca_components=pca.components_, pca_evr=pca.explained_variance_ratio_)
nn = NearestNeighbors(n_neighbors=K).fit(X); _, nbrs = nn.kneighbors(X)
for m, Y in L.items():
    fx = FADEx(X, Y, n_neighbors=K, feature_names=names); fx._fit_all()
    r2 = np.random.default_rng(1); cos = []
    for i in r2.choice(N, 300, replace=False):
        nb = nbrs[i][1:].copy(); r2.shuffle(nb); cut = int(0.7*len(nb)); tr, te = nb[:cut], nb[cut:]
        J = fx._compute_jac_ls(X[i], Y[i], X[tr], Y[tr])
        p, a = (X[te]-X[i]) @ J.T, Y[te]-Y[i]
        cos.append((np.sum(p*a,1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(a,axis=1)+1e-12)).mean())
    k = m.replace("-", "")
    out.update({f"Y_{k}": Y, f"phi_{k}": fx.all_phis, f"snd_{k}": fx.all_norms, f"jac_{k}": fx.all_jacs, f"cos_{k}": np.array(cos)})
    print(m, "hold-out cosine", round(float(np.mean(cos)),3), "| SND median", round(float(np.median(fx.all_norms)),2))
np.savez("all.npz", **out)

# --- explore ---
print("PCA explained variance", pca.explained_variance_ratio_.round(3))
Jp = out["jac_PCA"]; print("PCA: mean J vs components, max abs diff", np.abs(Jp.mean(0)-pca.components_).max().round(3),
      "| std of J across points (max)", Jp.std(0).max().round(4))
Yt = out["Y_tSNE"]; lob = KMeans(2, n_init=10, random_state=0).fit_predict(Yt)
for c in (0,1):
    m = lob==c; print(f"t-SNE lobe {c}: n={m.sum()} lat {Xr[m,6].mean():.2f}±{Xr[m,6].std():.2f} lon {Xr[m,7].mean():.2f} price {price[m].mean():.2f} x-centre {Yt[m,0].mean():.1f}")
print("share of lobe split explained by latitude>36:", max(np.mean((Xr[:,6]>36)==lob), np.mean((Xr[:,6]>36)!=lob)).round(3))
for a,b in [("tSNE","UMAP"),("tSNE","PCA"),("UMAP","PCA")]:
    rs=[spearmanr(out[f"phi_{a}"][i], out[f"phi_{b}"][i])[0] for i in range(N)]
    print(f"Spearman phi {a} vs {b}: {np.mean(rs):.3f}")
for k in ("PCA","tSNE","UMAP"):
    g = out[f"phi_{k}"].mean(0); o=np.argsort(-g); print(k, "mean phi:", [(names[j], round(float(g[j]),2)) for j in o])
    mg = np.linalg.norm(out[f"jac_{k}"],axis=1).mean(0); o=np.argsort(-mg); print(k, "mean |g|:", [(names[j], round(float(mg[j]),2)) for j in o])
