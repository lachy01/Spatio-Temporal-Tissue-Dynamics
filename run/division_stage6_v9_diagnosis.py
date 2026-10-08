"""V9 diagnosis (2026-10-08): allocation conditional on the mixture's posterior state, against V9.

    python run/division_stage6_v9_diagnosis.py

One Dirichlet-multinomial per type fails V9; this fits one profile for cycling and one for
non-cycling cells (weighted by the posterior w), and reports the expected and one simulated
draw's detection rates against the observed. Exploratory: not yet adopted.
"""
import sys, json, numpy as np, pandas as pd, scanpy as sc
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
sys.path.insert(0, "src"); sys.path.insert(0, "run")
import jax.numpy as jnp
from tissue_dynamics.mixture.data import load_config, _tirosh_genes
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis
from tissue_dynamics.mixture.model import build_data, responsibilities
import division_stage6_prepare as p6

cfg = load_config()
df = pd.read_parquet(cfg["data"]["table_parquet"])
df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
pe = json.load(open("outputs/hpc/point_estimate.json"))
basis = build_basis(df, cfg, verbose=False)
spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
D = build_data(df, basis, cfg, spec).as_jax()
w_all = np.asarray(responsibilities({k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}, D, spec))
lab = pd.read_parquet("outputs/division_classifier/labels.parquet", columns=["cell_type"])
ad = sc.read_10x_h5(f"{cfg['data']['xenium_dir']}/cell_feature_matrix.h5", gex_only=True)
ad.var_names_make_unique(); ad = ad[df["cell_id"].astype(str).to_numpy()].copy()
genes = list(ad.var_names); tir = [g for g in _tirosh_genes() if g in set(genes)]
Yg = np.asarray(ad.X[:, [genes.index(g) for g in tir]].todense()).astype(np.int64)
ctype = lab["cell_type"].to_numpy(); y = Yg.sum(1)

def dm_logpmf(X, n, alpha):
    a0 = alpha.sum()
    return (gammaln(n + 1) - gammaln(X + 1).sum(1) + gammaln(a0) - gammaln(n + a0)
            + (gammaln(X + alpha[None, :]) - gammaln(alpha)[None, :]).sum(1))

def fit_weighted(X, wt):
    n = X.sum(1)
    q = (wt[:, None] * X).sum(0) + 1e-6; q /= q.sum()
    f = lambda la: -np.sum(wt * dm_logpmf(X, n, np.exp(la) * q))
    return q, float(np.exp(minimize_scalar(f, bounds=(-4, 12), method="bounded").x))

def p_det(q, a, n):
    aq, ar = a * q[None, :], a * (1 - q)[None, :]
    nn = n[:, None].astype(float)
    return 1 - np.exp(gammaln(ar + nn) + gammaln(a) - gammaln(ar) - gammaln(a + nn))

rng = np.random.default_rng(1)
tot_bad_an = tot_bad_sim = tot = 0
for t in json.load(open("data/mixture_cells_major.phi_names.json")):
    m = ctype == t; Yt, yt, wt = Yg[m], y[m], w_all[m]; pos = yt >= 1
    obs = (Yt >= 1).mean(0); use = obs >= 0.01
    (q1, a1), (q0, a0) = fit_weighted(Yt[pos], wt[pos]), fit_weighted(Yt[pos], 1 - wt[pos])
    exp_det = (wt[:, None] * p_det(q1, a1, yt) + (1 - wt)[:, None] * p_det(q0, a0, yt)).mean(0)
    z = rng.uniform(size=len(yt)) < wt
    sim = np.zeros_like(Yt)
    sim[z] = p6.sim_dm(yt[z], q1, a1, rng); sim[~z] = p6.sim_dm(yt[~z], q0, a0, rng)
    rel_an = exp_det[use] / obs[use] - 1
    rel_sim = (sim >= 1).mean(0)[use] / obs[use] - 1
    tot += use.sum(); tot_bad_an += int(np.sum(np.abs(rel_an) > .1)); tot_bad_sim += int(np.sum(np.abs(rel_sim) > .1))
    worst = np.array(tir)[use][np.argsort(-np.abs(rel_an))[:3]]
    print(f"{t:<18} a1 {a1:7.1f} a0 {a0:7.1f} | expected: max {np.max(np.abs(rel_an)):.3f}, "
          f"{np.sum(np.abs(rel_an) > .1)}/{use.sum()} beyond 10% | one draw: max {np.max(np.abs(rel_sim)):.3f}, "
          f"{np.sum(np.abs(rel_sim) > .1)} beyond | worst {list(worst)}", flush=True)
print(f"beyond 10%: expected {tot_bad_an}/{tot}, one simulated draw {tot_bad_sim}/{tot}")
