"""MisTIC T2 (blocking): are the probabilities sane, and are they spatial?

    python run/mistic_t2_sanity.py
    python run/mistic_t2_sanity.py --box X0 X1 Y0 Y1   # one region (smoke tests)

1. Every retained transcript has p in [0, 1] (transcripts_mistic.parquet).
2. Nt <= N for every cell (cell_mistic.parquet).
3. On a random 1,000,000 transcripts, the distance from each to its own cell's
   polygon boundary -- positive inside, negative outside -- binned, against
   mean p. Misassigned transcripts sit near the edges of the cell they were
   given, so mean p should rise toward the boundary. Pass: mean p in the
   0-0.5 um bin exceeds mean p deep in the cell (>= 4 um) by more than 3
   combined standard errors, and mean p falls with distance across the inside
   bins (Spearman rho < -0.5). A flat curve means MisTIC is not capturing
   spatial misassignment, and the pipeline stops here.

Writes outputs/mistic/t2.md, t2_boundary.csv and t2_boundary.png; exits
non-zero on a failure.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics import mistic as M                               # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

N_SAMPLE, SEED = 1_000_000, 20261012
EDGES = [-np.inf, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, np.inf]
DEEP_UM = 4.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--box", type=float, nargs=4, default=None)
    a = ap.parse_args()

    import geopandas as gpd
    import pyarrow.parquet as pq
    import shapely
    from scipy.stats import spearmanr

    t0 = time.time()
    cfg = load_config()
    P = M.paths(cfg)
    out = os.environ.get("MISTIC_OUT", "outputs/mistic")
    os.makedirs(out, exist_ok=True)
    checks = {}

    # ---- 1. p in [0, 1] ------------------------------------------------------
    tx = pq.read_table(P["transcripts"], columns=["transcript_id", "cell_id", "p"],
                       read_dictionary=["cell_id"])
    p_all = tx["p"].to_numpy()
    checks["p in [0, 1] for every transcript"] = bool(
        np.all((p_all >= 0) & (p_all <= 1)) and not np.isnan(p_all).any())
    # ---- 2. Nt <= N ------------------------------------------------------------
    cells = M.load_cell_table(cfg)
    checks["Nt <= N for every cell"] = bool(np.all(cells["Nt"] <= cells["N"] + 1e-6))
    checks["Nt_nonB <= N_nonB for every cell"] = bool(
        np.all(cells["Nt_nonB"] <= cells["N_nonB"] + 1e-6))

    # ---- 3. p against distance to the cell's own boundary ------------------------
    src = pq.read_table(f"{P['inputs']}/detected_transcripts.parquet",
                        columns=["transcript_id", "x_location", "y_location"])
    if not np.array_equal(src["transcript_id"].to_numpy(), tx["transcript_id"].to_numpy()):
        sys.exit("transcripts_mistic.parquet is not in the input table's order")
    x, y = src["x_location"].to_numpy(), src["y_location"].to_numpy()
    pool = np.arange(len(x))
    if a.box:
        pool = pool[(x >= a.box[0]) & (x < a.box[1]) & (y >= a.box[2]) & (y < a.box[3])]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(pool, min(N_SAMPLE, len(pool)), replace=False))
    cid = tx["cell_id"].combine_chunks()
    cell = cid.dictionary.to_numpy(zero_copy_only=False).astype(str)[cid.indices.to_numpy()[idx]]
    poly = gpd.read_parquet(f"{P['inputs']}/cell_boundary_polygons.parquet")
    geom = poly.geometry.reindex(cell).to_numpy()
    px, py, pp = x[idx].astype(float), y[idx].astype(float), p_all[idx].astype(float)
    d = shapely.distance(shapely.points(px, py), shapely.get_exterior_ring(geom))
    inside = shapely.contains_xy(geom, px, py)
    sd = np.where(inside, d, -d)
    b = pd.cut(sd, EDGES, right=False, labels=False)
    rows = []
    for k in range(len(EDGES) - 1):
        m = b == k
        lab = ("outside the polygon" if k == 0
               else f"{EDGES[k]:g}-{EDGES[k + 1]:g}" if np.isfinite(EDGES[k + 1])
               else f">= {EDGES[k]:g}")
        rows.append({"distance to own boundary (um)": lab, "transcripts": int(m.sum()),
                     "mean p": float(pp[m].mean()) if m.any() else np.nan,
                     "se": float(pp[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else np.nan,
                     "evaluated (p > 0)": float(np.mean(pp[m] > 0)) if m.any() else np.nan,
                     "mean p, evaluated only": float(pp[m & (pp > 0)].mean())
                     if (m & (pp > 0)).any() else np.nan,
                     "mid": (EDGES[k] + EDGES[k + 1]) / 2 if np.isfinite(EDGES[k + 1]) and k > 0
                     else (EDGES[k] + 1.0 if k > 0 else -0.5)})
    T = pd.DataFrame(rows)
    T.drop(columns="mid").to_csv(f"{out}/t2_boundary.csv", index=False)

    near = (sd >= 0) & (sd < 0.5)
    deep = sd >= DEEP_UM
    diff = pp[near].mean() - pp[deep].mean()
    se = np.sqrt(pp[near].var(ddof=1) / near.sum() + pp[deep].var(ddof=1) / deep.sum())
    ins = T.iloc[1:].dropna(subset=["mean p"])
    rho = float(spearmanr(ins["mid"], ins["mean p"]).statistic)
    checks[f"mean p at 0-0.5 um exceeds mean p at >= {DEEP_UM:g} um by > 3 SE"] = bool(diff > 3 * se)
    checks["mean p falls with distance inside the cell (Spearman rho < -0.5)"] = bool(rho < -0.5)
    ok = all(checks.values())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(T))
    ax[0].errorbar(xs, T["mean p"], yerr=1.96 * T["se"], fmt="o-", color="#1f4e79", capsize=3,
                   label="all transcripts (p = 0 if not evaluated)")
    ax[0].plot(xs, T["mean p, evaluated only"], "s--", color="#c0504d", ms=4,
               label="evaluated transcripts only")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(T["distance to own boundary (um)"], rotation=40, ha="right", fontsize=8)
    ax[0].set_ylabel("mean p (MisTIC reassign_probs)")
    ax[0].set_xlabel("distance to own cell boundary (um; inside >= 0)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Mean p by distance to the host cell's boundary", fontsize=10)
    ax[1].bar(xs, T["transcripts"], color="#9dc3e6")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(T["distance to own boundary (um)"], rotation=40, ha="right", fontsize=8)
    ax[1].set_ylabel("transcripts in the sample")
    ax[1].set_title(f"{len(idx):,} random transcripts", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/t2_boundary.png", dpi=150)
    plt.close(fig)

    def md(df):
        cols = [c for c in df.columns if c != "mid"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float)
                                           else f"{r[c]:,}" if isinstance(r[c], (int, np.integer))
                                           else str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    text = [f"## T2. MisTIC sanity — **{'PASS' if ok else 'FAIL'}**", "",
            *[f"- {k}: **{'pass' if v else 'FAIL'}**" for k, v in checks.items()], "",
            f"Mean p at 0-0.5 um minus mean p at >= {DEEP_UM:g} um: **{diff:.4f}** "
            f"(SE {se:.4f}); Spearman rho of mean p on bin distance, inside bins: "
            f"**{rho:.2f}**.", "",
            md(T), "", "![T2](t2_boundary.png)"]
    if a.box:
        text.insert(2, f"_Restricted to the box {a.box} (smoke test)._\n")
    open(f"{out}/t2.md", "w").write("\n".join(text) + "\n")
    json.dump({"checks": checks, "diff": float(diff), "se": float(se), "rho": rho},
              open(f"{out}/t2.json", "w"), indent=1)
    print("\n".join(text[:len(checks) + 4]))
    print(f"T2: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
