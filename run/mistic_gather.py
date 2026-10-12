"""MisTIC step 1.5: join the tiles' probabilities back onto every transcript.

    python run/mistic_gather.py                  # the tiles in tiles.json
    python run/mistic_gather.py --dirs tile_all  # a whole-section fit instead

Reads `<mistic.dir>/tiles/<dir>/probs_core.parquet` (run/mistic_run.py) and the
input transcript table, and writes `<mistic.dir>/transcripts_mistic.parquet`:
one row per transcript MisTIC was given (qv >= 20, gene feature, assigned, host
cell in MisTIC's input), in the input table's order, with columns

    transcript_id, cell_id, gene, p

`p` is MisTIC's `reassign_probs`; transcripts MisTIC did not evaluate (no
candidate cell of another type within reach) get p = 0.

Also writes `<mistic.dir>/tiles_summary.csv` (one row per tile: sizes, epochs,
evaluated fraction, memory, time).

Checks, all fatal: every tile present (unless --allow-missing); no transcript
in two tiles' cores; every tile transcript found in the input table; p in [0, 1].
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics.mistic import paths                               # noqa: E402
from tissue_dynamics.mixture.data import load_config                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="tile directories under <mistic.dir>/tiles (default: tiles.json)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="gather whatever tiles exist (smoke tests); the rest get p = 0")
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = load_config()
    P = paths(cfg)
    if a.dirs is None:
        n_tiles = len(json.load(open(f"{P['inputs']}/tiles.json"))["tiles"])
        dirs = [f"tile_{k:02d}" for k in range(n_tiles)]
    else:
        dirs = a.dirs
    found = [d for d in dirs if os.path.exists(f"{P['tiles']}/{d}/probs_core.parquet")]
    missing = sorted(set(dirs) - set(found))
    if missing and not a.allow_missing:
        sys.exit(f"{len(missing)} tile(s) have no output: {missing[:10]}")
    print(f"{len(found)} tile(s) found" + (f", {len(missing)} missing" if missing else ""))

    tid, pv, rows = [], [], []
    for d in found:
        t = pq.read_table(f"{P['tiles']}/{d}/probs_core.parquet", columns=["transcript_id", "p"])
        tid.append(t["transcript_id"].to_numpy())
        pv.append(t["p"].to_numpy())
        s = json.load(open(f"{P['tiles']}/{d}/summary.json"))
        rows.append({"tile": d, **{k: s.get(k) for k in (
            "cells_fit", "cells_core", "transcripts_core", "evaluated_core",
            "evaluated_core_fraction", "p_core_mean_over_all_core_transcripts",
            "p_core_above_0.5", "epochs_run", "device", "peak_rss_gb")},
            "seconds": s["seconds"]["total"]})
    pd.DataFrame(rows).to_csv(os.path.join(cfg["mistic"]["dir"], "tiles_summary.csv"), index=False)
    tid = np.concatenate(tid) if tid else np.zeros(0, np.uint64)
    pv = np.concatenate(pv).astype(np.float32) if pv else np.zeros(0, np.float32)
    if len(np.unique(tid)) != len(tid):
        sys.exit("a transcript appears in more than one tile's core")
    if len(pv) and (np.nanmin(pv) < 0 or np.nanmax(pv) > 1 or np.isnan(pv).any()):
        sys.exit("p outside [0, 1] or NaN")
    order = np.argsort(tid)
    tid, pv = tid[order], pv[order]

    src = pq.read_table(f"{P['inputs']}/detected_transcripts.parquet",
                        columns=["transcript_id", "cell_id", "feature_name"],
                        read_dictionary=["cell_id", "feature_name"])
    t_all = src["transcript_id"].to_numpy()
    pos = np.searchsorted(tid, t_all)
    pos_c = np.minimum(pos, max(len(tid) - 1, 0))
    hit = (len(tid) > 0) & (tid[pos_c] == t_all) if len(tid) else np.zeros(len(t_all), bool)
    if int(hit.sum()) != len(tid):
        sys.exit(f"{len(tid) - int(hit.sum()):,} tile transcripts are not in the input table")
    p_all = np.zeros(len(t_all), np.float32)
    p_all[hit] = pv[pos_c[hit]]

    out = pa.table({"transcript_id": src["transcript_id"],
                    "cell_id": src["cell_id"], "gene": src["feature_name"],
                    "p": pa.array(p_all)})
    pq.write_table(out, P["transcripts"] + ".tmp", compression="zstd", row_group_size=8_000_000)
    os.replace(P["transcripts"] + ".tmp", P["transcripts"])
    print(f"wrote {P['transcripts']}: {len(t_all):,} transcripts, {hit.mean():.1%} evaluated by "
          f"MisTIC, mean p {p_all.mean():.4f}, p > 0.5 for {np.mean(p_all > 0.5):.2%}")


if __name__ == "__main__":
    main()
