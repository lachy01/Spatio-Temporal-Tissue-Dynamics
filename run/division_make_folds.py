"""Write the stored cell-to-fold map the division comparison loads (spec 4.1, V5).

    python run/division_make_folds.py            # on Spartan, via hpc/folds.slurm

Run this in the SAME environment the mixture CV ran in. MiniBatchKMeans is
deterministic within an environment but not across them: Spartan's unpinned
scikit-learn and this repo's local 1.6.1 return slightly different partitions
from the same seed, so folds regenerated locally do not reproduce the
mixture CV's (V5 failed on that, 2026-10-05). Generating the map once where
the CV ran, and storing it, makes every later stage use exactly that partition.

Writes outputs/division_classifier/folds.parquet and folds_meta.json, and checks
the fold sizes against the mixture CV's stored outputs if they are present.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from tissue_dynamics.division import paths as DP                    # noqa: E402

from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.inference import block_folds, spatial_blocks  # noqa: E402

OUT = DP.out_dir()


def main():
    import numpy
    import sklearn

    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"], columns=["cell_id", "x", "y"])
    xy = df[["x", "y"]].to_numpy(float)
    blocks = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
    folds = block_folds(blocks, int(cfg["model"].get("cv_folds", 5)))
    sizes = {f: [int(np.sum(folds != f)), int(np.sum(folds == f))] for f in range(5)}

    match = None
    cvp = [json.load(open(f)) for f in sorted(glob.glob(
        DP.hpc_dir() + "/cv_penalty/*.json"))]
    if cvp:
        seen = {}
        for r in cvp:
            seen.setdefault(int(r["fold"]), set()).add((int(r["n_train"]), int(r["n_test"])))
        match = all(seen.get(f) == {tuple(sizes[f])} for f in range(5))

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({"cell_id": df["cell_id"].astype(str).to_numpy(), "block": blocks,
                  "fold": folds}).to_parquet(f"{OUT}/folds.parquet", index=False)
    meta = dict(sha256=hashlib.sha256(folds.astype(np.int8).tobytes()).hexdigest(),
                fold_sizes_train_test=sizes, matches_mixture_cv=match,
                sklearn=sklearn.__version__, numpy=numpy.__version__,
                python=sys.version.split()[0], host=os.uname().nodename)
    json.dump(meta, open(f"{OUT}/folds_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
    if match is False:
        sys.exit("fold sizes do NOT match the mixture CV outputs in this environment")


if __name__ == "__main__":
    main()
