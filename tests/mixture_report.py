"""Shared helper for the mixture validation suite (spec section 6).

Each numbered test appends a section to `outputs/validation_report.md` carrying
its **numbers**, not just a pass/fail flag -- spec 6 is explicit about that, and
spec 10 requires the entropy and confound R2 alongside every reported fit.

Test sizes default to a value that keeps the whole suite runnable in minutes.
Set `MIXTURE_FULL=1` to run at the sizes the spec names (n = 100,000, 20-100
replicates); the reduced default is recorded in the report so a reader can see
which regime produced the numbers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

REPORT = ROOT / "outputs" / "validation_report.md"


def full_scale():
    return os.environ.get("MIXTURE_FULL", "") not in ("", "0", "false")


def size(full, reduced):
    """Spec-scale value when MIXTURE_FULL is set, else the fast default."""
    return full if full_scale() else reduced


def write(section, lines, passed=None, notes=None):
    """Append one test's results. `lines` is a list of strings or (key, value)."""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    body = [f"\n## {section}", ""]
    if passed is not None:
        body.append(f"**{'PASS' if passed else 'FAIL'}**")
        body.append("")
    if not full_scale():
        body.append("_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes "
                    "named in the spec)._")
        body.append("")
    for ln in lines:
        body.append(f"- {ln[0]}: **{ln[1]}**" if isinstance(ln, tuple) else str(ln))
    if notes:
        body += ["", notes]
    with open(REPORT, "a") as fh:
        fh.write("\n".join(body) + "\n")


def reset_report(header="# Mixture validation report"):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write(header + "\n")


def git_hash():
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def fit_summary(res, p, data, spec, extra=None):
    """The provenance block spec 10 requires with every fit."""
    import numpy as np

    from tissue_dynamics.mixture.model import (degeneracy_flags, normalised_entropy,
                                               responsibilities)

    D = data.as_jax() if hasattr(data, "as_jax") else data
    w = np.asarray(responsibilities(p, D, spec))
    ent = normalised_entropy(w)
    flags = degeneracy_flags(p, w, spec)
    out = [
        ("git", git_hash()),
        ("loglik", f"{res.loglik:.6f}"),
        ("||grad||inf", f"{res.grad_inf:.2e}"),
        ("iterations", res.n_iter),
        ("converged", res.converged),
        ("restart agreement", f"{100 * res.restart_agreement:.0f}%"
         if np.isfinite(res.restart_agreement) else "n/a"),
        ("wall time", f"{res.wall_time:.1f}s"),
        ("normalised entropy (spec 8.7)", f"{ent:.4f}"),
        ("rate ratio exp(e^delta)", f"{flags['rate_ratio']:.3f}"),
        ("mean w", f"{flags['mean_w']:.4f}"),
        ("degeneracy flags fired", flags["any"]),
    ]
    return out + (extra or []), ent, flags
