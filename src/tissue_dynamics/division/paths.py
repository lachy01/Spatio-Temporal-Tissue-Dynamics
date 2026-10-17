"""Where the division comparison reads and writes, per section.

Unset MIX_DATASET is the breast section (outputs/division_classifier,
outputs/hpc). MIX_DATASET=ovarian writes outputs/division_classifier_ovarian and
reads the ovarian mixture's outputs/hpc_ovarian (or MIX_OUT). DIV_VARIANT=<v>
(a label variant) appends _<v> to the output folder.
"""
from __future__ import annotations

import os


def dataset():
    return os.environ.get("MIX_DATASET") or None


def variant():
    """Label variant (DIV_VARIANT; division/labels.py VARIANTS), or None."""
    return os.environ.get("DIV_VARIANT") or None


def out_dir():
    ds = dataset()
    base = f"outputs/division_classifier_{ds}" if ds else "outputs/division_classifier"
    v = variant()
    return f"{base}_{v}" if v else base


def hpc_dir():
    ds = dataset()
    return os.environ.get("MIX_OUT", f"outputs/hpc_{ds}" if ds else "outputs/hpc")


def phi_names_path(cfg):
    return os.path.splitext(cfg["data"]["table_parquet"])[0] + ".phi_names.json"


def type_names(cfg):
    """Cell-type labels in PhiSpace column order, after the section's fold
    (`data.type_fold`; ovarian: Granulosa and Fallopian Tube Epithelium -> Other,
    listed last). On breast this is the PhiSpace names unchanged."""
    import json

    names = json.load(open(phi_names_path(cfg)))
    fold = cfg["data"].get("type_fold") or {}
    out = []
    for t in names:
        if t not in fold and t not in out:
            out.append(t)
    for t in names:
        f = fold.get(t)
        if f is not None and f not in out:
            out.append(f)
    return out


def focus_types(cfg):
    """The two types Stage 4 singles out in its paired differences and
    reliability panels: breast Cancer Epithelial and Plasmablasts (the
    over-dispersed, high-cycling types); set per section with
    `division_focus_types` (ovarian: Tumour, T & NK Cells)."""
    return list(cfg.get("division_focus_types", ["Cancer Epithelial", "Plasmablasts"]))
