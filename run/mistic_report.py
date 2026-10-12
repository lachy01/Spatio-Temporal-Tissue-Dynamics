"""Assemble outputs/mistic/report.md from the MisTIC test sections.

    python run/mistic_report.py

Each test script writes its own section (outputs/mistic/t1.md ... t6.md), so a
rerun replaces that section and nothing is ever appended twice. This joins
them under the decisions this implementation took, in spec order. A missing
section is listed as not yet run.
"""
import json
import os
import subprocess
import time

OUT = os.environ.get("MISTIC_OUT", "outputs/mistic")
SECTIONS = [("t1", "T1. Likelihood recovery (blocking)"),
            ("t2", "T2. MisTIC sanity (blocking)"),
            ("t3", "T3. Null calibration negative control (blocking)"),
            ("t4", "T4. Spillover control (report)"),
            ("t5", "T5. PhiSpace feature check (report)"),
            ("t6", "T6. Old vs new comparison (report)")]

HEADER = """# MisTIC integration: test report

Spec: `instructions/mistic_implementation_spec.md`. Pipeline, inputs and the
probability column: `outputs/mistic/README.md`.

## Decisions taken in implementing the spec

| item | spec | adopted | why |
| --- | --- | --- | --- |
| MisTIC fit | one run | **36 spatial tiles**, 50 um halo, p kept for core cells (spec 1.4) | MisTIC's prior-expression table has a row per (cell, gene): 3.5 billion for the section. Equal-count tiles of ~19,000 core cells; cores partition the cells. |
| unevaluated transcripts | -- | **p = 0** | MisTIC only scores transcripts with a candidate cell of another type within 5 um; it has no evidence of misassignment for the rest. |
| cells left out of MisTIC's input | drop, log | 15,312 empty, 514 invalid polygons, 579 unlabelled; **their transcripts p = 0** | 304 analysis cells (0.05%) among them keep their raw library size and m = 0. |
| library size in the models | Nt over all transcripts (notation table) | **Nt_nonB**: sum of q over genes outside the blocklist | the spec rules out any use of cycle-gene p, which the all-gene Nt would put in the offset; the models' N already excludes the blocklist. `model.mistic_library_size` switches it. |
| N_i in m_i = f N s | raw total | raw total over all genes, as the spec | s's denominator is the same all-gene total |
| technical null | NB, mean Nt e^b0 + theta m | as the spec, keeping the existing **iterative trimming** (P(Y >= y) < 0.001) | the trimming is how BASE-T/BASE-O's null has always been estimated; only the mean function changed |
| LOG arms (division comparison) | general rule | area spline removed, log Nt for log N; **no m term** | their outcome is a label, not a raw count, so the additive rule does not apply |
| BASE-O-neg (a Stage 7 variant) | -- | **unchanged** | its floor is the negative-codeword rate per um^2 times area: decoding noise scales with area, it is not a segmentation covariate |
| area elsewhere | -- | kept as a **diagnostic axis** only (confound R^2, T3 deciles) | not a model term |
| old area fits | -- | `model.technical: area` reproduces them | needed for T6 |
| T5 | -- | **deferred** | PhiSpace is trained in R; the weighted matrix (`data/mistic/cell_feature_matrix_mistic.h5`) is ready for it |

Known limitation: MisTIC's own model sees cycle-gene transcripts when it
scores other genes; they are part of the expression profiles it compares. No
covariate the models read uses p of a cycle-gene transcript.
"""


VALIDATION = "outputs/validation_report.md"
TEST01 = [("Test 01 with theta_m m", "with the theta_m m term"),
          ("Test 01 -- simulation recovery", "as configured (the comparison)")]
T1_MIXTURE_NOTE = """T1's mixture half is Test 01 (spec 6.1) with the theta_m m term in every
component's mean, beside the same test as configured. Both are the latest runs
in `outputs/validation_report.md`. At full scale on 2026-10-12 (git b14ce20,
weight_ridge 300) the two failed identically, on a0 alone: the intercept of
the cycling probability, estimated -1.707 against -1.735 in both, because the
ridge on the weight slopes narrows the spread of the logit and the unpenalised
intercept moves to keep mean pi. The failure predates the MisTIC term. theta_m,
b0, theta, delta and B were all recovered."""


def latest_section(path, prefix):
    """The last `## <prefix>...` section of a markdown file, demoted to ###."""
    import re

    if not os.path.exists(path):
        return None
    hits = [s for s in re.split(r"(?m)^## ", open(path).read()) if s.startswith(prefix)]
    return "### " + hits[-1].strip() if hits else None


def main():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    parts = [HEADER, f"_Assembled {time.strftime('%Y-%m-%d %H:%M')} at git {sha}._", ""]
    status = []
    for key, title in SECTIONS:
        path = os.path.join(OUT, f"{key}.md")
        if key == "t1":
            text = (open(path).read().strip() if os.path.exists(path)
                    else f"## {title}\n\n_Null half not yet run._")
            null_state = ("PASS" if "PASS" in text.splitlines()[0] else "FAIL"
                          if "FAIL" in text.splitlines()[0] else "not yet run")
            mix = [(latest_section(VALIDATION, p), lab) for p, lab in TEST01]
            mix_state = ("not yet run" if mix[0][0] is None else
                         "PASS" if "**PASS**" in mix[0][0] else "FAIL")
            text += "\n\n" + T1_MIXTURE_NOTE + "\n"
            for sec, lab in mix:
                if sec:
                    text += f"\n_Test 01 {lab}:_\n\n{sec}\n"
            parts.append(text + "\n")
            status.append((title, f"null: {null_state}; mixture: {mix_state}"))
            continue
        if key == "t5":
            parts.append("## T5. PhiSpace feature check (report)\n\nDeferred: PhiSpace is "
                         "trained in R. Its MisTIC input, X~ = sum of q per cell and gene, is "
                         "`data/mistic/cell_feature_matrix_mistic.h5`.\n")
            status.append((title, "deferred"))
        elif os.path.exists(path):
            text = open(path).read().strip()
            parts.append(text + "\n")
            first = text.splitlines()[0]
            status.append((title, "PASS" if "PASS" in first else "FAIL" if "FAIL" in first
                           else "reported"))
        else:
            parts.append(f"## {title}\n\n_Not yet run._\n")
            status.append((title, "not yet run"))
    summary = ["## Status", "", "| test | status |", "| --- | --- |"]
    summary += [f"| {t} | {s} |" for t, s in status]
    parts.insert(1, "\n".join(summary) + "\n")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(parts))
    print(json.dumps(dict(status), indent=1))


if __name__ == "__main__":
    main()
