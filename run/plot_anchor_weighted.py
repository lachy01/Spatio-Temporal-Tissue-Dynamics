"""Figures for the score-weighted anchor sensitivity (FINDINGS §14), from the stored
results of run/portrait_anchor_weighted.py (no rerun).

    python run/plot_anchor_weighted.py

Writes outputs/portrait/sensitivity/
  anchor_weighted_anchors.png      each archetype's anchor under each weighting
  anchor_weighted_fates.png        where the 400 trajectories end, unbounded and under
                                   OSDR q = 8, per weighting
  anchor_weighted_states.png       the composition of each run's leading settled state
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402

OUT = "outputs/portrait/sensitivity"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"
WEIGHTS = ["hard", "softmax t=0.25", "softmax t=0.5", "softmax t=1", "membership"]
SHORT = {"hard": "hard", "softmax t=0.25": "t = 0.25", "softmax t=0.5": "t = 0.5", "softmax t=1": "t = 1",
         "membership": "membership"}
LABEL = {"hard": "hard labels (§12)", "softmax t=0.25": "softmax t = 0.25", "softmax t=0.5": "softmax t = 0.5",
         "softmax t=1": "softmax t = 1", "membership": "membership (s + 1)/2"}
BOUNDS = {"unbounded": "unbounded", "OSDR p=8": "OSDR bound, q = 8"}
FATE = [("leading state", "#2a78d6"), ("other settled states", "#a9cbf2"), ("collapse (n < 1)", "#e34948"),
        ("still moving at t = 3000", "#eb6834"), ("other", "#e4e3df")]


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def anchors(d, names):
    cols = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(WEIGHTS)))
    fig, ax = plt.subplots(figsize=(8.2, 5.4), facecolor=SURF, layout="constrained")
    ax.set_facecolor(SURF)
    y = np.arange(len(names))[::-1]
    for i, n in enumerate(names):
        v = [d["anchors"][w][n] for w in WEIGHTS]
        ax.plot([min(v), max(v)], [y[i]] * 2, color=GRID, lw=3, zorder=1, solid_capstyle="round")
    for w, c in zip(WEIGHTS, cols):
        ax.scatter([d["anchors"][w][n] for n in names], y, s=58, color=c, edgecolor=SURF, lw=1.5, zorder=3,
                   label=LABEL[w])
    ax.axvline(0.098, color=MUTED, lw=0.9, ls="--", zorder=0)
    ax.text(0.0995, y[-1] - 0.75, "section mean 0.098", fontsize=8, color=INK2, va="center")
    ax.set_ylim(y[-1] - 1.1, y[0] + 0.6)
    ax.set_yticks(y, names)
    ax.set_xlabel("death anchor π̄ₖ (mean posterior cycling)")
    ax.set_title("Each archetype's anchor under each weighting", loc="left", fontsize=11.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    style(ax)
    fig.savefig(f"{OUT}/anchor_weighted_anchors.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def fates(d):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, facecolor=SURF, layout="constrained")
    for ax, (bkey, btitle) in zip(axes, BOUNDS.items()):
        ax.set_facecolor(SURF)
        for i, w in enumerate(WEIGHTS):
            f = d["runs"][f"{w}|{bkey}"]["fates"]
            att = f["attractors"]
            lead = att[0]["share"] if att else 0.0
            rest = sum(a["share"] for a in att[1:])
            parts = [lead, rest, f["collapse"], f["still moving"]]
            parts.append(max(0.0, 1.0 - sum(parts) - f["runaway"]) + f["runaway"])
            left = 0.0
            yy = len(WEIGHTS) - 1 - i
            for (lab, col), v in zip(FATE, parts):
                ax.barh(yy, v, left=left, height=0.62, color=col, edgecolor=SURF, lw=2)
                if v >= 0.07:
                    ax.text(left + v / 2, yy, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8.5,
                            color="white" if col in ("#2a78d6", "#e34948", "#eb6834") else INK)
                left += v
            if att:
                a = att[0]
                ax.text(1.01, yy, f"n {a['n']:.0f}, {a['types present']} types", va="center", fontsize=8,
                        color=INK2, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(WEIGHTS))[::-1], [LABEL[w] for w in WEIGHTS])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(btitle, loc="left", fontsize=11)
        ax.set_xlabel("share of 400 starts")
        style(ax)
    fig.legend(handles=[matplotlib.patches.Patch(color=c, label=l) for l, c in FATE], loc="lower center",
               ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Where trajectories end under each anchor weighting (leading state: n and types present, right)",
                 x=0.01, ha="left", fontsize=11.5)
    fig.savefig(f"{OUT}/anchor_weighted_fates.png", dpi=160, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def states(d, names):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURF, layout="constrained")
    vmax = max(a["X"][n] for w in WEIGHTS for b in BOUNDS for a in d["runs"][f"{w}|{b}"]["fates"]["attractors"][:1]
               for n in names)
    for ax, (bkey, btitle) in zip(axes, BOUNDS.items()):
        M = np.full((len(names), len(WEIGHTS)), np.nan)
        heads = []
        for j, w in enumerate(WEIGHTS):
            att = d["runs"][f"{w}|{bkey}"]["fates"]["attractors"]
            if att:
                M[:, j] = [att[0]["X"][n] for n in names]
                heads.append(f"{SHORT[w]}\n{100 * att[0]['share']:.0f}%, n {att[0]['n']:.0f}")
            else:
                heads.append(SHORT[w])
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(len(names)):
            for j in range(len(WEIGHTS)):
                v = M[i, j]
                ax.text(j, i, "absent" if v == 0 else f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color=MUTED if v == 0 else ("white" if v > 0.55 * vmax else INK))
        ax.set_xticks(range(len(WEIGHTS)), heads, fontsize=8.5)
        ax.set_xlabel("anchor weighting (softmax temperature t)", fontsize=9)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_title(f"{btitle}: the leading settled state", loc="left", fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.7, label="density, cells per (100 µm)²")
    fig.suptitle("Composition of each run's most common end state (column header: share of starts reaching it, "
                 "total density n)", x=0.01, ha="left", fontsize=11.5)
    fig.savefig(f"{OUT}/anchor_weighted_states.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    d = json.load(open(f"{OUT}/anchor_weighted.json"))
    names = list(d["anchors"]["hard"])
    anchors(d, names)
    fates(d)
    states(d, names)
    print(f"wrote {OUT}/anchor_weighted_{{anchors,fates,states}}.png")


if __name__ == "__main__":
    main()
