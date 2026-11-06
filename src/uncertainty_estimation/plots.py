"""Static plots for uncertainty evaluation (PNG). Colors follow the method, fixed order; line style is a
secondary encoding (three slots are below 3:1 contrast on the surface; methods.csv is the table view)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"U0": "#2a78d6", "U1": "#eb6834", "U2-eq": "#1baf7a", "U2-rank": "#eda100", "U2-f1": "#e87ba4"}
STYLE = {"U0": "-", "U1": "--", "U2-eq": "-.", "U2-rank": ":", "U2-f1": (0, (5, 1, 1, 1, 1, 1))}


def _axes(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)


def excess_risk_curve(m: dict) -> tuple[list[float], list[float]]:
    """Per accepted code k = 1..n (in score order): coverage k/n and risk(k) - best risk(k).
    risk(k) is the risk of the tie group code k belongs to (tied codes enter together); best risk(k) =
    max(0, k - n1) / k, the lowest achievable with k codes accepted. The mean of the excess is E-AURC."""
    n, n1 = m["n"], m["n_correct"]
    risks, prev = [], 0
    for p in m["risk_coverage"]:
        k = round(p["coverage"] * n)
        risks += [p["risk"]] * (k - prev)
        prev = k
    return ([k / n for k in range(1, n + 1)],
            [r - max(0, k - n1) / k for k, r in enumerate(risks, 1)])


def excess_risk_coverage(metrics: dict, methods: list[str], level: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _axes(ax, "Coverage (share of candidate codes auto-accepted)",
          "Excess risk (risk minus best possible risk)")
    for m in methods:
        x, y = excess_risk_curve(metrics[f"{m}|{level}"])
        ax.step([0] + x, [y[0]] + y, where="pre", color=COLOR[m], linestyle=STYLE[m], linewidth=2,
                label=f'{m}  (E-AURC {metrics[f"{m}|{level}"]["e_aurc"]:.3f})')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, None)
    ax.set_title(f"Excess risk–coverage, {level.replace('_', ' ')} (area = E-AURC, lower is better)",
                 color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability(metrics: dict, methods: list[str], level: str, path) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.4), sharey=True, facecolor=SURFACE)
    for ax, m in zip(axes, methods):
        bins = metrics[f"{m}|{level}"]["reliability"]
        _axes(ax, "Confidence", "Accuracy" if ax is axes[0] else "")
        ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, linestyle="--")
        ax.bar([(b["lo"] + b["hi"]) / 2 for b in bins], [b["acc"] for b in bins], width=0.09,
               color=COLOR[m], edgecolor=SURFACE, linewidth=1)
        for b in bins:
            ax.text((b["lo"] + b["hi"]) / 2, b["acc"] + 0.02, f'n={b["n"]}', ha="center", fontsize=7, color=TEXT_2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        x = metrics[f"{m}|{level}"]
        ax.set_title(f'{m}  ECE {x["ece"]:.3f} (chance {x["ece_floor_median"]:.3f})', color=TEXT, fontsize=10,
                     loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
