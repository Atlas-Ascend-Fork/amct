# -*- coding: UTF-8 -*-
"""Generate report figures from the checked HiFloat8 experiment JSON files.

The figures are derived only from evidence JSON committed beside this script.
They contain no manually entered accuracy or benchmark measurements.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
FIGURES = ROOT / "figures"

COLORS = {
    "red": "#C7000B",
    "red_light": "#FDEBEC",
    "green": "#2A7F62",
    "green_light": "#E5F3ED",
    "blue": "#2864A0",
    "blue_light": "#E8F0F8",
    "amber": "#C47A16",
    "amber_light": "#FBF0DF",
    "ink": "#1F2933",
    "muted": "#637282",
    "line": "#CCD4DB",
    "panel": "#F7F9FA",
    "white": "#FFFFFF",
}


def load_json(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["line"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "figure.facecolor": COLORS["white"],
            "axes.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["white"],
        }
    )


def save_figure(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_footer(fig, text):
    fig.text(0.5, 0.018, text, ha="center", color=COLORS["muted"], fontsize=8)


def draw_box(ax, xy, width, height, title, subtitle, facecolor, edgecolor):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    return box


def draw_arrow(ax, start, end, color=None, dashed=False):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.5,
        color=color or COLORS["muted"],
        linestyle="--" if dashed else "-",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def draw_workflow_inputs(ax):
    boxes = [
        ((0.02, 0.69), 0.15, 0.14, "BF16 Qwen3", "596M parameters", "panel", "line"),
        (
            (0.22, 0.69),
            0.16,
            0.14,
            "amct.quantize",
            "Cast / Quantile / OFMR",
            "blue_light",
            "blue",
        ),
        (
            (0.43, 0.69),
            0.16,
            0.14,
            "Native probe",
            "Minimal cast roundtrip",
            "amber_light",
            "amber",
        ),
    ]
    for xy, width, height, title, subtitle, face, edge in boxes:
        draw_box(ax, xy, width, height, title, subtitle, COLORS[face], COLORS[edge])
    draw_arrow(ax, (0.17, 0.76), (0.22, 0.76))
    draw_arrow(ax, (0.38, 0.76), (0.43, 0.76))


def draw_workflow_cast_path(ax):
    boxes = [
        (
            (0.65, 0.80),
            0.16,
            0.12,
            "Native cast",
            "Unavailable: 161002",
            "red_light",
            "red",
        ),
        (
            (0.65, 0.58),
            0.16,
            0.12,
            "amct_ops fallback",
            "Selected and verified",
            "green_light",
            "green",
        ),
        (
            (0.85, 0.58),
            0.13,
            0.12,
            "uint8 codes",
            "Real HiFloat8 bytes",
            "green_light",
            "green",
        ),
        (
            (0.65, 0.34),
            0.16,
            0.12,
            "Decode to BF16",
            "Quantized values retained",
            "blue_light",
            "blue",
        ),
        ((0.85, 0.34), 0.13, 0.12, "BF16 Linear", "PPL evaluation", "panel", "line"),
    ]
    for xy, width, height, title, subtitle, face, edge in boxes:
        draw_box(ax, xy, width, height, title, subtitle, COLORS[face], COLORS[edge])
    draw_arrow(ax, (0.59, 0.78), (0.65, 0.86), COLORS["red"], dashed=True)
    draw_arrow(ax, (0.59, 0.72), (0.65, 0.64), COLORS["green"])
    ax.text(0.608, 0.84, "try", fontsize=8, color=COLORS["red"])
    ax.text(0.602, 0.64, "fallback", fontsize=8, color=COLORS["green"])
    draw_arrow(ax, (0.81, 0.64), (0.85, 0.64), COLORS["green"])
    draw_arrow(ax, (0.915, 0.58), (0.75, 0.46), COLORS["blue"])
    draw_arrow(ax, (0.81, 0.40), (0.85, 0.40), COLORS["blue"])


def draw_workflow_quantization(ax):
    draw_workflow_inputs(ax)
    draw_workflow_cast_path(ax)


def draw_workflow_deployment(ax):
    boxes = [
        (
            (0.22, 0.18),
            0.16,
            0.12,
            "amct.convert",
            "Native deploy stage",
            "amber_light",
            "amber",
        ),
        (
            (0.47, 0.18),
            0.20,
            0.12,
            "DT_HIFLOAT8 quantize",
            "aclnnQuantize required",
            "red_light",
            "red",
        ),
        (
            (0.75, 0.18),
            0.20,
            0.12,
            "HiFloat8 MatMul",
            "Not reached in CANN 9.0.0",
            "red_light",
            "red",
        ),
    ]
    for xy, width, height, title, subtitle, face, edge in boxes:
        draw_box(ax, xy, width, height, title, subtitle, COLORS[face], COLORS[edge])
    draw_arrow(ax, (0.38, 0.24), (0.47, 0.24), COLORS["red"])
    draw_arrow(ax, (0.67, 0.24), (0.75, 0.24), COLORS["red"], dashed=True)
    ax.text(
        0.57,
        0.125,
        "Blocked: DT_HIFLOAT8 is not in the kernel dtype support list",
        ha="center",
        color=COLORS["red"],
        fontsize=9,
        fontweight="bold",
    )


def plot_workflow():
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "AMCT HiFloat8 execution path verified in this experiment",
        loc="left",
        pad=18,
        fontweight="bold",
    )
    ax.text(
        0,
        0.955,
        "The fallback completes a real encode/decode roundtrip; "
        "native low-bit MatMul remains a separate deploy capability.",
        fontsize=9.5,
        color=COLORS["muted"],
    )
    draw_workflow_quantization(ax)
    draw_workflow_deployment(ax)
    add_footer(fig, "Measured path: Ascend910_9362 | CANN 9.0.0 | AMCT 1.1.0")
    save_figure(fig, "hifloat8_execution_path.png")


def add_bar_labels(ax, bars, digits=3):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def configure_accuracy_panel(ax, by_id, ids, title, subtitle):
    names = ["Cast", "Quantile", "OFMR"]
    colors = [COLORS["amber"], COLORS["green"], COLORS["blue"]]
    deltas = [by_id[item]["delta"] for item in ids]
    bars = ax.bar(names, deltas, color=colors, width=0.62)
    ax.axhline(
        0.2,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.3,
        label="Reference delta = 0.2",
    )
    ax.set_title(title, fontweight="bold", pad=20)
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        color=COLORS["muted"],
        fontsize=9,
    )
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", color=COLORS["line"], linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    add_bar_labels(ax, bars)
    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_accuracy(summary):
    by_id = {row["id"]: row for row in summary["experiments"]}
    official_ids = [
        "cast_official",
        "quantile_official",
        "ofmr_official",
    ]
    controlled_ids = [
        "cast_controlled",
        "quantile_official",
        "ofmr_controlled",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharey=True)
    panels = [
        (axes[0], official_ids, "Official configurations", "Different granularity / layer scope"),
        (axes[1], controlled_ids, "Controlled comparison", "Tensor weights + skip lm_head"),
    ]
    for ax, ids, title, subtitle in panels:
        configure_accuracy_panel(ax, by_id, ids, title, subtitle)
    axes[0].set_ylabel("PPL increase vs BF16 baseline")
    fig.suptitle(
        "Qwen3-0.6B HiFloat8 accuracy on Wikitext-2",
        x=0.05,
        y=1.02,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    add_footer(
        fig,
        "Full evaluation: 73 x 4096 tokens | BF16 PPL 19.1651 | Lower delta is better",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, "ppl_accuracy_comparison.png")


def configure_benchmark_panel(ax, benchmark, dtype):
    rows = [row for row in benchmark["results"] if row["dtype"] == dtype]
    sizes = np.array([row["numel"] for row in rows])
    series = [
        ("encode", "throughput_mb_s", "o", "red", "Encode"),
        ("decode", "throughput_mb_s", "s", "blue", "Decode"),
        (
            "roundtrip",
            "effective_throughput_mb_s",
            "^",
            "green",
            "Roundtrip effective",
        ),
    ]
    for operation, metric, marker, color, label in series:
        ax.plot(
            sizes,
            [row[operation][metric] for row in rows],
            marker=marker,
            linewidth=2,
            color=COLORS[color],
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_title(dtype.upper(), fontweight="bold")
    ax.set_xlabel("Elements (log2 scale)")
    ax.grid(color=COLORS["line"], linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_benchmark(benchmark):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for ax, dtype in zip(axes, ("float16", "bfloat16"), strict=True):
        configure_benchmark_panel(ax, benchmark, dtype)
    axes[0].set_ylabel("Throughput (decimal MB/s)")
    fig.suptitle(
        "HiFloat8 cast throughput scales with tensor size",
        x=0.05,
        y=1.0,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    add_footer(
        fig,
        "10 warmups + 100 synchronized measurements | Roundtrip effective rate uses 4 bytes/element",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, "cast_operator_benchmark.png")


def configure_cost_panel(ax, names, colors, values, title, ylabel, digits):
    bars = ax.bar(names, values, color=colors, width=0.64)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", color=COLORS["line"], linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025 if max(values) else 0.02,
            f"{value:.{digits}f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_ylim(0, max(values) * 1.17 if max(values) else 1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_cost(summary):
    by_id = {row["id"]: row for row in summary["experiments"]}
    ids = ["baseline", "cast_official", "quantile_official", "ofmr_official"]
    names = ["BF16", "Cast", "Quantile", "OFMR"]
    colors = [COLORS["muted"], COLORS["amber"], COLORS["green"], COLORS["blue"]]
    evaluation = [by_id[item]["evaluation_seconds"] for item in ids]
    memory = [by_id[item]["peak_memory_bytes"] / (1024**3) for item in ids]
    calibration = [
        0.0
        if by_id[item]["calibration_seconds"] is None
        else by_id[item]["calibration_seconds"]
        for item in ids
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.5))
    panels = [
        (axes[0], evaluation, "Evaluation time", "seconds", 0),
        (axes[1], memory, "Peak allocated memory", "GiB", 2),
        (axes[2], calibration, "Calibration time", "seconds", 2),
    ]
    for ax, values, title, ylabel, digits in panels:
        configure_cost_panel(ax, names, colors, values, title, ylabel, digits)
    fig.suptitle(
        "Fake-quant execution cost is not native deployment performance",
        x=0.04,
        y=1.01,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    add_footer(
        fig,
        "The simulation keeps BF16 weights and inserts encode/decode work; no deployment speedup is claimed.",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(fig, "fake_quant_cost_and_memory.png")


def main():
    configure_style()
    summary = load_json("experiment_summary.json")
    benchmark = load_json("benchmark_hifloat8_cast.json")
    plot_workflow()
    plot_accuracy(summary)
    plot_benchmark(benchmark)
    plot_cost(summary)
    print(f"Generated four figures in {FIGURES}")


if __name__ == "__main__":
    main()
