"""Plot mechanism allocation for a set of models (Figure 3, Table 7, Figures 8-11).

Usage:
    python plot.py --results-dirs results/google__gemma-2-2b-it ... --out-dir plots
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "P052"          # Palatino (URW)
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

NEAR_POS_MAX = 2  # near-positional bands (d=1, 2) split out of unknown in the by-index plots

DISPLAY_NAMES = {
    "gemma-2-2b-it":                  "Gemma-2 2B",
    "gemma-2-9b-it":                  "Gemma-2 9B",
    "gemma-2-27b-it":                 "Gemma-2 27B",
    "gemma-3-4b-it":                  "Gemma-3 4B",
    "gemma-3-12b-it":                 "Gemma-3 12B",
    "gemma-3-27b-it":                 "Gemma-3 27B",
    "gemma-4-E2B-it":                 "Gemma-4 2B",
    "gemma-4-E4B-it":                 "Gemma-4 4B",
    "gemma-4-12B-it":                 "Gemma-4 12B",
    "gemma-4-31B-it":                 "Gemma-4 31B",
    "Qwen2.5-3B-Instruct":            "Qwen2.5 3B",
    "Qwen2.5-7B-Instruct":            "Qwen2.5 7B",
    "Qwen2.5-32B-Instruct":           "Qwen2.5 32B",
    "Qwen2.5-72B-Instruct":           "Qwen2.5 72B",
    "Llama-3.1-8B-Instruct":          "Llama-3.1 8B",
    "Llama-3.1-70B-Instruct":         "Llama-3.1 70B",
    "SmolLM3-3B":                     "SmolLM3 3B",
    "c4ai-command-r7b-12-2024":       "Command R7B",
    "Qwen3-1.7B":                     "Qwen3 1.7B",
    "Qwen3-4B":                       "Qwen3 4B",
    "Qwen3-8B":                       "Qwen3 8B",
    "Qwen3-14B":                      "Qwen3 14B",
}


def _savefig(fig, out_path, **kwargs):
    """Save out_path plus a PDF copy."""
    fig.savefig(out_path, **kwargs)
    stem, ext = os.path.splitext(out_path)
    if ext.lower() != ".pdf":
        fig.savefig(stem + ".pdf", **kwargs)


def _clean_label(label):
    # "google__gemma-2-2b-it" -> display name
    slug = label.split("__", 1)[-1]
    return DISPLAY_NAMES.get(slug, slug)


def _arch_of(clean_label):
    l = clean_label.lower()
    if any(k in l for k in ("gemma-2", "llama", "qwen")):
        return "rope"
    if any(k in l for k in ("gemma-3", "gemma-4", "smollm", "command")):
        return "hybrid"
    return "other"


def _pos_set_of(filename):
    """'pos-1-4-6-8' for multi-position runs, 'legacy' for last-token runs."""
    m = re.search(r"_(pos[-\d]+)_\d{8}\.csv$", os.path.basename(filename))
    return m.group(1) if m else "legacy"


def _cq_from_source(filename):
    """cat_to_query, field 4 of the CSV name."""
    try:
        return int(filename.replace(".csv", "").split("_")[4])
    except (IndexError, ValueError):
        return None


def _task_key_from_source(source_file):
    stem = re.sub(r"_mess\d+_fill\d+.*$", "", os.path.basename(source_file).replace(".csv", ""))
    m = re.match(r"^(.+?)_(\d+)_(\d+)_(\d+)_(\d+)_(.+)$", stem)
    return f"{m.group(6)}_cat{m.group(5)}" if m else stem


def load_results_dir(results_dir):
    """Return {label: DataFrame} with all of a model's multi-position CSVs."""
    label = os.path.basename(results_dir.rstrip("/"))
    by_set = {}
    for p in glob.glob(os.path.join(results_dir, "*.csv")):
        by_set.setdefault(_pos_set_of(p), []).append(p)
    pos_sets = sorted(k for k in by_set if k != "legacy")
    if len(pos_sets) > 1:
        raise ValueError(f"{label}: multiple position sets in one dir ({pos_sets})")
    csvs = by_set[pos_sets[0]] if pos_sets else []

    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        df["_source_file"] = os.path.basename(p)
        # |distance| for unknown samples, so plots can split unknown into d=1, 2, ... bands
        df["near_pos_dist"] = df["distance"].abs().where(df["patch_effect"] == "unknown")
        frames.append(df)
    return {label: pd.concat(frames, ignore_index=True)} if frames else {}


def _filter_by_cq(model_data, cq):
    out = {}
    for label, df in model_data.items():
        filtered = df[df["_source_file"].map(lambda f: _cq_from_source(f) == cq)]
        if len(filtered) > 0:
            out[label] = filtered
    return out


def print_mechanism_distribution(model_data, csv_path):
    """Table 7: counts of each outcome per model, written to csv_path."""
    rows = []
    for label, df in sorted(model_data.items()):
        row = {"model": _clean_label(label), "n": len(df),
               "n_legible": (df["patch_effect"] != "no_effect").sum()}
        for mech in ["no_effect", "positional", "keyload", "payload", "unknown"]:
            row[mech] = (df["patch_effect"] == mech).sum()
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("model")
    tbl.to_csv(csv_path)
    print(f"[saved] {csv_path}")
    return tbl


def _swatch_rpos_formula(ax, colors, fontsize=8, loc="upper right"):
    """r_pos = positional / (lexical + reflexive), each word led by its bar colour's swatch."""
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
    from matplotlib.patches import Rectangle

    sw, sh = 1.1 * fontsize, 0.7 * fontsize

    def term(mech):
        swatch = DrawingArea(sw, sh)
        swatch.add_artist(Rectangle((0, 0), sw, sh, facecolor=colors[mech], edgecolor="none"))
        return HPacker(children=[swatch, TextArea(mech, textprops=dict(fontsize=fontsize))],
                       align="center", pad=0, sep=0.35 * fontsize)

    num = term("positional")
    den = HPacker(children=[term("lexical"), TextArea("+", textprops=dict(fontsize=fontsize)),
                            term("reflexive")], align="center", pad=0, sep=0.4 * fontsize)
    den.set_figure(ax.figure)
    den_width = den.get_bbox(ax.figure.canvas.get_renderer()).width * 72 / ax.figure.dpi
    bar = DrawingArea(den_width, 1)
    bar.add_artist(Line2D([0, den_width], [0.5, 0.5], color="black", lw=0.7))
    frac = VPacker(children=[num, bar, den], align="center", pad=0, sep=2)
    formula = HPacker(children=[TextArea(r"$r_{pos}$ =", textprops=dict(fontsize=fontsize)), frac],
                      align="center", pad=0, sep=0.4 * fontsize)
    ax.add_artist(AnchoredOffsetbox(loc=loc, child=formula, frameon=False, pad=0.3, borderpad=0.2))


def plot_mechanism_stacked_bars(model_data, out_path, figsize=(8.15, 2.6)):
    """Figure 3: one stacked bar per model (positional, lexical, reflexive) with r_pos below
    on a log scale, models sorted by r_pos."""
    colors = {"positional": "#1f5db1", "lexical": "#fd9097", "reflexive": "#ffc9cd"}
    mechs = ("positional", "lexical", "reflexive")
    hybrid_color = "#e8646d"

    stats = {}
    for label, df in model_data.items():
        pe_col = df["patch_effect"]
        counts = {"positional": (pe_col == "positional").sum(), "lexical": (pe_col == "keyload").sum(),
                  "reflexive": (pe_col == "payload").sum()}
        total = sum(counts.values())
        semantic = counts["lexical"] + counts["reflexive"]
        if total == 0 or semantic == 0:
            continue
        stats[label] = {m: counts[m] / total for m in mechs}
        stats[label]["ratio"] = counts["positional"] / semantic

    labels = sorted(stats, key=lambda l: stats[l]["ratio"], reverse=True)
    names = [_clean_label(l) for l in labels]
    groups = [_arch_of(n) for n in names]
    x = np.arange(len(labels))
    ratios = np.array([stats[l]["ratio"] for l in labels])

    core = [i for i, g in enumerate(groups) if g in ("rope", "hybrid")]
    rope_idx = [i for i in core if groups[i] == "rope"]
    hyb_idx = [i for i in core if groups[i] == "hybrid"]
    assert [groups[i] for i in core] == ["rope"] * len(rope_idx) + ["hybrid"] * len(hyb_idx), \
        "RoPE and PE-hybrid models interleave in r_pos order"

    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=figsize,  # bars above, r_pos below
                                  gridspec_kw=dict(height_ratios=[1, 1], hspace=0))
    bottom = np.zeros(len(labels))
    for mech in mechs:
        h = np.array([stats[l][mech] for l in labels])
        ax.bar(x, h, 0.62, bottom=bottom, color=colors[mech], label=mech,
               edgecolor="white", linewidth=0.4, zorder=2)
        bottom += h
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=8)
    ax.set_ylabel("Allocation (%)", fontsize=9, labelpad=2)
    ax.tick_params(axis="y", pad=2)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    axr.set_xticks(x)
    axr.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
    curve = {"rope": colors["positional"], "hybrid": hybrid_color}
    if rope_idx and hyb_idx:
        for a in (axr, ax):
            a.axvline(x[hyb_idx[0]] - 0.5, color="gray", lw=1.0, ls="--", alpha=0.7, zorder=3)

    for g, idx in (("rope", rope_idx), ("hybrid", hyb_idx)):
        if idx:
            axr.plot(x[idx], ratios[idx], color=curve[g], lw=1.6, zorder=5)
            axr.scatter(x[idx], ratios[idx], s=28, color=curve[g], edgecolor="white", linewidth=0.9, zorder=6)

    # label the extremes and the two models either side of the boundary
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    to_label = {core[0], core[-1]}
    side = {core[0]: ("left", 0.15)}
    if rope_idx and hyb_idx:
        to_label |= {rope_idx[-1], hyb_idx[0]}
        side.update({rope_idx[-1]: ("center", -0.15), hyb_idx[0]: ("center", 0.15)})
    for i in sorted(to_label):
        ha, dx = side.get(i, ("center", 0))
        y = ratios[i] * (0.99 if i == core[0] else 1.15)
        axr.text(x[i] + dx, y, f"{ratios[i]:.2f}", ha=ha, va="bottom", fontsize=8.5, fontweight="bold",
                 color="#1a1a1a", zorder=7, path_effects=halo)
    axr.set_yscale("log")
    axr.set_ylim(0.23, 2.6)
    axr.set_yticks([0.25, 0.5, 1, 2])
    axr.set_yticklabels(["0.25", "0.5", "1", "2"], fontsize=8)
    axr.minorticks_off()
    axr.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    axr.set_axisbelow(True)
    axr.set_ylabel(r"$r_{pos}$ (log)", fontsize=9, labelpad=2)
    axr.tick_params(axis="y", pad=2)
    fig.align_ylabels([axr, ax])
    ax.tick_params(axis="x", length=0)
    _swatch_rpos_formula(axr, colors)

    # group brackets under the rotated names
    fig.canvas.draw()
    to_axes = axr.transAxes.inverted()
    label_bottom = min(to_axes.transform(t.get_window_extent())[:, 1].min() for t in axr.get_xticklabels())
    by = label_bottom - 0.04
    for g, txt, idx in (("rope", "RoPE Architectures", rope_idx),
                        ("hybrid", "PE Hybrid Architectures", hyb_idx)):
        if not idx:
            continue
        x0, x1 = x[idx[0]] - 0.35, x[idx[-1]] + 0.35
        trans = axr.get_xaxis_transform()
        axr.plot([x0, x0, x1, x1], [by + 0.03, by, by, by + 0.03], color=curve[g], lw=1.2,
                 transform=trans, clip_on=False)
        axr.text((x0 + x1) / 2, by - 0.025, txt, ha="center", va="top", fontsize=9,
                 fontweight="bold", color=curve[g], transform=trans)
    _savefig(fig, out_path, dpi=300, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close(fig)


def plot_mechanism_by_index(model_data, out_path):
    """One panel per model: mechanism shares by entity index (Figures 9-11)."""
    data_max = int(max(df["near_pos_dist"].max() for df in model_data.values()))
    near_pos_max = min(NEAR_POS_MAX, data_max)
    show_far = near_pos_max < data_max

    near_keys = [f"near_pos_d{d}" for d in range(1, near_pos_max + 1)]
    stack_order = ["positional"] + near_keys + (["unknown_far"] if show_far else []) + ["payload", "keyload"]
    near_shades = [plt.cm.Blues(x) for x in np.linspace(0.60, 0.15, near_pos_max)]
    colors_stack = (["#4878d0"] + near_shades + (["#aaaaaa" if near_pos_max else "#a6c8ec"] if show_far else [])
                    + ["#fdb462", "#1db377"])
    legend_labels = (["positional"] + [f"near-pos d={d}" for d in range(1, near_pos_max + 1)]
                     + ([f"mixed (d>{near_pos_max})" if near_pos_max else "unknown"] if show_far else [])
                     + ["reflexive", "lexical"])

    def _ratio(df):
        eff = df[df["patch_effect"] != "no_effect"]
        if len(eff) == 0:
            return 0
        pos = (eff["patch_effect"] == "positional").sum() + (eff["near_pos_dist"] <= near_pos_max).sum()
        lex = eff["patch_effect"].isin(["payload", "keyload"]).sum()
        return pos / lex if lex > 0 else float("inf")

    sorted_items = sorted(model_data.items(), key=lambda kv: _ratio(kv[1]), reverse=True)
    n_models = len(sorted_items)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True, squeeze=False)
    all_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in all_axes[n_models:]:
        ax.set_visible(False)

    n_idx = int(max(df["positional_index"].max() for df in model_data.values())) + 1
    for ax, (label, df) in zip(all_axes, sorted_items):
        counts = {k: np.zeros(n_idx) for k in stack_order}
        for i in range(n_idx):
            sub = df[df["positional_index"] == i]
            if len(sub) == 0:
                continue
            counts["positional"][i] = (sub["patch_effect"] == "positional").sum()
            counts["keyload"][i] = (sub["patch_effect"] == "keyload").sum()
            counts["payload"][i] = (sub["patch_effect"] == "payload").sum()
            for d, key in enumerate(near_keys, start=1):
                counts[key][i] = (sub["near_pos_dist"] == d).sum()
            if show_far:
                counts["unknown_far"][i] = ((sub["patch_effect"] == "unknown").sum()
                                            - sum(counts[k][i] for k in near_keys))
        totals = sum(counts[k] for k in stack_order)
        totals[totals == 0] = 1
        ax.stackplot(range(n_idx), [counts[k] / totals for k in stack_order], labels=legend_labels,
                     colors=colors_stack, alpha=0.9)
        ax.set_title(_clean_label(label), fontsize=9)
        ax.set_xlabel("Entity index")
        ax.set_xticks(range(0, n_idx, max(1, n_idx // 10)))
        ax.set_xlim(0, n_idx - 1)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5, color="white")
    all_axes[0].set_ylabel("Fraction of samples")

    handles, labels_leg = all_axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels_leg[::-1], fontsize=8, ncol=1,
               bbox_to_anchor=(1.01, 0.5), loc="center left")
    fig.suptitle("Mechanism distribution by entity index (averaged over tasks)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close(fig)


def plot_mechanism_by_index_by_cq(model_data, out_path_base):
    """plot_mechanism_by_index once per query category, saved as <base>_cq{0,1,2}."""
    stem, ext = os.path.splitext(out_path_base)
    for cq in [0, 1, 2]:
        filtered = _filter_by_cq(model_data, cq)
        if filtered:
            plot_mechanism_by_index(filtered, f"{stem}_cq{cq}{ext}")






def plot_rank_heatmap(model_data, out_path):
    """Figure 8: r_pos per task and model, cells coloured by the model's rank on that task."""
    task_model_ratio = {}
    for label, df in model_data.items():
        for src, grp in df.groupby("_source_file"):
            eff = grp[grp["patch_effect"] != "no_effect"]
            if len(eff) == 0:
                continue
            pos = (eff["patch_effect"] == "positional").sum() / len(eff)
            sem = eff["patch_effect"].isin(["payload", "keyload"]).sum() / len(eff)
            task_model_ratio.setdefault(_task_key_from_source(src), {})[label] = pos / (sem + 1e-9)

    all_labels = list(model_data)
    tasks = sorted((t for t, d in task_model_ratio.items() if len(d) == len(all_labels)),
                   key=lambda t: (t.rsplit("_cat", 1)[-1], t.rsplit("_cat", 1)[0]))
    # order models by pooled r_pos, as in plot_mechanism_stacked_bars
    pe_cols = {m: model_data[m]["patch_effect"] for m in all_labels}
    pooled = {m: (pe_cols[m] == "positional").sum() / pe_cols[m].isin(["payload", "keyload"]).sum()
              for m in all_labels}
    rope = sorted([m for m in all_labels if _arch_of(_clean_label(m)) == "rope"], key=pooled.get, reverse=True)
    hybrid = sorted([m for m in all_labels if _arch_of(_clean_label(m)) == "hybrid"], key=pooled.get, reverse=True)
    ordered = rope + hybrid
    n_rope, n_models = len(rope), len(ordered)

    ratio_mat = pd.DataFrame({m: {t: task_model_ratio[t][m] for t in tasks} for m in ordered})
    rank_mat = ratio_mat.rank(axis=1, ascending=False).astype(int)
    rank_colors = np.vstack([plt.cm.Blues(np.linspace(0.9, 0.45, n_rope)),
                             plt.cm.Purples(np.linspace(0.45, 0.9, n_models - n_rope))])

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.4), max(8, len(tasks) * 0.4)))
    for ti, task in enumerate(tasks):
        for mi, model in enumerate(ordered):
            rank = rank_mat.loc[task, model]
            ax.add_patch(plt.Rectangle((mi, ti), 1, 1, color=rank_colors[rank - 1]))
            ax.text(mi + 0.5, ti + 0.5, f"{rank}\n{ratio_mat.loc[task, model]:.2f}",
                    ha="center", va="center", fontsize=6.5, linespacing=1.0,
                    color="white" if rank <= 2 or rank >= n_models - 1 else "black")
    ax.axvline(n_rope, color="black", linewidth=1.5)
    ax.set_xlim(0, n_models)
    ax.set_ylim(0, len(tasks))
    ax.set_xticks([i + 0.5 for i in range(n_models)])
    ax.set_xticklabels([_clean_label(m) for m in ordered], rotation=35, ha="right", fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(tasks))])
    ax.set_yticklabels(tasks, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Positional/lexical ratio rank per task (green=positional, red=lexical)", fontsize=11)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dirs", nargs="+", required=True, help="One results directory per model.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    model_data = {}
    for d in args.results_dirs:
        model_data.update(load_results_dir(d))
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)

    print_mechanism_distribution(model_data, out("mechanism_distribution.csv"))
    plot_mechanism_stacked_bars(model_data, out("mechanism_stacked_bars.png"))
    plot_rank_heatmap(model_data, out("positional_rank_heatmap.png"))
    plot_mechanism_by_index_by_cq(model_data, out("mechanism_by_index.png"))


if __name__ == "__main__":
    main()
