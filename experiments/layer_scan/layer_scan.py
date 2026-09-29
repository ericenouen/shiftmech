"""Run the mechanism analysis at every layer of one model, to choose the patching layer.

Usage:
    python layer_scan.py --model-id google/gemma-3-4b-it --token-positions -1 -4 -6 -8
    python layer_scan.py --model-id google/gemma-3-4b-it --token-positions -1 -4 -6 -8 --messiness 4
"""

import argparse
import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoConfig

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SCAN_SAMPLES = 1000
NUM_INSTANCES = 20
SCHEMA = "boxes"
CAT_TO_QUERY = 1  # Q: which box? A: box letter

COLORS = {"positional": "#4878d0", "keyload": "#1db377", "payload": "#f59c42",
          "unknown": "#aec6e8", "no_effect": "#cccccc"}
LABELS = {"positional": "positional", "keyload": "lexical", "payload": "reflexive",
          "unknown": "unknown", "no_effect": "no effect"}


def num_layers(model_id):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return cfg.get_text_config().num_hidden_layers


def find_csv(out_dir, model_id, layer, messiness, pos_tag):
    pattern = (f"{model_id.split('/')[-1]}_{NUM_INSTANCES}_{SCAN_SAMPLES}_{layer}_{CAT_TO_QUERY}"
               f"_{SCHEMA}_mess{messiness}_fill0{pos_tag}_*.csv")
    matches = sorted(glob.glob(os.path.join(out_dir, pattern)))
    if not pos_tag:
        # "fill0_*" also matches multi-position "fill0_pos..." files
        matches = [m for m in matches if "_fill0_pos" not in os.path.basename(m)]
    return matches[-1] if matches else None


def run_layer(model_id, layer, out_dir, messiness, token_positions, completion, pos_tag):
    """Run tasks/dist.py at one layer, unless its CSV already exists."""
    csv = find_csv(out_dir, model_id, layer, messiness, pos_tag)
    if csv:
        return csv
    cmd = [
        sys.executable, "-m", "tasks.dist",
        "--model-id", model_id,
        "--schema-name", SCHEMA,
        "--num-instances", str(NUM_INSTANCES),
        "--num-samples", str(SCAN_SAMPLES),
        "--layer", str(layer),
        "--cat-to-query", str(CAT_TO_QUERY),
        "--cat-indices-to-query", str([1 - CAT_TO_QUERY]),
        "--messiness", str(messiness),
        "--out-dir", out_dir,
    ]
    if messiness == 4:
        # the messiness-4 answer never appears in the original prompt
        cmd.append("--open-vocab")
    if token_positions:
        cmd += ["--token-positions", " ".join(str(p) for p in token_positions)]
    if completion:
        cmd.append("--completion")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return find_csv(out_dir, model_id, layer, messiness, pos_tag)


def plot_scan(csvs, model_name, out_path):
    """One stacked bar per layer: the share of each patch effect."""
    layers = sorted(csvs)
    shares = {layer: pd.read_csv(csvs[layer])["patch_effect"].value_counts(normalize=True)
              for layer in layers}
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.7), 4))
    bottom = np.zeros(len(layers))
    for cat in COLORS:
        vals = np.array([shares[layer].get(cat, 0.0) for layer in layers])
        ax.bar(layers, vals, bottom=bottom, label=LABELS[cat], color=COLORS[cat], width=0.8)
        bottom += vals
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction")
    ax.set_title(f"Layer scan: {model_name}")
    ax.set_ylim(0, 1)
    ax.set_xticks(layers)
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--messiness", type=int, default=0, help="0 for the standard scan, 4 for the reflexive scan.")
    p.add_argument("--token-positions", nargs="+", type=int, default=None,
                   help="Positions to patch (default: -1).")
    p.add_argument("--completion", action="store_true", help="Completion prompt format, for base models.")
    args = p.parse_args()

    model_name = args.model_id.split("/")[-1]
    pos_tag = "_pos" + "".join(str(p) for p in args.token_positions) if args.token_positions else ""
    scan_dir = os.path.join(RESULTS_DIR, f"{SCHEMA}_{CAT_TO_QUERY}", model_name,
                            ("layer_scan" if args.messiness == 0 else f"layer_scan_mess{args.messiness}") + pos_tag)
    os.makedirs(scan_dir, exist_ok=True)

    csvs = {}
    for layer in range(num_layers(args.model_id)):
        csvs[layer] = run_layer(args.model_id, layer, scan_dir, args.messiness,
                                args.token_positions, args.completion, pos_tag)
    plot_scan(csvs, model_name, os.path.join(scan_dir, "layer_scan.png"))


if __name__ == "__main__":
    main()
