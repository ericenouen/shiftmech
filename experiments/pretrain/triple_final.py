#!/usr/bin/env python
"""Table 1: mechanism allocation for the EdenQiao PE triple.

Reads the boxes cq=1 runs written by run_mech_allocation.sh. Shares are percentages
of legible samples (no_effect excluded); r_pos = P / (L + R) is a ratio of raw counts,
with SE(r) = r * sqrt(1/P + 1/(L+R)).

Sampling is unseeded, so a rerun agrees with the paper within a couple of SEs.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# (context, label, model directory suffix), in table order
ARMS = [
    ("16K", "RoPE", "full-100b"),
    ("16K", "SWA RoPE", "SWA-128-100b"),
    ("16K", "SWA NoPE", "SWA-128-Nope-100b"),
    ("32K", "RoPE", "full-100b-32k-5b"),
    ("32K", "SWA RoPE", "SWA-128-100b-32k-5b"),
    ("32K", "SWA NoPE", "SWA-128-Nope-100b-32k-5b"),
]


def shares(patch_effect):
    """Percentages of legible mass, plus r_pos and its standard error."""
    legible = (patch_effect != "no_effect").sum()
    n = {k: (patch_effect == k).sum() for k in ("positional", "keyload", "payload", "unknown")}
    P, D = n["positional"], n["keyload"] + n["payload"]
    r = P / D
    return {
        "positional": P / legible * 100,
        "lexical": n["keyload"] / legible * 100,
        "reflexive": n["payload"] / legible * 100,
        "unknown": n["unknown"] / legible * 100,
        "r": r,
        "se": r * np.sqrt(1 / P + 1 / D),
        "n": legible,
    }


rows = []
for context, label, suffix in ARMS:
    hits = glob.glob(os.path.join(RES, f"EdenQiao__Rethinking-Hybrid-S5-{suffix}", "*clozescan10k*.csv"))
    if len(hits) != 1:
        sys.exit(f"expected one clozescan10k CSV for {context} {label}, found {len(hits)}")
    patch_effect = pd.read_csv(hits[0], usecols=["patch_effect"])["patch_effect"].to_numpy()
    rows.append((context, label, shares(patch_effect)))

print(f"{'ctx':<5}{'model':<10}{'P':>7}{'L':>7}{'R':>7}{'Unk':>7}{'r_pos':>8}{'SE':>8}{'N':>8}")
print("-" * 67)
last = None
for context, label, s in rows:
    if last is not None and context != last:
        print()
    last = context
    print(f"{context:<5}{label:<10}{s['positional']:7.1f}{s['lexical']:7.1f}{s['reflexive']:7.1f}"
          f"{s['unknown']:7.1f}{s['r']:8.3f}{s['se']:8.4f}{s['n']:8d}")
