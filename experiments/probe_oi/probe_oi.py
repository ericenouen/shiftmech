"""Linear probes for the ordering ID (--target index) or the semantic key (--target object)
at each entity's answer token.

Usage:
    python probe_oi.py --model-id EdenQiao/Rethinking-Hybrid-S5-full-100b --completion --layers 15
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "CausalAbstraction"))
sys.path.insert(0, REPO_ROOT)

from grammar.schemas import SCHEMA_BOXES
from grammar.task_to_causal_model import (
    multi_order_multi_schema_task_to_lookbacks_generic_causal_model,
)
from training import (
    get_counterfactual_datasets,
    ppkn_simpler_counterfactual_template_split_key_loc,
    sample_answerable_question_template,
)
from tasks.dist import (
    _get_layer_module, find_answer_indices, format_prompt, make_completion_prompt,
    to_str_tokens,
)

NUM_INSTANCES = 20


def build_prompts(tokenizer, schema, cq, n_prompts, completion):
    """Yield (prompt, entity token indices, entity objects)."""
    causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
        [schema], NUM_INSTANCES, fillers=False, num_fillers_per_item=0,
    )
    ds, _, _ = get_counterfactual_datasets(
        None, [schema], num_samples=n_prompts * 3, num_instances=NUM_INSTANCES,
        cat_indices_to_query=[i for i in range(len(schema.categories)) if i != cq],
        answer_cat_id=cq, do_assert=True, do_filter=False,
        counterfactual_template=ppkn_simpler_counterfactual_template_split_key_loc,
        causal_models={schema.name: causal_model},
        sample_an_answerable_question=sample_answerable_question_template,
    )
    for sample in ds[schema.name][schema.name]:
        if completion:
            prompt = make_completion_prompt(sample["input"], schema, cq)
        else:
            prompt = format_prompt(tokenizer, sample["input"]["raw_input"])
        toks = to_str_tokens(tokenizer, prompt)
        idx, _, _ = find_answer_indices(toks, schema, cq, sample["input"]["metadata"], NUM_INSTANCES)
        if len(idx) != NUM_INSTANCES:
            continue
        obj = [sample["input"].get(f"Object.{1 - cq}.{k}") for k in range(NUM_INSTANCES)]
        yield prompt, idx, obj


def collect(model, tokenizer, schema, cq, n_prompts, layers, completion, target, vocab):
    """Residual at each entity's answer token for each layer in `layers`, with labels."""
    X = {L: [] for L in layers}
    y = []
    kept = 0
    for prompt, idx, obj in build_prompts(tokenizer, schema, cq, n_prompts, completion):
        if kept >= n_prompts:
            break
        grab = {}
        handles = []
        for L in layers:
            def hook(mod, inp, L=L):
                hs = inp[0]
                if hs.dim() == 3 and hs.size(1) > max(idx):
                    grab[L] = hs[0, idx, :].detach().float().cpu().numpy()
            handles.append(_get_layer_module(model, L).register_forward_pre_hook(hook))
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            # the EdenQiao remote code needs use_cache=False and an explicit mask
            model(ids, attention_mask=torch.ones_like(ids), logits_to_keep=1, use_cache=False)
        for h in handles:
            h.remove()
        if len(grab) != len(layers):
            continue
        for L in layers:
            X[L].append(grab[L])
        if target == "index":
            y.append(np.arange(NUM_INSTANCES))
        else:
            y.append(np.array([vocab.index(o) if o in vocab else -1 for o in obj]))
        kept += 1

    y = np.concatenate(y)
    keep = y >= 0  # drop entities whose object is off-vocabulary
    return {L: np.concatenate(X[L], 0)[keep] for L in layers}, y[keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--cat-to-query", type=int, default=1)
    p.add_argument("--target", choices=["index", "object"], default="index")
    p.add_argument("--completion", action="store_true", help="Completion prompt format, for base models.")
    p.add_argument("--layers", type=int, nargs="+", default=[15])
    p.add_argument("--n-prompts", type=int, default=1000, help="Training prompts per repeat.")
    p.add_argument("--holdout-prompts", type=int, default=1000, help="Test prompts per repeat.")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--out", required=True, help="Output path, without .csv")
    args = p.parse_args()

    schema, cq = SCHEMA_BOXES, args.cat_to_query
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    print(f"[+] Loading {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", trust_remote_code=True, device_map="auto")
    model.eval()

    # schema.items is keyed by category name
    vocab = sorted(schema.items[schema.categories[1 - cq]]) if args.target == "object" else None
    collect_args = (model, tok, schema, cq)
    kw = dict(layers=args.layers, completion=args.completion, target=args.target, vocab=vocab)

    # each repeat draws fresh train and test prompts and refits
    acc = np.zeros((args.repeats, len(args.layers)))
    for r in range(args.repeats):
        Xtr, ytr = collect(*collect_args, args.n_prompts, **kw)
        Xte, yte = collect(*collect_args, args.holdout_prompts, **kw)
        for j, L in enumerate(args.layers):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            acc[r, j] = (clf.fit(Xtr[L], ytr).predict(Xte[L]) == yte).mean()
        print(f"[repeat {r + 1}/{args.repeats}] " + "  ".join(
            f"L{L}={acc[r, j]:.4f}" for j, L in enumerate(args.layers)), flush=True)

    df = pd.DataFrame([dict(layer=L, acc=acc[:, j].mean(), acc_std=acc[:, j].std(ddof=1),
                            **{f"rep{r}": acc[r, j] for r in range(args.repeats)})
                       for j, L in enumerate(args.layers)])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out + ".csv", index=False)
    print(df[["layer", "acc", "acc_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
