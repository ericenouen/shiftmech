"""Run the mechanism analysis for one model on every task at a fixed layer.

Usage:
    python run.py --model-id google/gemma-2-2b-it --layer 18 --token-positions -1 -4 -6 -8
"""

import argparse
import glob
import os
import sys

from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "CausalAbstraction"))
sys.path.insert(0, REPO_ROOT)

from grammar.schemas import (
    SCHEMA_FILLING_LIQUIDS,
    SCHEMA_MUSIC_PERFORMANCE,
    SCHEMA_PEOPLE_AND_OBJECTS,
    SCHEMA_PROGRAMMING_PEOPLE_DICT,
    SCHEMA_LAB_EXPERIMENTS,
    SCHEMA_CHEMISTRY_EXPERIMENTS,
    SCHEMA_TRANSPORTATION,
    SCHEMA_SPORTS_EVENTS,
    SCHEMA_SPACE_OBSERVATIONS,
    SCHEMA_BOXES,
)
from grammar.task_to_causal_model import (
    multi_order_multi_schema_task_to_lookbacks_generic_causal_model,
)
from training import (
    get_counterfactual_datasets,
    ppkn_simpler_counterfactual_template_split_key_loc,
    sample_answerable_question_template,
)
from tasks.dist import get_dist

ALL_SCHEMAS = [
    SCHEMA_FILLING_LIQUIDS,
    SCHEMA_MUSIC_PERFORMANCE,
    SCHEMA_PEOPLE_AND_OBJECTS,
    SCHEMA_PROGRAMMING_PEOPLE_DICT,
    SCHEMA_LAB_EXPERIMENTS,
    SCHEMA_CHEMISTRY_EXPERIMENTS,
    SCHEMA_TRANSPORTATION,
    SCHEMA_SPORTS_EVENTS,
    SCHEMA_SPACE_OBSERVATIONS,
    SCHEMA_BOXES,
]

NUM_INSTANCES = 20
MESSINESS = 0

# these reject flash-attention 2 (gemma-4: head_dim 512)
FORCE_EAGER = {"google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "google/gemma-4-31B-it",
               "google/gemma-4-12B-it", "meta-llama/Llama-3.1-70B-Instruct"}
# gemma-2 silently falls back to eager under flash-attention 2
FORCE_SDPA = {"google/gemma-2-2b-it", "google/gemma-2-9b-it", "google/gemma-2-27b-it"}


def out_name(model_id, layer, num_samples, cat_to_query, schema_name, tag, date):
    tag_part = f"_{tag}" if tag else ""
    return (f"{model_id.split('/')[-1]}_{NUM_INSTANCES}_{num_samples}_{layer}_{cat_to_query}"
            f"_{schema_name}_mess{MESSINESS}_fill0{tag_part}_{date}.csv")


def run_battery(model_id, layer, completion, num_samples, cat_to_query_list, out_dir,
                schemas, tag, token_positions):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[+] Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    attn = ("eager" if model_id in FORCE_EAGER else
            "sdpa" if model_id in FORCE_SDPA else "flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", trust_remote_code=True,
        attn_implementation=attn, device_map="auto",
    )
    model.eval()

    combos = [(schema, cq) for schema in schemas for cq in range(len(schema.categories))
              if cat_to_query_list is None or cq in cat_to_query_list]
    for done, (schema, cat_to_query) in enumerate(combos, 1):
        # skip tasks already on disk (date-only glob, so pos-tagged files don't match untagged runs)
        pattern = out_name(model_id, layer, num_samples, cat_to_query, schema.name, tag, "[0-9]*")
        if glob.glob(os.path.join(out_dir, pattern)):
            print(f"[{done}/{len(combos)}] skip  {schema.name} cq={cat_to_query}")
            continue
        print(f"[{done}/{len(combos)}] run   {schema.name} cq={cat_to_query}")

        causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
            [schema], NUM_INSTANCES, fillers=False, num_fillers_per_item=0,
        )
        train_ds, _, _ = get_counterfactual_datasets(
            None,
            [schema],
            num_samples=num_samples,
            num_instances=NUM_INSTANCES,
            cat_indices_to_query=[i for i in range(len(schema.categories)) if i != cat_to_query],
            answer_cat_id=cat_to_query,
            do_assert=True,
            do_filter=False,
            counterfactual_template=ppkn_simpler_counterfactual_template_split_key_loc,
            causal_models={schema.name: causal_model},
            sample_an_answerable_question=sample_answerable_question_template,
        )
        out_path = os.path.join(out_dir, out_name(model_id, layer, num_samples, cat_to_query, schema.name,
                                                  tag, datetime.now().strftime("%Y%m%d")))
        get_dist(
            model, tokenizer, model_id, train_ds, schema,
            NUM_INSTANCES, num_samples, layer, cat_to_query,
            MESSINESS, generate=False, completion=completion, out_path=out_path,
            token_positions=token_positions,
        )

    print(f"\n[done] results in {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Run the mechanism analysis on every task at a fixed layer.")
    p.add_argument("--model-id", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--completion", action="store_true", help="Completion prompt format, for base models.")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--cat-to-query", type=int, default=None, help="Query category (default: all).")
    p.add_argument("--schemas", nargs="+", default=None, help="Schema names (default: all).")
    p.add_argument("--out-dir", default=None, help="Default: results/<model>/ next to this script.")
    p.add_argument("--tag", default="", help="Suffix for output filenames.")
    p.add_argument("--token-positions", nargs="+", type=int, default=None,
                   help="Token positions to patch (default: -1).")
    args = p.parse_args()

    if args.token_positions is not None and args.token_positions != [-1]:
        pos_tag = "pos" + "".join(str(p) for p in args.token_positions)
        args.tag = f"{args.tag}_{pos_tag}" if args.tag else pos_tag

    schemas = ALL_SCHEMAS if args.schemas is None else [s for s in ALL_SCHEMAS if s.name in args.schemas]
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(__file__), "results", args.model_id.replace("/", "__"))

    run_battery(args.model_id, args.layer, args.completion, args.num_samples,
                None if args.cat_to_query is None else [args.cat_to_query], args.out_dir,
                schemas, args.tag, args.token_positions)


if __name__ == "__main__":
    main()
