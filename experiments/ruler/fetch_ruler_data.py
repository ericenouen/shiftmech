"""Download hotpotqa.json for RULER's qa_2 task (upstream's URL is dead).
"""
import json, os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "RULER", "scripts", "data", "synthetic", "json", "hotpotqa.json")

def main():
    if os.path.exists(OUT) and os.path.getsize(OUT) > 1_000_000:
        print(f"[skip] {OUT} already present ({os.path.getsize(OUT)/1e6:.0f} MB)")
        return 0
    from datasets import load_dataset
    print("[+] downloading hotpot_qa (distractor, validation) from HuggingFace ...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)
    out = []
    for r in ds:
        ctx = r["context"]
        out.append({
            "_id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            # qa.py does ''.join(p) over the paragraph list, so keep sentences as a list
            "context": [[t, s] for t, s in zip(ctx["title"], ctx["sentences"])],
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"[+] wrote {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB, {len(out)} questions)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
