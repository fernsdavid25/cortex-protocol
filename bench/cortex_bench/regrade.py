"""Re-grade committed hypotheses under a DIFFERENT judge, without re-running the reader.

Why this exists
---------------
The published headline was produced with reader ``gemini-3.5-flash`` AND judge
``gemini-3.5-flash`` — the same model family on both sides. That is a self-preference
risk: a judge may favour text produced by its own family. The honest fix is to re-score
the *already committed* answers with a cross-family judge and publish BOTH columns.

This is a RE-GRADE, not a re-run: the reader never executes again, so the only cost is
the judge call. Hypotheses live in ``*_hypotheses.jsonl`` next to every results file.

Usage
-----
Free plumbing check (no API calls at all)::

    python -m cortex_bench.regrade --hypotheses <path> --variant s --judge offline --sample 20

Cost preview — builds the real prompts, counts tokens, calls nothing::

    python -m cortex_bench.regrade --hypotheses <path> --variant s --judge openai --dry-run

The real thing (needs OPENAI_API_KEY)::

    python -m cortex_bench.regrade --hypotheses <path> --variant s \
        --judge openai --judge-model gpt-4o-2024-08-06 --sample 300 --seed 0 \
        --out bench/results/regrade_gpt4o

Every judged item reuses the canonical per-question-type LongMemEval prompt via
``judge.get_anscheck_prompt``, so the re-grade is directly comparable to the upstream
methodology and to competitors who publish under a gpt-4o judge.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .dataset import download, load_instances
from .judge import Judge, get_anscheck_prompt
from .memory_system import QAInstance

# Judge replies are capped at 10 tokens ("yes"/"no"), so output cost is a rounding error.
_JUDGE_MAX_OUTPUT_TOKENS = 10

# USD per 1M tokens. Standard (non-batch) tier. Update when a vendor changes list price.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def _approx_tokens(text: str) -> int:
    """Rough token count (~4 chars/token). Only used for the cost PREVIEW, never for billing."""
    return max(1, len(text) // 4)


def load_hypotheses(path: Path) -> dict[str, dict]:
    """Read ``{question_id: row}`` from a hypotheses jsonl, skipping malformed lines."""
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id")
            if qid:
                rows[qid] = row
    return rows


def _category(instance: QAInstance) -> str:
    """Report bucket: abstention is overlaid on the 6 base types, so it gets its own row."""
    return "abstention" if instance.is_abstention else instance.question_type


def _cohens_kappa(both_yes: int, both_no: int, a_only: int, b_only: int) -> float:
    """Cohen's kappa for two binary raters over the same items. 1.0 = perfect agreement."""
    n = both_yes + both_no + a_only + b_only
    if n == 0:
        return 0.0
    p_observed = (both_yes + both_no) / n
    p_a_yes = (both_yes + a_only) / n
    p_b_yes = (both_yes + b_only) / n
    p_expected = p_a_yes * p_b_yes + (1 - p_a_yes) * (1 - p_b_yes)
    if p_expected >= 1.0:
        return 1.0
    return (p_observed - p_expected) / (1 - p_expected)


def main(argv: list[str] | None = None) -> dict:
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=True)

    ap = argparse.ArgumentParser(description="Re-grade committed hypotheses under another judge.")
    ap.add_argument("--hypotheses", required=True, help="path to a *_hypotheses.jsonl")
    ap.add_argument("--benchmark", default="longmemeval", choices=["longmemeval", "locomo"])
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--data", help="explicit dataset path (overrides --variant)")
    ap.add_argument("--data-dir", default="bench/data")
    ap.add_argument("--judge", default="openai", choices=["openai", "gemini", "offline"])
    ap.add_argument("--judge-model", default=None, help="defaults per backend (openai: gpt-4o-2024-08-06)")
    ap.add_argument("--judge-votes", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0, help="judge a random N-item subset (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the deterministic sample")
    ap.add_argument("--dry-run", action="store_true", help="build prompts + estimate cost, call nothing")
    ap.add_argument("--out", default="", help="directory to write the regraded jsonl + summary")
    args = ap.parse_args(argv)

    # --- load the answers we already paid for -------------------------------------------------
    hyp_path = Path(args.hypotheses)
    if not hyp_path.exists():
        raise SystemExit(f"hypotheses file not found: {hyp_path}")
    hypotheses = load_hypotheses(hyp_path)

    # --- load the questions/reference answers -------------------------------------------------
    if args.benchmark == "locomo":
        from .locomo import load_locomo

        instances = load_locomo(args.data or f"{args.data_dir}/locomo10.json")
    else:
        instances = load_instances(args.data or download(args.variant, args.data_dir))
    by_id = {inst.question_id: inst for inst in instances}

    pairs = [(by_id[qid], row) for qid, row in hypotheses.items() if qid in by_id]
    missing = len(hypotheses) - len(pairs)
    if not pairs:
        raise SystemExit("no hypothesis rows matched the loaded dataset — wrong --variant?")

    pairs.sort(key=lambda p: p[0].question_id)  # deterministic order before sampling
    if args.sample and args.sample < len(pairs):
        random.Random(args.seed).shuffle(pairs)
        pairs = pairs[: args.sample]
        pairs.sort(key=lambda p: p[0].question_id)

    print(f"hypotheses: {len(hypotheses)}  matched: {len(pairs)}  unmatched: {missing}")

    # --- cost preview -------------------------------------------------------------------------
    prompts = [
        get_anscheck_prompt(
            inst.question_type, inst.question, inst.answer, row.get("hypothesis", ""),
            abstention=inst.is_abstention,
        )
        for inst, row in pairs
    ]
    in_tokens = sum(_approx_tokens(p) for p in prompts)
    out_tokens = len(prompts) * _JUDGE_MAX_OUTPUT_TOKENS
    model_name = args.judge_model or ("gpt-4o-2024-08-06" if args.judge == "openai" else "")
    price = _PRICES.get(model_name)
    print(f"\nprompts built: {len(prompts)}")
    print(f"est. input tokens:  {in_tokens:,}  (~4 chars/token heuristic)")
    print(f"est. output tokens: {out_tokens:,}  (max_tokens={_JUDGE_MAX_OUTPUT_TOKENS})")
    if price:
        cost = in_tokens / 1e6 * price[0] + out_tokens / 1e6 * price[1]
        print(f"est. cost on {model_name}: ${cost:.2f}  (batch API would halve this)")

    if args.dry_run:
        print("\n--- sample prompt (first item) ---")
        print(prompts[0][:1200])
        print("\n[dry-run] no API calls made.")
        return {"dry_run": True, "items": len(prompts), "est_input_tokens": in_tokens}

    # --- re-grade ------------------------------------------------------------------------------
    judge = Judge(backend=args.judge, model=args.judge_model, votes=args.judge_votes)
    both_yes = both_no = old_only = new_only = 0
    per_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "old": 0, "new": 0})
    out_rows: list[dict] = []

    for i, (inst, row) in enumerate(pairs, 1):
        old = bool(row.get("correct", False))
        new = judge.grade(inst, row.get("hypothesis", ""))
        cat = _category(inst)
        per_cat[cat]["n"] += 1
        per_cat[cat]["old"] += int(old)
        per_cat[cat]["new"] += int(new)
        if old and new:
            both_yes += 1
        elif not old and not new:
            both_no += 1
        elif old:
            old_only += 1
        else:
            new_only += 1
        out_rows.append(
            {"question_id": inst.question_id, "category": cat, "old_correct": old, "new_correct": new}
        )
        if i % 25 == 0 or i == len(pairs):
            print(f"  judged {i}/{len(pairs)}", flush=True)

    n = len(pairs)
    old_acc = (both_yes + old_only) / n
    new_acc = (both_yes + new_only) / n
    agreement = (both_yes + both_no) / n
    kappa = _cohens_kappa(both_yes, both_no, old_only, new_only)

    # ASCII only: this prints to a Windows cp1252 console in the maintainer's environment.
    print(f"\n=== re-grade: {args.benchmark} | judge {judge.backend}/{judge.model} | n={n} ===")
    print(f"old (committed) accuracy : {old_acc:.4f}")
    print(f"new (re-graded) accuracy : {new_acc:.4f}   delta {new_acc - old_acc:+.4f}")
    print(f"raw agreement            : {agreement:.4f}   Cohen's kappa {kappa:.4f}")
    print(f"flips: old-correct -> new-wrong {old_only}   old-wrong -> new-correct {new_only}")
    print(f"\n{'category':<30}{'n':>5}{'old':>9}{'new':>9}{'delta':>9}")
    for cat in sorted(per_cat):
        c = per_cat[cat]
        o, nw = c["old"] / c["n"], c["new"] / c["n"]
        print(f"{cat:<30}{c['n']:>5}{o:>9.3f}{nw:>9.3f}{nw - o:>+9.3f}")

    summary = {
        "benchmark": args.benchmark,
        "judge_backend": judge.backend,
        "judge_model": judge.model,
        "n": n,
        "sampled": bool(args.sample),
        "seed": args.seed,
        "old_accuracy": round(old_acc, 4),
        "new_accuracy": round(new_acc, 4),
        "delta": round(new_acc - old_acc, 4),
        "raw_agreement": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "flips_old_correct_new_wrong": old_only,
        "flips_old_wrong_new_correct": new_only,
        "per_category": {
            c: {"n": v["n"], "old": round(v["old"] / v["n"], 4), "new": round(v["new"] / v["n"], 4)}
            for c, v in sorted(per_cat.items())
        },
    }

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        tag = f"regrade_{judge.backend}"
        with (outdir / f"{tag}_rows.jsonl").open("w", encoding="utf-8") as fh:
            for row in out_rows:
                fh.write(json.dumps(row) + "\n")
        (outdir / f"{tag}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {outdir / (tag + '_rows.jsonl')} and {outdir / (tag + '_summary.json')}")

    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
