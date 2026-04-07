#!/usr/bin/env python3
import json
import re
import statistics as stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"

# Put your three JSONL outputs here
FILES = {
    "think": "outputs/aime_qwen3_think_template_think_agnos.jsonl",
    "default": "outputs/aime_qwen3_think_template.jsonl",
    "no_think": "outputs/aime_qwen3_think_template_nothink.jsonl",
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def n_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def robust_split(full_text: str) -> Tuple[str, str, Dict[str, bool]]:
    """
    Try several ways to split reasoning vs final answer.

    Returns:
        thinking, answer, flags
    """
    flags = {
        "has_open_think": "<think>" in full_text,
        "has_close_think": "</think>" in full_text,
        "used_regex_split": False,
        "fallback_open_only": False,
        "fallback_no_tags": False,
    }

    # 1) Best case: exact <think>...</think>
    m = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL | re.IGNORECASE)
    if m:
        flags["used_regex_split"] = True
        return m.group(1).strip(), full_text[m.end():].strip(), flags

    # 2) If we have an opening tag but no closing tag, split after the first opening tag
    open_idx = full_text.find("<think>")
    if open_idx != -1:
        flags["fallback_open_only"] = True
        after_open = full_text[open_idx + len("<think>"):].strip()
        # Heuristic: if the model emits a clear answer marker, split there
        answer_markers = [
            r"\n\s*Answer\s*:",
            r"\n\s*Final\s*Answer\s*:",
            r"\n\s*The answer is\s*",
            r"\n\s*\\boxed\{",
        ]
        for pat in answer_markers:
            mm = re.search(pat, after_open, re.IGNORECASE)
            if mm:
                return after_open[:mm.start()].strip(), after_open[mm.start():].strip(), flags

        # Otherwise treat everything after <think> as reasoning
        return after_open, "", flags

    # 3) No tags at all: we cannot know exactly, so keep everything as answer
    flags["fallback_no_tags"] = True
    return "", full_text.strip(), flags


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(values: List[int]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }

    vals = sorted(values)
    n = len(vals)

    def percentile(p: float) -> float:
        if n == 1:
            return float(vals[0])
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return float(vals[f])
        return float(vals[f] * (c - k) + vals[c] * (k - f))

    return {
        "count": n,
        "mean": float(stats.mean(vals)),
        "median": float(stats.median(vals)),
        "stdev": float(stats.pstdev(vals)) if n > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "q25": percentile(0.25),
        "q75": percentile(0.75),
    }


def analyze_file(path: str) -> Dict:
    rows = load_jsonl(path)

    reasoning_tokens = []
    answer_tokens = []
    total_tokens = []

    per_sample = []
    empty_think_count = 0
    missing_close_count = 0
    missing_any_tag_count = 0

    for row in rows:
        samples = row.get("samples", [])
        for s in samples:
            raw_full = s.get("full_output", "") or ""
            thinking = s.get("thinking", "")
            answer_text = s.get("answer_text", "")

            # Prefer stored split fields if they look non-empty; otherwise recover from raw text
            if not thinking and not answer_text and raw_full:
                thinking, answer_text, flags = robust_split(raw_full)
            else:
                # Still inspect the raw text for split failures
                _, _, flags = robust_split(raw_full if raw_full else (thinking + "\n" + answer_text))

            rt = n_tokens(thinking)
            at = n_tokens(answer_text)
            tt = rt + at

            if rt == 0:
                empty_think_count += 1
            if not flags["has_close_think"]:
                missing_close_count += 1
            if flags["fallback_no_tags"]:
                missing_any_tag_count += 1

            reasoning_tokens.append(rt)
            answer_tokens.append(at)
            total_tokens.append(tt)

            per_sample.append({
                "reasoning_tokens": rt,
                "answer_tokens": at,
                "total_tokens": tt,
                "has_open_think": flags["has_open_think"],
                "has_close_think": flags["has_close_think"],
                "used_regex_split": flags["used_regex_split"],
                "fallback_open_only": flags["fallback_open_only"],
                "fallback_no_tags": flags["fallback_no_tags"],
            })

    # Per-question averages, so long questions with many samples do not dominate everything
    question_level = []
    for row in rows:
        samples = row.get("samples", [])
        q_reason = []
        q_ans = []
        q_total = []
        for s in samples:
            thinking = s.get("thinking", "")
            answer_text = s.get("answer_text", "")
            q_reason.append(n_tokens(thinking))
            q_ans.append(n_tokens(answer_text))
            q_total.append(n_tokens(thinking) + n_tokens(answer_text))

        if q_reason:
            question_level.append({
                "mean_reasoning": stats.mean(q_reason),
                "mean_answer": stats.mean(q_ans),
                "mean_total": stats.mean(q_total),
            })

    q_reason_vals = [x["mean_reasoning"] for x in question_level]
    q_ans_vals = [x["mean_answer"] for x in question_level]
    q_total_vals = [x["mean_total"] for x in question_level]

    return {
        "file": path,
        "num_questions": len(rows),
        "num_samples": len(per_sample),
        "sample_level": {
            "reasoning": summarize(reasoning_tokens),
            "answer": summarize(answer_tokens),
            "total": summarize(total_tokens),
            "empty_think_count": empty_think_count,
            "missing_close_count": missing_close_count,
            "missing_any_tag_count": missing_any_tag_count,
        },
        "question_level": {
            "reasoning": summarize(q_reason_vals),
            "answer": summarize(q_ans_vals),
            "total": summarize(q_total_vals),
        },
    }


def print_report(report: Dict) -> None:
    print("\n" + "=" * 80)
    print(f"FILE: {report['file']}")
    print(f"Questions: {report['num_questions']}")
    print(f"Samples:   {report['num_samples']}")

    s = report["sample_level"]
    q = report["question_level"]

    print("\n--- SAMPLE-LEVEL TOKENS ---")
    for name in ["reasoning", "answer", "total"]:
        x = s[name]
        print(
            f"{name:10s}  mean={x['mean']:.2f}  median={x['median']:.2f}  "
            f"std={x['stdev']:.2f}  min={x['min']:.0f}  q25={x['q25']:.0f}  "
            f"q75={x['q75']:.0f}  max={x['max']:.0f}"
        )

    print("\n--- QUESTION-LEVEL MEANS ---")
    for name in ["reasoning", "answer", "total"]:
        x = q[name]
        print(
            f"{name:10s}  mean={x['mean']:.2f}  median={x['median']:.2f}  "
            f"std={x['stdev']:.2f}  min={x['min']:.2f}  q25={x['q25']:.2f}  "
            f"q75={x['q75']:.2f}  max={x['max']:.2f}"
        )

    print("\n--- SPLIT HEALTH ---")
    print(f"Empty thinking count:        {s['empty_think_count']}")
    print(f"Missing closing </think>:    {s['missing_close_count']}")
    print(f"Missing any think tags:      {s['missing_any_tag_count']}")


def main():
    all_reports = []
    for mode, path in FILES.items():
        if not Path(path).exists():
            print(f"Skipping missing file: {path}")
            continue

        report = analyze_file(path)
        all_reports.append((mode, report))
        print_report(report)

    print("\n" + "=" * 80)
    print("MODE COMPARISON")
    print("=" * 80)
    for mode, report in all_reports:
        s = report["sample_level"]
        q = report["question_level"]
        print(
            f"{mode:10s} | "
            f"sample_mean_total={s['total']['mean']:.2f} | "
            f"sample_mean_reasoning={s['reasoning']['mean']:.2f} | "
            f"sample_mean_answer={s['answer']['mean']:.2f} | "
            f"question_mean_total={q['total']['mean']:.2f}"
        )


if __name__ == "__main__":
    main()