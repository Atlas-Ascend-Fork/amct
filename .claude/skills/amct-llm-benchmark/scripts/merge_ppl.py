"""Merge per-chunk NLL dumps from multiple ppl_eval.py runs into a single PPL.

Use when a full PPL run can't fit under a single Bash invocation's 600s timeout
(typical when subagent context). Run ppl_eval.py twice (or more) with
--start_chunk / --max_chunks / --nll_dump, then merge:

    python merge_ppl.py part1.json part2.json [part3.json ...]

Validates that the partitions tile the full chunk set without gaps/overlap.
"""
import argparse
import json
import math


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parts", nargs="+", help="NLL dump JSON files (from ppl_eval --nll_dump)")
    args = p.parse_args()

    chunks = []  # (start, end, seqlen, total, nlls)
    for path in args.parts:
        with open(path) as f:
            d = json.load(f)
        chunks.append((d["start_chunk"], d["end_chunk"], d["seqlen"],
                       d["total_chunks"], d["nlls"]))

    chunks.sort()
    seqlen = chunks[0][2]
    total = chunks[0][3]
    for s, e, sl, t, _ in chunks:
        assert sl == seqlen and t == total, f"seqlen/total mismatch in {chunks}"

    covered = []
    for s, e, _, _, n in chunks:
        covered.append((s, e, len(n)))
    print(f"[merge] partitions: {covered}, total_chunks={total}")

    # Check coverage
    expected = 0
    for s, e, n in covered:
        assert s == expected, f"Gap before chunk {s}; expected start {expected}"
        assert e - s == n, f"Chunk count mismatch in [{s}, {e})"
        expected = e

    all_nlls = [n for _, _, _, _, ns in chunks for n in ns]
    nsamples = len(all_nlls)

    nll_sum = sum(all_nlls)
    ppl = math.exp(nll_sum / (nsamples * seqlen))
    print(f"\n[merge] merged PPL = {ppl:.4f}  ({nsamples}/{total} chunks "
          f"{'(FULL)' if nsamples == total else f'(PARTIAL, {nsamples}/{total})'})")


if __name__ == "__main__":
    main()
