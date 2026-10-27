"""Compare two --save-rows files of the same candidate (a run and its repeat): decision agreement and probability drift.

    python tools/interpolation/compare_runs.py rows/x.jsonl rows_rep/x.jsonl
First output line is a one-line verdict.
"""
import json
import sys


def load(path):
    with open(path) as f:
        header = json.loads(f.readline())
        rows = {(r["line"], r["name"]): r for r in map(json.loads, f)}
    return header, rows


def main(a, b):
    _, ra = load(a)
    _, rb = load(b)
    keys = ra.keys() & rb.keys()
    agree, max_dp, sum_dp = 0, 0.0, 0.0
    for k in keys:
        pa, pb = ra[k]["p"], rb[k]["p"]
        agree += max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__)
        d = max(abs(x - y) for x, y in zip(pa, pb))
        max_dp, sum_dp = max(max_dp, d), sum_dp + d
    n = len(keys)
    acc = lambda r: sum(max(range(len(x["p"])), key=x["p"].__getitem__) == x["gold"] for x in r.values()) / len(r)
    verdict = "IDENTICAL" if max_dp == 0 else ("OK" if agree == n and max_dp < 1e-2 else "DIFFERS")
    print(f"{verdict}: {n} decisions in both ({len(ra)} / {len(rb)}), argmax agreement {agree}/{n}, "
          f"max |dp| {max_dp:.2e}, mean |dp| {sum_dp / max(n, 1):.2e}, accuracy {acc(ra):.4f} / {acc(rb):.4f}")
    for k in sorted(keys - {k for k in keys if ra[k]["p"] == rb[k]["p"]})[:20]:
        print("  differs:", k, ra[k]["p"][:4], rb[k]["p"][:4])


if __name__ == "__main__":
    main(*sys.argv[1:3])
