#!/usr/bin/env python3
"""Check finished runs and test blends from the probability files the notebook saves.

It needs a runs.json that maps a run id to its probability file (see runs.example.json). Every
probability file must come from the same folds, so rows line up, and the notebook already
guarantees that as long as SEED is never changed.

Commands, run from the project folder with a Python that has numpy, pandas and scikit-learn.
  python blend_check.py runs.json table
  python blend_check.py runs.json check 55 --base "30,31,32" --ref submission-best.csv
  python blend_check.py runs.json blend "30,31,32,43:2" --base "30,31,32" --ref submission-best.csv --write out.csv

A blend is a comma separated list of run ids, each with an optional weight after a colon, for
example 43:2 gives run 43 twice the weight. All scores are macro F1 on the out-of-fold flip
probabilities. The bootstrap interval compares a blend against the --base blend.
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


def load_probs(path):
    path = str(path)
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        return {n[:-4]: np.load(io.BytesIO(z.read(n)), allow_pickle=True) for n in z.namelist() if n.endswith(".npy")}
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_runs(runs_json):
    spec = json.loads(Path(runs_json).read_text())["runs"]
    root = Path(runs_json).resolve().parent
    runs, csvs = {}, {}
    for rid, entry in spec.items():
        probs = entry if isinstance(entry, str) else entry["probs"]
        runs[rid] = load_probs(root / probs)
        if isinstance(entry, dict) and entry.get("csv"):
            csvs[rid] = root / entry["csv"]
    return runs, csvs


def tail(r):
    return ["/".join(str(x).split("/")[-2:]) for x in r["oof_filepaths"]]


def align_rows(runs):
    """Reorder the OOF rows of runs that list the same files in another order (a different machine can list a folder differently)."""
    first = next(iter(runs.values()))
    order = tail(first)
    for rid, r in runs.items():
        t = tail(r)
        if t == order or sorted(t) != sorted(order):
            continue
        pos = {x: i for i, x in enumerate(t)}
        idx = np.array([pos[x] for x in order])
        for key in ("oof_filepaths", "oof_targets", "oof_probs_flip", "oof_probs_tta"):
            r[key] = r[key][idx]
        print(f"run {rid} listed the training files in another order, rows re-aligned by file name")


def check_alignment(runs):
    first = next(iter(runs.values()))
    t0 = tail(first)
    ok_rows = all(tail(r) == t0 for r in runs.values())
    ok_test = all((r["test_filenames"] == first["test_filenames"]).all() for r in runs.values())
    ok_targets = all((r["oof_targets"] == first["oof_targets"]).all() for r in runs.values())
    print(f"alignment, rows {ok_rows}, test order {ok_test}, targets {ok_targets}")
    if not (ok_rows and ok_test and ok_targets):
        sys.exit("the runs are not aligned, do not blend them")


def parse_blend(text):
    weights = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        rid, _, w = part.partition(":")
        weights[rid.strip()] = float(w) if w else 1.0
    return weights


def mix(weights, runs, key):
    total = sum(weights.values())
    return sum(runs[rid][key] * w for rid, w in weights.items()) / total


def macro_f1(y, probs):
    return f1_score(y, probs.argmax(1), average="macro")


def bootstrap(y, a, b, reps=1000, seed=0):
    pa, pb = a.argmax(1), b.argmax(1)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro"))
    return float(np.mean(diffs)), np.percentile(diffs, [2.5, 97.5])


def labels_of(probs, classes):
    return np.array(classes)[probs.argmax(1)]


def read_ref(path, names):
    if not path:
        return None
    ref = pd.read_csv(path)
    if not (ref.iloc[:, 0].to_numpy() == names).all():
        sys.exit(f"{path} does not list the test files in the same order as the probability files")
    return ref.iloc[:, 1].to_numpy()


def describe_test(probs, classes, ref_labels):
    pred = probs.argmax(1)
    counts = dict(zip(classes, np.bincount(pred, minlength=len(classes)).tolist()))
    line = f"  test class counts {counts}"
    if ref_labels is not None:
        lab = np.array(classes)[pred]
        line += f", differs from the reference file on {int((lab != ref_labels).sum())} labels"
    print(line)


def cmd_table(runs, classes, y):
    rows = []
    for rid, r in runs.items():
        rows.append((rid, macro_f1(y, r["oof_probs_flip"]), macro_f1(y, r["oof_probs_tta"])))
    print(f"{'run':>8s} {'OOF flip':>9s} {'OOF no flip':>12s}")
    for rid, f_flip, f_tta in sorted(rows, key=lambda r: -r[1]):
        print(f"{rid:>8s} {f_flip:9.4f} {f_tta:12.4f}")


def cmd_check(args, runs, csvs, classes, y, names):
    rid = args.run
    r = runs[rid]
    pred = r["oof_probs_flip"].argmax(1)
    per_class = f1_score(y, pred, average=None)
    recalls = recall_score(y, pred, average=None)
    print(f"run {rid}")
    print(f"  OOF macro F1 {macro_f1(y, r['oof_probs_flip']):.4f} (no flip {macro_f1(y, r['oof_probs_tta']):.4f})")
    for c, f, rc in zip(classes, per_class, recalls):
        print(f"  class {c}, F1 {f:.4f}, recall {rc:.4f}")
    ref = read_ref(args.ref, names)
    describe_test(r["test_probs"], classes, ref)
    if rid in csvs:
        csv = pd.read_csv(csvs[rid])
        same = (csv.iloc[:, 0].to_numpy() == names).all() and (csv.iloc[:, 1].to_numpy() == labels_of(r["test_probs"], classes)).all()
        print(f"  submission csv matches the probability file, {bool(same)}")
    if args.vs and args.vs in runs:
        wrong_a = pred != y
        wrong_b = runs[args.vs]["oof_probs_flip"].argmax(1) != y
        share = (wrong_a & wrong_b).sum() / max(1, wrong_a.sum())
        print(f"  share of its OOF errors that run {args.vs} also makes, {share:.3f}")
    if args.base:
        base = parse_blend(args.base)
        both = dict(base)
        both[rid] = both.get(rid, 0) + 1.0
        fb, fn = macro_f1(y, mix(base, runs, "oof_probs_flip")), macro_f1(y, mix(both, runs, "oof_probs_flip"))
        _, ci = bootstrap(y, mix(both, runs, "oof_probs_flip"), mix(base, runs, "oof_probs_flip"))
        print(f"  base blend OOF {fb:.4f}, with this run {fn:.4f} ({fn - fb:+.4f}), bootstrap CI {ci[0]:+.4f} to {ci[1]:+.4f}")
        describe_test(mix(both, runs, "test_probs"), classes, ref)


def cmd_blend(args, runs, classes, y, names):
    weights = parse_blend(args.spec)
    missing = [k for k in weights if k not in runs]
    if missing:
        sys.exit(f"unknown run ids {missing}")
    oof, test = mix(weights, runs, "oof_probs_flip"), mix(weights, runs, "test_probs")
    print(f"blend {args.spec}")
    print(f"  OOF macro F1 {macro_f1(y, oof):.4f}, accuracy {float((oof.argmax(1) == y).mean()):.4f}")
    if args.base:
        base = parse_blend(args.base)
        _, ci = bootstrap(y, oof, mix(base, runs, "oof_probs_flip"))
        fb = macro_f1(y, mix(base, runs, "oof_probs_flip"))
        print(f"  base blend OOF {fb:.4f}, difference {macro_f1(y, oof) - fb:+.4f}, bootstrap CI {ci[0]:+.4f} to {ci[1]:+.4f}")
    describe_test(test, classes, read_ref(args.ref, names))
    if args.write:
        pd.DataFrame({"filename": names, "label": labels_of(test, classes)}).to_csv(args.write, index=False)
        print(f"  wrote {args.write}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    c = sub.add_parser("check")
    c.add_argument("run")
    c.add_argument("--base")
    c.add_argument("--ref")
    c.add_argument("--vs", help="run id to compare OOF errors against")
    b = sub.add_parser("blend")
    b.add_argument("spec")
    b.add_argument("--base")
    b.add_argument("--ref")
    b.add_argument("--write")
    args = ap.parse_args()

    runs, csvs = load_runs(args.runs_json)
    align_rows(runs)
    check_alignment(runs)
    first = next(iter(runs.values()))
    classes = [str(x) for x in first["classes"]]
    y, names = first["oof_targets"], first["test_filenames"]
    if args.cmd == "table":
        cmd_table(runs, classes, y)
    elif args.cmd == "check":
        cmd_check(args, runs, csvs, classes, y, names)
    else:
        cmd_blend(args, runs, classes, y, names)


if __name__ == "__main__":
    main()
