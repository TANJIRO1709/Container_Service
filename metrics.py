"""Exact replica of the challenge metric: macro-averaged per-entity F0.5.

Per entity with truth G and prediction P:
  G empty, P empty      -> 1.0
  exactly one empty     -> 0.0
  otherwise             -> 1.25*TP / (|P| + 0.25*|G|)   (0 if TP == 0)

Usage:
  python metrics.py --gt dataset/train/train_ground_truth.tsv --pred my_val_preds.tsv
  (optional) --ids val_ids.txt   restrict scoring to a validation subset of S1 ids
"""
import argparse
from collections import defaultdict


def load_lists(path: str) -> dict:
    """Read a 2-column TSV (s1_id \t comma-list) into {s1_id: set(ids)}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            sid, _, rest = line.partition("\t")
            out[sid] = {x.strip() for x in rest.split(",") if x.strip()}
    return out


def f05_entity(g: set, p: set) -> float:
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    return 0.0 if tp == 0 else 1.25 * tp / (len(p) + 0.25 * len(g))


def macro_f05(gt: dict, pred: dict, ids=None, breakdown_key=None) -> dict:
    """ids: iterable of S1 ids to score (default: all in gt).
    Missing predictions count as empty. breakdown_key: optional fn(s1_id)->group."""
    ids = list(gt.keys()) if ids is None else list(ids)
    total = 0.0
    groups = defaultdict(lambda: [0.0, 0])
    single_n = single_ok = 0
    for sid in ids:
        g = gt.get(sid, set())
        p = pred.get(sid, set())
        s = f05_entity(g, p)
        total += s
        if not g:
            single_n += 1
            single_ok += s == 1.0
        if breakdown_key:
            k = breakdown_key(sid)
            groups[k][0] += s
            groups[k][1] += 1
    res = {
        "macro_f05": total / max(len(ids), 1),
        "n": len(ids),
        "singleton_n": single_n,
        "singleton_acc": single_ok / max(single_n, 1),
    }
    if breakdown_key:
        res["by_group"] = {k: v[0] / v[1] for k, v in groups.items()}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ids", help="file with one S1 id per line to restrict scoring")
    a = ap.parse_args()
    gt, pred = load_lists(a.gt), load_lists(a.pred)
    ids = None
    if a.ids:
        with open(a.ids) as f:
            ids = [l.strip() for l in f if l.strip()]
    r = macro_f05(gt, pred, ids)
    for k, v in r.items():
        print(f"{k:>15}: {v:.5f}" if isinstance(v, float) else f"{k:>15}: {v}")
