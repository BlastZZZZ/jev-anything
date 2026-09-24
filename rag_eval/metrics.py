"""Frozen retrieval metrics used by the reported experiment."""


def rank_metrics(ranking, gold, ks):
    """ranking: list of item ids, best first. gold: set of ids."""
    out = {}
    hits = [1 if r in gold else 0 for r in ranking]
    for k in ks:
        top = hits[:k]
        out[f"r@{k}"] = sum(top) / len(gold)
        out[f"all@{k}"] = 1.0 if all(g in ranking[:k] for g in gold) else 0.0
    mrr = 0.0
    for i, h in enumerate(hits):
        if h:
            mrr = 1.0 / (i + 1)
            break
    out["mrr"] = mrr
    return out


def agg(rows):
    keys = [k for k in rows[0] if k != "id"]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys} | {"n": len(rows)}
