"""Combine a retriever's ordering with a calibrated decision model's ordering."""

import math


def fuse_rankings(dense_docids, relevance, *, reranker_weight, rank_constant):
    if not 0 <= reranker_weight <= 1 or not math.isfinite(reranker_weight):
        raise ValueError("reranker_weight must be in [0, 1]")
    if rank_constant <= 0 or not math.isfinite(rank_constant):
        raise ValueError("rank_constant must be positive")
    if len(set(dense_docids)) != len(dense_docids):
        raise ValueError("Duplicate retrieval candidates")
    if any(d not in relevance or not math.isfinite(relevance[d]) for d in dense_docids):
        raise ValueError("Every retrieved candidate needs a finite relevance score")
    model_order = sorted(dense_docids, key=lambda d: -relevance[d])
    model_rank = {d: rank for rank, d in enumerate(model_order, 1)}
    scores = {
        d: (1 - reranker_weight) / (rank_constant + rank)
        + reranker_weight / (rank_constant + model_rank[d])
        for rank, d in enumerate(dense_docids, 1)
    }
    return sorted(dense_docids, key=lambda d: -scores[d])
