import pytest

from rag_eval.fusion import fuse_rankings


def test_endpoints_and_stable_model_ties_preserve_retrieval_information():
    dense = ["a", "b", "c"]
    scores = {"a": 0.1, "b": 0.9, "c": 0.9}
    assert fuse_rankings(dense, scores, reranker_weight=0, rank_constant=1) == dense
    assert fuse_rankings(dense, scores, reranker_weight=1, rank_constant=1) == [
        "b",
        "c",
        "a",
    ]
    assert (
        fuse_rankings(
            dense, {d: 1 for d in dense}, reranker_weight=0.4, rank_constant=1
        )
        == dense
    )


def test_fusion_keeps_every_candidate_and_rejects_missing_scores():
    dense = ["a", "b", "c", "d"]
    scores = dict(zip(dense, [0.1, 0.2, 0.8, 0.7]))
    ranking = fuse_rankings(dense, scores, reranker_weight=0.4, rank_constant=1)
    assert len(ranking) == len(dense) and set(ranking) == set(dense)
    assert ranking[0] == "a" and ranking.index("c") < ranking.index("b")
    with pytest.raises(ValueError, match="finite relevance"):
        fuse_rankings(dense, {"a": 0.1}, reranker_weight=0.4, rank_constant=1)
