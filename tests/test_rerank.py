import pytest

from retrieval.rerank import Reranker, doc_text, rerank_hits
from retrieval.search import Hit


def H(i, sub=0, caption="", ents=("e",)):
    return Hit(image_id=f"i{i}", score=1.0, entity_ids=[f"{e}{i}" for e in ents], caption=caption,
               source_sub=sub, dup_group=f"i{i}", entity_names=["Tên"])


class Fake:
    def __init__(self, fn):
        self.fn, self.seen = fn, 0

    def score(self, query, docs):
        self.seen += len(docs)
        return [self.fn(query, d) for d in docs]


def test_never_scores_more_than_24_pairs():
    f = Fake(lambda q, d: 0.5)
    rerank_hits("q", [H(i) for i in range(40)], f, top_k=10)
    assert f.seen == 24


def test_orders_within_intent_by_score():
    hits = [H(1, caption="xấu"), H(2, caption="tốt"), H(3, caption="xấu")]
    out = rerank_hits("q", hits, Fake(lambda q, d: 1.0 if "tốt" in d else 0.0), top_k=3)
    assert out[0].image_id == "i2"


def test_rerank_keeps_every_intent_represented():
    hits = [H(i, sub=i % 2, caption="a" if i % 2 == 0 else "b") for i in range(20)]
    out = rerank_hits(["q0", "q1"], hits, Fake(lambda q, d: 1.0 if "a" in d.split()[-1] else 0.0), top_k=6)
    assert {h.source_sub for h in out} == {0, 1}
    assert sum(h.source_sub == 1 for h in out) >= 2


def test_each_hit_scored_against_its_own_sub_query():
    seen = []
    f = Fake(lambda q, d: seen.append(q) or 0.0)
    rerank_hits(["hổ", "mèo"], [H(1, sub=0), H(2, sub=1)], f, top_k=2)
    assert sorted(seen) == ["hổ", "mèo"]


def test_doc_text_contains_entity_and_caption_and_is_trimmed():
    d = doc_text(H(1, caption="x" * 2000))
    assert "Tên" in d and len(d) < 600


@pytest.mark.slow
def test_relevant_doc_scores_higher():
    r = Reranker()
    s = r.score("áo bà ba", ["Áo bà ba: người phụ nữ mặc áo bà ba", "Chợ nổi Cái Răng: thuyền bán trái cây"])
    assert s[0] > s[1]


@pytest.mark.slow
def test_truncates_to_128_tokens():
    r = Reranker()
    enc = r.tokenize("áo bà ba", ["từ " * 3000])
    assert enc["input_ids"].shape[1] <= 128 and r.max_length == 128


@pytest.mark.slow
def test_default_reranker_is_dynamically_quantized_on_cpu():
    import torch
    r = Reranker()
    assert any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in r.model.modules())


def test_multi_intent_is_scored_in_a_single_batched_call():
    class Batched:
        def __init__(self):
            self.calls = []

        def score_pairs(self, pairs):
            self.calls.append(list(pairs))
            return [1.0 if "tốt" in d else 0.0 for _, d in pairs]

    b = Batched()
    hits = [H(1, sub=0, caption="xấu"), H(2, sub=0, caption="tốt"), H(3, sub=1, caption="xấu"), H(4, sub=1, caption="tốt")]
    out = rerank_hits(["hổ", "mèo"], hits, b, top_k=4)
    assert len(b.calls) == 1 and len(b.calls[0]) == 4
    assert {q for q, _ in b.calls[0]} == {"hổ", "mèo"}
    assert next(h.image_id for h in out if h.source_sub == 0) == "i2"


def test_rerank_hits_does_not_mutate_the_input_hits():
    hits = [H(1, caption="xấu"), H(2, caption="tốt")]
    rerank_hits("q", hits, Fake(lambda q, d: 0.99 if "tốt" in d else 0.01), top_k=2)
    assert [h.score for h in hits] == [1.0, 1.0]  # điểm gốc giữ nguyên, kể cả khi luồng chạy muộn
