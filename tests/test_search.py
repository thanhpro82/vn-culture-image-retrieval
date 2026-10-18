import numpy as np
from qdrant_client import QdrantClient

from fakes import FakeEmbedder, hash_vec
from retrieval.index import build_doc, create_collection, upsert_points
from retrieval.qparse import ParsedQuery, SubQuery
from retrieval.search import Hit, interleave, search


def H(image_id, score=1.0, ents=("e",), sub=0, dup=None):
    return Hit(image_id=image_id, score=score, entity_ids=list(ents), caption="", source_sub=sub,
               dup_group=dup or image_id)


# ---------- interleave ----------
def test_each_intent_gets_quota():
    a = [H(f"a{i}", 1 - i * 0.01, ents=(f"ea{i}",), sub=0) for i in range(10)]
    b = [H(f"b{i}", 0.1 - i * 0.001, ents=(f"eb{i}",), sub=1) for i in range(10)]
    out = interleave([a, b], top_k=10)
    assert len(out) == 10
    assert sum(h.source_sub == 1 for h in out) >= 4


def test_max_per_entity():
    a = [H(f"a{i}", 1 - i * 0.01, ents=("same",)) for i in range(10)]
    out = interleave([a], top_k=10, max_per_entity=3)
    assert len(out) == 3


def test_dedupe_same_image_and_dup_group():
    a = [H("x", 1.0, ents=("e1",)), H("y", 0.9, ents=("e2",), dup="x"), H("z", 0.8, ents=("e3",))]
    b = [H("x", 0.7, ents=("e1",), sub=1)]
    assert [h.image_id for h in interleave([a, b], top_k=10)] == ["x", "z"]


def test_fewer_than_k_when_not_enough():
    assert len(interleave([[H("a", ents=("e1",))], [H("b", ents=("e2",), sub=1)]], top_k=10)) == 2


def test_one_intent_empty_does_not_block_others():
    a = [H(f"a{i}", ents=(f"e{i}",)) for i in range(5)]
    assert len(interleave([[], a], top_k=4)) == 4


# ---------- search qua Qdrant :memory: với embedder giả ----------
ROWS = [
    {"image_id": "00000000000000a1", "entity_ids": ["e1"], "entity_names": ["Áo bà ba"], "caption": "người mặc áo bà ba"},
    {"image_id": "00000000000000a2", "entity_ids": ["e2"], "entity_names": ["Múa hổ"], "caption": "điệu múa hổ"},
    {"image_id": "00000000000000a3", "entity_ids": ["e3"], "entity_names": ["Chợ nổi"], "caption": "chợ nổi miền tây"},
]


def _client():
    c = QdrantClient(":memory:")
    create_collection(c, "images")
    emb = FakeEmbedder()
    docs = [build_doc(r) for r in ROWS]
    dense, sparse = emb.text_bge(docs)
    upsert_points(c, "images", ROWS, {"img": np.stack([hash_vec(r["caption"], 768) for r in ROWS]),
                                      "txt": dense, "sparse": sparse})
    return c


def test_hybrid_search_finds_matching_doc_first():
    pq = ParsedQuery("single", [SubQuery(vi="áo bà ba", en="ao ba ba")])
    hits = search(pq, FakeEmbedder(), _client(), top_k=3)
    assert hits[0].image_id == "00000000000000a1"


def test_multi_hop_entities_are_searched_separately():
    pq = ParsedQuery("multi_hop", [SubQuery(vi="động vật và trang phục", en="x", ents=["Múa hổ", "Áo bà ba"])])
    ids = {h.image_id for h in search(pq, FakeEmbedder(), _client(), top_k=4)}
    assert {"00000000000000a1", "00000000000000a2"} <= ids


def test_search_rejects_unknown_branch():
    import pytest
    pq = ParsedQuery("single", [SubQuery(vi="áo bà ba", en="ao ba ba")])
    with pytest.raises(ValueError):
        search(pq, FakeEmbedder(), _client(), top_k=3, branches=("img", "bogus"))


def test_branches_select_which_signals_are_used():
    c = QdrantClient(":memory:")
    create_collection(c, "images")
    rows = [  # ảnh 1: tên thực thể khớp query nhưng caption (vector ảnh) không; ảnh 2: ngược lại
        {"image_id": "00000000000000b1", "entity_ids": ["x"], "entity_names": ["Áo bà ba"], "caption": "xyz"},
        {"image_id": "00000000000000b2", "entity_ids": ["y"], "entity_names": ["Khác"], "caption": "không liên quan"},
    ]
    emb = FakeEmbedder()
    dense, sparse = emb.text_bge([build_doc(r) for r in rows])
    image_looks_like = ["xyz", "áo bà ba"]  # vector ảnh độc lập với văn bản
    upsert_points(c, "images", rows, {"img": np.stack([hash_vec(t, 768) for t in image_looks_like]),
                                      "txt": dense, "sparse": sparse})
    pq = ParsedQuery("single", [SubQuery(vi="áo bà ba", en="ao ba ba")])
    assert search(pq, emb, c, top_k=1, branches=("txt", "sparse"))[0].image_id.endswith("b1")
    assert search(pq, emb, c, top_k=1, branches=("img",))[0].image_id.endswith("b2")
