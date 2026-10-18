import zlib

import numpy as np
from qdrant_client import QdrantClient

from retrieval.index import build_doc, create_collection, upsert_points
from retrieval.qparse import ParsedQuery, SubQuery, strip_accents
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
def _hv(text, dim):
    v = np.zeros(dim, dtype="float32")
    for w in strip_accents(text).split():
        v[zlib.crc32(w.encode()) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEmbedder:
    def text_siglip(self, texts):
        return np.stack([_hv(t, 768) for t in texts])

    def text_bge(self, texts):
        dense = np.stack([_hv(t, 1024) for t in texts])
        sparse = [{zlib.crc32(w.encode()) % 100000: 1.0 for w in strip_accents(t).split()} for t in texts]
        return dense, sparse


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
    upsert_points(c, "images", ROWS, {"img": np.stack([_hv(r["caption"], 768) for r in ROWS]),
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
