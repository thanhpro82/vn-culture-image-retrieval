import numpy as np
from qdrant_client import QdrantClient

from retrieval.index import build_doc, create_collection, upsert_points

ROW = {
    "image_id": "00000000000000aa", "url": "http://x/1.jpg",
    "entity_ids": ["e1"], "entity_names": ["Lễ hội Leo cột mỡ"],
    "caption": "Một người đàn ông leo cột", "page_title": "Vui nhộn trò chơi dân gian",
    "branch": "C › Lễ hội", "region": "Nhiều địa phương", "culture_type": "Lễ hội dân gian",
    "entity_desc": "Trò chơi dân gian",
}


def test_build_doc_contains_entity_and_caption():
    doc = build_doc(ROW)
    assert "Lễ hội Leo cột mỡ" in doc
    assert "Một người đàn ông leo cột" in doc
    assert "Vui nhộn trò chơi dân gian" in doc


def test_build_doc_tolerates_missing_fields():
    doc = build_doc({**ROW, "caption": None, "region": None, "entity_desc": None})
    assert "Lễ hội Leo cột mỡ" in doc and "None" not in doc


def test_create_and_upsert_roundtrip():
    client = QdrantClient(":memory:")
    create_collection(client, "t")
    rows = [ROW, {**ROW, "image_id": "00000000000000bb", "url": "http://x/2.jpg"}]
    vecs = {
        "img": np.random.rand(2, 768).astype("float32"),
        "txt": np.random.rand(2, 1024).astype("float32"),
        "sparse": [{1: 0.5, 7: 0.2}, {3: 0.9}],
    }
    upsert_points(client, "t", rows, vecs)
    assert client.count("t").count == 2
    pts = client.retrieve("t", ids=[int("aa", 16)], with_payload=True)
    assert pts[0].payload["image_id"] == "00000000000000aa"
    assert pts[0].payload["entity_names"] == ["Lễ hội Leo cột mỡ"]


def test_payload_has_dup_group_defaulting_to_own_id():
    client = QdrantClient(":memory:")
    create_collection(client, "t2")
    vecs = {"img": np.random.rand(2, 768), "txt": np.random.rand(2, 1024), "sparse": [{1: 1.0}, {2: 1.0}]}
    rows = [{**ROW, "dup_group": "g1"}, {**ROW, "image_id": "00000000000000bb"}]
    upsert_points(client, "t2", rows, vecs)
    got = {p.payload["image_id"]: p.payload["dup_group"]
           for p in client.scroll("t2", with_payload=True, limit=10)[0]}
    assert got == {"00000000000000aa": "g1", "00000000000000bb": "00000000000000bb"}


def test_upsert_accepts_numpy_list_columns_from_parquet():
    client = QdrantClient(":memory:")
    create_collection(client, "t3")
    row = {**ROW, "entity_ids": np.array(["e1", "e2"]), "entity_names": np.array(["A", "B"])}
    vecs = {"img": np.random.rand(1, 768), "txt": np.random.rand(1, 1024), "sparse": [{1: 1.0}]}
    upsert_points(client, "t3", [row], vecs)
    p = client.scroll("t3", with_payload=True, limit=1)[0][0]
    assert p.payload["entity_ids"] == ["e1", "e2"] and p.payload["entity_names"] == ["A", "B"]
    assert "A" in build_doc(row) and "B" in build_doc(row)
