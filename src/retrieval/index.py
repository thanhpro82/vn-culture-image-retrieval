"""Tạo collection Qdrant (3 vector mỗi ảnh) và nạp điểm."""
import numpy as np
from qdrant_client import QdrantClient, models

IMG_DIM, TXT_DIM = 768, 1024


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def _list(v) -> list:
    """Cột danh sách từ parquet là mảng numpy; chuẩn hoá về list thuần, None -> []."""
    return [] if v is None else [str(x) for x in list(v)]


def build_doc(row) -> str:
    """Ghép văn bản có ngữ cảnh để lập chỉ mục text (dense bge-m3 và sparse)."""
    head = [f"Thực thể: {' / '.join(n for n in _list(row.get('entity_names')) if n)}"]
    for label, key in (("Phân loại", "branch"), ("Địa bàn", "region"), ("Thể loại", "culture_type")):
        if _s(row.get(key)):
            head.append(f"{label}: {_s(row.get(key))}")
    parts = ["[" + " | ".join(head) + "]"]
    for label, key in (("Mô tả", "entity_desc"), ("Chi tiết ảnh", "caption"), ("Nguồn", "page_title")):
        if _s(row.get(key)):
            parts.append(f"{label}: {_s(row.get(key))}")
    return ". ".join(parts)


def create_collection(client: QdrantClient, name: str) -> None:
    quant = models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True))
    client.create_collection(
        collection_name=name,
        vectors_config={
            "img": models.VectorParams(size=IMG_DIM, distance=models.Distance.COSINE,
                                       quantization_config=quant),
            "txt": models.VectorParams(size=TXT_DIM, distance=models.Distance.COSINE,
                                       quantization_config=quant),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


def upsert_points(client: QdrantClient, name: str, rows: list[dict], vectors: dict) -> None:
    points = []
    for i, row in enumerate(rows):
        sp = vectors["sparse"][i]
        points.append(models.PointStruct(
            id=int(row["image_id"], 16),
            vector={
                "img": np.asarray(vectors["img"][i], dtype="float32").tolist(),
                "txt": np.asarray(vectors["txt"][i], dtype="float32").tolist(),
                "sparse": models.SparseVector(indices=list(sp.keys()), values=list(sp.values())),
            },
            payload={
                "image_id": row["image_id"],
                "entity_ids": _list(row.get("entity_ids")),
                "entity_names": _list(row.get("entity_names")),
                "caption": _s(row.get("caption")),
                "page_title": _s(row.get("page_title")),
                "branch": _s(row.get("branch")),
                "region": _s(row.get("region")),
                "dup_group": _s(row.get("dup_group")) or row["image_id"],
            },
        ))
    client.upsert(collection_name=name, points=points)


def _dup_groups(manifest, hamming: int = 4) -> dict[str, str]:
    """image_id -> đại diện nhóm (id nhỏ nhất) cho ảnh gần trùng; ảnh đơn không có khoá."""
    from retrieval.download import near_duplicates
    ph = dict(zip(manifest["image_id"], manifest["phash"], strict=True))
    out: dict[str, str] = {}
    for g in near_duplicates({k: v for k, v in ph.items() if isinstance(v, str)}, hamming):
        rep = min(g)
        out.update({i: rep for i in g})
    return out


def build(data_dir: str = "Data", batch: int = 64, collection: str = "images", limit: int | None = None):
    """Embed ảnh đã tải (SigLIP 2) rồi text (bge-m3) tuần tự, lưu .npy trung gian, nạp Qdrant."""
    import pickle
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    from retrieval.config import get_settings
    from retrieval.embed import Embedder

    d = Path(data_dir)
    m = pd.read_parquet(d / "manifest.parquet")
    st = pd.read_parquet(d / "status.parquet")
    m = m[m["image_id"].isin(set(st[st["ok"]]["image_id"]))].reset_index(drop=True)
    if limit:
        m = m.head(limit)
    groups = _dup_groups(m)
    m["dup_group"] = [groups.get(i, i) for i in m["image_id"]]
    rows = m.to_dict("records")
    emb = Embedder()
    work = d / "embeddings"
    work.mkdir(exist_ok=True)
    tag = f"{len(rows)}"
    f_img, f_txt = work / f"img_{tag}.npy", work / f"txt_{tag}.npy"
    f_sp = work / f"sparse_{tag}.pkl"
    if not f_img.exists():
        out = []
        for i in range(0, len(rows), batch):
            ims = [Image.open(d / "images" / f"{r['image_id']}.webp") for r in rows[i:i + batch]]
            out.append(emb.image(ims))
            if (i // batch) % 50 == 0:
                print(f"img {i}/{len(rows)}", flush=True)
        np.save(f_img, np.vstack(out))
    emb.unload("siglip")
    if not f_txt.exists():
        docs = [build_doc(r) for r in rows]
        dense, sparse = [], []
        for i in range(0, len(docs), 256):
            dv, sv = emb.text_bge(docs[i:i + 256])
            dense.append(dv)
            sparse += sv
            if (i // 256) % 20 == 0:
                print(f"txt {i}/{len(docs)}", flush=True)
        np.save(f_txt, np.vstack(dense))
        f_sp.write_bytes(pickle.dumps(sparse))
    vecs = {"img": np.load(f_img), "txt": np.load(f_txt), "sparse": pickle.loads(f_sp.read_bytes())}
    client = QdrantClient(url=get_settings().qdrant_url, timeout=120)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    create_collection(client, collection)
    for i in range(0, len(rows), 500):
        upsert_points(client, collection, rows[i:i + 500],
                      {k: v[i:i + 500] for k, v in vecs.items()})
    print("points:", client.count(collection).count)


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["build"]:
        build(limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
