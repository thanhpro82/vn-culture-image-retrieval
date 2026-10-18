"""Tìm kiếm lai (ảnh + text + sparse, RRF) cho từng ý, rồi gộp theo hạn ngạch."""
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from retrieval.qparse import ParsedQuery

BRANCHES = ("img", "txt", "sparse")
PREFETCH = 50
PER_UNIT = 30


@dataclass
class Hit:
    image_id: str
    score: float
    entity_ids: list[str]
    caption: str
    source_sub: int
    dup_group: str = ""
    page_title: str = ""
    entity_names: list[str] | None = None


def _units(parsed: ParsedQuery) -> list[tuple[str, str]]:
    """(văn bản cho SigLIP, văn bản cho bge-m3) của mỗi 'ý' cần tìm riêng."""
    out: list[tuple[str, str]] = []
    for s in parsed.subs:
        if parsed.qt == "multi_hop" and s.ents:  # mỗi thực thể suy ra được tìm riêng
            out += [(e, e) for e in s.ents]
        else:
            out.append((s.en or s.vi, " ".join([s.vi, *s.ents])))
    return out


def search(parsed: ParsedQuery, embedder, client: QdrantClient, top_k: int = 10,
           collection: str = "images", max_per_entity: int = 3,
           branches: tuple[str, ...] = BRANCHES, timings: dict | None = None) -> list[Hit]:
    if not branches or not set(branches) <= set(BRANCHES):
        raise ValueError(f"branches phải là tập con của {BRANCHES}")
    units = _units(parsed)
    t_a = time.perf_counter()
    img_q = embedder.text_siglip([u[0] for u in units])
    dense_q, sparse_q = embedder.text_bge([u[1] for u in units])
    t_b = time.perf_counter()
    reqs = []
    for i in range(len(units)):
        sp = sparse_q[i]
        all_pf = {
            "img": models.Prefetch(query=img_q[i].tolist(), using="img", limit=PREFETCH),
            "txt": models.Prefetch(query=dense_q[i].tolist(), using="txt", limit=PREFETCH),
            "sparse": models.Prefetch(query=models.SparseVector(indices=list(sp), values=list(sp.values())),
                                      using="sparse", limit=PREFETCH),
        }
        reqs.append(models.QueryRequest(
            prefetch=[all_pf[b] for b in branches],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=PER_UNIT, with_payload=True))
    responses = client.query_batch_points(collection, requests=reqs)
    if timings is not None:
        timings["embed_ms"] = (t_b - t_a) * 1000
        timings["qdrant_ms"] = (time.perf_counter() - t_b) * 1000
    per_unit = [[Hit(image_id=p.payload["image_id"], score=float(p.score),
                     entity_ids=list(p.payload.get("entity_ids", [])), caption=p.payload.get("caption", ""),
                     source_sub=i, dup_group=p.payload.get("dup_group", p.payload["image_id"]),
                     page_title=p.payload.get("page_title", ""),
                     entity_names=list(p.payload.get("entity_names", [])))
                 for p in r.points] for i, r in enumerate(responses)]
    return interleave(per_unit, top_k=top_k, max_per_entity=max_per_entity)


def interleave(per_sub: list[list[Hit]], top_k: int, max_per_entity: int = 3) -> list[Hit]:
    """Vòng tròn giữa các ý; bỏ ảnh trùng (cùng id hoặc dup_group); tối đa N ảnh mỗi thực thể."""
    seen_img: set[str] = set()
    seen_dup: set[str] = set()
    per_ent: dict[str, int] = {}
    cursors = [0] * len(per_sub)
    out: list[Hit] = []

    def take(i: int) -> bool:
        lst = per_sub[i]
        while cursors[i] < len(lst):
            h = lst[cursors[i]]
            cursors[i] += 1
            grp = h.dup_group or h.image_id
            if h.image_id in seen_img or grp in seen_dup:
                continue
            if any(per_ent.get(e, 0) >= max_per_entity for e in h.entity_ids):
                continue
            seen_img.add(h.image_id)
            seen_dup.add(grp)
            for e in h.entity_ids:
                per_ent[e] = per_ent.get(e, 0) + 1
            out.append(h)
            return True
        return False

    while len(out) < top_k:
        progressed = False
        for i in range(len(per_sub)):
            if len(out) >= top_k:
                break
            progressed |= take(i)
        if not progressed:
            break
    return out
