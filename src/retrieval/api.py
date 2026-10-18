"""FastAPI: /search, /embed, /rerank, /healthz. Thành phần được tiêm qua Services để dễ test."""
import asyncio
import hmac
import os
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from retrieval.config import Settings
from retrieval.grounding import EntityCatalog, ground
from retrieval.qparse import ParsedQuery, fallback_parse, parse_query
from retrieval.rerank import rerank_hits
from retrieval.search import BRANCHES, _units, search
from retrieval.storage import ImageStore, LocalStore

WEB_DIR = Path(os.environ.get("WEB_DIR", "web"))
CANDIDATES = 16  # ngân sách rerank, xem ADR-002
CACHE_SIZE = 512


@dataclass
class Services:
    settings: Settings
    embedder: Any
    reranker: Any
    qdrant: Any
    llm: Any
    catalog: EntityCatalog | None
    store: ImageStore
    collection: str = "images"
    budget_ms: float = 2700.0  # trần thời gian một truy vấn; rerank chỉ chạy khi còn dư địa
    rerank_ema_ms: float = 800.0  # ước lượng thời gian rerank, cập nhật theo số đo thật


class SearchIn(BaseModel):
    query: str = Field(max_length=500)
    top_k: int = Field(default=10, ge=1, le=20)
    mode: Literal["hybrid", "image", "text"] = "hybrid"  # để đối chứng; mặc định là hybrid
    rerank: bool = True

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = unicodedata.normalize("NFC", v).strip()
        if not v:
            raise ValueError("query rỗng")
        return v


class EmbedIn(BaseModel):
    texts: list[Annotated[str, Field(max_length=512)]] = Field(min_length=1, max_length=64)
    space: Literal["image", "text"] = "image"


class RerankIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    documents: list[Annotated[str, Field(max_length=2000)]] = Field(min_length=1, max_length=64)
    top_n: int = Field(default=10, ge=1, le=64)


def create_app(svc: Services) -> FastAPI:
    app = FastAPI(title="VN culture image retrieval")
    cache: OrderedDict[str, ParsedQuery] = OrderedDict()

    def auth(x_api_key: Annotated[str | None, Header()] = None) -> None:
        key = svc.settings.api_key
        if not key or not x_api_key or not hmac.compare_digest(x_api_key, key):
            raise HTTPException(status_code=401, detail="API key không hợp lệ")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    async def understand(q: str) -> ParsedQuery:
        key = q.casefold()
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        names = svc.catalog.names if svc.catalog else []
        parsed = await parse_query(q, svc.llm, timeout_ms=svc.settings.gemini_timeout_ms, entity_names=names)
        if parsed != fallback_parse(q, names):  # không nhớ kết quả fallback (LLM có thể chỉ lỗi tạm)
            cache[key] = parsed
            if len(cache) > CACHE_SIZE:
                cache.popitem(last=False)
        return parsed

    async def run_search(body: SearchIn):
        t0 = time.perf_counter()
        parsed = await understand(body.query)
        if svc.catalog:
            parsed = ground(parsed, svc.catalog)
        t1 = time.perf_counter()
        branches = {"hybrid": BRANCHES, "image": ("img",), "text": ("txt", "sparse")}[body.mode]
        parts: dict = {}
        hits = await asyncio.to_thread(search, parsed, svc.embedder, svc.qdrant, CANDIDATES,
                                       svc.collection, 3, branches, parts)
        t2 = time.perf_counter()
        elapsed_ms = (t2 - t0) * 1000
        reranked = False
        top = hits[: body.top_k]
        if body.rerank and elapsed_ms + svc.rerank_ema_ms < svc.budget_ms:
            queries = [u[1] for u in _units(parsed)]
            remaining = max(svc.budget_ms - elapsed_ms, 200.0) / 1000
            try:  # giới hạn cứng: quá hạn thì trả thứ tự hybrid, không để một lần rerank chậm phá ngân sách
                top = await asyncio.wait_for(
                    asyncio.to_thread(rerank_hits, queries, hits, svc.reranker, body.top_k, CANDIDATES),
                    timeout=remaining)
                reranked = True
                took = (time.perf_counter() - t2) * 1000
                svc.rerank_ema_ms = 0.7 * svc.rerank_ema_ms + 0.3 * took
            except TimeoutError:
                top = hits[: body.top_k]
                svc.rerank_ema_ms = max(svc.rerank_ema_ms, remaining * 1000)
        t3 = time.perf_counter()
        return {
            "query_type": parsed.qt,
            "reranked": reranked,
            "subs": [{"vi": s.vi, "en": s.en, "ents": s.ents} for s in parsed.subs],
            "results": [{"image_id": h.image_id, "url": svc.store.url(h.image_id), "caption": h.caption,
                         "entities": h.entity_names or [], "page_title": h.page_title,
                         "score": round(h.score, 4)} for h in top],
            "timings_ms": {"parse_ms": round((t1 - t0) * 1000, 1), "retrieve_ms": round((t2 - t1) * 1000, 1),
                           "embed_ms": round(parts.get("embed_ms", 0.0), 1),
                           "qdrant_ms": round(parts.get("qdrant_ms", 0.0), 1),
                           "rerank_ms": round((t3 - t2) * 1000, 1), "total_ms": round((t3 - t0) * 1000, 1)},
        }

    @app.post("/search", dependencies=[Depends(auth)])
    async def search_endpoint(body: SearchIn):
        return await run_search(body)

    @app.post("/demo/search")
    async def demo_search_endpoint(body: SearchIn):
        """Dành cho giao diện web công khai (không nhúng khoá vào trang)."""
        return await run_search(body)

    @app.get("/", include_in_schema=False)
    def index():
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="thiếu web/index.html")
        return FileResponse(page, media_type="text/html")

    @app.post("/embed", dependencies=[Depends(auth)])
    async def embed_endpoint(body: EmbedIn):
        if body.space == "image":
            vecs = await asyncio.to_thread(svc.embedder.text_siglip, body.texts)
        else:
            vecs, _ = await asyncio.to_thread(svc.embedder.text_bge, body.texts)
        return {"space": body.space, "dim": len(vecs[0]), "vectors": [v.tolist() for v in vecs]}

    @app.post("/rerank", dependencies=[Depends(auth)])
    async def rerank_endpoint(body: RerankIn):
        scores = await asyncio.to_thread(svc.reranker.score, body.query, body.documents)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: body.top_n]
        return {"results": [{"index": i, "score": float(scores[i]), "document": body.documents[i]}
                            for i in order]}

    if isinstance(svc.store, LocalStore):
        app.mount("/media", StaticFiles(directory=svc.store.root), name="media")
    return app
