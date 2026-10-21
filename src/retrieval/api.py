"""FastAPI: /search, /embed, /rerank, /healthz. Thành phần được tiêm qua Services để dễ test."""
import asyncio
import hmac
import os
import time
import unicodedata
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from retrieval.config import Settings
from retrieval.grounding import EntityCatalog, entities_supported, ground
from retrieval.qparse import ParsedQuery, fallback_parse, parse_query
from retrieval.rerank import rerank_hits
from retrieval.search import BRANCHES, _units, search
from retrieval.storage import ImageStore, LocalStore

WEB_DIR = Path(os.environ.get("WEB_DIR", "web"))
CANDIDATES = 16  # ngân sách rerank, xem ADR-002
CACHE_SIZE = 512
LOW_CONF_CHECK = 10  # số ảnh đầu dùng để kiểm tra bằng chứng cho cờ độ tin cậy thấp (chọn theo số đo, xem docs/eval-report.md)


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
    max_inflight: int = 6  # số /search đang xử lý tối đa; vượt thì trả 503 thay vì xếp hàng vô hạn
    demo_rate_per_min: int = 20  # giới hạn /demo/search theo client
    search_reranker: Any = None  # bộ rerank riêng cho /search (ưu tiên endpoint ngoài); /rerank luôn dùng reranker local


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
    # Phần nặng CPU (embed, rerank) chạy lần lượt: hai yêu cầu cùng dùng đủ luồng sẽ tranh nhau và
    # chậm cả hai. Thời gian chờ Gemini (không dùng CPU) vẫn chồng lên nhau giữa các yêu cầu.
    cpu_slot = asyncio.Semaphore(1)

    def auth(x_api_key: Annotated[str | None, Header()] = None) -> None:
        key = svc.settings.api_key
        # so sánh trên bytes: compare_digest với chuỗi không ASCII ném TypeError (thành lỗi 500 thay vì 401)
        if not key or not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), key.encode("utf-8")):
            raise HTTPException(status_code=401, detail="API key không hợp lệ")

    @app.get("/healthz")
    async def healthz():
        try:  # Qdrant tắt thì API không phục vụ được: báo 503 để healthcheck của Docker/Caddy thấy
            await asyncio.to_thread(svc.qdrant.get_collection, svc.collection)
        except Exception:  # noqa: BLE001 - mọi lỗi kết nối hoặc thiếu collection đều là "không khoẻ"
            return JSONResponse({"status": "degraded", "qdrant": "unavailable"}, status_code=503)
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

    async def _rerank_job(queries, hits, top_k):
        """Chạy trong một task riêng: giữ khoá CPU đến khi luồng rerank thật sự xong, kể cả khi người gọi
        đã bỏ chờ vì quá hạn. Ước lượng thời gian chỉ cập nhật bằng thời gian chạy đo được (sau khi có khoá)."""
        async with cpu_slot:
            t = time.perf_counter()
            try:
                return await asyncio.to_thread(rerank_hits, queries, hits, svc.search_reranker or svc.reranker, top_k, CANDIDATES)
            finally:
                took = (time.perf_counter() - t) * 1000
                svc.rerank_ema_ms = 0.7 * svc.rerank_ema_ms + 0.3 * took

    async def run_search(body: SearchIn):
        t0 = time.perf_counter()
        parsed = await understand(body.query)
        proposed_ents = [e for s in parsed.subs for e in s.ents]
        proposed = bool(proposed_ents)
        if svc.catalog:
            parsed = ground(parsed, svc.catalog)
        low_confidence = proposed and not any(s.ents for s in parsed.subs)  # mọi thực thể LLM đề xuất đều không có trong dữ liệu
        t1 = time.perf_counter()
        branches = {"hybrid": BRANCHES, "image": ("img",), "text": ("txt", "sparse")}[body.mode]
        parts: dict = {}
        async with cpu_slot:
            hits = await asyncio.to_thread(search, parsed, svc.embedder, svc.qdrant, CANDIDATES,
                                           svc.collection, 3, branches, parts)
        t2 = time.perf_counter()
        if low_confidence and entities_supported(
                proposed_ents, [n for h in hits[:LOW_CONF_CHECK] for n in (h.entity_names or [])], pooled=True):
            low_confidence = False  # tên đề xuất không khớp danh mục nhưng ảnh trả về đúng là thực thể đó: không cảnh báo
        elapsed_ms = (t2 - t0) * 1000
        reranked = False
        top = hits[: body.top_k]
        if body.rerank and elapsed_ms + svc.rerank_ema_ms < svc.budget_ms:
            queries = [u[1] for u in _units(parsed)]
            remaining = max(svc.budget_ms - elapsed_ms, 200.0) / 1000
            try:  # giới hạn cứng: quá hạn thì trả thứ tự hybrid, không để một lần rerank chậm phá ngân sách
                job = asyncio.ensure_future(_rerank_job(queries, hits, body.top_k))
                job.add_done_callback(lambda f: f.cancelled() or f.exception())  # không để lỗi muộn thành cảnh báo
                top = await asyncio.wait_for(asyncio.shield(job), timeout=remaining)
                reranked = True
            except TimeoutError:
                top = hits[: body.top_k]
        t3 = time.perf_counter()
        return {
            "query_type": parsed.qt,
            "reranked": reranked,
            "low_confidence": low_confidence,
            "subs": [{"vi": s.vi, "en": s.en, "ents": s.ents} for s in parsed.subs],
            "results": [{"image_id": h.image_id, "url": svc.store.url(h.image_id), "caption": h.caption,
                         "entities": h.entity_names or [], "page_title": h.page_title,
                         "score": round(h.score, 4)} for h in top],
            "timings_ms": {"parse_ms": round((t1 - t0) * 1000, 1), "retrieve_ms": round((t2 - t1) * 1000, 1),
                           "embed_ms": round(parts.get("embed_ms", 0.0), 1),
                           "qdrant_ms": round(parts.get("qdrant_ms", 0.0), 1),
                           "rerank_ms": round((t3 - t2) * 1000, 1), "total_ms": round((t3 - t0) * 1000, 1)},
        }

    inflight = {"n": 0}
    hits_by_client: dict[str, deque] = {}

    async def admitted_search(body: SearchIn):
        if inflight["n"] >= svc.max_inflight:  # quá tải: từ chối nhanh thay vì xếp hàng vô hạn
            raise HTTPException(status_code=503, detail="Máy chủ đang bận", headers={"Retry-After": "2"})
        inflight["n"] += 1
        try:
            return await run_search(body)
        finally:
            inflight["n"] -= 1

    @app.post("/search", dependencies=[Depends(auth)])
    async def search_endpoint(body: SearchIn):
        return await admitted_search(body)

    @app.post("/demo/search")
    async def demo_search_endpoint(body: SearchIn, request: Request):
        """Dành cho giao diện web công khai (không nhúng khoá vào trang), có giới hạn tần suất theo client."""
        client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            request.client.host if request.client else "?")
        now = time.monotonic()
        q = hits_by_client.setdefault(client, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= svc.demo_rate_per_min:
            raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu, thử lại sau",
                                headers={"Retry-After": str(int(60 - (now - q[0])) + 1)})
        q.append(now)
        if len(hits_by_client) > 10_000:  # bộ nhớ có trần
            hits_by_client.pop(next(iter(hits_by_client)))
        return await admitted_search(body)

    @app.get("/", include_in_schema=False)
    def index():
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="thiếu web/index.html")
        return FileResponse(page, media_type="text/html")

    @app.post("/embed", dependencies=[Depends(auth)])
    async def embed_endpoint(body: EmbedIn):
        async with cpu_slot:  # dùng chung khoá CPU với /search: hai việc nặng không tranh nhau luồng
            if body.space == "image":
                vecs = await asyncio.to_thread(svc.embedder.text_siglip, body.texts)
            else:
                vecs, _ = await asyncio.to_thread(svc.embedder.text_bge, body.texts)
        return {"space": body.space, "dim": len(vecs[0]), "vectors": [v.tolist() for v in vecs]}

    @app.post("/rerank", dependencies=[Depends(auth)])
    async def rerank_endpoint(body: RerankIn):
        async with cpu_slot:
            scores = await asyncio.to_thread(svc.reranker.score, body.query, body.documents)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: body.top_n]
        return {"results": [{"index": i, "score": float(scores[i]), "document": body.documents[i]}
                            for i in order]}

    if isinstance(svc.store, LocalStore):
        app.mount("/media", StaticFiles(directory=svc.store.root), name="media")
    return app
