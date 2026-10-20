# Image Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Web demo tìm ảnh theo văn bản tiếng Việt trên >= 80.000 ảnh, latency < 3 giây, chạy bằng `docker compose`.

**Architecture:** Index ngoại tuyến (manifest từ Excel, tải ảnh, embed bằng SigLIP 2 và bge-m3, nạp Qdrant 3 vector/ảnh). Serve trên CPU: Gemini tách query, cổng kiểm tra thực thể, hybrid search + RRF, gộp theo thực thể kèm hạn ngạch, rerank, trả top 10.

**Tech Stack:** Python 3.12, FastAPI, Qdrant, SigLIP 2 B/16, bge-m3, ONNX Runtime, Gemini API, k6, Docker, GitHub Actions, AWS S3.

**Spec:** `docs/superpowers/specs/2026-10-18-image-retrieval-design.md`

## Global Constraints

- Hạn nộp 2026-10-22; mọi mục bắt buộc xong trước tối 2026-10-21.
- Index và truy xuất tối thiểu 80.000 ảnh; latency mỗi query < 3 giây (cả single, multi-intent, multi-hop) trên toàn bộ index.
- Chất lượng đo bằng Recall@10: mỗi query trả 10 ảnh, chỉ cần 1 ảnh trúng ground truth.
- Bước tìm ảnh chỉ dùng embedding và/hoặc rerank, không dùng agent.
- LLM: Gemini 3.5 Flash Lite, một lần gọi cho mỗi query, timeout cứng 1.200 ms, có fallback regex. Không truyền `$schema` vào `response_schema`.
- Endpoint: `/search`, `/embed`, `/rerank`, `/healthz`; xác thực bằng header `X-API-Key` (trừ `/healthz`).
- Rerank: tối đa 24 cặp, `max_length=128`. Tối đa 3 ảnh mỗi thực thể trong top 10.
- Nhãn ảnh: lấy file recheck mới nhất, nếu không có thì file final; ô `Giữ/Bỏ` trống là giữ, `x` là bỏ. Chỉ dùng sheet "Ảnh đã thẩm định".
- Bí mật chỉ nằm trong `.env`; không commit `.env` và `Data/`.
- Ngôn ngữ Python; mọi số latency trong báo cáo phải là số đo thật.

## Review Focus

Các đầu vào mà spec ngụ ý nhưng không task nào kiểm tra tự nhiên; mỗi dòng đã có test ở task chủ.

1. Query tiếng Việt không dấu ("ao ba ba cach tan") hoặc sai chính tả: vẫn phải ra kết quả hợp lý (Task 8).
2. Query rỗng, chỉ khoảng trắng, hoặc rất dài (> 500 ký tự): trả lỗi 422 rõ ràng, không treo (Task 10).
3. Query có thực thể không tồn tại trong dữ liệu ("con sư tử" khi không có): không trả ảnh sai chủ đề, báo ít ảnh hơn 10 thay vì độn (Task 7, 8).
4. Gemini lỗi, timeout hoặc trả JSON hỏng: hệ thống vẫn trả kết quả bằng fallback (Task 6).
5. Link ảnh chết hoặc ảnh chưa tải được: không xuất hiện trong kết quả và không làm vỡ giao diện (Task 3, 11).

---

## File Structure

```
src/retrieval/
  config.py        # đọc .env -> Settings
  manifest.py      # Excel -> manifest.parquet
  download.py      # tải ảnh, WebP, pHash, báo cáo link sống
  storage.py       # ImageStore (local + S3)
  embed.py         # SigLIP 2 + bge-m3
  index.py         # tạo collection Qdrant, nạp điểm
  qparse.py        # Gemini + fallback
  grounding.py     # cổng kiểm tra thực thể
  search.py        # hybrid, gộp, hạn ngạch
  rerank.py        # cross-encoder
  api.py           # FastAPI
web/index.html     # giao diện
eval/build_testset.py, eval/evaluate.py
loadtest/script.js
tests/             # pytest, mỗi module một file
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, README.md
```

---

### Task 1: Khung dự án, cấu hình và quy trình làm việc

Quy trình lấy từ repo cũ `VIVI-cabin-copilot`, rút gọn cho một người trong 4 ngày.

**Files:**
- Create: `pyproject.toml`, `src/retrieval/__init__.py`, `src/retrieval/config.py`, `tests/test_config.py`
- Create (quy trình): `CLAUDE.md`, `WORKLOG.md`, `Makefile`, `docs/adr/` (rỗng, có `README.md`), `eval/datasets/` (rỗng, có `README.md`), `deploy/` (rỗng)
- Process rules: nhánh `feature/<giai-doạn>` + PR vào `main`; commit `type(scope): mô tả`; marker pytest `slow` và `integration` cho test cần model hoặc Qdrant thật, CI chỉ chạy `-m "not slow and not integration"`; số liệu trong báo cáo là ảnh chụp có ngày, đo lại trước khi trích.

**Interfaces:**
- Produces: `class Settings` (pydantic-settings) với các trường `aws_access_key_id, aws_secret_access_key, aws_region, s3_bucket, s3_prefix, gemini_api_key, gemini_model, gemini_timeout_ms: int = 1200, api_key, qdrant_url`; `get_settings() -> Settings` (cache).

- [ ] **Step 1: Viết test.** `test_settings_reads_env(monkeypatch)`: đặt `API_KEY=abc`, `GEMINI_TIMEOUT_MS=900`; assert `get_settings().api_key == "abc"` và `gemini_timeout_ms == 900`. `test_timeout_default`: không đặt biến, assert `== 1200`.
- [ ] **Step 2: Chạy `pytest tests/test_config.py -v`, kỳ vọng FAIL (module chưa có).**
- [ ] **Step 3: Cài đặt `Settings` và `get_settings()`** trong `config.py`; `pyproject.toml` khai báo phụ thuộc và `pytest`.
- [ ] **Step 4: Chạy lại, kỳ vọng PASS.**
- [ ] **Step 5: Quy trình.** `CLAUDE.md` (repo là gì, lệnh chạy test, quy tắc đo lại số liệu); `Makefile` với `lint`, `test`, `check` (`ruff` và `pytest -m "not slow and not integration"`); `WORKLOG.md` có bảng ngày: việc, trạng thái, bằng chứng; ghi vào `docs/adr/README.md` mẫu bảng cổng quyết định (ngưỡng đề xuất | số đo | căn cứ). Ba ADR sẽ được điền ở Task 3 (link sống), 9 (rerank), 10 (độ trễ).
- [ ] **Step 6: Commit** `chore(scaffold): settings, makefile, worklog and process docs`.

### Task 2: Manifest từ Excel

**Files:**
- Create: `src/retrieval/manifest.py`, `tests/test_manifest.py`, `tests/fixtures/make_fixture.py`

**Interfaces:**
- Produces: `build_manifest(data_dir: Path) -> pd.DataFrame` với cột `image_id, url, entity_ids (list[str]), entity_names (list[str]), caption, page_title, domain, resolution, phash, dhash, branch, region, culture_type, entity_desc`. `image_id = sha1(url)[:16]`. `write_manifest(df, path: Path) -> None` (parquet).

- [ ] **Step 1: Viết test trên workbook giả nhỏ** (tạo trong `make_fixture.py`, có sheet "Thực thể (Summary)" và "Ảnh đã thẩm định"):
  - `test_blank_is_keep_x_is_drop`: một ảnh trống `Giữ/Bỏ`, một ảnh `x`; chỉ ảnh trống còn lại.
  - `test_recheck_overrides_final`: ảnh `x` ở final nhưng trống ở recheck => còn; ngược lại => mất.
  - `test_short_rows_are_padded`: dòng bị cắt ngắn (không có ô cuối) vẫn được tính là giữ.
  - `test_url_in_two_entities_is_one_row`: cùng URL ở 2 thực thể => 1 dòng, `entity_ids` có 2 phần tử.
  - `test_skips_lock_files`: file `~$x.xlsx` bị bỏ qua.
- [ ] **Step 2: Chạy, kỳ vọng FAIL.**
- [ ] **Step 3: Cài đặt `build_manifest`.** Đọc bằng openpyxl read-only, đệm dòng theo độ dài header, file recheck đọc sau để ghi đè nhãn theo khoá `(entity_id, url)`, gộp theo `url`, ghép metadata thực thể từ sheet Summary.
- [ ] **Step 4: Chạy test, kỳ vọng PASS.**
- [ ] **Step 5: Kiểm tra trên dữ liệu thật:** `python -m retrieval.manifest Data/incoming data/manifest.parquet`. Kỳ vọng in `rows=92339` (số URL khác nhau đã xác minh).
- [ ] **Step 6: Commit** `feat: build image manifest from team 3 excel`.

### Task 3: Tải ảnh và đo tỉ lệ link sống

**Files:**
- Create: `src/retrieval/download.py`, `tests/test_download.py`

**Interfaces:**
- Consumes: `build_manifest` output (parquet).
- Produces: `async fetch_image(session, url: str) -> bytes | None` (timeout 15 giây, thử lại 2 lần, giới hạn 6 kết nối mỗi domain); `to_webp(raw: bytes, max_side: int = 640) -> bytes`; `probe(manifest_path: Path, n: int, seed: int) -> dict` trả `{"n", "ok", "alive_rate", "p50_ms", "mean_kb"}`; `download_all(manifest_path: Path, out_dir: Path) -> None` ghi `status.parquet` (`image_id, ok, reason`).

- [ ] **Step 1: Viết test với server HTTP giả (aiohttp test server):** `test_404_returns_none`, `test_truncated_image_returns_none` (byte cụt không giải mã được), `test_to_webp_limits_side` (ảnh 2000x1000 => cạnh dài 640), `test_retries_on_503_then_ok`.
- [ ] **Step 2: Chạy, kỳ vọng FAIL.**
- [ ] **Step 3: Cài đặt.** Dùng `aiohttp`, `tenacity`, `Pillow`. Không giả `Referer`, không tắt kiểm tra SSL. Ghi lý do lỗi (`http_404`, `timeout`, `decode`).
- [ ] **Step 4: Chạy test, kỳ vọng PASS.**
- [ ] **Step 5: Đo mẫu:** `python -m retrieval.download probe data/manifest.parquet --n 200 --seed 1`. Ghi `alive_rate`, `p50_ms`, `mean_kb` vào `docs/data-notes.md`. **Điểm quyết định:** nếu `alive_rate * 92339 < 80000` thì dừng và báo bạn trước khi tiếp tục.
- [ ] **Step 6: Gần trùng.** Thêm `near_duplicates(phashes: dict[str, str], max_hamming: int) -> list[set[str]]` và test `test_groups_images_within_threshold`. Ngưỡng `max_hamming` chọn bằng cách đo trên dữ liệu thật (xem vài cặp ở các ngưỡng 6, 10, 14 bằng mắt), không dùng nguyên số 30 của research vì pHash 64-bit khác PDQ 256-bit. Mỗi nhóm giữ một ảnh có độ phân giải cao nhất; ảnh bị loại không vào index.
- [ ] **Step 7: Commit** `feat: image downloader, link-health probe, near-duplicate grouping`.

### Task 4: Lưu trữ ảnh (local + S3)

**Files:**
- Create: `src/retrieval/storage.py`, `tests/test_storage.py`

**Interfaces:**
- Produces: `class ImageStore(Protocol)`: `put(image_id: str, data: bytes) -> None`, `exists(image_id: str) -> bool`, `url(image_id: str, expires: int = 3600) -> str`. `LocalStore(root: Path)`, `S3Store(bucket: str, prefix: str, region: str)`; `make_store(settings) -> ImageStore` (S3 nếu có `s3_bucket`, ngược lại local).

- [ ] **Step 1: Viết test:** `test_local_roundtrip` (put rồi exists là True, url trỏ tới `/media/<id>.webp`); `test_s3_url_is_presigned` dùng `moto` (assert URL có `X-Amz-Signature`); `test_make_store_without_bucket_is_local`.
- [ ] **Step 2: Chạy, kỳ vọng FAIL.** Step 3: cài đặt bằng `boto3`. Step 4: PASS.
- [ ] **Step 5: Kiểm tra S3 thật** (sau khi bạn điền `.env`): `python -m retrieval.storage selftest` đẩy 1 file 1 KB, đọc lại, xoá. Kỳ vọng in `ok`.
- [ ] **Step 6: Commit** `feat: image store with local and S3 backends`.

### Task 5: Embedding và nạp Qdrant

**Files:**
- Create: `src/retrieval/embed.py`, `src/retrieval/index.py`, `tests/test_embed.py`, `tests/test_index.py`

**Interfaces:**
- Produces: `class Embedder`: `image(batch: list[PIL.Image]) -> np.ndarray` (N x 768, chuẩn hoá L2); `text_siglip(texts: list[str]) -> np.ndarray` (N x 768); `text_bge(texts: list[str]) -> tuple[np.ndarray, list[dict[int, float]]]` (dense N x 1024 và sparse). `build_doc(row) -> str` ghép `[Thực thể | Phân loại | Địa bàn | Thể loại]. Mô tả. Chi tiết ảnh. Nguồn.` `create_collection(client, name: str) -> None` với 3 vector đặt tên `img`, `txt`, `sparse` (scalar INT8, `always_ram=True`); `upsert_points(client, name, rows, vectors) -> None` (payload gồm `image_id, entity_ids, entity_names, caption, page_title, branch, region`).

- [ ] **Step 1: Viết test:** `test_build_doc_contains_entity_and_caption`; `test_image_embedding_shape_and_norm` (vector 768 chiều, chuẩn gần 1); `test_vietnamese_text_matches_matching_caption` (cosine của "áo bà ba" với ảnh áo bà ba mẫu lớn hơn với ảnh phong cảnh); `test_create_and_upsert_roundtrip` dùng Qdrant `:memory:` (đếm điểm bằng số hàng đã nạp).
- [ ] **Step 2: Chạy, kỳ vọng FAIL.** Step 3: cài đặt. Embedder tải `google/siglip2-base-patch16-256` và `BAAI/bge-m3`, chạy FP16 trên GPU khi có, tuần tự từng model để vừa 4 GB VRAM. Index chạy theo lô 64 và lưu `.npy` trung gian để chạy lại không phải embed lại.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Chạy index thật:** `python -m retrieval.index build`. Kỳ vọng in số điểm bằng số ảnh `ok` ở Task 3 và ghi thời gian vào `docs/data-notes.md`.
- [ ] **Step 6: Commit** `feat: embeddings and qdrant index`.

### Task 6: Hiểu query bằng Gemini, có fallback

**Files:**
- Create: `src/retrieval/qparse.py`, `tests/test_qparse.py`

**Interfaces:**
- Produces: `@dataclass SubQuery(vi: str, en: str, ents: list[str], loc: str, cat: str)`; `@dataclass ParsedQuery(qt: Literal["single","multi_intent","multi_hop"], subs: list[SubQuery])`; `async parse_query(q: str, client, timeout_ms: int = 1200) -> ParsedQuery`; `fallback_parse(q: str, entity_names: list[str]) -> ParsedQuery`.

- [ ] **Step 1: Viết test với client giả:** `test_valid_json_parsed`; `test_timeout_uses_fallback` (client ngủ 2 giây => trả `fallback_parse`, tổng thời gian < 1,5 giây); `test_invalid_json_uses_fallback`; `test_fallback_splits_on_va` ("hồ bơi và công viên nước ở Phú Quốc" => 2 `SubQuery`, cả hai chứa "Phú Quốc"); `test_schema_has_no_dollar_schema_key` (assert `"$schema" not in schema`).
- [ ] **Step 2: FAIL. Step 3: cài đặt.** Prompt ép JSON ngắn, tối đa 4 thực thể mỗi ý, `thinking_budget=0`. Step 4: PASS.
- [ ] **Step 5: Gọi thật 3 query mẫu** (đơn ý, đa ý, đa bước), in kết quả và thời gian. Ghi vào `docs/data-notes.md`.
- [ ] **Step 6: Commit** `feat: gemini query parsing with fallback`.

### Task 7: Cổng kiểm tra thực thể

**Files:**
- Create: `src/retrieval/grounding.py`, `tests/test_grounding.py`

**Interfaces:**
- Consumes: `ParsedQuery`; danh mục thực thể từ manifest.
- Produces: `class EntityCatalog.from_manifest(df) -> EntityCatalog`; `ground(parsed: ParsedQuery, catalog) -> ParsedQuery` (bỏ thực thể không có trong catalog, khớp mờ không dấu bằng RapidFuzz; giữ nguyên `vi/en` của sub-query).

- [ ] **Step 1: Viết test:** `test_unknown_entity_removed`; `test_accent_insensitive_match` ("ao ba ba" khớp "Áo bà ba"); `test_all_removed_keeps_original_text_query` (không còn thực thể nào thì giữ `vi` gốc để vẫn search bằng văn bản).
- [ ] **Step 2: FAIL. Step 3: cài đặt. Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: entity grounding gate`.

### Task 8: Hybrid search, gộp và hạn ngạch

**Files:**
- Create: `src/retrieval/search.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `Embedder`, Qdrant client, `ParsedQuery`.
- Produces: `search(parsed: ParsedQuery, embedder, client, top_k: int = 10) -> list[Hit]` với `Hit(image_id, score, entity_ids, caption, source_sub: int)`; `interleave(per_sub: list[list[Hit]], top_k: int, max_per_entity: int = 3) -> list[Hit]`.

- [ ] **Step 1: Viết test cho `interleave`:** `test_each_intent_gets_quota` (2 ý, ý A có điểm cao hơn hẳn => top 10 vẫn có >= 4 ảnh từ ý B); `test_max_per_entity` (không quá 3 ảnh cùng thực thể); `test_dedupe_same_image`; `test_fewer_than_k_when_not_enough` (không độn ảnh). Test tích hợp trên Qdrant `:memory:` nhỏ: `test_accentless_query_finds_entity` ("ao ba ba" trả ảnh thuộc thực thể áo bà ba).
- [ ] **Step 2: FAIL. Step 3: cài đặt.** Mỗi sub-query dùng `query_batch_points` với 3 prefetch (`img`, `txt`, `sparse`) hợp nhất bằng RRF `k=60`; sau đó gọi `interleave`.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: hybrid search with intent quotas`.

### Task 9: Rerank

**Files:**
- Create: `src/retrieval/rerank.py`, `tests/test_rerank.py`

**Interfaces:**
- Produces: `class Reranker`: `score(query: str, docs: list[str]) -> list[float]` (cắt `max_length=128`); `rerank_hits(query: str, hits: list[Hit], top_k: int = 10, max_pairs: int = 24) -> list[Hit]`. Giữ các ảnh bắt buộc theo hạn ngạch của Task 8 (rerank chỉ sắp xếp lại trong từng ý, không xoá ý nào khỏi top 10).

- [ ] **Step 1: Viết test:** `test_relevant_doc_scores_higher`; `test_truncates_to_128_tokens`; `test_never_scores_more_than_24_pairs`; `test_rerank_keeps_every_intent_represented`.
- [ ] **Step 2: FAIL. Step 3: cài đặt** `bge-reranker-base` (PyTorch CPU trước). Step 4: PASS.
- [ ] **Step 5: Đo thật:** in p50/p95 chấm 24 cặp trên CPU của bạn. **Điểm quyết định:** nếu p95 > 700 ms thì xuất ONNX INT8 (thêm bước vào task này), nếu vẫn > 700 ms thì tắt rerank cho query có >= 3 ý.
- [ ] **Step 6: Commit** `feat: cross-encoder rerank`.

### Task 10: API

**Files:**
- Create: `src/retrieval/api.py`, `tests/test_api.py`

**Interfaces:**
- Produces: `POST /search {query: str, top_k: int = 10}` => `{results: [{image_id, url, caption, entities, score}], query_type, timings_ms: {...}}`; `POST /embed {texts: list[str]}` => `{vectors: list[list[float]]}`; `POST /rerank {query, documents: list[str], top_n}` => `{results: [{index, score}]}`; `GET /healthz` => `{"status": "ok"}`.

- [ ] **Step 1: Viết test (TestClient):** `test_missing_api_key_401`; `test_healthz_no_key_needed`; `test_empty_query_422`; `test_query_over_500_chars_422`; `test_search_returns_at_most_10_with_urls`; `test_response_has_timings`; `test_embed_returns_vectors_of_same_count`.
- [ ] **Step 2: FAIL. Step 3: cài đặt.** Khởi động nạp model một lần (lifespan), embed query gốc song song với lời gọi Gemini, ghi `timings_ms` từng bước. Cache LRU cho kết quả parse. Step 4: PASS.
- [ ] **Step 5: Đo end-to-end:** 30 query mẫu từ 3 loại, ghi p50/p95 vào `docs/data-notes.md`. **Điểm quyết định:** p95 > 2.500 ms thì quay lại tối ưu Task 9 trước khi làm tiếp.
- [ ] **Step 6: Commit** `feat: fastapi endpoints`.

### Task 11: Giao diện web

**Files:**
- Create: `web/index.html`; Modify: `src/retrieval/api.py` (phục vụ file tĩnh `/`)

- [ ] **Step 1: Viết test:** `test_root_serves_html` (GET `/` trả 200 và chứa `<form`).
- [ ] **Step 2: FAIL. Step 3: cài đặt** một trang: ô nhập, nút tìm, lưới ảnh, tên thực thể dưới mỗi ảnh, hiển thị thời gian, ảnh lỗi tải thì ẩn ô đó (`onerror`). Không phụ thuộc thư viện ngoài. Step 4: PASS.
- [ ] **Step 5: Mở trang thủ công, thử 3 loại query, xác nhận ảnh hiện.**
- [ ] **Step 6: Commit** `feat: search ui`.

### Task 12: Docker và compose

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`

- [ ] **Step 1:** `Dockerfile` multi-stage (builder cài phụ thuộc, runtime slim), người dùng không phải root, `HEALTHCHECK` gọi `/healthz`. Models và snapshot nạp từ volume, không đóng gói vào image.
- [ ] **Step 2:** `docker-compose.yml` gồm `api`, `qdrant` (volume snapshot), `caddy` (chỉ bật cho deploy). `depends_on` với `condition: service_healthy`.
- [ ] **Step 3: Chạy `docker compose up -d --build`; `curl http://localhost:8000/healthz`.** Kỳ vọng `{"status":"ok"}`. Ghi kích thước image và thời gian khởi động vào `docs/data-notes.md`.
- [ ] **Step 4: Commit** `feat: docker and compose`.

### Task 13: Test set và đánh giá

**Files:**
- Create: `eval/build_testset.py`, `eval/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces: `build_testset(manifest_path, n_per_type: int, seed: int) -> list[dict]` (`{query, type, gold_entity_ids}`); `recall_at_k(results: list[list[str]], gold: list[set[str]], k: int = 10) -> float`; `mrr_at_k(...) -> float`; `intent_coverage(...) -> float`.

- [ ] **Step 1: Viết test:** `test_recall_hit_if_any_in_top10`; `test_recall_zero_when_none`; `test_mrr_first_hit_rank_2` (kỳ vọng 0.5); `test_testset_queries_not_copied_from_captions` (không query nào trùng nguyên văn một caption đã index).
- [ ] **Step 2: FAIL. Step 3: cài đặt.** Single: tên thực thể. Multi-intent: ghép 2 thực thể cùng địa bàn bằng "và". Multi-hop: dùng danh mục (nhánh phân loại) làm khái niệm; mẫu multi-hop và một phần hai loại còn lại để bạn duyệt tay (file `eval/review.csv`). Chia nhóm theo `entity_id`.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Chạy `python eval/evaluate.py` cho `image-only` và `hybrid`;** ghi bảng Recall@10, MRR@10, độ phủ ý, theo từng loại, vào `docs/eval-report.md`.
- [ ] **Step 6: Commit** `feat: test set and evaluation`.

### Task 14: Load test k6

**Files:**
- Create: `loadtest/script.js`, `loadtest/run.sh`

- [ ] **Step 1:** `script.js` gọi `/search`, `/embed`, `/rerank` với `X-API-Key` từ biến môi trường, danh sách query lấy từ test set; ngưỡng `p(95)<2800` cho `/search` và `http_req_failed<0.01`; `handleSummary` xuất `reports/loadtest-report.html` và `.json`.
- [ ] **Step 2: Chạy `bash loadtest/run.sh`** trong compose. Kỳ vọng thoát 0 và có file báo cáo. Lưu thông lượng (req/s) và p50/p95/p99 vào `docs/perf-report.md`.
- [ ] **Step 3: Commit** `feat: reproducible k6 load test`.

### Task 15: CI/CD, deploy, tài liệu

**Files:**
- Create: `.github/workflows/ci.yml`, `README.md`, `docs/architecture.md`

- [ ] **Step 1:** Workflow `ci.yml`: chạy `pytest`, build image, `docker compose up` với dữ liệu mẫu nhỏ, k6 smoke 30 giây, tải báo cáo lên artifacts. Job `deploy` (chỉ nhánh `main`) qua SSH khi đã có VPS và secrets.
- [ ] **Step 2:** `README.md`: các bước chạy từ đầu (lấy dữ liệu, `.env`, index, compose), lệnh chạy test, đánh giá, load test.
- [ ] **Step 3:** `docs/architecture.md`: sơ đồ Mermaid kiến trúc, luồng dữ liệu, schema payload Qdrant, cách xử lý 3 loại query, lý do chọn, hạn chế.
- [ ] **Step 4: Đẩy lên GitHub, xác nhận workflow xanh.** Kỳ vọng tick xanh trên tab Actions.
- [ ] **Step 5 (nếu có VPS):** deploy, trỏ tên miền qua Caddy, chạy lại k6 trên máy chủ thật và lưu số liệu.
- [ ] **Step 6: Commit** `docs: readme, architecture, ci`.

---

## Lịch gắn với task

| Ngày | Task |
|---|---|
| 18-19/10 | 1, 2, 3, 4, 5 |
| 20/10 | 6, 7, 8, 9, 10, 11, 12 |
| 21/10 | 13, 14, 15 |
| 22/10 | Dự phòng và nộp |

Các mục "làm nếu còn thời gian" trong spec chỉ bắt đầu sau Task 15.
