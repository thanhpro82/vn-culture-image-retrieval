"""Tải ảnh, chuẩn hoá WebP, đo tỉ lệ link sống và gom ảnh gần trùng."""
import argparse
import asyncio
import io
import random
import statistics
import time
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
from PIL import Image

TIMEOUT_S = 15
RETRIES = 2  # thử lại 2 lần sau lần đầu
PER_HOST = 6
RETRY_STATUS = {429, 500, 502, 503, 504}
UA = "Mozilla/5.0 (compatible; vn-culture-image-retrieval-capstone/0.1)"


async def _fetch(session: aiohttp.ClientSession, url: str) -> tuple[bytes | None, str]:
    reason = "unknown"
    for attempt in range(RETRIES + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                if r.status in RETRY_STATUS and attempt < RETRIES:
                    reason = f"http_{r.status}"
                    await asyncio.sleep(0.05 * (attempt + 1))
                    continue
                if r.status != 200:
                    return None, f"http_{r.status}"
                raw = await r.read()
        except TimeoutError:
            reason = "timeout"
            continue
        except aiohttp.ClientError as e:
            reason = f"net_{type(e).__name__}"
            continue
        try:
            Image.open(io.BytesIO(raw)).load()  # ảnh cụt byte sẽ lỗi ở đây
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            return None, "decode"
        return raw, "ok"
    return None, reason


async def fetch_image(session: aiohttp.ClientSession, url: str) -> bytes | None:
    return (await _fetch(session, url))[0]


def to_webp(raw: bytes, max_side: int = 640) -> bytes:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, "WEBP", quality=80)
    return out.getvalue()


def _session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        headers={"User-Agent": UA}, connector=aiohttp.TCPConnector(limit=64, limit_per_host=PER_HOST)
    )


async def _probe(urls: list[str]) -> dict:
    lat: list[float] = []
    sizes: list[int] = []
    ok = 0
    sem = asyncio.Semaphore(32)
    reasons: dict[str, int] = {}
    async with _session() as s:
        async def one(u):
            nonlocal ok
            async with sem:
                t = time.perf_counter()
                raw, why = await _fetch(s, u)
                lat.append((time.perf_counter() - t) * 1000)
                reasons[why] = reasons.get(why, 0) + 1
                if raw:
                    ok += 1
                    sizes.append(len(to_webp(raw)))
        await asyncio.gather(*(one(u) for u in urls))
    return {
        "n": len(urls), "ok": ok, "alive_rate": ok / len(urls) if urls else 0.0,
        "p50_ms": statistics.median(lat) if lat else 0.0,
        "mean_kb": (sum(sizes) / len(sizes) / 1024) if sizes else 0.0,
        "reasons": reasons,
    }


def probe(manifest_path: Path, n: int, seed: int) -> dict:
    urls = pd.read_parquet(manifest_path, columns=["url"])["url"].tolist()
    sample = random.Random(seed).sample(urls, min(n, len(urls)))
    return asyncio.run(_probe(sample))


async def _download_all(df: pd.DataFrame, out_dir: Path, concurrency: int) -> pd.DataFrame:
    sem = asyncio.Semaphore(concurrency)
    res: list[dict] = []
    async with _session() as s:
        async def one(image_id, url):
            dst = out_dir / f"{image_id}.webp"
            if dst.exists():
                res.append({"image_id": image_id, "ok": True, "reason": "cached"})
                return
            async with sem:
                raw, why = await _fetch(s, url)
            if raw:
                dst.write_bytes(await asyncio.to_thread(to_webp, raw))
            res.append({"image_id": image_id, "ok": bool(raw), "reason": why})
        await asyncio.gather(*(one(i, u) for i, u in zip(df["image_id"], df["url"], strict=True)))
    return pd.DataFrame(res)


def download_all(manifest_path: Path, out_dir: Path, concurrency: int = 48) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(manifest_path, columns=["image_id", "url"])
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # xen kẽ domain, tránh nghẽn per-host
    status = asyncio.run(_download_all(df, out_dir, concurrency))
    status.to_parquet(out_dir.parent / "status.parquet", index=False)


def near_duplicates(phashes: dict[str, str], max_hamming: int) -> list[set[str]]:
    """Gom ảnh có khoảng cách Hamming của pHash 64-bit <= max_hamming (union-find)."""
    ids = list(phashes)
    if len(ids) < 2:
        return []
    arr = np.array([int(phashes[i], 16) for i in ids], dtype=np.uint64)
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    block = 512
    for start in range(0, len(arr), block):
        chunk = arr[start:start + block]
        dist = np.bitwise_count(chunk[:, None] ^ arr[None, :])
        rows, cols = np.nonzero(dist <= max_hamming)
        for r, c in zip(rows, cols, strict=True):
            a, b = start + int(r), int(c)
            if a < b:
                parent[find(a)] = find(b)
    groups: dict[int, set[str]] = {}
    for i, name in enumerate(ids):
        groups.setdefault(find(i), set()).add(name)
    return [g for g in groups.values() if len(g) > 1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("manifest")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    d = sub.add_parser("all")
    d.add_argument("manifest")
    d.add_argument("out_dir")
    d.add_argument("--concurrency", type=int, default=48)
    args = ap.parse_args()
    if args.cmd == "probe":
        print(probe(Path(args.manifest), args.n, args.seed))
    else:
        download_all(Path(args.manifest), Path(args.out_dir), args.concurrency)
