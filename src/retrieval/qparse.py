"""Hiểu query bằng một lần gọi Gemini Flash Lite; lỗi hoặc quá hạn thì dùng fallback quy tắc."""
import asyncio
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal, Protocol

import aiohttp

QT = ("single", "multi_intent", "multi_hop")
MAX_SUBS, MAX_ENTS = 4, 4

# Không có khoá "$schema": Gemini trả 400 nếu có.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "qt": {"type": "STRING", "enum": list(QT)},
        "subs": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "vi": {"type": "STRING"}, "en": {"type": "STRING"},
            "ents": {"type": "ARRAY", "items": {"type": "STRING"}}},
            "required": ["vi", "en", "ents"]}},
    },
    "required": ["qt", "subs"],
}

PROMPT = (
    "Tìm ảnh văn hoá, địa danh Việt Nam. Trả JSON: qt = single | multi_intent (nhiều ý ghép bằng \"và\") "
    "| multi_hop (cần suy ra quan hệ hoặc khái niệm rộng); subs = mỗi ý một phần tử "
    "{vi: câu tiếng Việt giữ địa danh chung, en: mô tả thị giác tiếng Anh ngắn, "
    "ents: tối đa 4 thực thể cụ thể có thật (multi_hop thì suy ra từ quan hệ hoặc khái niệm)}. Truy vấn: {q}"
)


@dataclass
class SubQuery:
    vi: str
    en: str = ""
    ents: list[str] = field(default_factory=list)
    loc: str = ""
    cat: str = ""


@dataclass
class ParsedQuery:
    qt: Literal["single", "multi_intent", "multi_hop"]
    subs: list[SubQuery]


class LLMClient(Protocol):
    async def generate(self, prompt: str, schema: dict) -> str: ...


def build_request(prompt: str, schema: dict = RESPONSE_SCHEMA) -> dict:
    """thinkingBudget=0 bị Gemini 3.5 từ chối (HTTP 400, đo 2026-10-18); dùng thinkingLevel=minimal."""
    return {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema,
                                 "temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}}


class GeminiClient:
    """Giữ một phiên aiohttp dùng lại giữa các lần gọi để khỏi bắt tay TLS lại (~100-200 ms)."""

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://generativelanguage.googleapis.com"):
        self.api_key, self.model, self.base_url = api_key, model, base_url
        self._session: aiohttp.ClientSession | None = None

    async def generate(self, prompt: str, schema: dict) -> str:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        async with self._session.post(url, json=build_request(prompt, schema),
                                      headers={"x-goog-api-key": self.api_key}) as r:
            r.raise_for_status()
            data = await r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").casefold()


_SPLIT = re.compile(r"\s+(?:và|với)\s+|\s*,\s*", re.IGNORECASE)
_LOC = re.compile(r"\s+(?:ở|tại)\s+(.+)$", re.IGNORECASE)


def fallback_parse(q: str, entity_names: list[str]) -> ParsedQuery:
    q = unicodedata.normalize("NFC", q).strip()
    loc_m = _LOC.search(q)
    loc = loc_m.group(1).strip() if loc_m else ""
    body = q[: loc_m.start()] if loc_m else q
    parts = [p.strip() for p in _SPLIT.split(body) if p.strip()][:MAX_SUBS] or [q]
    norm = {strip_accents(n): n for n in entity_names}
    subs = []
    for p in parts:
        vi = f"{p} ở {loc}" if loc else p
        key = strip_accents(vi)
        ents = [orig for n, orig in norm.items() if n and n in key][:MAX_ENTS]
        subs.append(SubQuery(vi=vi, en=vi, ents=ents, loc=loc))
    return ParsedQuery("single" if len(subs) == 1 else "multi_intent", subs)


def _from_json(text: str) -> ParsedQuery | None:
    try:
        d = json.loads(text)
        if d["qt"] not in QT:
            return None
        subs = [SubQuery(vi=str(s["vi"]), en=str(s.get("en") or s["vi"]),
                         ents=[str(e) for e in s.get("ents", [])][:MAX_ENTS],
                         loc=str(s.get("loc") or ""), cat=str(s.get("cat") or ""))
                for s in d["subs"][:MAX_SUBS] if str(s.get("vi", "")).strip()]
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return ParsedQuery(d["qt"], subs) if subs else None


async def parse_query(q: str, client: LLMClient, timeout_ms: int = 1600,
                      entity_names: list[str] | None = None) -> ParsedQuery:
    try:
        text = await asyncio.wait_for(client.generate(PROMPT.replace("{q}", q), RESPONSE_SCHEMA),
                                      timeout=timeout_ms / 1000)
        parsed = _from_json(text)
    except (TimeoutError, aiohttp.ClientError, KeyError, IndexError, ValueError, TypeError, AttributeError):
        parsed = None  # Gemini không bao giờ được làm hỏng tìm kiếm: mọi lỗi đều rơi về fallback
    return parsed or fallback_parse(q, entity_names or [])
