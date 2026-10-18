"""Cổng kiểm tra thực thể: bỏ thực thể do LLM sinh ra mà không có trong dữ liệu."""
import unicodedata
from dataclasses import dataclass, field, replace

import pandas as pd
from rapidfuzz import fuzz, process

from retrieval.qparse import MAX_ENTS, ParsedQuery, strip_accents

TYPO_MIN = 88  # fuzz.ratio trên cả chuỗi (không tách từ) để chịu gõ sai nhỏ
STRIPPED_PHRASE_MIN = 6  # bỏ dấu làm nhập nhằng từ ngắn (hổ/hồ/hò), nên cụm bỏ dấu phải đủ dài
PER_ENT = 3


def _fold(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold().strip()


@dataclass
class EntityCatalog:
    names: list[str]
    folded: list[str] = field(default_factory=list)  # giữ dấu
    keys: list[str] = field(default_factory=list)  # bỏ dấu
    _fpad: list[str] = field(default_factory=list)
    _kpad: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.folded = [_fold(n) for n in self.names]
        self.keys = [strip_accents(n) for n in self.names]
        self._fpad = [f" {x} " for x in self.folded]
        self._kpad = [f" {x} " for x in self.keys]

    @classmethod
    def from_manifest(cls, df: pd.DataFrame) -> "EntityCatalog":
        seen: dict[str, str] = {}
        for lst in df["entity_names"]:
            for n in (lst if lst is not None else []):
                if n:
                    seen.setdefault(_fold(str(n)), str(n))
        return cls(names=list(seen.values()))

    def match(self, ent: str) -> list[str]:
        fe, ke = _fold(ent), strip_accents(ent).strip()
        if len(fe) < 2:
            return []
        pf, pk = f" {fe} ", f" {ke} "
        hits: dict[int, int] = {}
        for i, (a, b) in enumerate(zip(self._fpad, self._kpad, strict=True)):
            if pf in a:  # cụm liền kề, đúng dấu
                hits[i] = 100 if fe == self.folded[i] else 95
            elif len(ke) >= STRIPPED_PHRASE_MIN and pk in b:  # cụm liền kề, bỏ dấu
                hits[i] = 100 if ke == self.keys[i] else 92
        for _, score, i in process.extract(ke, self.keys, scorer=fuzz.ratio,
                                           score_cutoff=TYPO_MIN, limit=PER_ENT):
            hits[i] = max(hits.get(i, 0), int(score))
        ranked = sorted(hits, key=lambda i: (-hits[i], len(self.names[i])))
        return [self.names[i] for i in ranked[:PER_ENT]]


def ground(parsed: ParsedQuery, catalog: EntityCatalog) -> ParsedQuery:
    subs = []
    for s in parsed.subs:
        found: list[str] = []
        for e in s.ents:
            for n in catalog.match(e):
                if n not in found:
                    found.append(n)
        subs.append(replace(s, ents=found[:MAX_ENTS]))
    return replace(parsed, subs=subs)
