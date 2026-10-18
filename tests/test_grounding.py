import pandas as pd

from retrieval.grounding import EntityCatalog, ground
from retrieval.qparse import ParsedQuery, SubQuery

DF = pd.DataFrame({"entity_names": [["Áo bà ba cách tân"], ["Lễ hội Leo cột mỡ"], ["Chiến khu D (Đông Nam Bộ)"]]})
CAT = EntityCatalog.from_manifest(DF)


def _pq(ents, vi="truy vấn"):
    return ParsedQuery("multi_hop", [SubQuery(vi=vi, en="q", ents=ents)])


def test_unknown_entity_removed():
    out = ground(_pq(["Sư tử", "Lễ hội Leo cột mỡ"]), CAT)
    assert out.subs[0].ents == ["Lễ hội Leo cột mỡ"]


def test_accent_insensitive_match():
    out = ground(_pq(["ao ba ba"]), CAT)
    assert out.subs[0].ents == ["Áo bà ba cách tân"]


def test_typo_still_matches():
    out = ground(_pq(["Lễ hội Leo cot mo"]), CAT)
    assert out.subs[0].ents == ["Lễ hội Leo cột mỡ"]


def test_all_removed_keeps_original_text_query():
    out = ground(_pq(["Sư tử", "Cá voi"], vi="động vật sống trong rừng"), CAT)
    assert out.subs[0].ents == [] and out.subs[0].vi == "động vật sống trong rừng"
    assert out.qt == "multi_hop"


def test_matched_names_are_deduplicated():
    out = ground(_pq(["Chiến khu D", "chien khu d"]), CAT)
    assert out.subs[0].ents == ["Chiến khu D (Đông Nam Bộ)"]


def test_scattered_tokens_do_not_match():
    cat = EntityCatalog.from_manifest(pd.DataFrame({"entity_names": [
        ["Đền Hùng Phú Thọ giỗ Tổ quốc lễ"], ["Nhà thờ Sóc Trăng"], ["Đồ sứ ký kiểu hiệu Tự Đức"]]}))
    assert ground(_pq(["Phú Quốc", "Nha Trang", "sư tử"]), cat).subs[0].ents == []


def test_short_accented_word_does_not_match_other_accents():
    cat = EntityCatalog.from_manifest(pd.DataFrame({"entity_names": [["Múa hổ"], ["Hồ Gươm"], ["Hò Huế"]]}))
    assert ground(_pq(["hổ"]), cat).subs[0].ents == ["Múa hổ"]
