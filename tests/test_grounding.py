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


# ---------- cờ độ tin cậy dựa trên bằng chứng ở kết quả ----------
def test_entities_supported_when_all_words_appear_in_one_result_text():
    from retrieval.grounding import entities_supported
    texts = ["Nhà thờ chính tòa Phát Diệm (Nhà thờ đá)", "Chùa Một Cột"]
    assert entities_supported(["Nhà thờ Phát Diệm"], texts)            # từ chen ở giữa vẫn tính
    assert entities_supported(["Phát Diệm nhà thờ"], texts)             # thứ tự từ không quan trọng
    assert entities_supported(["Sư tử", "Nhà thờ Phát Diệm"], texts)    # chỉ cần một thực thể được hậu thuẫn


def test_entities_not_supported_when_words_are_split_across_results_or_absent():
    from retrieval.grounding import entities_supported
    texts = ["Tranh Ngũ hổ", "Miếu Ông Hổ", "Làng tranh Đông Hồ"]
    assert not entities_supported(["Hổ Đông Dương"], texts)             # các từ nằm ở các tên khác nhau
    assert not entities_supported(["VinFast"], texts)
    assert not entities_supported([], texts)
    assert not entities_supported(["Nhà thờ"], [])
