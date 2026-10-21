import pytest

from retrieval import search as S
from retrieval.qparse import ParsedQuery, SubQuery


def P(qt, vi="di tích Hải Dương", en="Hai Duong heritage", ents=("Bánh đậu xanh", "Đền Kiếp Bạc")):
    return ParsedQuery(qt=qt, subs=[SubQuery(vi=vi, en=en, ents=list(ents))])


def test_current_policy_keeps_the_old_behaviour_for_comparison():
    assert S._units(P("single"), "current") == [("Hai Duong heritage", "di tích Hải Dương Bánh đậu xanh Đền Kiếp Bạc")]
    assert S._units(P("multi_hop"), "current") == [("Bánh đậu xanh", "Bánh đậu xanh"), ("Đền Kiếp Bạc", "Đền Kiếp Bạc")]


def test_default_policy_is_both():
    assert S.UNIT_POLICY == "both"
    assert S._units(P("single")) == [("Hai Duong heritage", "di tích Hải Dương")]
    assert S._units(P("multi_hop"))[0] == ("Hai Duong heritage", "di tích Hải Dương")


def test_vi_only_drops_entities_from_single_text_but_keeps_multi_hop_split():
    assert S._units(P("single"), "vi_only") == [("Hai Duong heritage", "di tích Hải Dương")]
    assert S._units(P("multi_hop"), "vi_only") == S._units(P("multi_hop"), "current")


def test_mh_keep_query_adds_original_unit_first():
    u = S._units(P("multi_hop"), "mh_keep_query")
    assert u[0] == ("Hai Duong heritage", "di tích Hải Dương")
    assert ("Bánh đậu xanh", "Bánh đậu xanh") in u and len(u) == 3
    assert S._units(P("single"), "mh_keep_query") == S._units(P("single"), "current")


def test_both_combines_the_two_changes():
    assert S._units(P("single"), "both") == [("Hai Duong heritage", "di tích Hải Dương")]
    assert S._units(P("multi_hop"), "both")[0] == ("Hai Duong heritage", "di tích Hải Dương")


def test_ignore_ents_treats_every_query_as_plain_text():
    assert S._units(P("multi_hop"), "ignore_ents") == [("Hai Duong heritage", "di tích Hải Dương")]


def test_units_are_capped():
    p = ParsedQuery(qt="multi_hop", subs=[SubQuery(vi="q", en="q", ents=[f"e{i}" for i in range(9)])])
    assert len(S._units(p, "mh_keep_query")) == S.MAX_UNITS


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        S._units(P("single"), "không-có")


def test_module_default_is_used_when_no_policy_given(monkeypatch):
    monkeypatch.setattr(S, "UNIT_POLICY", "ignore_ents")
    assert S._units(P("multi_hop")) == [("Hai Duong heritage", "di tích Hải Dương")]


def _mi(*subs):
    from retrieval.qparse import ParsedQuery, SubQuery
    return ParsedQuery("multi_intent", [SubQuery(vi=v, en=e, ents=list(ents)) for v, e, ents in subs])


def test_anchor_gop_ydieu_kien_khong_co_thuc_the_vao_y_chinh():
    p = _mi(("Một ngôi đình cổ có mái ngói", "old communal house", ["Đình"]), ("sân rộng", "wide yard", []))
    assert S._units(p, "anchor") == [("old communal house wide yard", "Một ngôi đình cổ có mái ngói sân rộng")]
    # chính sách mặc định giữ nguyên hành vi cũ
    assert len(S._units(p, "both")) == 2


def test_anchor_giu_nguyen_khi_moi_y_deu_co_thuc_the_hoac_khong_co_y_nao_co_thuc_the():
    full = _mi(("áo dài", "ao dai", ["Áo dài"]), ("nón lá", "conical hat", ["Nón lá"]))
    assert S._units(full, "anchor") == S._units(full, "both") and len(S._units(full, "anchor")) == 2
    none = _mi(("chiều muộn", "late", []), ("mưa nhẹ", "rain", []))
    assert S._units(none, "anchor") == S._units(none, "both")


def test_anchor_chi_ap_dung_cho_multi_intent():
    from retrieval.qparse import ParsedQuery, SubQuery
    mh = ParsedQuery("multi_hop", [SubQuery(vi="a", en="a", ents=["X"]), SubQuery(vi="b", en="b", ents=[])])
    assert S._units(mh, "anchor") == S._units(mh, "both")
