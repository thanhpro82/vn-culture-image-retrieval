from eval.make_review_sheet import build_sheet, gold_entities

NAMES = {"a": ["Hát chèo"], "b": ["Hát chèo", "Chèo cổ"], "c": ["Rối nước"], "d": ["Rối nước"]}


def T(i, kind, intents):
    return {"id": f"{kind[:2]}-{i:04d}", "type": kind, "query": f"q{i}", "intents": intents}


def test_gold_entities_sorted_by_image_count():
    assert gold_entities(["a", "b", "c"], NAMES)[0] == ("Hát chèo", 2)


def test_sheet_samples_per_type_and_is_reproducible():
    tests = [T(i, "single", [["a", "b"]]) for i in range(10)] + [T(20 + i, "multi_intent", [["a"], ["c", "d"]]) for i in range(10)]
    s1, s2 = build_sheet(tests, NAMES, 3, 0), build_sheet(tests, NAMES, 3, 0)
    assert list(s1["id"]) == list(s2["id"])
    assert (s1["type"] == "single").sum() == 3 and (s1["type"] == "multi_intent").sum() == 3
    assert (s1["verdict"] == "").all()


def test_multi_intent_rows_describe_each_intent():
    row = build_sheet([T(1, "multi_intent", [["a"], ["c", "d"]])], NAMES, 1, 0).iloc[0]
    assert "Ý 1" in row["gold"] and "Ý 2" in row["gold"] and "Rối nước (2)" in row["gold"]


def test_unknown_image_ids_do_not_crash():
    assert gold_entities(["zzz"], NAMES) == []
