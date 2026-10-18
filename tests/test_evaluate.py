import numpy as np
import pandas as pd
from eval.build_testset import build_testset
from eval.evaluate import intent_coverage, mrr_at_k, recall_at_k


def test_recall_hit_if_any_in_top10():
    res = [["a", "b", "c"] + ["x"] * 7 + ["g"]]  # g ở hạng 11
    assert recall_at_k(res, [{"c"}], k=10) == 1.0
    assert recall_at_k(res, [{"g"}], k=10) == 0.0


def test_recall_zero_when_none_and_mean_over_queries():
    assert recall_at_k([["a"], ["b"]], [{"a"}, {"z"}], k=10) == 0.5


def test_mrr_first_hit_rank_2():
    assert mrr_at_k([["x", "g", "y"]], [{"g"}], k=10) == 0.5


def test_intent_coverage_is_fraction_of_intents_hit():
    res = [["a", "b"]]
    assert intent_coverage(res, [[{"a"}, {"z"}]], k=10) == 0.5
    assert intent_coverage(res, [[{"a"}, {"b"}]], k=10) == 1.0


def _manifest():
    rows = []
    for e in range(12):
        region = "Nam Bộ" if e < 6 else "Bắc Bộ"
        culture = "Trang phục" if e < 6 else "Lễ hội"
        for k in range(4):
            rows.append({"image_id": f"{e:02d}{k:02d}", "entity_ids": np.array([f"E{e}"]),
                         "entity_names": np.array([f"Thực thể số {e} (phụ chú)"]),
                         "caption": f"mô tả bí mật {e}-{k}", "region": region, "culture_type": culture,
                         "branch": "B"})
    return pd.DataFrame(rows)


def test_testset_has_three_types_with_nonempty_gold():
    ts = build_testset(_manifest(), n_per_type=3, seed=1)
    assert {t["type"] for t in ts} == {"single", "multi_intent", "multi_hop"}
    assert all(all(len(g) > 0 for g in t["intents"]) for t in ts)
    assert [len(t["intents"]) for t in ts if t["type"] == "multi_intent"] == [2, 2, 2]


def test_testset_queries_not_copied_from_captions():
    m = _manifest()
    ts = build_testset(m, n_per_type=3, seed=1)
    caps = {c.casefold() for c in m["caption"]}
    assert not any(t["query"].casefold() in caps for t in ts)


def test_testset_is_deterministic_and_strips_parenthetical():
    a = build_testset(_manifest(), n_per_type=3, seed=7)
    b = build_testset(_manifest(), n_per_type=3, seed=7)
    assert a == b
    assert all("(" not in t["query"] for t in a)
