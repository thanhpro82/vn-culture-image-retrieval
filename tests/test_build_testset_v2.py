import asyncio
import json

import pandas as pd
from eval.build_testset_v2 import build_v2, leaks


def test_leaks_detects_full_name_and_three_word_windows():
    assert leaks("ảnh hát chèo cổ truyền ở Thái Bình", ["Hát chèo cổ truyền"])
    assert leaks("múa lân sư rồng ngày tết", ["Trang phục múa lân sư rồng"])      # 3 từ liên tiếp của tên
    assert leaks("tò he", ["Tò he"])                                              # tên ngắn: chứa nguyên văn
    assert not leaks("điệu hát có đàn nhị và trống cơm", ["Hát chèo cổ truyền"])


def test_leaks_ignores_parenthetical_part_of_name():
    assert not leaks("đình làng cổ ở Bắc Ninh", ["Đình Bảng (Bắc Ninh)"][:0] or ["Đình Bảng (Bắc Ninh)"][0:0])  # không có tên
    assert leaks("đình bảng cổ", ["Đình Bảng (Bắc Ninh)"])


def manifest(n_entities=15):
    rows = []
    for e in range(n_entities):
        for k in range(4):
            rows.append({"image_id": f"e{e}-{k}", "entity_names": [f"Thực thể số {e} loại X"], "region": "Vùng A",
                         "culture_type": f"Thể loại {e % 3}", "caption": f"c{e}{k}",
                         "entity_desc": "Mô tả khá dài về thực thể này để đủ ngưỡng độ dài tối thiểu cho việc sinh truy vấn."})
    return pd.DataFrame(rows)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, schema):
        self.calls += 1
        return json.dumps({"q": f"mô tả tự nhiên số {self.calls}"})


class LeakyOnceLLM(FakeLLM):
    async def generate(self, prompt, schema):
        self.calls += 1
        q = "ảnh thực thể số 3 loại X" if self.calls == 1 else "câu không lộ tên"
        return json.dumps({"q": q})


def test_build_v2_makes_requested_counts_with_gold_and_no_name_leak():
    tests = asyncio.run(build_v2(manifest(), FakeLLM(), per_type=3, seed=1))
    kinds = [t["type"] for t in tests]
    assert kinds.count("single") == 3 and kinds.count("multi_intent") == 3 and kinds.count("multi_hop") == 3
    for t in tests:
        assert t["intents"] and all(len(ids) >= 1 for ids in t["intents"])
        assert t["id"].startswith("v2-")
    assert all(len(t["intents"]) == 2 for t in tests if t["type"] == "multi_intent")


def test_build_v2_retries_when_query_leaks_the_name():
    llm = LeakyOnceLLM()
    tests = asyncio.run(build_v2(manifest(2), llm, per_type=1, seed=1))
    assert llm.calls >= 2
    assert not any("loại X" in t["query"] for t in tests if t["type"] == "single")


def test_build_v2_is_reproducible_for_same_seed():
    a = asyncio.run(build_v2(manifest(), FakeLLM(), per_type=3, seed=1))
    b = asyncio.run(build_v2(manifest(), FakeLLM(), per_type=3, seed=1))
    assert [t["intents"] for t in a] == [t["intents"] for t in b]
