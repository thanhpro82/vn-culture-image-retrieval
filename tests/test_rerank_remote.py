import json

import httpx
import pytest

from retrieval.rerank import HttpRerankClient, RemoteFallbackScorer


class Local:
    def __init__(self):
        self.pairs_seen = []

    def score_pairs(self, pairs):
        self.pairs_seen.append(list(pairs))
        return [0.1] * len(pairs)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def remote_ok(calls):
    def call(query, docs):
        calls.append((query, list(docs)))
        return [0.9] * len(docs)
    return call


def test_uses_remote_when_available():
    calls, local = [], Local()
    s = RemoteFallbackScorer(local, remote_ok(calls), rpm=40)
    out = s.score("q", ["a", "b"])
    assert out == [0.9, 0.9]
    assert calls == [("q", ["a", "b"])]
    assert local.pairs_seen == []


def test_pairs_grouped_per_query_keep_order():
    calls = []

    def call(query, docs):
        calls.append(query)
        return [float(len(query) * 10 + i) for i, _ in enumerate(docs)]

    s = RemoteFallbackScorer(Local(), call, rpm=40)
    out = s.score_pairs([("a", "d0"), ("bb", "d1"), ("a", "d2"), ("bb", "d3")])
    assert out == [10.0, 20.0, 11.0, 21.0]
    assert calls == ["a", "bb"]  # mỗi query khác nhau một lần gọi


def test_falls_back_to_local_on_error_and_opens_circuit():
    clock, local = Clock(), Local()
    attempts = []

    def boom(query, docs):
        attempts.append(1)
        raise httpx.ConnectError("tunnel chết")

    s = RemoteFallbackScorer(local, boom, rpm=40, cooldown_s=30, clock=clock)
    assert s.score("q", ["a"]) == [0.1]
    assert s.score("q", ["a"]) == [0.1]  # mạch đang mở: không thử lại endpoint ngoài
    assert len(attempts) == 1
    clock.t = 31.0  # hết thời gian chờ: thử lại
    s.score("q", ["a"])
    assert len(attempts) == 2
    assert s.fallbacks == 3


def test_rate_limit_falls_back_without_calling_remote():
    clock, calls, local = Clock(), [], Local()
    s = RemoteFallbackScorer(local, remote_ok(calls), rpm=2, clock=clock)
    s.score("q", ["a"])
    s.score("q", ["a"])
    assert s.score("q", ["a"]) == [0.1]  # hết 2 lượt trong phút này
    assert len(calls) == 2
    clock.t = 61.0
    assert s.score("q", ["a"]) == [0.9]


def test_group_needing_more_requests_than_budget_goes_local():
    clock, calls, local = Clock(), [], Local()
    s = RemoteFallbackScorer(local, remote_ok(calls), rpm=1, clock=clock)
    out = s.score_pairs([("a", "x"), ("b", "y")])  # cần 2 lượt, chỉ có 1
    assert out == [0.1, 0.1]
    assert calls == []


def test_wrong_length_from_remote_is_treated_as_failure():
    local = Local()
    s = RemoteFallbackScorer(local, lambda q, d: [0.5], rpm=40)
    assert s.score("q", ["a", "b"]) == [0.1, 0.1]


def test_http_client_sends_utf8_and_orders_by_index():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.2},
                                                      {"index": 0, "relevance_score": 0.7}]})

    c = HttpRerankClient("https://x.example/", "sk-test", "jina-reranker-v2", timeout_s=1,
                         transport=httpx.MockTransport(handler))
    assert c("thủ đô", ["Hà Nội", "Paris"]) == [0.7, 0.2]
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["query"] == "thủ đô"
    assert seen["body"]["model"] == "jina-reranker-v2"


def test_http_client_raises_on_error_status():
    c = HttpRerankClient("https://x.example", "k", "m", timeout_s=1,
                         transport=httpx.MockTransport(lambda r: httpx.Response(429, json={})))
    with pytest.raises(httpx.HTTPStatusError):
        c("q", ["a"])
