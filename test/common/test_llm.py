import copy
import threading

import pytest
from botocore.exceptions import ClientError

from src.common import llm
from src.common.trace import Trace


class FakeClient:
    """Returns queued replies; can reject `temperature` like the Anthropic/OpenAI models do (real botocore ClientError)."""
    def __init__(self, replies, rejects_temperature=False):
        self.replies, self.rejects_temperature, self.calls = list(replies), rejects_temperature, []

    def converse(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.rejects_temperature and "temperature" in kwargs["inferenceConfig"]:
            raise ClientError({"Error": {"Code": "ValidationException",
                                         "Message": "`temperature` is deprecated for this model."}}, "Converse")
        text = self.replies.pop(0)
        return {"output": {"message": {"content": [{"text": text}]}},
                "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 3,
                          "cacheWriteInputTokens": 2}, "stopReason": "end_turn"}


@pytest.fixture
def fake(monkeypatch):
    def install(replies, **kw):
        client = FakeClient(replies, **kw)
        monkeypatch.setattr(llm, "client", lambda: client)
        monkeypatch.setattr(llm, "_NO_TEMPERATURE", set())
        real = llm.load_config()                       # test-only model: accepts temperature, no prompt caching
        monkeypatch.setattr(llm, "load_config", lambda: {**real, "models": {**real["models"], "test-model": "test.model"}})
        return client
    return install


def test_parse_json_handles_fences_and_surrounding_text():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Here you go: {"a": [1, 2]} done') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        llm.parse_json("no json here")


def test_bad_json_is_retried_with_error_and_both_attempts_traced(fake):
    client = fake(["not json", '{"codes": []}'])
    trace = Trace()
    result = llm.call("test-model", "sys", [{"role": "user", "content": "hi"}], trace, "tool_agent.turn_1")
    assert result.data == {"codes": []} and result.attempts == 2
    assert [(e.attempt, e.status) for e in trace.events] == [(1, "retry"), (2, "ok")]
    assert "invalid" in client.calls[1]["messages"][-1]["content"][0]["text"]
    assert result.usage == {"inputTokens": 20, "outputTokens": 10, "cacheReadInputTokens": 6,
                            "cacheWriteInputTokens": 4}


def test_check_failure_triggers_retry(fake):
    fake(['{"codes": "oops"}', '{"codes": []}'])
    def check(d):
        if not isinstance(d["codes"], list):
            raise ValueError("codes must be a list")
    result = llm.call("test-model", "sys", [{"role": "user", "content": "hi"}], Trace(), "s", check=check)
    assert result.attempts == 2


def test_raises_after_max_retries(fake):
    fake(["bad"] * 3)                                  # max_retries=2 -> 3 attempts
    with pytest.raises(llm.LLMError):
        llm.call("test-model", "sys", [{"role": "user", "content": "hi"}], Trace(), "s")


def test_temperature_dropped_and_remembered_for_models_that_reject_it(fake):
    client = fake(['{"a": 1}', '{"a": 2}'], rejects_temperature=True)
    t1, t2 = Trace(), Trace()
    llm.call("claude-opus-5-5", "sys", [{"role": "user", "content": "hi"}], t1, "s")
    llm.call("claude-opus-5-5", "sys", [{"role": "user", "content": "hi"}], t2, "s")
    assert ["temperature" in c["inferenceConfig"] for c in client.calls] == [True, False, False]
    assert [t.events[0].outputs["temperature_sent"] for t in (t1, t2)] == [None, None]   # what was actually sent


def test_temperature_sent_recorded_when_supported(fake):
    fake(['{"a": 1}'])
    trace = Trace()
    llm.call("test-model", "sys", [{"role": "user", "content": "hi"}], trace, "s")
    assert trace.events[0].outputs["temperature_sent"] == 0.7


def test_other_validation_errors_are_not_swallowed(fake, monkeypatch):
    client = fake([])
    def boom(**kw):
        raise ClientError({"Error": {"Code": "ValidationException", "Message": "input too long"}}, "Converse")
    monkeypatch.setattr(client, "converse", boom)
    with pytest.raises(ClientError):
        llm.call("test-model", "sys", [{"role": "user", "content": "hi"}], Trace(), "s")


def test_client_is_shared_across_threads(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setattr(llm, "_CLIENT", None)
    got = []
    threads = [threading.Thread(target=lambda: got.append(llm.client())) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len({id(c) for c in got}) == 1


PARTS = [{"role": "user", "content": [{"text": "ICL"}, {"cache": True}, {"text": "ENC"}, {"cache": True},
                                      {"text": "STEP"}]}]


def test_cache_points_sent_for_cache_models(fake):
    client = fake(['{"a": 1}'])
    llm.call("claude-sonnet-5-5", "sys", PARTS, Trace(), "s")
    sent = client.calls[0]
    assert sent["system"][-1] == {"cachePoint": {"type": "default"}}
    assert [list(b) for b in sent["messages"][0]["content"]] == [["text"], ["cachePoint"], ["text"], ["cachePoint"], ["text"]]


def test_cache_markers_stripped_for_other_models(fake):
    client = fake(['{"a": 1}'])
    llm.call("test-model", "sys", PARTS, Trace(), "s")
    sent = client.calls[0]
    assert sent["system"] == [{"text": "sys"}]
    assert [b["text"] for b in sent["messages"][0]["content"]] == ["ICL", "ENC", "STEP"]


def test_usage_sums_cache_fields_over_attempts(fake):
    fake(["bad", '{"a": 1}'])
    r = llm.call("claude-sonnet-5-5", "sys", PARTS, Trace(), "s")
    assert r.usage == {"inputTokens": 20, "outputTokens": 10, "cacheReadInputTokens": 6, "cacheWriteInputTokens": 4}


@pytest.mark.integration
def test_real_bedrock_call_returns_json():
    trace = Trace()
    result = llm.call("claude-opus-5-5", "Reply with JSON only.",
                      [{"role": "user", "content": 'Return {"ok": true}'}], trace, "smoke")
    assert result.data == {"ok": True} and trace.events[-1].outputs["usage"]["inputTokens"] > 0
