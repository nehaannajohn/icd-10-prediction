"""Bedrock wrapper: plain-JSON replies with retries, temperature only where supported, every attempt traced."""
import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from src.common.config import load_config
from src.common.trace import Trace

_NO_TEMPERATURE: set[str] = set()      # model ids that rejected `temperature` (learned at runtime)
_CACHE_POINT = {"cachePoint": {"type": "default"}}
USAGE_KEYS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens")


class LLMError(RuntimeError):
    """Raised when a reply is still invalid after all retries."""


@dataclass
class LLMResult:
    data: dict                         # parsed (and checked) JSON
    raw: str                           # final raw reply text
    usage: dict = field(default_factory=dict)   # summed over attempts: inputTokens, outputTokens
    attempts: int = 1


_CLIENT, _CLIENT_LOCK = None, threading.Lock()


def client():
    """One Bedrock client shared by all threads (created once, under a lock).
    A per-thread client would come from a different boto3 Session with its own exception classes."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
                raise RuntimeError("AWS_PROFILE and AWS_REGION must be set in the environment (see README)")
            _CLIENT = boto3.Session().client("bedrock-runtime", config=Config(
                retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=300))
        return _CLIENT


def parse_json(text: str) -> dict:
    """Parse a JSON object from a reply; tolerates ```json fences and text around the object."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in reply")
        return json.loads(candidate[start:end + 1])


def _content(content: str | list[dict], cache: bool) -> list[dict]:
    """Message content -> Converse blocks. A str becomes one text block. A list of parts
    ({"text": ...} or {"cache": True}) keeps the text; cache markers become cachePoint blocks
    when `cache` is on and are dropped otherwise."""
    if isinstance(content, str):
        return [{"text": content}]
    blocks = []
    for part in content:
        if part.get("cache"):
            if cache:
                blocks.append(_CACHE_POINT)
        else:
            blocks.append({"text": part["text"]})
    return blocks


def _converse(model_id: str, system: str, messages: list[dict], temperature: float | None, max_tokens: int,
              cache: bool = False):
    """Returns (response, temperature actually sent or None). With `cache`, a cachePoint follows the
    system prompt and replaces each {"cache": True} marker in messages."""
    cfg = {"maxTokens": max_tokens}
    if temperature is not None and model_id not in _NO_TEMPERATURE:
        cfg["temperature"] = temperature
    kwargs = {"modelId": model_id, "system": [{"text": system}] + ([_CACHE_POINT] if cache else []),
              "inferenceConfig": cfg,
              "messages": [{"role": m["role"], "content": _content(m["content"], cache)} for m in messages]}
    try:
        return client().converse(**kwargs), cfg.get("temperature")
    except ClientError as e:
        if (e.response.get("Error", {}).get("Code") != "ValidationException"
                or "temperature" not in str(e).lower() or "temperature" not in cfg):
            raise
        _NO_TEMPERATURE.add(model_id)
        cfg.pop("temperature")
        return client().converse(**kwargs), None


def call(model: str, system: str, messages: list[dict], trace: Trace, stage: str,
         check: Callable[[dict], None] | None = None, temperature: float | None = None) -> LLMResult:
    """Call `model` (a name from config.models) and return its JSON reply.

    `messages`: [{"role": "user" | "assistant", "content": str | [{"text": ...} | {"cache": True}, ...]}, ...]
    Cache markers become Bedrock cachePoints for models in llm.prompt_cache_models, else are dropped.
    `check(data)`: optional; raise ValueError to reject the reply (e.g. schema problems) and trigger a retry.
    On a bad reply the model's answer and the error are appended to the conversation and the call is retried,
    up to llm.max_retries times. Every attempt is a separate trace event named `<stage>.llm`.
    """
    config = load_config()
    model_id = config["models"][model]
    max_tokens = config["prediction"]["max_output_tokens"]
    temperature = config["prediction"]["temperature"] if temperature is None else temperature
    cache = model in config["llm"].get("prompt_cache_models", [])
    convo, usage = list(messages), dict.fromkeys(USAGE_KEYS, 0)
    for attempt in range(1, config["llm"]["max_retries"] + 2):
        with trace.stage(f"{stage}.llm", attempt=attempt, model=model, model_id=model_id,
                         prompt_cache=cache, system=system, messages=convo) as event:
            response, sent = _converse(model_id, system, convo, temperature, max_tokens, cache)
            event.outputs["temperature_sent"] = sent
            raw = "".join(c.get("text", "") for c in response["output"]["message"]["content"])
            for k in usage:
                usage[k] += response["usage"].get(k, 0)
            event.outputs.update(raw=raw, usage=response["usage"], stop_reason=response.get("stopReason"))
            try:
                data = parse_json(raw)
                if check:
                    check(data)
            except (ValueError, json.JSONDecodeError) as e:
                event.status, event.error = "retry", f"{type(e).__name__}: {e}"
                convo = convo + [{"role": "assistant", "content": raw},
                                 {"role": "user", "content": f"Your reply was invalid: {e}. "
                                                             "Reply again with only the corrected JSON object."}]
                continue
            event.outputs["parsed"] = data
            return LLMResult(data=data, raw=raw, usage=usage, attempts=attempt)
    raise LLMError(f"{stage}: invalid reply from {model} after {attempt} attempts: {event.error}")
