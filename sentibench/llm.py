"""Provider-neutral structured-output calls (GPT through the API, or a local model via Ollama).

Every LLM classification goes through `complete_json`: a system prompt, a user
prompt and a strict JSON schema in, the raw JSON text out. The caller parses
and validates the answer; a reply that does not parse counts as a failure.

Each call also returns usage metadata: tokens, latency and cost in USD.
Adapted from ElicitSense's `llm.py`; the Anthropic path is left out because no
key is configured for this project, and adding a provider is one function.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .pricing import cost_usd, embedding_cost_usd

PROVIDERS = ("openai", "ollama")

# Sentiment prompts are short; a small context keeps the KV cache small on a 16 GB laptop.
OLLAMA_CONTEXT = 4096
OLLAMA_MAX_OUTPUT = 1024

# Transient API errors (rate limits, 5xx, timeouts) are retried with backoff;
# anything else is returned to the caller as a failed item.
RETRIES = 4


class LLMError(RuntimeError):
    pass


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines into the environment without overriding variables already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_model_spec(spec: str) -> tuple[str, str]:
    """'openai:gpt-6.1-sol' or 'ollama:qwen3:8b' -> (provider, model)."""
    provider, sep, model = spec.partition(":")
    if not sep or provider not in PROVIDERS or not model:
        raise ValueError(f"model spec must be provider:model with provider in {PROVIDERS}: {spec!r}")
    return provider, model


def _retrying(call):
    import openai

    for attempt in range(RETRIES + 1):
        try:
            return call()
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as e:
            if attempt == RETRIES:
                raise LLMError(f"API error after {RETRIES} retries: {type(e).__name__}") from e
            time.sleep(2 ** attempt * 2)


_client = None


def _openai_client():
    global _client
    if _client is None:
        import openai

        _client = openai.OpenAI(timeout=120, max_retries=0)
    return _client


def _openai(system: str, prompt: str, schema: dict, name: str, model: str,
            reasoning_effort: str | None) -> tuple[str, dict]:
    kwargs = {}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    response = _retrying(lambda: _openai_client().responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        **kwargs,
    ))
    if response.status == "incomplete":
        raise LLMError(f"response incomplete: {response.incomplete_details}")
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    raise LLMError(f"model declined the request ({part.refusal})")
    u = response.usage
    return response.output_text, {
        "provider": "openai",
        "model": response.model,
        "input_tokens": u.input_tokens,
        "cached_input_tokens": u.input_tokens_details.cached_tokens or 0,
        "output_tokens": u.output_tokens,
        "reasoning_tokens": u.output_tokens_details.reasoning_tokens or 0,
    }


def _ollama(system: str, prompt: str, schema: dict, name: str, model: str,
            reasoning_effort: str | None) -> tuple[str, dict]:
    """Local open-weight model through Ollama's native chat API, which constrains
    decoding to the JSON schema. A "+think" suffix turns on the model's reasoning mode."""
    label = model
    model, think = model.removesuffix("+think"), model.endswith("+think")
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    body = {
        "model": model,
        "stream": False,
        "format": schema,
        "think": think,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {"num_ctx": OLLAMA_CONTEXT, "num_predict": OLLAMA_MAX_OUTPUT},
    }
    request = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise LLMError(f"Ollama error {e.code}: {e.read().decode(errors='replace')[:200]}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"could not reach Ollama at {host} (is `ollama serve` running?)") from e
    except TimeoutError as e:
        raise LLMError("Ollama did not answer within 10 minutes") from e
    if data.get("done_reason") == "length":
        raise LLMError("output hit the token limit before the JSON was complete")
    if data.get("prompt_eval_count", 0) >= OLLAMA_CONTEXT:
        raise LLMError("prompt filled the context window and was truncated")
    return data["message"]["content"], {
        "provider": "ollama",
        "model": label,
        "input_tokens": data.get("prompt_eval_count", 0),
        "cached_input_tokens": 0,
        "output_tokens": data.get("eval_count", 0),
        "reasoning_tokens": 0,
    }


def complete_json(system: str, prompt: str, schema: dict, name: str, spec: str,
                  reasoning_effort: str | None = None) -> tuple[str, dict]:
    provider, model = parse_model_spec(spec)
    call = {"openai": _openai, "ollama": _ollama}[provider]
    started = time.perf_counter()
    text, meta = call(system, prompt, schema, name, model, reasoning_effort)
    meta["latency_s"] = round(time.perf_counter() - started, 3)
    meta["cost_usd"] = cost_usd(provider, meta["model"], meta["input_tokens"],
                                meta["cached_input_tokens"], meta["output_tokens"])
    return text, meta


def embed(texts: list[str], model: str = "text-embedding-3-small") -> tuple[list[list[float]], dict]:
    """One batched embeddings request. Returns vectors in input order and usage metadata."""
    started = time.perf_counter()
    response = _retrying(lambda: _openai_client().embeddings.create(model=model, input=texts))
    vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
    tokens = response.usage.prompt_tokens
    return vectors, {"model": response.model, "input_tokens": tokens,
                     "latency_s": round(time.perf_counter() - started, 3),
                     "cost_usd": embedding_cost_usd(response.model, tokens)}
