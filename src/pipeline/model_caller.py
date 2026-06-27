import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from src.pipeline.validator import build_correction_message, parse_json
from src.schema import LLMExtraction

_JSON_RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON. "
    "Please output ONLY the JSON object, starting with { and ending with }, no markdown."
)


@dataclass
class ModelResponse:
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


def _provider_config(provider: str) -> dict:
    """Per-provider connection config, keyed by the model-string prefix.

    Any OpenAI-compatible runner (vllm, LM Studio, Mistral local, …) can be added
    here without code changes — only base_url / api_key / json mechanism differ.
    """
    if provider == "openai":
        return {
            "base_url": None,  # default OpenAI endpoint
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "timeout": 120.0,
            "structured_outputs": True,  # native response_format=LLMExtraction
            "json_format": None,
        }
    if provider == "ollama":
        return {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "api_key": "ollama",  # ollama ignores the key but the SDK requires one
            "timeout": 1800.0,  # local generation can be slow on long records
            "structured_outputs": False,
            "json_format": "json",  # ollama grammar-constrained JSON output
        }
    # Default: any other OpenAI-compatible local runner, configured via env.
    return {
        "base_url": os.environ.get("LLM_BASE_URL"),
        "api_key": os.environ.get("LLM_API_KEY", "none"),
        "timeout": 600.0,
        "structured_outputs": False,
        "json_format": None,
    }


def _is_reasoning_model(model_name: str) -> bool:
    """Return True for OpenAI o-series reasoning models (o1, o3, o4, etc.)."""
    return model_name.lower().startswith(("o1", "o3", "o4"))


def _strip_provider(model: str) -> str:
    """Drop the provider prefix: 'openai:o4-mini' → 'o4-mini', 'ollama:mistral:7b' → 'mistral:7b'."""
    return model.split(":", 1)[1] if ":" in model else model


def _base_kwargs(model_name: str, max_tokens: int) -> dict:
    if _is_reasoning_model(model_name):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens, "temperature": 0.0}


def _usage(response) -> tuple[int, int]:
    try:
        return response.usage.prompt_tokens, response.usage.completion_tokens
    except AttributeError:
        return 0, 0


# Models that require the native Ollama /api/chat endpoint for grammar-constrained
# JSON decoding. The /v1 OpenAI-compat endpoint silently ignores format:json for
# these models, causing free-text output. Substring match, case-insensitive.
_NATIVE_JSON_MODELS = ("openbiollm",)


def _needs_native_json(model_name: str) -> bool:
    name = model_name.lower()
    return any(m in name for m in _NATIVE_JSON_MODELS)


def _ollama_root(base_url: Optional[str]) -> str:
    """Native-API root for Ollama, derived from the OpenAI-compatible base URL."""
    url = base_url or "http://localhost:11434/v1"
    return url[: -len("/v1")] if url.endswith("/v1") else url.rstrip("/")


def _ollama_post(url: str, payload: dict, timeout: float) -> dict:
    """POST a JSON payload to Ollama's native API and return the parsed response.

    Isolated so tests can mock the HTTP boundary without a running server.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _invoke_ollama(
    prompt: str, model_name: str, max_tokens: int, num_ctx: Optional[int], cfg: dict
) -> tuple[str, int, int, float]:
    """Single native ``/api/chat`` call. ``num_ctx`` is omitted when None so the
    server default applies. ``prompt_eval_count`` reflects the tokens actually
    ingested — lower than the prompt size means the model truncated it."""
    options: dict = {"num_predict": max_tokens, "temperature": 0.0}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",  # ollama grammar-constrained JSON output
        "options": options,
    }
    url = _ollama_root(cfg["base_url"]) + "/api/chat"
    t0 = time.monotonic()
    response = _ollama_post(url, payload, cfg["timeout"])
    latency_ms = (time.monotonic() - t0) * 1000
    raw = response.get("message", {}).get("content", "") or ""
    in_tok = response.get("prompt_eval_count", 0) or 0
    out_tok = response.get("eval_count", 0) or 0
    return raw, in_tok, out_tok, latency_ms


def _call_ollama_native_with_retry(
    prompt: str, model_name: str, max_tokens: int, num_ctx: Optional[int], model_label: str, cfg: dict
) -> ModelResponse:
    """Ollama native path. Mirrors ``_call_local_with_retry`` — one schema/JSON
    correction retry — but over the native endpoint that honors ``num_ctx``."""
    raw, in_tok, out_tok, latency = _invoke_ollama(prompt, model_name, max_tokens, num_ctx, cfg)

    correction = _correction_for(raw)
    if correction is not None:
        raw2, in2, out2, lat2 = _invoke_ollama(
            prompt + correction, model_name, max_tokens, num_ctx, cfg
        )
        raw, in_tok, out_tok, latency = raw2, in_tok + in2, out_tok + out2, latency + lat2

    return ModelResponse(raw, in_tok, out_tok, latency, model_label)


def _call_structured(
    client: OpenAI, prompt: str, model_name: str, max_tokens: int, model_label: str
) -> ModelResponse:
    """OpenAI path — native structured outputs. The API cannot return an invalid
    schema, so no schema-shape retry is needed."""
    t0 = time.monotonic()
    completion = client.chat.completions.parse(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format=LLMExtraction,
        **_base_kwargs(model_name, max_tokens),
    )
    latency_ms = (time.monotonic() - t0) * 1000
    raw = completion.choices[0].message.content or ""
    in_tok, out_tok = _usage(completion)
    return ModelResponse(raw, in_tok, out_tok, latency_ms, model_label)


def _invoke_local(
    client: OpenAI, prompt: str, model_name: str, max_tokens: int, cfg: dict,
    num_ctx: Optional[int] = None,
) -> tuple[str, int, int, float]:
    t0 = time.monotonic()
    kwargs = _base_kwargs(model_name, max_tokens)
    extra_body: dict = {}
    if cfg.get("json_format"):
        extra_body["format"] = cfg["json_format"]
    if num_ctx is not None:
        # Ollama accepts num_ctx via options in the OpenAI-compat /v1 endpoint.
        extra_body["options"] = {"num_ctx": num_ctx}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    raw = response.choices[0].message.content or ""
    in_tok, out_tok = _usage(response)
    return raw, in_tok, out_tok, latency_ms


def _correction_for(raw: str) -> Optional[str]:
    """Decide whether a local response needs a retry and what to tell the model.

    Returns None if the response parses and validates, otherwise the correction
    text to append to the prompt: a JSON-parse hint (item 10) or a schema-specific
    ValidationError message (item 11).
    """
    data = parse_json(raw)
    if data is None:
        return _JSON_RETRY_SUFFIX
    try:
        LLMExtraction.model_validate(data)
    except ValidationError as e:
        return build_correction_message(e)
    return None


def _call_local_with_retry(
    client: OpenAI, prompt: str, model_name: str, max_tokens: int, model_label: str, cfg: dict,
    num_ctx: Optional[int] = None,
) -> ModelResponse:
    raw, in_tok, out_tok, latency = _invoke_local(client, prompt, model_name, max_tokens, cfg, num_ctx)

    correction = _correction_for(raw)
    if correction is not None:
        raw2, in2, out2, lat2 = _invoke_local(
            client, prompt + correction, model_name, max_tokens, cfg, num_ctx
        )
        raw, in_tok, out_tok, latency = raw2, in_tok + in2, out_tok + out2, latency + lat2

    return ModelResponse(raw, in_tok, out_tok, latency, model_label)


def call_model(
    prompt: str, model: str, max_tokens: int = 4096, num_ctx: Optional[int] = None
) -> ModelResponse:
    provider = model.split(":")[0]
    cfg = _provider_config(provider)
    model_name = _strip_provider(model)

    if provider == "ollama" and _needs_native_json(model_name):
        # Native /api/chat path: honours format:json grammar constraint.
        # Note: calls on this path are not captured by mlflow.openai.autolog().
        return _call_ollama_native_with_retry(prompt, model_name, max_tokens, num_ctx, model, cfg)

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=cfg["timeout"])
    if cfg["structured_outputs"]:
        return _call_structured(client, prompt, model_name, max_tokens, model)
    return _call_local_with_retry(client, prompt, model_name, max_tokens, model, cfg, num_ctx)
