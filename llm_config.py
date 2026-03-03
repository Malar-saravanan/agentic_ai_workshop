"""
LLM helper — build Groq LLMs and auto-retry on rate limits.

Retry: cycles all 3 models first, then waits and repeats.
  8b-instant → 70b-versatile → gemma2-9b → (wait) → repeat
"""

import os, re, time
from crewai import LLM

os.environ["LITELLM_NUM_RETRIES"] = "1"

DEFAULT_MODELS = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",
    "groq/gemma2-9b-it",
]


def get_models():
    """Read model list from .env or fall back to defaults."""
    raw = os.getenv("LLM_MODEL_CANDIDATES", "")
    if raw.strip():
        return [m.strip() for m in raw.split(",") if m.strip()]
    return DEFAULT_MODELS


def build_llm(model="", temperature=0.7, max_tokens=500, force_react=False):
    """Create a CrewAI LLM. force_react=True disables function calling."""
    if not model:
        model = get_models()[0]
    llm = LLM(model=model, temperature=temperature, max_tokens=max_tokens)
    if force_react:
        llm.supports_function_calling = lambda: False
    return llm


def _is_retryable(e):
    s = str(e).lower()
    return "rate_limit" in s or "rate limit" in s or "429" in s \
        or "tool_use_failed" in s or "none or empty" in s


def _parse_wait(error_msg):
    match = re.search(r"try again in (\d+\.?\d*)s", error_msg.lower())
    return float(match.group(1)) + 1 if match else 10


def run_with_retry(crew_fn, llm=None, rounds=2):
    """Run crew_fn, cycling all models on failure. Waits between rounds."""
    models = get_models()
    original = llm.model if llm else None
    last_err = None

    for rd in range(rounds):
        if rd > 0 and last_err:
            wait = _parse_wait(str(last_err))
            print(f"  [All models failed — waiting {wait:.0f}s, round {rd+1}]")
            time.sleep(wait)

        for model in models:
            if llm:
                llm.model = model
            try:
                result = crew_fn()
                if llm and original:
                    llm.model = original
                return result
            except Exception as e:
                if _is_retryable(e):
                    print(f"  [{model.split('/')[-1]} failed — trying next]")
                    last_err = e
                else:
                    if llm and original:
                        llm.model = original
                    raise

    if llm and original:
        llm.model = original
    raise last_err
