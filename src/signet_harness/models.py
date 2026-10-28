"""The model table. Every model is called through OpenRouter (https://openrouter.ai/api/v1).

alias -> OpenRouter model id (what requests send), the canonical slug that id
pointed to when the table was written, USD per 1M tokens (input, output),
reasoning effort to request (None = the model's default, "on" = switch reasoning
on for models that have no effort levels), tier, and optionally "provider": an
OpenRouter provider-routing object sent with every request. OpenRouter only
accepts the id in requests; the canonical slug is its permanent dated name.
Preflight fails if an id now resolves to a different canonical slug, so results
don't shift mid-run without anyone noticing; the served model is logged on every
call either way. Reasoning tokens are billed as output. Prices are from
OpenRouter's /api/v1/models on Oct 27, 2026: preflight prints the live price, and
OpenRouter's reported cost is what gets logged. This table is only for estimates
and as a fallback.

Switching models is a settings change, not a code change: `models = [...]` or
`judge_model = "..."` in signet.toml, SIGNET_MODELS / SIGNET_JUDGE_MODEL, or
--models on the command line. New models go in [models_table.<alias>] in
signet.toml.
"""

from . import settings

MODELS = {
    # Budget tier: what the team is paying for now. DeepSeek + Gemma is the default pair.
    "deepseek-flash": {"slug": "deepseek/deepseek-v4.1-flash", "canonical": "deepseek/deepseek-v4.1-flash-20260910",
                       "price": (0.14, 0.42), "reasoning": None, "tier": "budget"},
    # The US open-weight model (swapped in for Nemotron at the project sponsor's request,
    # Oct 28, 2026). Thinking is an on/off switch with no effort levels and an unclear
    # default, so it's switched on explicitly. Many providers serve it at fp4; the
    # provider filter keeps requests on fp8 or better ("unknown" endpoints are left out).
    # Track 2 found it almost never calls optional tools (1 of 45 A3 runs).
    "gemma-4-31b": {"slug": "google/gemma-4-31b-it", "canonical": "google/gemma-4-31b-it-20260402",
                    "price": (0.20, 0.40), "reasoning": "on", "tier": "budget",
                    "provider": {"quantizations": ["fp8", "bf16", "fp16", "fp32"]}},
    # The previous US open-weight model (Track 2 runs/pilot). Reasoning is on by default.
    # Its listing doesn't confirm tool calling everywhere; preflight checks supported_parameters.
    "nemotron-3.5-super-vl": {"slug": "nvidia/nemotron-3.5-super-vl", "canonical": "nvidia/nemotron-3.5-super-vl-20261015",
                              "price": (0.15, 0.50), "reasoning": None, "tier": "budget"},
    # GLM 5.3 always reasons and defaults to effort "max", which multiplies output
    # tokens; "low" keeps it inside the budget.
    "glm-5.3": {"slug": "z-ai/glm-5.3", "canonical": "z-ai/glm-5.3-20260816",
                "price": (0.0699, 1.40), "reasoning": "low", "tier": "budget"},
    # The judge, at low effort to keep scoring cheap.
    "gpt-6-luna": {"slug": "openai/gpt-6-luna", "canonical": "openai/gpt-6-luna-20260922",
                   "price": (0.10, 0.50), "reasoning": "low", "tier": "budget"},
    # Funded tier: ready for when more credits are approved.
    "gpt-6-sol": {"slug": "openai/gpt-6-sol", "canonical": "openai/gpt-6-sol-20260922",
                  "price": (2.00, 10.00), "reasoning": "low", "tier": "funded"},
    "gemini-4-argon": {"slug": "google/gemini-4-argon", "canonical": "google/gemini-4-argon-20261013",
                       "price": (2.00, 10.00), "reasoning": None, "tier": "funded"},
    "gemini-3.8-flash": {"slug": "google/gemini-3.8-flash", "canonical": "google/gemini-3.8-flash-20260902",
                         "price": (0.825, 4.125), "reasoning": None, "tier": "funded"},
}


def table() -> dict:
    """The built-in table plus any [models_table] entries from signet.toml."""
    return {**MODELS, **settings.get().extra_models}


def resolve(model: str) -> dict:
    """Alias, OpenRouter id or canonical slug -> {"alias", "slug", "price", "reasoning", "tier", "canonical"?, "provider"?}.
    "slug" is always the id that requests send. Unknown ids pass through unpinned."""
    models = table()
    if model in models:
        return {"alias": model, **models[model]}
    known = next((a for a, m in models.items() if model in (m["slug"], m.get("canonical"))), None)
    if known:
        return {"alias": known, **models[known]}
    return {"alias": model, "slug": model, "price": None, "reasoning": None, "tier": "custom"}


def price_for(slug: str):
    return next((m["price"] for m in table().values() if m["slug"] == slug), None)
