"""Team-wide settings. Read from `signet.toml` (the nearest one at or above the
current directory, or the file named by SIGNET_CONFIG), then SIGNET_* environment
variables, then command-line flags. Everything that should stay fixed across
streams lives here, so it's set in one place and logged with every run.

The TTS_* variables from the Track 2 harness still work as aliases."""

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

CONFIG_NAME = "signet.toml"

# setting -> environment variables, first match wins
ENV = {
    "experiment": ("SIGNET_EXPERIMENT",),
    "runs_dir": ("SIGNET_RUNS_DIR",),
    "models": ("SIGNET_MODELS", "TTS_MODELS"),
    "judge_model": ("SIGNET_JUDGE_MODEL", "TTS_JUDGE_MODEL"),
    "embed_model": ("SIGNET_EMBED_MODEL", "TTS_EMBED_MODEL"),
    "reasoning_override": ("SIGNET_REASONING_EFFORT", "TTS_REASONING_EFFORT"),
    "judge_reasoning_override": ("SIGNET_JUDGE_REASONING_EFFORT", "TTS_JUDGE_REASONING_EFFORT"),
}


@dataclass(frozen=True)
class Settings:
    workspace: Path
    # Default experiment for the CLI: a path (relative to the workspace) or a folder name under experiments/.
    experiment: str | None = None
    # Where run folders go. None = results/<the experiment's RESULTS_SUBDIR>.
    runs_dir: Path | None = None
    cache_dir: Path | None = None
    # Models under test, run in parallel. DeepSeek V4.1 Flash + Gemma 4 31B is the team's
    # default pair; every other model in models.MODELS is one setting away.
    models: tuple = ("deepseek-flash", "gemma-4-31b")
    judge_model: str = "gpt-6-luna"
    embed_model: str = "openai/text-embedding-3-small"
    reasoning_override: str | None = None
    judge_reasoning_override: str | None = None
    temperature: float = 0.0
    top_k: int = 5
    seeds: tuple = (0, 1, 2)
    max_tool_steps: int = 4          # model turns allowed before a final answer is forced
    workers: int = 8
    # Extra or overridden model-table entries from [models_table.<alias>] in signet.toml.
    extra_models: dict = field(default_factory=dict)
    config_file: Path | None = None

    @property
    def cache(self) -> Path:
        return self.cache_dir or self.workspace / ".signet-cache"


def find_config(start: Path | None = None) -> Path | None:
    if os.environ.get("SIGNET_CONFIG"):
        return Path(os.environ["SIGNET_CONFIG"]).resolve()
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / CONFIG_NAME).is_file():
            return d / CONFIG_NAME
    return None


def _split(value: str) -> tuple:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def load(start: Path | None = None) -> Settings:
    path = find_config(start)
    raw = tomllib.loads(path.read_text()) if path else {}
    workspace = path.parent if path else (start or Path.cwd()).resolve()
    values = {"workspace": workspace, "config_file": path}
    for key in ("experiment", "judge_model", "embed_model", "reasoning_override", "judge_reasoning_override",
                "temperature", "top_k", "max_tool_steps", "workers"):
        if key in raw:
            values[key] = raw[key]
    for key in ("models", "seeds"):
        if key in raw:
            values[key] = tuple(raw[key])
    for key in ("runs_dir", "cache_dir"):
        if key in raw:
            values[key] = workspace / raw[key]
    values["extra_models"] = {alias: {**spec, "price": tuple(spec["price"]) if spec.get("price") else None}
                              for alias, spec in raw.get("models_table", {}).items()}
    for key, names in ENV.items():
        val = next((os.environ[n] for n in names if os.environ.get(n)), None)
        if val is None:
            continue
        if key == "models":
            values[key] = _split(val)
        elif key == "runs_dir":
            values[key] = Path(val).resolve()
        else:
            values[key] = val
    return Settings(**values)


_current: Settings | None = None


def get() -> Settings:
    global _current
    if _current is None:
        _current = load()
    return _current


def configure(settings: Settings | None = None, **overrides) -> Settings:
    """Replace the active settings (CLI flags, tests). Returns the new settings."""
    global _current
    _current = replace(settings or get(), **overrides)
    return _current


def reset():
    global _current
    _current = None
