"""Configuration and filesystem paths.

Settings live in ``~/.config/vimask/config.toml``; every key is optional.
Generated data (vector index, help excerpt index, missing.log) lives under
``~/.local/share/vimask`` unless ``data_dir`` overrides it.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Repository root when running from a checkout; the KB ships next to the code.
REPO_ROOT = PACKAGE_ROOT.parent.parent


def default_kb_dir() -> Path:
    env = os.environ.get("VIMASK_KB")
    if env:
        return Path(env).expanduser()
    for candidate in (REPO_ROOT / "kb", PACKAGE_ROOT / "kb"):
        if candidate.is_dir():
            return candidate
    return REPO_ROOT / "kb"


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "vimask" / "config.toml"


def default_data_dir() -> Path:
    env = os.environ.get("VIMASK_DATA")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "vimask"


@dataclass
class Config:
    # Ollama
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    selector_model: str = "qwen2.5:3b-instruct"
    ollama_timeout: float = 20.0
    keep_alive: int = -1
    num_predict: int = 128
    seed: int = 7

    # Retrieval
    top_k: int = 8
    lexical_weight: float = 0.6
    vector_weight: float = 0.4
    # Fast path (skip the selector) when fused top-1 score and margin exceed
    # these. Lexical-only mode uses the conservative pair.
    fast_score: float = 0.80
    fast_margin: float = 0.20
    fast_score_lexical_only: float = 0.85
    fast_margin_lexical_only: float = 0.25
    # Below this fused top-1 score the query goes straight to fallback.
    fallback_score: float = 0.30
    # Without the LLM selector nothing can disambiguate, so answers need a
    # stronger retrieval score than this or they fall back to "not found".
    answer_score_no_selector: float = 0.55
    # Score boost for entries whose mode matches the session mode.
    mode_boost: float = 0.05

    # Paths
    kb_dir: Path = field(default_factory=default_kb_dir)
    data_dir: Path = field(default_factory=default_data_dir)

    # Editors used by the verifier and composer. Empty string = search PATH.
    vim_path: str = "vim"
    nvim_path: str = "nvim"
    editor_timeout: float = 5.0

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def missing_log(self) -> Path:
        return self.data_dir / "missing.log"

    @property
    def history_file(self) -> Path:
        return self.data_dir / "history"


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    cfg = Config()
    if not path.is_file():
        return cfg
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    # Accept both flat keys and [section] tables; tables are flattened.
    flat: dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    valid = {f.name: f.type for f in fields(Config)}
    for key, value in flat.items():
        if key not in valid:
            continue
        if key in ("kb_dir", "data_dir"):
            value = Path(str(value)).expanduser()
        setattr(cfg, key, value)
    return cfg
