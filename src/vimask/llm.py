"""Thin Ollama client wrapper with availability probing.

All network access of the application goes through here (localhost only).
Every method degrades to ``None`` on failure so the engine can fall back to
lexical-only retrieval without the selector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config


@dataclass
class OllamaStatus:
    reachable: bool
    models: list[str] = field(default_factory=list)
    error: str = ""

    def has(self, model: str) -> bool:
        base = model.split(":")[0]
        for m in self.models:
            if m == model or (":" not in model and m.split(":")[0] == base):
                return True
        return any(m.startswith(model + ":") for m in self.models)


class OllamaClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client: Any = None
        self._status: OllamaStatus | None = None

    def _get_client(self):
        if self._client is None:
            import ollama  # imported lazily: optional at runtime

            self._client = ollama.Client(host=self.cfg.ollama_host, timeout=self.cfg.ollama_timeout)
        return self._client

    def status(self, refresh: bool = False) -> OllamaStatus:
        if self._status is not None and not refresh:
            return self._status
        try:
            resp = self._get_client().list()
            models = []
            for m in getattr(resp, "models", None) or resp.get("models", []):
                name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None) or ""
                if name:
                    models.append(name)
            self._status = OllamaStatus(True, models)
        except Exception as exc:  # connection refused, timeout, import error
            self._status = OllamaStatus(False, [], error=str(exc).splitlines()[0] if str(exc) else repr(exc))
        return self._status

    @property
    def available(self) -> bool:
        return self.status().reachable

    def embed(self, text: str) -> np.ndarray | None:
        vecs = self.embed_many([text])
        if not vecs:
            return None
        return np.asarray(vecs[0], dtype=float)

    def embed_many(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        try:
            resp = self._get_client().embed(model=self.cfg.embed_model, input=texts, keep_alive=self.cfg.keep_alive)
        except Exception:
            return None
        embeddings = getattr(resp, "embeddings", None) or (resp.get("embeddings") if isinstance(resp, dict) else None)
        if not embeddings:
            return None
        return [list(map(float, v)) for v in embeddings]

    def chat_json(self, system: str, user: str, schema: dict, *, num_predict: int | None = None) -> dict | None:
        """One chat turn constrained to ``schema``; returns the parsed object."""
        try:
            resp = self._get_client().chat(
                model=self.cfg.selector_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema,
                options={
                    "temperature": 0,
                    "seed": self.cfg.seed,
                    "num_predict": num_predict or self.cfg.num_predict,
                },
                keep_alive=self.cfg.keep_alive,
                stream=False,
            )
        except Exception:
            return None
        message = getattr(resp, "message", None) or (resp.get("message") if isinstance(resp, dict) else None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if not content:
            return None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
