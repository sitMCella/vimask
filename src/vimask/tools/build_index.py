"""Index builder: vector index over the KB (needs Ollama) and the help
excerpt index (needs a Vim or Neovim runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import Config
from ..help_index import HelpIndex, runtime_doc_dir
from ..kb import KnowledgeBase
from ..llm import OllamaClient
from ..retrieval import build_vector_index
from ..vimrun import find_editor


def build_vectors(cfg: Config, kb: KnowledgeBase, client: OllamaClient) -> bool:
    status = client.status(refresh=True)
    if not status.reachable:
        print(f"  Ollama not reachable at {cfg.ollama_host}: skipping vector index", file=sys.stderr)
        return False
    if not status.has(cfg.embed_model):
        print(f"  embedding model {cfg.embed_model} not pulled: `ollama pull {cfg.embed_model}`", file=sys.stderr)
        return False

    def embed_many(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch = 64
        for i in range(0, len(texts), batch):
            vecs = client.embed_many(texts[i : i + batch])
            if vecs is None:
                raise RuntimeError("embedding request failed")
            out.extend(vecs)
        return out

    vi = build_vector_index(kb, embed_many, cfg.embed_model)
    vi.save(cfg.index_dir)
    print(f"  vector index: {len(vi.ids)} entries, dim {vi.matrix.shape[1]}, model {cfg.embed_model}", file=sys.stderr)
    return True


def build_help_index(cfg: Config, flavor_preference: tuple[str, ...] = ("vim", "nvim")) -> Path | None:
    for fl in flavor_preference:
        exe = find_editor(fl, cfg.vim_path if fl == "vim" else cfg.nvim_path)
        doc = runtime_doc_dir(fl, exe) if exe else None
        if doc:
            hi = HelpIndex.build(doc)
            hi.save(cfg.index_dir, source=str(doc))
            print(f"  help excerpt index: {len(hi.chunks)} chunks from {doc}", file=sys.stderr)
            return doc
    print("  no Vim/Neovim runtime found: help excerpt index not built", file=sys.stderr)
    return None
