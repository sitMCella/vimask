# vimask

Offline terminal companion that answers "how do I do X in Vim?" with verified
keystrokes, a one-line description and a before/after example. See
`docs/design.md` for the architecture: the LLM never writes keystrokes; every
answer is rendered from a knowledge-base entry that was executed in a real
headless Vim and Neovim by `vimask kb verify`.

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.12 | runtime |
| Docker + Docker Compose | any recent | runs Ollama |

## Setup

```sh
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/vimask kb build        # verify KB, build help-excerpt index (+ vector index if Ollama runs)
.venv/bin/vimask doctor          # shows which mode will run
.venv/bin/vimask                 # REPL
.venv/bin/vimask "join lines"    # one-shot
```

Optional (full mode):

```bash
docker compose up -d
docker compose ps
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:3b-instruct
docker compose exec ollama ollama list

.venv/bin/vimask kb build 
```

Without Ollama the tool runs lexical-only with the same correctness guarantees.

## Commands

- `vimask [--mode insert|visual] [--flavor nvim] [--lexical-only] [--plain] "question"`
- `vimask kb verify|build|check|stats|import`
- `vimask eval [--setup lexical|retrieval|full|all] -v`
- `vimask doctor`

Examples assume `shiftwidth=4 expandtab nostartofline` and no user config.
Tests: `.venv/bin/python -m pytest`.
