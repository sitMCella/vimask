# vimask — Design Document

Status: Draft v0.2
Date: 2026-11-09

Changes in v0.2 (design review): hybrid lexical + embedding retrieval with a
real degraded mode; reverse lookup path; mode as boost instead of filter;
two-stage selector with margin-based confidence; composer `arg` slot and
catalogue stage; placeholder policy for parameterized commands; schema
additions (`setup_keys`, verification tiers, cursor marker); verifier
details (`normal!`, keycode translation, per-flavor tags, restricted mode);
verbatim `:help` excerpts in fallback; light multi-turn context; eval set
moved to milestone 1.

## 1. Overview

`vimask` is a terminal chat application that answers "how do I do X in Vim?"
questions with the exact keystroke sequence, a one-line explanation, and a
before/after example showing the effect on a buffer. It runs fully offline on an
Apple Silicon (M1) Mac, using a local LLM served by [Ollama](https://ollama.com).

It is meant to be used side by side with Vim (tmux split, a second terminal
tab, or Vim's own `:terminal`) as a cheat sheet and working companion.

Example session:

```
vimask> delete everything inside the parentheses

  di(      Delete inside the nearest enclosing ( ).

  before:  print(foo, ba┃r)
  after:   print(┃)

  see:     :help i(
```

## 2. Requirements

### 2.1 Functional

| ID   | Requirement |
|------|-------------|
| F1   | Interactive terminal chat (REPL) that accepts natural-language questions. |
| F2   | Answer with the keystroke combination for the requested functionality. |
| F3   | Answers are concise: keystrokes, one-line description, example, help tag. |
| F4   | Include a before/after example of the buffer when the command has a visible text effect. |
| F5   | Answers must be correct. When the tool does not know, it says so instead of guessing. |
| F6   | Optional context: current mode (normal/insert/visual), and Vim vs Neovim flavour. |
| F7   | Optional one-shot CLI usage (`vimask "join lines"`) for use from inside Vim. |
| F8   | Reverse lookup: "what does `ciw` do?" answers with the entry for `ciw`. |
| F9   | Works without the LLM selector (retrieval only) and without Ollama at all (lexical only), with reduced recall but no reduction in correctness. |

### 2.2 Non-functional

| ID   | Requirement |
|------|-------------|
| N1   | Runs locally on an Apple M1 Mac (8 GB minimum, 16 GB+ recommended), no network at query time. |
| N2   | Response latency under ~2 s for typical queries on M1 with the model already loaded. Cold start (model load) is reported separately and excluded. |
| N3   | Zero hallucinated keystrokes: every keystroke shown must come from a verified source. |
| N4   | Small footprint: one Python package plus Ollama and one or two small models. |

## 3. Key design decision: the LLM never writes keystrokes

Requirement F5/N3 ("no hallucinations") cannot be met by asking a small local
model to generate Vim commands from its own memory — 3B–8B models routinely
invent mappings, confuse Vim and Emacs, or mix up `p`/`P`, `c`/`s`, etc.

Therefore the architecture is **retrieval + selection, not generation**:

1. A curated, machine-verified **knowledge base (KB)** holds every command the
   tool can ever output: keystrokes, description, mode, example, help tag.
2. The LLM's only jobs are (a) understanding the user's question and
   (b) choosing which KB entry/entries answer it, returning **entry IDs**.
3. The final answer is **rendered from the KB entry by code**, not by the LLM.
   The LLM's free text is never shown as a command.
4. If no KB entry matches with sufficient confidence, the tool answers
   "Not found in knowledge base" and shows the closest `:help` topics with a
   verbatim excerpt from the documentation.

This makes correctness a property of the data (which can be tested), not of the
model (which cannot be fully trusted).

### 3.1 Relationship to RAG

This design is a RAG variant in which the generation step is replaced by
selection. Three distinct uses of retrieval were considered:

| Use | Decision |
|-----|----------|
| Classic RAG: chunk `:help` text, retrieve chunks, let the LLM write the answer from them. | **Rejected for the answer body.** A 3B model paraphrases and garbles keystrokes taken from chunks; longer context threatens N2; output is non-deterministic. |
| Retrieval over `:help` text **without** generation: show the verbatim excerpt. | **Adopted for fallback (§5 step 7).** Quoted text cannot hallucinate. Also used offline to draft KB entries and aliases. |
| Retrieval over the KB (lexical + embeddings) to produce candidates for the selector. | **Adopted, hybrid (§4.1).** The eval set decides whether embeddings add enough over lexical search to justify the Ollama dependency at query time. |

The LLM selector itself is **conditional**: it is only invoked when retrieval is
ambiguous (§5 step 4). With ~500 well-aliased entries, lexical + embedding
retrieval alone is expected to resolve most queries.

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Terminal UI (REPL)                     │
│          prompt_toolkit input  ·  rich output rendering      │
└───────────────┬──────────────────────────────▲───────────────┘
                │ question (+ last turn)       │ rendered answer
┌───────────────▼──────────────────────────────┴───────────────┐
│                         Query Engine                          │
│  1. normalize   2. exact / reverse lookup   3. hybrid retrieve│
│  4. select (LLM, conditional)   5. validate   6. render        │
│  7. fallback (verbatim :help excerpt)                         │
└──────┬─────────────────┬────────────────┬──────────────┬─────┘
       │                 │                │              │
┌──────▼──────┐  ┌───────▼───────┐ ┌──────▼───────┐ ┌────▼────────────┐
│ Knowledge   │  │ Lexical index │ │ Vector index │ │ Ollama server   │
│ base (YAML) │  │ (BM25, always │ │ (embeddings, │ │ localhost:11434 │
│ verified    │  │  available)   │ │  optional)   │ │ embed + chat    │
└──────▲──────┘  └───────────────┘ └──────────────┘ └─────────────────┘
       │ build + verify (offline, CI)
┌──────┴───────────────────────────────────────────────┐
│ KB tooling: importer from Vim :help docs, example     │
│ verifier (headless vim/nvim), index builder,          │
│ :help chunker (for fallback excerpts)                 │
└──────────────────────────────────────────────────────┘
```

### 4.1 Components

**Terminal UI** — Python, `prompt_toolkit` for line editing/history,
`rich` for coloured output. Commands: `/mode normal|insert|visual`,
`/flavor vim|nvim`, `/more` (show alternatives), `/missing` (log the last
question as a KB gap), `/help`, `/quit`.

**Query Engine** — orchestrates the pipeline in §5.

**Knowledge Base** — YAML files under `kb/`, one entry per command (§6).
Version controlled, human reviewable.

**Retrieval indexes** — two indexes over the same text per entry
(description + aliases + example questions), built by `vimask kb build`:

- *Lexical index*: BM25 (`rank_bm25`, pure Python). Always available, no
  Ollama needed. Vim vocabulary is keyword-heavy (yank, register, macro,
  text object), so lexical search carries much of the load.
- *Vector index*: embeddings in a single `.npy` file plus an ID list.
  A few thousand vectors at most, so brute-force cosine similarity is enough;
  no vector DB. Optional: if Ollama is unreachable, retrieval runs lexical-only.

Scores are fused with reciprocal rank fusion (RRF). The index file records
the embedding model name and a hash of the KB; a mismatch at startup triggers
a rebuild (or a warning in lexical-only mode) so a config change can never
silently degrade retrieval.

**Help excerpt index** — `:help` files chunked by tag (the text between one
`*tag*` and the next), indexed lexically and by embedding. Used only in
fallback to show verbatim documentation; never fed to the LLM for generation.

**Ollama** — local inference server. Two models:

| Role       | Model (default)            | Size on disk | Notes |
|------------|----------------------------|--------------|-------|
| Embeddings | `nomic-embed-text`         | ~270 MB      | Fast, good for short queries. |
| Selector   | `qwen2.5:3b-instruct`      | ~2 GB        | Fits 8 GB M1. Supports JSON output. |
| Selector (16 GB+) | `qwen2.5:7b-instruct` or `llama3.1:8b` | ~4.7 GB | Better intent understanding. |

Ollama runs natively on Apple Silicon with Metal acceleration, satisfying N1.
Model names are configurable in `~/.config/vimask/config.toml`. The client
sends `keep_alive: -1` so the selector stays resident between queries; cold
load (2–5 s) happens once per session and is excluded from N2.

**KB tooling** — offline scripts (not used at query time):
importer, verifier, index builder, help chunker (§7).

## 5. Query pipeline

1. **Normalize** — lowercase, strip filler, attach session context (mode,
   flavor, and the previous turn's question and chosen entry ID, if any, for
   follow-ups like "and how do I undo that?").

2. **Exact and reverse lookup** — no LLM call:
   - If the query contains a quoted or backticked token, or any token that
     equals a KB `keys` value (`what does ciw do`, `` `dap` ``), return that
     entry directly (reverse lookup, F8).
   - If the normalized query equals an alias, return that entry directly.

3. **Retrieve (hybrid)** — run BM25 and, when Ollama is available, embedding
   search; fuse with RRF; take top-k (k≈8).
   - **Flavor** is a filter (vim/nvim); most entries list both.
   - **Mode** is a *boost*, not a filter. A user in insert mode asking "delete
     a word" must still get normal-mode `dw`; the renderer adds a
     "press `<Esc>` first" hint when the entry's mode differs from the
     session mode.
   - Compute `margin = score(top1) − score(top2)`.

4. **Select (LLM, conditional)** — skipped when the top-1 fused score is
   above the high threshold *and* the margin is above the margin threshold;
   then top-1 is taken directly (fast path, no LLM latency). Otherwise:

   *Stage A — single entry.* Send the question (plus previous turn, if any)
   and the k candidates as `(ID, keys, short description)` to the selector
   with a constrained JSON schema (Ollama `format` parameter):

   ```json
   { "ids": ["delete-inside-parens"], "compose": false }
   ```

   `ids` is ordered best-first and may be empty. Candidates include `keys`
   because that helps disambiguation (`di(` vs `da(`) and costs nothing in
   safety: step 5 restricts output to candidate IDs regardless. The model is
   **not** asked for a self-reported confidence; small models report it badly.
   `compose: true` means "this question needs an operator + motion
   composition" and routes to Stage B (§5.1).

   System prompt (abridged):
   > You choose which of the listed Vim commands answers the user's question.
   > Return only IDs from the list, best first. If none answers it, return an
   > empty list. If the question combines an action with a count, motion or
   > text object that is not in the list, set compose to true. Never invent
   > commands.

   `temperature: 0`, fixed `seed`. This makes output stable in practice but is
   not strictly deterministic on Metal; tests do not depend on it (§7.3).

5. **Validate** — drop any ID not in the candidate list. Confidence is
   derived by code, not reported by the model:
   - *high*: selector's first ID equals retrieval top-1, or the fast path
     was taken;
   - *medium*: selector's first ID is in retrieval top-3;
   - *low*: anything else, or empty `ids`.
   Low confidence goes to fallback. Medium is shown with alternatives.

6. **Render** — format the selected entries from KB data with a fixed
   template (§8). At most 1 primary answer + up to 2 alternatives (`/more`
   shows the rest).

7. **Fallback** — print "Not in knowledge base." followed by the top 3 help
   chunks from the help excerpt index: tag plus the first few lines of the
   documentation, quoted verbatim. Real tags and real text, therefore safe.
   The question is appended to `missing.log` (§13).

### 5.1 Compositional queries

Many Vim commands are compositions (operator + count + motion/text object),
e.g. "delete the next 3 words" → `d3w`. To support these without generation:

- The KB stores **operators** (`d`, `c`, `y`, `>`, `gU`, …), **motions**
  (`w`, `e`, `$`, `}`, `f{char}`, …) and **text objects** (`iw`, `a"`, `i(`, …)
  as typed entries (`kind: operator | motion | textobject`).

- **Stage B — composition.** Triggered by `compose: true` from Stage A.
  Retrieval top-k cannot be relied on to surface `op-delete` and `mo-word` as
  separate candidates, so Stage B sends the **full catalogue** of operators,
  motions and text objects (roughly 80 items as `(ID, keys, description)`,
  which fits a 3B context easily) and asks for:

  ```json
  { "count": 3, "operator": "op-delete", "target": "mo-word", "arg": null }
  ```

  `arg` is the single-character argument required by `f`, `F`, `t`, `T`,
  `r`, `` ` `` and `'` (`dt)` = operator `d`, target `t{char}`, arg `)`).

- A deterministic **composer** validates that each part exists, that `arg` is
  present exactly when the target requires one and is a single printable
  character, that the combination is legal (operator + motion/text object;
  count within 1–99), and builds the keystroke string.

- The example is generated by running the composed keys in headless Vim on a
  **sample buffer chosen per target** (§7.2): each motion/text-object entry
  carries a `sample:` buffer in which its effect is visible (e.g. a sentence
  with at least five words after the cursor for `w`; a line with nested
  parentheses for `i(`). Headless execution takes ~50 ms (`vim -Es`) to
  ~150 ms (`nvim --headless`). Vim is launched in restricted mode (`-Z`), so
  no shell can be reached even if a bad composition slipped through. If no
  editor is found on `PATH`, the composed answer is rendered without an
  example and with a note.

So even composed answers are grounded in verified parts and an actual Vim
execution, never in model text.

### 5.2 Parameterized commands

Ex commands and some normal-mode commands take user text:
`:%s/{pattern}/{replacement}/g`, `:{range}d`, `q{register}`.

Policy for v1: **placeholders are shown literally**, highlighted, with a
one-line legend ("replace `{pattern}` and `{replacement}`"). The LLM never
fills them. This is a deliberate restriction: filling slots means the LLM
writes command text, which §3 forbids. A later version may allow the LLM to
fill *typed* slots (pattern, replacement, register) with code-side escaping
and a distinct colour marking the text as user-supplied; this is listed as
future work in §14.

## 6. Knowledge base schema

```yaml
id: delete-inside-parens
keys: "di("
kind: command              # command | operator | motion | textobject | ex
mode: normal               # normal | insert | visual | cmdline | ex
flavor: [vim, nvim]
category: editing/text-objects
description: Delete inside the nearest enclosing ( ).
aliases:
  - delete inside parentheses
  - remove arguments of function call
  - clear contents of brackets
help_tag: "i("
alternatives: [delete-around-parens, change-inside-parens]
setup_keys: null           # optional; keys run before `keys` to create state
example:                   # optional; omitted for non-textual commands
  before: "print(foo, ba┃r)"   # '┃' (U+2503) marks the cursor
  after:  "print(┃)"
sample: null               # motions/text objects only: buffer used by composer
verified: example          # none | tag | example; set by verifier, never by hand
```

Notes:
- The cursor marker is `┃` (U+2503), not `|`, because `|` appears in example
  text (`a || b`, `:s/x/y/ | w`). The renderer shows it as a highlighted
  cursor block.
- `setup_keys` creates state that the command depends on: `yy` before `p`,
  `qaxq` before `@a`, `ma` before `` `a ``, a first edit before `.`. The
  verifier runs `setup_keys` then `keys` and compares only the result of
  `keys`.
- Entries without visible text effect (e.g. `:w`, `<C-w>v`) omit `example`
  and may have an `effect:` field ("Splits window vertically"). They can only
  reach `verified: tag`.
- Ex commands store the full command (`:%s/old/new/g`) and, if
  parameterized, placeholders (`:%s/{pattern}/{replacement}/g`); see §5.2.
- Verification tiers: `none` (not indexed), `tag` (help tag exists in both
  flavors' `doc/tags`; no example or example not runnable), `example` (tag
  check passed and the example reproduced in every listed flavor). The index
  builder includes `tag` and `example` entries only.

Initial scope: ~400–600 entries covering movement, editing, text objects,
registers, search/replace, visual mode, macros, marks, windows/buffers/tabs,
folding, and common ex commands.

## 7. Ensuring correctness

### 7.1 Sources

- Primary source: Vim's own documentation (`$VIMRUNTIME/doc/*.txt`,
  especially `index.txt`, `motion.txt`, `change.txt`), which is the
  authoritative reference and is already on the laptop.
- The **importer** parses `index.txt` (which lists every default key with tag
  and one-line description) to bootstrap entries with correct `keys`,
  `mode` and `help_tag`. Aliases and examples are then added and reviewed
  by hand (an LLM may *draft* aliases offline from the help excerpt index,
  but a human reviews them; aliases do not affect correctness of keystrokes,
  only retrieval).
- Neovim's `index.txt` differs slightly from Vim's; the importer runs against
  both and records the flavor in which each key was found.

### 7.2 Example verifier

Every `example` is executed in a real editor, so examples cannot be wrong:

```
nvim --headless --clean -u NONE -Z   (or: vim -Es -u NONE -N -Z)
  1. load `before` into a buffer, place cursor at '┃' (marker removed)
  2. run setup_keys, if any, with  :execute "normal! " . keys
  3. run `keys` the same way
  4. read buffer + cursor, compare with `after`
```

Details:
- `normal!` (with bang) is used so user mappings can never interfere, even
  though `-u NONE` already disables them.
- Keys are stored in Vim notation (`<Esc>`, `<C-v>`, `<CR>`). The verifier
  translates them to escape sequences inside a double-quoted Vimscript string
  (`"\<Esc>"`) before `:execute`. Ex commands are run as
  `normal! :%s/old/new/g\<CR>`.
- `-Z` (restricted mode) blocks `:!`, `system()` and friends in both the
  verifier and the query-time composer.
- Runs for both `vim` and `nvim` when `flavor` lists both; a mismatch
  restricts the entry's flavor or fails the build.
- Help tags are checked against the `tags` file of **each** listed flavor's
  `doc` directory; tags differ between Vim and Neovim.
- Runs in CI and as `vimask kb verify`. The verifier writes the `verified`
  tier; the index builder ignores `verified: none`.

### 7.3 Evaluation set

A test file of question → expected-ID pairs (including paraphrases, typos,
compositions and out-of-scope questions that must return "not found").
Started in milestone 1 with ~50–100 questions and grown to ~200 by
milestone 3, because retrieval thresholds for the LLM-free mode (milestone 2)
already need it.

Metrics tracked per configuration (lexical only, embeddings only, hybrid,
hybrid + selector; per selector model):

- **Top-1 accuracy** (target ≥ 90%)
- **False answer rate** — wrong entry shown with high confidence
  (target ≤ 1%; prefer "not found" over a wrong answer)
- **Selector invocation rate** — share of queries that leave the fast path
- **Latency p50/p95** on M1, warm model

Used to choose default model, k, fusion weights, the fast-path thresholds and
the fallback threshold. Tests assert on aggregate metrics with tolerance, not
on exact per-query model output, since Metal inference is not bit-exact.

## 8. Output format

Fixed template, rendered by code:

```
  <keys>     <description>            [mode, if not normal]
  [press <Esc> first — you are in insert mode]

  before:  <line with cursor>
  after:   <line with cursor>

  see:     :help <tag>
  also:    <alt keys> (<short desc>), ...
```

Rules:
- Maximum ~8 lines per answer by default.
- Special keys rendered in Vim notation: `<Esc>`, `<C-v>`, `<CR>`, `<leader>`.
- Placeholders (`{pattern}`) rendered highlighted with a one-line legend.
- Multi-line examples allowed (up to 5 lines) for commands like `J`, `>>`, `dap`.
- No prose paragraphs; no LLM-written text in the answer body.

## 9. Integration with Vim (companion mode)

- **Side pane**: run `vimask` in a tmux split or second terminal.
- **Inside Vim**: `:terminal vimask` (Vim 8+/Neovim).
- **One-shot**: `:!vimask "swap two characters"` prints a single answer.
- **Optional plugin (later)**: a tiny Vim/Lua plugin mapping `<leader>?` that
  passes the current mode and filetype to `vimask --oneshot` and shows the
  result in a popup/floating window.

## 10. Technology stack

| Concern        | Choice |
|----------------|--------|
| Language       | Python 3.11+ (install via Homebrew/uv; macOS system 3.9 is too old) |
| Packaging      | `uv` / `pipx install vimask` |
| LLM runtime    | Ollama (Homebrew `brew install ollama` or app) |
| Ollama client  | `ollama` Python package (HTTP to `localhost:11434`) |
| UI             | `prompt_toolkit`, `rich` |
| KB             | YAML (`pyyaml` / `ruamel.yaml`), validated with `pydantic` |
| Lexical index  | `rank_bm25` |
| Vector index   | `numpy` + single `.npy` file |
| Verifier       | `nvim --headless` and `vim -Es` via `subprocess` |
| Tests          | `pytest` (KB verification, eval set, unit tests) |

## 11. Project layout

```
vimask/
├── docs/design.md
├── kb/                    # verified YAML entries, by category
├── eval/questions.yaml    # evaluation set
├── src/vimask/
│   ├── cli.py             # entry point, REPL, one-shot mode
│   ├── engine.py          # query pipeline
│   ├── lookup.py          # exact / reverse / alias lookup
│   ├── retrieval.py       # BM25 + embeddings, RRF fusion
│   ├── selector.py        # Ollama selector calls (stage A, stage B), JSON schemas
│   ├── composer.py        # operator+motion composition, headless example
│   ├── render.py          # output template
│   ├── kb.py              # schema, loading
│   └── tools/
│       ├── import_help.py # bootstrap from $VIMRUNTIME/doc/index.txt
│       ├── chunk_help.py  # :help excerpt index for fallback
│       ├── verify.py      # headless example verification
│       └── build_index.py
└── tests/
```

## 12. Setup on M1

```sh
brew install ollama python@3.12 uv
ollama serve &                       # or launch the Ollama app
ollama pull nomic-embed-text
ollama pull qwen2.5:3b-instruct      # 7b variant on 16 GB+ machines
uv tool install .                    # from repo
vimask kb build                    # verify KB + build index
vimask
```

`vimask doctor` checks: Ollama reachable, models pulled, index present and
matching the configured embedding model and KB hash, Vim/Neovim found, and
reports which mode will run (full, retrieval-only, lexical-only).

## 13. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Selector picks the wrong (but real) entry | Code-derived confidence from retrieval agreement, eval set, show alternatives, "not found" bias. |
| KB coverage gaps | "Not found" + verbatim help excerpts; `/missing` and automatic logging of fallback questions to `~/.local/share/vimask/missing.log` to guide KB growth. |
| Vim vs Neovim behavioural differences | `flavor` field, verifier and tag check run against both. |
| User's custom mappings change behaviour | Out of scope for v1; answers describe defaults (`-u NONE`). Stated in `/help`. |
| Low-RAM 8 GB M1 | 3B selector default; exact/alias path and retrieval fast path avoid the LLM for most queries; `keep_alive: -1` avoids reloads. |
| Latency over 2 s | Fast path skips the selector; candidate descriptions kept short; `num_predict` capped; measured in milestone 3 before N2 is confirmed. |
| Ollama not running | Clear error + `doctor`; lexical-only retrieval with conservative fast-path threshold; no LLM, no embeddings, correctness unchanged (F9). |
| Composer runs model-chosen keys in Vim | Parts restricted to verified operators/motions/text objects; `-Z` restricted mode; no ex commands composed. |
| Embedding model changed in config | Index stores model name + KB hash; rebuilt on mismatch. |

## 14. Out of scope (v1)

- Plugin-specific commands (fugitive, telescope, etc.).
- Reading the user's `.vimrc` / custom mappings.
- Editing the user's buffer directly.
- Free-form explanations or tutorials generated by the LLM. A possible later
  `/explain` command would generate prose from retrieved help chunks, clearly
  labelled as model-written.
- LLM-filled placeholders in parameterized commands (§5.2).
- Multi-turn dialogue beyond carrying the previous question and answer.

## 15. Milestones

1. **M1 – KB core**: schema, importer from `index.txt`, verifier (tiers,
   `setup_keys`, keycode translation), ~150 hand-reviewed entries with
   examples, first 50–100 eval questions.
2. **M2 – Retrieval REPL**: lexical + embedding retrieval with RRF, exact and
   reverse lookup, renderer, fast-path thresholds tuned on the eval set;
   works without LLM selector and in lexical-only mode.
3. **M3 – LLM selector**: Stage A JSON-constrained selection, code-derived
   confidence, eval set to ~200 questions, latency measurement, help excerpt
   fallback.
4. **M4 – Composer**: Stage B catalogue selection, `arg` slot, per-target
   sample buffers, live headless example generation under `-Z`.
5. **M5 – Companion**: one-shot mode, Vim plugin popup, mode/flavor context,
   "press `<Esc>` first" hint.
