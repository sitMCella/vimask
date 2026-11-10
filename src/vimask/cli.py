"""Command-line entry point: REPL, one-shot mode, KB tooling, doctor, eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

from . import __version__
from .config import Config, REPO_ROOT, default_config_path, load_config
from .engine import Engine, Session
from .help_index import HelpIndex
from .kb import load_kb
from .llm import OllamaClient
from .render import Answer, render_more, render_response, plain_response
from .retrieval import VectorIndex
from .vimrun import editors_available

HELP_TEXT = """\
Ask in plain English, e.g.  delete everything inside the parentheses
Reverse lookup:             what does ciw do        or     `dap`

Commands:
  /mode normal|insert|visual|off   tell vimask which mode you are in
  /flavor vim|nvim                 Vim or Neovim
  /more                            show more matches for the last question
  /missing                         log the last question as a KB gap
  /debug                           show how the last answer was found
  /help                            this text
  /quit                            exit (also Ctrl-D)

Answers describe default Vim behaviour (no .vimrc, no plugins). Examples
assume shiftwidth=4 expandtab nostartofline.
"""


def build_engine(cfg: Config, *, use_llm: bool = True, use_embeddings: bool = True, quiet: bool = False) -> Engine:
    kb = load_kb(cfg.kb_dir)
    client = OllamaClient(cfg) if use_llm or use_embeddings else None
    loader = lambda: HelpIndex.load(cfg.index_dir)  # noqa: E731
    engine = Engine(
        cfg,
        kb,
        client=client,
        use_selector=use_llm,
        use_embeddings=use_embeddings,
        help_index_loader=loader,
    )
    if not quiet:
        for w in (engine.index_warning, engine.selector_warning):
            if w:
                print(f"note: {w}", file=sys.stderr)
    return engine


# ------------------------------------------------------------------- REPL


def repl(engine: Engine, session: Session, console: Console, debug: bool = False) -> int:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    try:
        engine.cfg.history_file.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(engine.cfg.history_file))
    except OSError:
        history = None
    ps = PromptSession(history=history)
    console.print(Text(f"vimask {__version__} — {engine.mode_name}. /help for commands.", style="dim"))
    last_question: str | None = None
    while True:
        try:
            line = ps.prompt("vimask> ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            cmd, arg = cmd.lower(), arg.strip().lower()
            if cmd in ("quit", "q", "exit"):
                return 0
            if cmd == "help":
                console.print(HELP_TEXT)
            elif cmd == "mode":
                if arg in ("normal", "insert", "visual"):
                    session.mode = arg
                    console.print(Text(f"  mode: {arg}", style="dim"))
                elif arg in ("off", "none", ""):
                    session.mode = None
                    console.print(Text("  mode: unknown", style="dim"))
                else:
                    console.print(Text("  usage: /mode normal|insert|visual|off", style="red"))
            elif cmd == "flavor":
                if arg in ("vim", "nvim"):
                    session.flavor = arg
                    console.print(Text(f"  flavor: {arg}", style="dim"))
                else:
                    console.print(Text("  usage: /flavor vim|nvim", style="red"))
            elif cmd == "more":
                render_more(console, session.last_more)
            elif cmd == "missing":
                if last_question:
                    engine.log_missing_question(last_question)
                    console.print(Text(f"  logged to {engine.cfg.missing_log}", style="dim"))
                else:
                    console.print(Text("  no question yet", style="dim"))
            elif cmd == "debug":
                t = engine.last_trace
                console.print(Text(f"  path={t.path} confidence={t.confidence} embeddings={t.used_embeddings} "
                                   f"selector={t.selector_called} margin={t.margin}", style="dim"))
                console.print(Text(f"  normalized={t.normalized!r}", style="dim"))
                for i, (eid, sc) in enumerate(t.top[:8]):
                    console.print(Text(f"  {i + 1}. {eid} {sc}", style="dim"))
                if t.note:
                    console.print(Text(f"  note: {t.note}", style="dim"))
            else:
                console.print(Text(f"  unknown command /{cmd}; /help lists commands", style="red"))
            continue

        resp = engine.answer(line, session)
        render_response(console, resp)
        if isinstance(resp, Answer):
            session.last_question = line
        last_question = line
        if debug:
            t = engine.last_trace
            console.print(Text(f"  [debug] path={t.path} conf={t.confidence} top={t.top[:3]} margin={t.margin}", style="dim"))


# --------------------------------------------------------------- commands


def cmd_doctor(cfg: Config) -> int:
    ok = True
    print(f"vimask {__version__}")
    print(f"config:        {default_config_path()}{'' if default_config_path().is_file() else ' (defaults)'}")
    print(f"KB dir:        {cfg.kb_dir}")
    try:
        kb = load_kb(cfg.kb_dir)
        indexed = kb.indexed()
        print(f"KB:            {len(kb)} entries, {len(indexed)} verified "
              f"({sum(1 for e in indexed if e.verified == 'example')} with examples), hash {kb.kb_hash}")
    except Exception as exc:
        print(f"KB:            ERROR {exc}")
        return 1
    eds = editors_available(cfg.vim_path, cfg.nvim_path)
    for fl, path in eds.items():
        print(f"{fl + ':':<14} {path or 'not found'}")
    client = OllamaClient(cfg)
    st = client.status()
    if st.reachable:
        print(f"Ollama:        reachable at {cfg.ollama_host} ({len(st.models)} models)")
        for role, model in (("embed", cfg.embed_model), ("selector", cfg.selector_model)):
            have = st.has(model)
            print(f"  {role + ':':<12} {model} {'pulled' if have else 'MISSING — ollama pull ' + model}")
    else:
        print(f"Ollama:        not reachable at {cfg.ollama_host} ({st.error})")
    vi = VectorIndex.load(cfg.index_dir)
    if vi is None:
        print("vector index:  missing (run `vimask kb build`)")
    elif not vi.matches(cfg.embed_model, kb.kb_hash):
        print(f"vector index:  STALE (model {vi.model}, kb {vi.kb_hash}); run `vimask kb build`")
    else:
        print(f"vector index:  ok ({len(vi.ids)} vectors, {vi.model})")
    hi = (cfg.index_dir / HelpIndex.FILE).is_file()
    print(f"help index:    {'ok' if hi else 'missing (run `vimask kb build`)'}")
    engine = Engine(cfg, kb, client=client, help_index_loader=None)
    print(f"mode:          {engine.mode_name}")
    return 0 if ok else 1


def cmd_kb(cfg: Config, args: argparse.Namespace) -> int:
    from .tools import build_index, import_help, verify

    if args.kb_cmd == "verify":
        reports = verify.verify_kb(cfg, only=set(args.ids) if args.ids else None, write=not args.dry_run, verbose=args.verbose)
        print(verify.summarize(reports), file=sys.stderr)
        return 0 if all(r.ok for r in reports) else 1
    if args.kb_cmd == "build":
        rc = 0
        if not args.skip_verify:
            reports = verify.verify_kb(cfg, write=True)
            print(verify.summarize(reports), file=sys.stderr)
            if not all(r.ok for r in reports):
                rc = 1
        kb = load_kb(cfg.kb_dir)
        build_index.build_help_index(cfg)
        build_index.build_vectors(cfg, kb, OllamaClient(cfg))
        return rc
    if args.kb_cmd == "import":
        out = Path(args.out) if args.out else cfg.kb_dir / "_import"
        counts = import_help.import_all(out, editors_available(cfg.vim_path, cfg.nvim_path))
        for sec, n in counts.items():
            print(f"  {sec:<24} {n}")
        print(f"drafts written to {out}")
        return 0
    if args.kb_cmd == "check":
        kb = load_kb(cfg.kb_dir)
        index = import_help.index_lookup(editors_available(cfg.vim_path, cfg.nvim_path))
        problems = 0
        for e in kb:
            keys = index.get(e.help_tag)
            if keys is None:
                continue  # tag not in index.txt (e.g. option or concept tag); verifier checks tags file
            if e.kind in ("ex", "operator") or e.placeholders or any(k.replace("{motion}", "") in e.keys for k in keys):
                continue
            if e.keys not in keys:
                print(f"  {e.id}: keys {e.keys!r} differ from index.txt {sorted(keys)} for tag {e.help_tag}")
                problems += 1
        print(f"{problems} discrepancies")
        return 0 if problems == 0 else 1
    if args.kb_cmd == "stats":
        kb = load_kb(cfg.kb_dir)
        from collections import Counter

        c = Counter(e.category.split("/")[0] for e in kb)
        tiers = Counter(e.verified for e in kb)
        kinds = Counter(e.kind for e in kb)
        print(f"{len(kb)} entries; tiers {dict(tiers)}; kinds {dict(kinds)}")
        for cat, n in sorted(c.items()):
            print(f"  {cat:<20} {n}")
        return 0
    return 2


def cmd_eval(cfg: Config, args: argparse.Namespace) -> int:
    from .evaluate import load_eval, run_eval

    path = Path(args.file) if args.file else REPO_ROOT / "eval" / "questions.yaml"
    items = load_eval(path)
    configs = []
    if args.setup in ("all", "lexical"):
        configs.append(("lexical-only", dict(use_llm=False, use_embeddings=False)))
    if args.setup in ("all", "retrieval"):
        configs.append(("hybrid, no selector", dict(use_llm=False, use_embeddings=True)))
    if args.setup in ("all", "full"):
        configs.append(("full", dict(use_llm=True, use_embeddings=True)))
    rc = 0
    for name, kw in configs:
        engine = build_engine(cfg, quiet=True, **kw)
        engine.log_missing = False
        m = run_eval(engine, items)
        print(f"[{name}] {engine.mode_name}")
        print(f"  {m.summary()}")
        if args.verbose:
            for o in m.outcomes:
                if not o.correct:
                    print(f"  WRONG {o.item.q!r}: expected {o.item.expect_keys or o.item.expect} got {o.got_keys} ({o.got}, {o.confidence}, {o.path})")
        if m.false_answer_rate > 0.01:
            rc = 1
    return rc


# ------------------------------------------------------------------- main


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vimask", description="How do I do X in Vim? Answers from a verified knowledge base.")
    p.add_argument("--version", action="version", version=f"vimask {__version__}")
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("--mode", choices=["normal", "insert", "visual"], help="your current Vim mode")
    p.add_argument("--flavor", choices=["vim", "nvim"], default=None)
    p.add_argument("--no-llm", action="store_true", help="never call the LLM selector")
    p.add_argument("--lexical-only", action="store_true", help="no Ollama at all (no embeddings, no selector)")
    p.add_argument("--debug", action="store_true", help="print retrieval trace after each answer")
    p.add_argument("--plain", action="store_true", help="no colours")
    sub = p.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="one-shot question (default when a question is given)")
    ask.add_argument("question", nargs="+")

    sub.add_parser("doctor", help="check Ollama, models, index, editors")

    kb = sub.add_parser("kb", help="knowledge-base tooling")
    kbs = kb.add_subparsers(dest="kb_cmd", required=True)
    v = kbs.add_parser("verify", help="verify help tags and examples in vim/nvim")
    v.add_argument("ids", nargs="*", help="only these entry IDs")
    v.add_argument("--dry-run", action="store_true", help="do not write verified tiers back")
    v.add_argument("-v", "--verbose", action="store_true")
    b = kbs.add_parser("build", help="verify KB, build help excerpt index and vector index")
    b.add_argument("--skip-verify", action="store_true")
    i = kbs.add_parser("import", help="draft entries from $VIMRUNTIME/doc/index.txt")
    i.add_argument("--out")
    kbs.add_parser("check", help="cross-check keys/help_tag against index.txt")
    kbs.add_parser("stats", help="KB statistics")

    ev = sub.add_parser("eval", help="run the evaluation set")
    ev.add_argument("--file")
    ev.add_argument("--setup", choices=["all", "lexical", "retrieval", "full"], default="lexical",
                    help="which configuration(s) to evaluate")
    ev.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = make_parser()
    # `vimask "join lines"` works without the `ask` subcommand: the first
    # positional token that is not a subcommand starts the question.
    takes_value = {"--config", "--mode", "--flavor"}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in takes_value:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok not in {"ask", "doctor", "kb", "eval"}:
            argv.insert(i, "ask")
        break
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)

    if args.command == "doctor":
        return cmd_doctor(cfg)
    if args.command == "kb":
        return cmd_kb(cfg, args)
    if args.command == "eval":
        return cmd_eval(cfg, args)

    use_llm = not (args.no_llm or args.lexical_only)
    use_embeddings = not args.lexical_only
    session = Session(mode=args.mode, flavor=args.flavor or "vim")
    console = Console(color_system=None if args.plain else "auto", highlight=False)

    if args.command == "ask":
        engine = build_engine(cfg, use_llm=use_llm, use_embeddings=use_embeddings, quiet=True)
        question = " ".join(args.question)
        resp = engine.answer(question, session)
        if args.plain:
            sys.stdout.write(plain_response(resp))
        else:
            render_response(console, resp)
        if args.debug:
            t = engine.last_trace
            print(f"[debug] path={t.path} conf={t.confidence} top={t.top[:3]} margin={t.margin}", file=sys.stderr)
        return 0 if isinstance(resp, Answer) else 1

    engine = build_engine(cfg, use_llm=use_llm, use_embeddings=use_embeddings)
    return repl(engine, session, console, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
