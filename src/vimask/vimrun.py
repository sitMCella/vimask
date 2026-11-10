"""Headless editor runner shared by the verifier and the composer.

Runs a key sequence (Vim notation, e.g. ``ci"``, ``<Esc>``, ``:%s/a/b/<CR>``)
against a small buffer in a real ``vim`` or ``nvim`` and returns the
resulting buffer and cursor. Everything happens with ``-u NONE`` (no user
config, no mappings) and ``normal!`` (mappings ignored even if present).

Sandboxing: Vim is started with ``-Z`` (restricted mode: no ``:!``,
``system()`` …). Neovim has no ``-Z``; there ``'shell'`` is pointed at
``/usr/bin/false`` so any shell escape fails. The composer additionally
never composes ex commands, so model-chosen input can only ever be a
verified operator + motion/text object.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .kb import join_cursor, split_cursor

# Options applied to every run so examples are reproducible and readable.
# These are stated to the user as assumptions in /help.
BASE_OPTIONS = [
    "nomore",
    "noswapfile",
    "nowritebackup",
    "nobackup",
    "shiftwidth=4",
    "expandtab",
    "tabstop=4",
    "softtabstop=0",
    "nostartofline",
    "nowrapscan",
    "noautoindent",
    "textwidth=0",
    "formatoptions=tcq",
    "nojoinspaces",
    "columns=80",
    "lines=24",
]


# Vim key notation we translate: <Esc>, <CR>, <C-x>, <C-\>, <S-Tab>, <M-a>, <F5>,
# <lt>, <Bar>, <Space>, <Up> ... Anything else (e.g. the regex atom \<cat\>) is
# typed literally.
_SPECIAL_KEY_RE = re.compile(r"<(?:[CSMAD]-(?:\\|[^\s<>])|[A-Za-z]{2,}|F\d{1,2})>")


class EditorNotFound(RuntimeError):
    pass


@dataclass
class RunResult:
    ok: bool
    lines: list[str]
    lnum: int
    col: int
    error: str = ""

    @property
    def text(self) -> str:
        return join_cursor(self.lines, self.lnum, self.col)


def find_editor(flavor: str, configured: str = "") -> str | None:
    """Return an executable path for ``vim`` or ``nvim`` or None."""
    name = configured or flavor
    path = shutil.which(name)
    if path:
        return path
    if flavor == "vim":
        for cand in ("/usr/bin/vim", "/opt/homebrew/bin/vim"):
            if os.access(cand, os.X_OK):
                return cand
    return None


def vim_string_literal(keys: str) -> str:
    """Encode Vim-notation keys as the body of a double-quoted Vimscript
    string: ``<Esc>`` becomes ``\\<Esc>``; ``"`` and ``\\`` are escaped."""
    out = []
    i = 0
    while i < len(keys):
        m = _SPECIAL_KEY_RE.match(keys, i)
        if m:
            out.append("\\" + m.group(0))
            i = m.end()
            continue
        ch = keys[i]
        if ch in ('"', "\\"):
            out.append("\\" + ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def normalize_keys_for_run(keys: str) -> str:
    """Ex commands stored as ``:cmd`` get an implicit ``<CR>``."""
    if keys.startswith(":") and "<CR>" not in keys:
        return keys + "<CR>"
    return keys


def _script(lnum: int, col: int, keys: str, setup_keys: str | None, options: list[str], out: Path) -> str:
    opts = " ".join(BASE_OPTIONS + list(options))
    seq = (setup_keys or "") + normalize_keys_for_run(keys)
    # Capture the position without leaving insert mode: <C-\><C-o> runs one
    # ex command and returns to the current mode (in normal mode it is a no-op
    # prefix, so the ':' command simply runs).
    capture = "\\<C-\\>\\<C-o>:let g:vh_pos = getpos('.')\\<CR>"
    body = vim_string_literal(seq)
    out_lit = str(out).replace("\\", "\\\\").replace("'", "''")
    return "\n".join(
        [
            f"set {opts}",
            "let g:vh_pos = []",
            f"call cursor({lnum}, {col})",
            "try",
            f'  execute "normal! {body}{capture}"',
            "catch",
            f"  call writefile(['ERROR', v:exception], '{out_lit}')",
            "  qa!",
            "endtry",
            "if empty(g:vh_pos) | let g:vh_pos = getpos('.') | endif",
            f"call writefile(['OK', string(g:vh_pos[1]), string(g:vh_pos[2])] + getline(1, '$'), '{out_lit}')",
            "qa!",
            "",
        ]
    )


def run_keys(
    before: str,
    keys: str,
    *,
    flavor: str = "vim",
    setup_keys: str | None = None,
    options: list[str] | None = None,
    editor: str | None = None,
    timeout: float = 5.0,
) -> RunResult:
    """Execute ``setup_keys`` + ``keys`` on ``before`` (with cursor marker)."""
    exe = editor or find_editor(flavor)
    if not exe:
        raise EditorNotFound(f"{flavor} not found on PATH")
    lines, lnum, col = split_cursor(before)
    if lnum == 0:
        lnum, col = 1, 1
    with tempfile.TemporaryDirectory(prefix="vimask-") as tmp:
        tmpdir = Path(tmp)
        buf = tmpdir / "buffer.txt"
        out = tmpdir / "out.txt"
        script = tmpdir / "run.vim"
        buf.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.write_text(_script(lnum, col, keys, setup_keys, options or [], out), encoding="utf-8")
        if flavor == "nvim":
            cmd = [
                exe,
                "--headless",
                "--clean",
                "-u",
                "NONE",
                "-i",
                "NONE",
                "-n",
                "--cmd",
                "set shell=/usr/bin/false shellcmdflag= noloadplugins",
                "-c",
                f"source {script}",
                str(buf),
            ]
        else:
            cmd = [
                exe,
                "-Es",
                "-u",
                "NONE",
                "-N",
                "-Z",
                "-i",
                "NONE",
                "-n",
                "--cmd",
                "set noloadplugins",
                "-c",
                f"source {script}",
                str(buf),
            ]
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "TERM": "dumb", "LANG": "en_US.UTF-8"}
        try:
            subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, [], 0, 0, error="timeout (incomplete key sequence?)")
        if not out.exists():
            return RunResult(False, [], 0, 0, error="editor produced no output")
        raw = out.read_text(encoding="utf-8", errors="replace").split("\n")
        if raw and raw[-1] == "":
            raw.pop()
    if not raw or raw[0] != "OK":
        msg = raw[1] if len(raw) > 1 else "unknown error"
        return RunResult(False, [], 0, 0, error=msg)
    lnum_out, col_out = int(raw[1]), int(raw[2])
    return RunResult(True, raw[3:], lnum_out, col_out)


def editors_available(vim_path: str = "vim", nvim_path: str = "nvim") -> dict[str, str | None]:
    return {"vim": find_editor("vim", vim_path), "nvim": find_editor("nvim", nvim_path)}
