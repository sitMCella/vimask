"""Help excerpt index for the fallback path (design §4.1, §5 step 7).

``:help`` files are chunked by tag: the text from one ``*tag*`` definition to
the next. Chunks are searched lexically (BM25) and quoted verbatim; they are
never fed to the LLM.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from .retrieval import tokenize

_TAG_RE = re.compile(r"\*([^*\s|]+)\*")
_HEADER_RE = re.compile(r"\*[^*]+\*\s*$")


@dataclass
class HelpChunk:
    tag: str
    file: str
    text: str
    all_tags: list[str] = field(default_factory=list)

    def excerpt_lines(self, n: int = 4) -> list[str]:
        out = []
        for line in self.text.split("\n"):
            stripped = line.rstrip()
            if not stripped:
                if out:
                    break
                continue
            if _HEADER_RE.search(stripped) and not out:
                continue  # skip the tag header line itself
            out.append(stripped.replace("\t", "    "))
            if len(out) >= n:
                break
        return out


def runtime_doc_dir(flavor: str, exe: str | None = None) -> Path | None:
    """Locate ``$VIMRUNTIME/doc`` for vim or nvim by asking the editor."""
    exe = exe or shutil.which(flavor)
    if not exe:
        return None
    try:
        if flavor == "nvim":
            cmd = [exe, "--headless", "--clean", "-c", 'echo $VIMRUNTIME', "-c", "qa!"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            out = (proc.stderr or proc.stdout).strip().splitlines()
        else:
            cmd = [exe, "-es", "-u", "NONE", "-c", "redir! >> /dev/stdout | echo $VIMRUNTIME | redir END", "-c", "qa!"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            out = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return None
    for line in reversed(out):
        p = Path(line.strip()) / "doc"
        if p.is_dir():
            return p
    return None


def chunk_help_file(path: Path) -> list[HelpChunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    chunks: list[HelpChunk] = []
    current_tags: list[str] = []
    buf: list[str] = []

    def flush():
        if current_tags and buf:
            body = "\n".join(buf).strip("\n")
            if body.strip():
                chunks.append(HelpChunk(current_tags[0], path.name, body, list(current_tags)))

    for line in lines:
        tags = _TAG_RE.findall(line)
        # A line whose tags sit at the end (after text or alone) starts a chunk.
        if tags and _HEADER_RE.search(line.rstrip()):
            flush()
            current_tags = tags
            buf = [line]
        else:
            buf.append(line)
    flush()
    return chunks


def chunk_help_dir(doc_dir: Path) -> list[HelpChunk]:
    chunks: list[HelpChunk] = []
    for p in sorted(doc_dir.glob("*.txt")):
        if p.name in ("tags",):
            continue
        chunks.extend(chunk_help_file(p))
    return chunks


class HelpIndex:
    FILE = "help_chunks.json"

    def __init__(self, chunks: list[HelpChunk]) -> None:
        self.chunks = chunks
        self._bm25: BM25Okapi | None = None

    @classmethod
    def build(cls, doc_dir: Path) -> "HelpIndex":
        return cls(chunk_help_dir(doc_dir))

    def save(self, index_dir: Path, source: str = "") -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "source": source,
            "chunks": [{"tag": c.tag, "file": c.file, "text": c.text, "tags": c.all_tags} for c in self.chunks],
        }
        (index_dir / self.FILE).write_text(json.dumps(data))

    @classmethod
    def load(cls, index_dir: Path) -> "HelpIndex | None":
        p = index_dir / cls.FILE
        if not p.is_file():
            return None
        data = json.loads(p.read_text())
        return cls([HelpChunk(c["tag"], c["file"], c["text"], c.get("tags", [])) for c in data["chunks"]])

    def _ensure(self) -> None:
        if self._bm25 is None:
            docs = [tokenize(" ".join(c.all_tags) + " " + c.text[:1200]) for c in self.chunks]
            self._bm25 = BM25Okapi(docs) if docs else None

    def search(self, query: str, k: int = 3) -> list[HelpChunk]:
        self._ensure()
        if not self._bm25:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            out.append(self.chunks[i])
            if len(out) >= k:
                break
        return out


def load_tags(doc_dir: Path) -> set[str]:
    tags_file = doc_dir / "tags"
    if not tags_file.is_file():
        return set()
    out = set()
    for line in tags_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if line and not line.startswith("!_TAG"):
            out.add(line.split("\t", 1)[0])
    return out
