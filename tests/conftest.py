from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vimask.config import Config
from vimask.kb import load_kb

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def kb():
    return load_kb(REPO / "kb")


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.kb_dir = REPO / "kb"
    c.data_dir = tmp_path / "data"
    return c


def has_editor(name: str) -> bool:
    return shutil.which(name) is not None or (name == "vim" and Path("/usr/bin/vim").exists())


needs_vim = pytest.mark.skipif(not has_editor("vim"), reason="vim not installed")
needs_nvim = pytest.mark.skipif(not has_editor("nvim"), reason="nvim not installed")
