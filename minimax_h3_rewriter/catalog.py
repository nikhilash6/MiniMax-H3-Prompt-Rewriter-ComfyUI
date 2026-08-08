"""The user-editable list of base models and adapters offered by the node.

The shipped ``models.json`` is a seed, not the live file: it is copied into the
ComfyUI user directory on first use and read from there afterwards, so updating
the node pack never overwrites a list somebody has curated. The node's "Open
model list" button opens that copy.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass

from .constants import ADAPTER_REPO

log = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_FILE = os.path.join(PACKAGE_DIR, "models.json")
USER_SUBDIR = "minimax_h3_rewriter"
FILE_NAME = "models.json"

FORMAT_TRANSFORMERS = "transformers"
FORMAT_GGUF = "gguf"
FORMATS = (FORMAT_TRANSFORMERS, FORMAT_GGUF)

PLACEHOLDER = "REPLACE_ME"


@dataclass
class CatalogEntry:
    name: str
    repo: str
    fmt: str = FORMAT_TRANSFORMERS
    file: str = ""
    mmproj: str = ""
    download_gb: float = 0.0
    vram: str = ""
    note: str = ""

    @property
    def is_gguf(self) -> bool:
        return self.fmt == FORMAT_GGUF

    @property
    def label(self) -> str:
        parts = [self.name]
        if self.download_gb:
            parts.append(f"{self.download_gb:g} GB download")
        if self.vram:
            parts.append(self.vram)
        label = " · ".join(parts)
        if self.note:
            label += f" — {self.note}"
        return label


@dataclass
class AdapterSpec:
    repo: str
    file: str = ""
    download_gb: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.repo) and PLACEHOLDER not in self.repo


def user_file() -> str:
    """Path of the live list, seeded from the packaged copy on first use."""
    try:
        import folder_paths

        base = os.path.join(folder_paths.get_user_directory(), USER_SUBDIR)
    except Exception:
        base = os.path.join(PACKAGE_DIR, "_user")

    path = os.path.join(base, FILE_NAME)
    if not os.path.isfile(path):
        try:
            os.makedirs(base, exist_ok=True)
            shutil.copyfile(SEED_FILE, path)
            log.info("[minimax_h3_rewriter.catalog] seeded model list at %s", path)
        except OSError as error:
            log.warning("[minimax_h3_rewriter.catalog] could not seed %s: %s", path, error)
            return SEED_FILE
    return path


def _read(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        log.warning("[minimax_h3_rewriter.catalog] %s is unreadable (%s)", path, error)
        return {}


def _data() -> dict:
    data = _read(user_file())
    if not data.get("models"):
        data = _read(SEED_FILE)
    return data


def _entries(data: dict, key: str) -> list[CatalogEntry]:
    entries = []
    for raw in data.get(key, []):
        if not isinstance(raw, dict) or not raw.get("repo") or not raw.get("name"):
            continue
        fmt = str(raw.get("format") or FORMAT_TRANSFORMERS).lower()
        if fmt not in FORMATS:
            log.warning("[minimax_h3_rewriter.catalog] unknown format %r in %r", fmt, raw.get("name"))
            continue
        try:
            entries.append(
                CatalogEntry(
                    name=str(raw["name"]),
                    repo=str(raw["repo"]),
                    fmt=fmt,
                    file=str(raw.get("file") or ""),
                    mmproj=str(raw.get("mmproj") or ""),
                    download_gb=float(raw.get("download_gb") or 0.0),
                    vram=str(raw.get("vram") or ""),
                    note=str(raw.get("note") or ""),
                )
            )
        except (TypeError, ValueError):
            log.warning("[minimax_h3_rewriter.catalog] skipping malformed entry %r", raw)
    return entries


def load() -> list[CatalogEntry]:
    """Base models for the LoRA rewriter, from the live list or the seed."""
    return _entries(_data(), "models")


def writers() -> list[CatalogEntry]:
    """General-purpose models offered by the guided writer nodes.

    A list seeded before this section existed has no ``writers`` key at all, and
    the seed is only ever copied once. Rather than rewrite somebody's file, an
    absent section falls back to the packaged one — the same rule ``adapter``
    follows, and for the same reason.
    """
    entries = _entries(_data(), "writers")
    return entries or _entries(_read(SEED_FILE), "writers")


def captioners() -> list[CatalogEntry]:
    """Multimodal models offered by the reference captioner node.

    Shorter than it looks like it should be. Publishing a GGUF and an mmproj is
    not the same as llama.cpp's ``mtmd`` being able to load them -- several
    current models abort outright -- so this list holds only the ones that have
    actually been run.
    """
    entries = _entries(_data(), "captioners")
    return entries or _entries(_read(SEED_FILE), "captioners")


def _adapter_from(data: dict, fmt: str) -> AdapterSpec:
    raw = (data.get("adapters") or {}).get(fmt) or {}
    if fmt == FORMAT_TRANSFORMERS:
        return AdapterSpec(
            repo=str(raw.get("repo") or ADAPTER_REPO),
            download_gb=float(raw.get("download_gb") or 0.0),
        )
    return AdapterSpec(
        repo=str(raw.get("repo") or ""),
        file=str(raw.get("file") or ""),
        download_gb=float(raw.get("download_gb") or 0.0),
    )


def adapter(fmt: str) -> AdapterSpec:
    """The adapter to pair with a base model of the given format.

    A live list seeded before an adapter had a home still carries the placeholder,
    and the seed is only ever copied once. Rather than rewrite somebody's file,
    an unconfigured entry falls back to the packaged value — a real repository
    always beats a placeholder, and a real entry the user wrote always wins.
    """
    spec = _adapter_from(_data(), fmt)
    if spec.configured:
        return spec
    fallback = _adapter_from(_read(SEED_FILE), fmt)
    return fallback if fallback.configured else spec


def reveal() -> str:
    """Open the list in whatever the desktop uses for .json files."""
    import subprocess
    import sys

    path = user_file()
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opening the user's own config file
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
    return path
