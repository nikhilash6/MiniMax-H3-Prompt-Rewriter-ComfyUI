"""The user-editable list of base models and adapters offered by the node.

The shipped ``models.json`` is a seed, not the live file: it is copied into the
ComfyUI user directory on first use and read from there afterwards, so updating
the node pack never overwrites a list somebody has curated. The node's "Open
model list" button opens that copy.

**New entries are merged in, though**, because "we will not overwrite your list"
turned into "you will never see a model added after you installed" -- a silent
one. Somebody on 0.6.0 who updated to 0.6.2 kept getting the old quant list, with
nothing anywhere to say the node knew about more.

The merge is set algebra, not a version comparison. Beside the lists the live
file records ``seed_offered``: every name the packaged list has *ever* put in
front of this installation. An update then adds exactly

    names in the seed  -  names in your file  -  names you were already offered

so a model you deleted stays deleted, a model you renamed is not duplicated, and
a genuinely new one arrives. The one exception is unavoidable and happens once:
a file written before this mechanism existed has no record of what it was
offered, so on the first update everything missing is added back, including
anything deleted by hand. The previous file is kept beside it as ``.bak``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass

from .constants import ADAPTER_REPO

log = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_FILE = os.path.join(PACKAGE_DIR, "models.json")
USER_SUBDIR = "minimax_h3_rewriter"
FILE_NAME = "models.json"
BACKUP_SUFFIX = ".bak"

FORMAT_TRANSFORMERS = "transformers"
FORMAT_GGUF = "gguf"
FORMATS = (FORMAT_TRANSFORMERS, FORMAT_GGUF)

PLACEHOLDER = "REPLACE_ME"

SECTIONS = ("models", "writers", "captioners")

OFFERED_KEY = "seed_offered"
VERSION_KEY = "seed_version"


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


def _names(entries) -> list[str]:
    if not isinstance(entries, list):
        return []
    return [str(raw["name"]) for raw in entries if isinstance(raw, dict) and raw.get("name")]


def merge(live: dict, seed: dict) -> tuple[dict, list[str]]:
    """Fold new seed entries into a live list. Returns ``(merged, what changed)``.

    Pure, so the rule is testable without a filesystem: nothing here reads or
    writes anything.
    """
    merged = dict(live)
    offered = dict(merged.get(OFFERED_KEY) or {})
    changes: list[str] = []

    for section in SECTIONS:
        available = seed.get(section)
        if not isinstance(available, list) or not available:
            continue

        current = merged.get(section)
        if not isinstance(current, list):
            # The section did not exist at all -- this installation predates it.
            merged[section] = list(available)
            changes.append(f"{section}: added {len(available)} (section is new)")
        else:
            known = set(_names(current))
            seen = set(offered.get(section) or [])
            fresh = [
                raw for raw in available
                if isinstance(raw, dict) and raw.get("name")
                and raw["name"] not in known and raw["name"] not in seen
            ]
            if fresh:
                merged[section] = current + fresh
                changes.append(f"{section}: added {', '.join(_names(fresh))}")

        offered[section] = sorted(set(offered.get(section) or []) | set(_names(available)))

    if offered == (live.get(OFFERED_KEY) or {}) and not changes:
        return merged, changes

    merged[OFFERED_KEY] = offered
    version = seed.get(VERSION_KEY)
    if version:
        merged[VERSION_KEY] = version
    return merged, changes


def _write(path: str, data: dict) -> None:
    """Replace the live file atomically, keeping one step back as ``.bak``."""
    directory = os.path.dirname(path) or "."
    handle, staging = tempfile.mkstemp(prefix=FILE_NAME, suffix=".part", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        if os.path.isfile(path):
            shutil.copyfile(path, path + BACKUP_SUFFIX)
        os.replace(staging, path)
    except OSError:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise


_DATA_CACHE: dict[tuple, dict] = {}


def _data() -> dict:
    """The live list, with anything new from the packaged one folded in.

    Cached per file identity: ``INPUT_TYPES`` runs on every graph validation and
    re-reading two files each time would be wasteful. Writing the merge back
    changes the mtime, so the next call re-reads and finds nothing left to do.
    """
    path = user_file()
    try:
        stat = os.stat(path)
        key = (os.path.normcase(path), stat.st_size, int(stat.st_mtime))
    except OSError:
        key = None

    if key is not None:
        cached = _DATA_CACHE.get(key)
        if cached is not None:
            return cached

    live = _read(path)
    seed = _read(SEED_FILE)
    if not live:
        return seed

    merged, changes = merge(live, seed)
    if merged != live and path != SEED_FILE:
        try:
            _write(path, merged)
            for line in changes:
                log.info("[minimax_h3_rewriter.catalog] %s", line)
            if changes:
                log.info(
                    "[minimax_h3_rewriter.catalog] merged into %s (previous copy at %s)",
                    path, path + BACKUP_SUFFIX,
                )
        except OSError as error:
            log.warning("[minimax_h3_rewriter.catalog] could not update %s: %s", path, error)
        else:
            try:
                stat = os.stat(path)
                key = (os.path.normcase(path), stat.st_size, int(stat.st_mtime))
            except OSError:
                key = None

    if key is not None:
        _DATA_CACHE.clear()
        _DATA_CACHE[key] = merged
    return merged


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
    """General-purpose models offered by the guided writer nodes."""
    return _entries(_data(), "writers")


def captioners() -> list[CatalogEntry]:
    """Multimodal models offered by the reference captioner node.

    Shorter than it looks like it should be. Publishing a GGUF and an mmproj is
    not the same as llama.cpp's ``mtmd`` being able to load them -- several
    current models abort outright -- so this list holds only the ones that have
    actually been run.
    """
    return _entries(_data(), "captioners")


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
