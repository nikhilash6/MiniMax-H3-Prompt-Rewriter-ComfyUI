"""Finding a usable Qwen3.6-27B and deciding whether it will actually work.

The adapter is bound to one base model, but that base ships in many builds — the
52 GB bf16 original plus FP8, AWQ, GPTQ and NVFP4 repackings around 19-29 GB.
Every one of them keeps the same architecture fingerprint in ``config.json``, so
a 4 KB fetch answers "is this the right model" before any of the weights move.

What differs is the *runtime*: a quantized checkpoint needs its own loader
package, and PEFT needs a LoRA dispatcher for that layer type or the adapter
cannot be attached at all. Both are knowable up front, so a repository is
reported as usable, usable-after-a-pip-install, or unsupported — never
discovered 20 GB into a download.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from dataclasses import dataclass, field

from .constants import install_command

log = logging.getLogger(__name__)

#: ``qwen3_5`` is the full multimodal config; ``qwen3_5_text`` is the language
#: model on its own. The adapter only ever touches language-model weights, so a
#: text-only repack is just as usable — and roughly a third of the download.
MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
HIDDEN_SIZE = 5120
NUM_LAYERS = 64
VOCAB_SIZE = 248320

SCAN_FOLDERS = ("LLM", "transformers", "diffusers")
GGUF_FOLDERS = ("LLM", "unet_gguf", "transformers")
CONFIG_NAME = "config.json"

#: what llama.cpp calls this architecture in general.architecture
GGUF_ARCH = "qwen35"
GGUF_SCAN_DEPTH = 2

#: quant_method -> (pip package, import name, PEFT can attach LoRA)
#:
#: 'fp8' looks self-contained -- Transformers ships the integration and PEFT
#: wraps its FP8Linear like any other nn.Linear -- but the forward pass calls
#: out to the `kernels` package, and fails only once generation starts. Listing
#: it here moves that failure to before the download.
QUANT_RUNTIME = {
    "none": ("", "", True),
    "fp8": ("kernels", "kernels", True),
    "bitsandbytes": ("bitsandbytes", "bitsandbytes", True),
    "bitsandbytes_4bit": ("bitsandbytes", "bitsandbytes", True),
    "bitsandbytes_8bit": ("bitsandbytes", "bitsandbytes", True),
    "awq": ("autoawq", "awq", True),
    "gptq": ("gptqmodel", "gptqmodel", True),
    "hqq": ("hqq", "hqq", True),
    "eetq": ("eetq", "eetq", True),
    "aqlm": ("aqlm", "aqlm", True),
    "torchao": ("torchao", "torchao", True),
    "compressed-tensors": ("compressed-tensors", "compressed_tensors", False),
    "modelopt": ("nvidia-modelopt", "modelopt", False),
    "quanto": ("optimum-quanto", "optimum_quanto", False),
    "fbgemm_fp8": ("fbgemm-gpu", "fbgemm_gpu", False),
}

#: quant_method -> what the pip install still does not buy you.
#:
#: Naming the package is honest but incomplete for FP8: `kernels` is a loader,
#: and the matrix multiply itself is downloaded from the Hub the first time a
#: token is generated. Somebody who installs the package and hits *that* has
#: paid for a 28.8 GB download twice over, so the caveat travels with the
#: instruction rather than living only in the README.
QUANT_CAVEAT = {
    "fp8": (
        "installing it is only half of it: the FP8 matmul is a Triton kernel that "
        "transformers then downloads from 'kernels-community/finegrained-fp8' on the "
        "first generation, and that needs a build matching this torch and CUDA version"
    ),
}


@dataclass
class ModelReport:
    """What a ``config.json`` says about a candidate base model."""

    source: str
    usable: bool = False
    architecture_ok: bool = False
    quant_method: str = "none"
    missing_package: str = ""
    lora_supported: bool = True
    details: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "usable" if self.usable else "NOT usable"
        quant = self.quant_method if self.quant_method != "none" else "unquantized"
        lines = [f"{self.source}: {head} ({quant})"]
        lines.extend(f"  - {problem}" for problem in self.problems)
        return "\n".join(lines)


def _text_config(config: dict) -> dict:
    return config.get("text_config") or config


def quant_method(config: dict) -> str:
    for holder in (config, _text_config(config)):
        quant = holder.get("quantization_config")
        if isinstance(quant, dict) and quant.get("quant_method"):
            return str(quant["quant_method"]).lower()
    return "none"


def is_prequantized(config: dict) -> bool:
    return quant_method(config) != "none"


def _installed(import_name: str) -> bool:
    if not import_name:
        return True
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def evaluate(config: dict | None, source: str) -> ModelReport:
    """Judge a candidate from its ``config.json`` alone."""
    report = ModelReport(source=source)
    if not config:
        report.problems.append("config.json is missing or unreadable")
        return report

    text = _text_config(config)
    report.details = {
        "model_type": config.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "vocab_size": text.get("vocab_size"),
    }

    expected = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "vocab_size": VOCAB_SIZE,
    }
    mismatched = [
        f"{key} is {report.details.get(key)!r}, the adapter needs {value!r}"
        for key, value in expected.items()
        if report.details.get(key) != value
    ]
    if report.details.get("model_type") not in MODEL_TYPES:
        mismatched.insert(
            0,
            f"model_type is {report.details.get('model_type')!r}, "
            f"the adapter needs one of {' or '.join(MODEL_TYPES)}",
        )
    report.architecture_ok = not mismatched
    if mismatched:
        report.problems.append(
            "this is not Qwen3.6-27B, so the adapter's LoRA layers do not exist in it"
        )
        report.problems.extend(mismatched)
        return report

    report.quant_method = quant_method(config)
    package, import_name, lora_supported = QUANT_RUNTIME.get(
        report.quant_method, (report.quant_method, report.quant_method.replace("-", "_"), False)
    )
    report.lora_supported = lora_supported

    if not lora_supported:
        # Deliberately *instead of* the missing-package line, not alongside it.
        # The package would load the weights fine; PEFT would still have nowhere
        # to hang the LoRA, so the run fails exactly as it does now. Printing an
        # install command next to "this cannot work" invites somebody to spend a
        # pip install and a 20 GB download proving the second line right.
        report.problems.append(
            f"PEFT has no LoRA dispatcher for '{report.quant_method}' layers, so the adapter "
            f"cannot be attached to this build. No package changes that: pick a bf16 or "
            f"bitsandbytes 4-bit entry from the model list instead."
        )
    elif not _installed(import_name):
        report.missing_package = package
        message = (
            f"the '{report.quant_method}' checkpoint needs the '{package}' package, "
            f"which is not installed in this Python environment. Install it with:\n"
            f"      {install_command(package)}"
        )
        caveat = QUANT_CAVEAT.get(report.quant_method)
        if caveat:
            message += f"\n    Note: {caveat}."
        report.problems.append(message)

    report.usable = report.architecture_ok and lora_supported and not report.missing_package
    return report


def read_local_config(directory: str) -> dict | None:
    path = os.path.join(directory, CONFIG_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def fetch_remote_config(repo_id: str, revision: str = "main") -> dict | None:
    """Fetch only ``config.json`` — 4 KB against a 20-52 GB download."""
    import requests

    from .download import _headers, access_token, endpoint

    url = f"{endpoint()}/{repo_id}/resolve/{revision}/{CONFIG_NAME}"
    try:
        response = requests.get(url, headers=_headers(access_token()), timeout=(15, 30))
    except Exception as error:
        log.warning("[minimax_h3_rewriter.fetch_remote_config] %s: %s", repo_id, error)
        return None
    if response.status_code >= 400:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def inspect_local(directory: str) -> ModelReport:
    return evaluate(read_local_config(directory), directory)


def inspect_repo(repo_id: str, revision: str = "main") -> ModelReport:
    return evaluate(fetch_remote_config(repo_id, revision), repo_id)


def _hf_cache_roots() -> list[str]:
    home = os.environ.get("HF_HOME")
    if home:
        yield_paths = [os.path.join(home, "hub")]
    else:
        yield_paths = [os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")]
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        yield_paths.insert(0, hub)
    return [path for path in yield_paths if os.path.isdir(path)]


def _comfy_roots(names: tuple[str, ...] = SCAN_FOLDERS) -> list[str]:
    try:
        import folder_paths
    except ImportError:
        return []

    roots = []
    for name in names:
        try:
            candidates = folder_paths.get_folder_paths(name)
        except KeyError:
            continue
        # A registered folder need not exist: extra_model_paths.yaml maps them in
        # from anywhere and some entries are simply wrong.
        roots.extend(path for path in candidates if os.path.isdir(path))
    return roots


def _snapshot_dirs(cache_root: str) -> list[str]:
    found = []
    try:
        entries = os.listdir(cache_root)
    except OSError:
        return found
    for entry in entries:
        if not entry.startswith("models--"):
            continue
        snapshots = os.path.join(cache_root, entry, "snapshots")
        if not os.path.isdir(snapshots):
            continue
        try:
            revisions = os.listdir(snapshots)
        except OSError:
            continue
        found.extend(os.path.join(snapshots, revision) for revision in revisions)
    return found


def scan_local() -> list[tuple[str, str]]:
    """Return ``(label, directory)`` for every local checkpoint that fits.

    Both the ComfyUI model folders and the Hugging Face cache are searched, so a
    copy pulled by any other tool is found rather than downloaded twice. A
    directory only qualifies once its weights are actually there: a Hugging Face
    cache entry can hold nothing but ``config.json``, and offering that as a
    choice would fail at load time instead of here.
    """
    from .paths import base_model_is_complete

    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def consider(directory: str, name: str) -> None:
        key = os.path.normcase(os.path.abspath(directory))
        if key in seen:
            return
        seen.add(key)
        config = read_local_config(directory)
        if not config:
            return
        report = evaluate(config, directory)
        if not report.architecture_ok or not base_model_is_complete(directory):
            return
        label = name
        if report.quant_method != "none":
            label += f" [{report.quant_method}]"
        if not report.usable:
            label += " (unusable)"
        found.append((label, directory))

    for root in _comfy_roots():
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for entry in entries:
            consider(os.path.join(root, entry), entry)

    for cache_root in _hf_cache_roots():
        for snapshot in _snapshot_dirs(cache_root):
            repo = os.path.basename(os.path.dirname(os.path.dirname(snapshot)))
            pretty = repo.replace("models--", "", 1).replace("--", "/")
            consider(snapshot, f"HF cache: {pretty}@{os.path.basename(snapshot)[:8]}")

    return found


_GGUF_INFO_CACHE: dict[tuple[str, int, int], tuple[str, str]] = {}


def gguf_info(path: str) -> tuple[str, str]:
    """``(general.architecture, general.type)`` of a GGUF, from its header only.

    The type matters: a converted LoRA carries the *same* architecture as the
    model it was trained on, so architecture alone would offer a 3.5 GB adapter
    as if it were a base model.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return "", ""
    key = (os.path.normcase(path), stat.st_size, int(stat.st_mtime))
    cached = _GGUF_INFO_CACHE.get(key)
    if cached is not None:
        return cached

    arch, kind = "", ""
    try:
        from gguf import GGUFReader

        reader = GGUFReader(path, "r")
        for field_name, target in (("general.architecture", "arch"), ("general.type", "kind")):
            field = reader.fields.get(field_name)
            if field is None:
                continue
            value = str(field.contents())
            if target == "arch":
                arch = value
            else:
                kind = value
        if not kind:
            kind = "adapter" if reader.fields.get("adapter.type") is not None else "model"
    except Exception:
        log.debug("[minimax_h3_rewriter.gguf_info] %s unreadable", path, exc_info=True)

    _GGUF_INFO_CACHE[key] = (arch, kind)
    return arch, kind


def gguf_architecture(path: str) -> str:
    return gguf_info(path)[0]


def _gguf_candidates(root: str, depth: int) -> list[str]:
    found = []
    stack = [(root, 0)]
    while stack:
        directory, level = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if level < depth:
                    stack.append((entry.path, level + 1))
            elif entry.name.lower().endswith(".gguf"):
                found.append(entry.path)
    return found


def _scan_gguf(kind: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    for root in _comfy_roots(GGUF_FOLDERS):
        for path in _gguf_candidates(root, GGUF_SCAN_DEPTH):
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            arch, file_kind = gguf_info(path)
            if arch != GGUF_ARCH or file_kind != kind:
                continue
            try:
                size = os.path.getsize(path) / 1024 ** 3
            except OSError:
                size = 0.0
            found.append((f"{os.path.basename(path)} [gguf, {size:.1f} GB]", path))

    return found


def scan_local_gguf() -> list[tuple[str, str]]:
    """Return ``(label, path)`` for local GGUF *base models* of this architecture.

    Only the header is read, and the answer is cached per file identity, so a
    folder of large quants costs no more than a stat each after the first pass.
    """
    return _scan_gguf("model")


def scan_local_gguf_adapters() -> list[tuple[str, str]]:
    """Return ``(label, path)`` for local GGUF LoRA adapters for this architecture."""
    return _scan_gguf("adapter")
