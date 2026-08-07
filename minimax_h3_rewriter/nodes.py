"""The ComfyUI nodes exposed by this package."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from . import catalog, cli_engine, discovery, download, engine, gguf_engine, llamacpp
from .catalog import FORMAT_GGUF, FORMAT_TRANSFORMERS
from .constants import (
    ADAPTER_FILES,
    ADAPTER_REPO,
    ATTN_IMPLEMENTATIONS,
    BASE_MODEL_REPO,
    BASE_SKIP_SUFFIXES,
    DURATION_MAX,
    DURATION_MIN,
    OUTPUT_FIELDS,
    QUANTIZATIONS,
    RESOLUTIONS,
)
from .fields import split_fields
from .paths import adapter_is_complete, base_model_is_complete, models_root, resolve_source
from .progress import NodeProgress, TransferReporter

log = logging.getLogger(__name__)

CATEGORY = "MiniMax-H3"
OPTIONS_TYPE = "H3_REWRITER_OPTIONS"

DEFAULT_OPTIONS = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "attn_implementation": "sdpa",
    "adapter": ADAPTER_REPO,
    "use_lora": True,
    "auto_download": True,
    "gpu_layers": -1,
    "n_ctx": 8192,
    "llama_backend": "auto",
}

BASE_SPEC = {
    "default_repo": BASE_MODEL_REPO,
    "complete": base_model_is_complete,
    "allow": None,
    "skip_suffixes": BASE_SKIP_SUFFIXES,
    "label": "Base model",
}
ADAPTER_SPEC = {
    "default_repo": ADAPTER_REPO,
    "complete": adapter_is_complete,
    "allow": ADAPTER_FILES,
    "skip_suffixes": (),
    "label": "Prompt-rewriter LoRA",
}

LOCAL_PREFIX = "on disk: "


@dataclass
class Choice:
    """Where a chosen model lives and how it has to be run."""

    reference: str
    fmt: str = FORMAT_TRANSFORMERS
    file: str = ""
    local: bool = False


_MODEL_MAP: dict[str, Choice] = {}


def _build_model_map() -> dict[str, Choice]:
    mapping: dict[str, Choice] = {}
    try:
        for entry in catalog.load():
            mapping[entry.label] = Choice(reference=entry.repo, fmt=entry.fmt, file=entry.file)
    except Exception:
        log.warning("[minimax_h3_rewriter._build_model_map] catalog unreadable", exc_info=True)
    try:
        for label, path in discovery.scan_local():
            mapping[f"{LOCAL_PREFIX}{label}"] = Choice(reference=path, local=True)
    except Exception:
        log.warning("[minimax_h3_rewriter._build_model_map] local scan failed", exc_info=True)
    try:
        for label, path in discovery.scan_local_gguf():
            mapping[f"{LOCAL_PREFIX}{label}"] = Choice(reference=path, fmt=FORMAT_GGUF, local=True)
    except Exception:
        log.warning("[minimax_h3_rewriter._build_model_map] gguf scan failed", exc_info=True)

    _MODEL_MAP.clear()
    _MODEL_MAP.update(mapping)
    return mapping


def model_choices() -> list[str]:
    choices = list(_build_model_map())
    return choices or [BASE_MODEL_REPO]


def _resolve_model_choice(choice: str) -> Choice:
    found = _MODEL_MAP.get(choice)
    if found is None:
        # The map is built when the frontend asks for object_info; rebuild in case
        # this process never served that request, or the list changed since.
        found = _build_model_map().get(choice)
    if found is not None:
        return found
    if choice and choice.lower().endswith(".gguf") and os.path.isfile(choice):
        return Choice(reference=choice, fmt=FORMAT_GGUF, local=True)
    if choice and ("/" in choice or os.path.isabs(choice)):
        return Choice(reference=choice)
    raise RuntimeError(
        f"'{choice}' is not in the model list any more. Pick another entry, or add it back "
        f"with the 'Open model list' button ({catalog.user_file()})."
    )


def _verify_base_model(reference: str, progress: NodeProgress) -> None:
    """Refuse a wrong or unloadable base model before any weights move."""
    if os.path.isdir(reference):
        report = discovery.inspect_local(reference)
    else:
        repo_id, local_dir = resolve_source(reference, BASE_MODEL_REPO)
        if os.path.isdir(local_dir) and discovery.read_local_config(local_dir):
            report = discovery.inspect_local(local_dir)
        elif repo_id:
            progress.text(f"Checking {repo_id}", force=True)
            report = discovery.inspect_repo(repo_id)
            if not report.details:
                return
        else:
            return

    if not report.usable:
        raise RuntimeError("This base model cannot run the prompt-rewriter LoRA.\n" + report.summary())
    log.info(
        "[minimax_h3_rewriter._verify_base_model] %s is usable (%s)",
        report.source, report.quant_method,
    )


def _fetch(repo_id: str, dest_dir: str, allow, skip_suffixes, progress: NodeProgress) -> None:
    reporter = TransferReporter(progress, 1, f"Downloading {repo_id}")
    download.sync_repo(
        repo_id,
        dest_dir,
        allow=allow,
        skip_suffixes=skip_suffixes,
        on_progress=reporter,
        on_status=lambda message: progress.text(message, force=True),
        on_total=reporter.set_total,
    )


def _ensure_present(value: str, spec: dict, auto_download: bool, progress: NodeProgress) -> str:
    """Return a local directory holding the model, downloading it when allowed."""
    repo_id, local_dir = resolve_source(value, spec["default_repo"])
    if spec["complete"](local_dir):
        return local_dir

    if not repo_id:
        raise RuntimeError(
            f"{spec['label']} was not found in '{local_dir}'. Point it at a Hugging Face "
            f"repository id (for example '{spec['default_repo']}') or a complete local folder."
        )
    if not auto_download:
        raise RuntimeError(
            f"{spec['label']} is missing from '{local_dir}' and auto_download is off. "
            f"Enable it, or fetch '{repo_id}' manually into that folder."
        )

    _fetch(repo_id, local_dir, spec["allow"], spec["skip_suffixes"], progress)
    if not spec["complete"](local_dir):
        raise RuntimeError(f"{spec['label']} is still incomplete after downloading into '{local_dir}'.")
    return local_dir


def _ensure_file(repo_id: str, filename: str, label: str, auto_download: bool, progress: NodeProgress) -> str:
    """Return a local path to one file of a repository, fetching it when allowed."""
    if not filename:
        raise RuntimeError(f"{label}: no file name given for repository '{repo_id}'.")

    destination = os.path.join(models_root(), filename)
    if os.path.isfile(destination) and os.path.getsize(destination) > 0:
        return destination
    if not auto_download:
        raise RuntimeError(
            f"{label} is missing from '{destination}' and auto_download is off. "
            f"Enable it, or fetch '{filename}' from '{repo_id}' into that folder."
        )

    _fetch(repo_id, models_root(), (filename,), (), progress)
    if not os.path.isfile(destination):
        raise RuntimeError(f"{label}: '{filename}' was not present after downloading from '{repo_id}'.")
    return destination


def _resolve_adapter(fmt: str, setting: str, auto_download: bool, progress: NodeProgress) -> str:
    """Locate the adapter matching the base model's format."""
    value = (setting or "").strip()

    if fmt == FORMAT_TRANSFORMERS:
        return _ensure_present(value or ADAPTER_REPO, ADAPTER_SPEC, auto_download, progress)

    if value.lower().endswith(".gguf"):
        for candidate in (value, os.path.join(models_root(), value)):
            if os.path.isfile(candidate):
                return candidate
        raise RuntimeError(f"Prompt-rewriter LoRA: '{value}' does not exist.")

    spec = catalog.adapter(FORMAT_GGUF)
    if not spec.configured:
        try:
            nearby = [path for _label, path in discovery.scan_local_gguf_adapters()]
        except Exception:
            nearby = []
        hint = f"\nAdapters found on disk:\n  " + "\n  ".join(nearby) if nearby else ""
        raise RuntimeError(
            "No GGUF build of the prompt-rewriter LoRA is configured. Put the path to a "
            "converted '.gguf' adapter in the options node's 'adapter' field, or set "
            f"adapters.gguf.repo in {catalog.user_file()}. Turning 'use_lora' off runs the "
            "plain base model instead." + hint
        )
    return _ensure_file(spec.repo, spec.file, "Prompt-rewriter LoRA", auto_download, progress)


class MiniMaxH3RewriterOptions:
    """Decoding and loading settings for the prompt rewriter."""

    DESCRIPTION = (
        "Optional settings for the MiniMax-H3 Prompt Rewriter. Leave it unconnected and the "
        "node uses the decoding parameters the adapter was published with."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
                "attn_implementation": (
                    list(ATTN_IMPLEMENTATIONS),
                    {"default": "sdpa", "tooltip": "Non-GGUF models only."},
                ),
            },
            "optional": {
                "adapter": (
                    "STRING",
                    {
                        "default": ADAPTER_REPO,
                        "tooltip": (
                            "Repository id or local folder of the LoRA. For a GGUF base model, "
                            "give the path to a converted '.gguf' adapter instead."
                        ),
                    },
                ),
                "use_lora": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Turn off to run the plain Qwen3.6-27B baseline."},
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Fetch missing weights from Hugging Face. Turn off to fail instead.",
                    },
                ),
                "gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 999,
                        "tooltip": (
                            "GGUF only: layers to put on the GPU. -1 is all of them; lower it to "
                            "fit a smaller card, at the cost of speed."
                        ),
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 2048,
                        "max": 131072,
                        "step": 1024,
                        "tooltip": "GGUF only: context size llama.cpp allocates.",
                    },
                ),
                "llama_backend": (
                    list(llamacpp.BACKENDS),
                    {
                        "default": "auto",
                        "tooltip": (
                            "GGUF only, and only when llama-cpp-python is absent: which official "
                            "llama.cpp build to fetch. 'auto' takes CUDA on Windows with a "
                            "supported NVIDIA card (511 MB, about twice as fast) and Vulkan "
                            "otherwise. Pick 'vulkan' to keep the download at 34 MB."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (OPTIONS_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, **kwargs):
        options = dict(DEFAULT_OPTIONS)
        options.update(kwargs)
        return (options,)


class MiniMaxH3PromptRewriter:
    """Rewrite a short prompt into a structured MiniMax-H3 T2VA description."""

    DESCRIPTION = (
        "Runs the LightX2V MiniMax-H3 T2VA Prompt Rewriter LoRA on Qwen3.6-27B and returns a "
        "structured audio-video description. Weights are downloaded on first use, with progress "
        "shown on the node. GGUF models run through llama-cpp-python when it is installed, and "
        "otherwise through the official llama.cpp binaries, fetched on first use."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "The short prompt to expand into an H3 audio-video description.",
                    },
                ),
                "model": (
                    model_choices(),
                    {
                        "tooltip": (
                            "Base model. Entries prefixed 'on disk:' are already downloaded; the "
                            "rest are fetched on first use. GGUF entries need no extra install: "
                            "without llama-cpp-python the node fetches the official llama.cpp "
                            "binaries. Use the button to edit the list."
                        ),
                    },
                ),
                "resolution": (
                    list(RESOLUTIONS),
                    {"default": "16:9", "tooltip": "Target aspect ratio the rewrite is composed for."},
                ),
                "duration": (
                    "INT",
                    {
                        "default": 10,
                        "min": DURATION_MIN,
                        "max": DURATION_MAX,
                        "step": 1,
                        "tooltip": "Target clip length in seconds; drives shot count and pacing.",
                    },
                ),
                "quantization": (
                    list(QUANTIZATIONS),
                    {
                        "default": "nf4",
                        "tooltip": (
                            "How to load an unquantized checkpoint: nf4 needs about 16 GB of VRAM, "
                            "int8 about 28 GB, bfloat16 about 54 GB. Ignored for GGUF models and "
                            "for checkpoints that are already quantized."
                        ),
                    },
                ),
                "greedy": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Deterministic decoding. Turn off to sample; see the options node.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        # Capped at 2**32-1 so "randomize" cannot produce a value
                        # the backends have to fold, which would make the seed on
                        # screen differ from the one that was used.
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Keep the 27B model in VRAM after the rewrite. Leave off when the same "
                            "GPU has to run MiniMax-H3 video generation afterwards."
                        ),
                    },
                ),
            },
            "optional": {
                "options": (OPTIONS_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("rewritten_prompt",) + OUTPUT_FIELDS
    FUNCTION = "rewrite"
    CATEGORY = CATEGORY

    def rewrite(
        self,
        prompt,
        model,
        resolution,
        duration,
        quantization,
        greedy,
        seed,
        keep_model_loaded,
        options=None,
        unique_id=None,
    ):
        if not (prompt or "").strip():
            raise ValueError("prompt must not be empty")

        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)

        progress = NodeProgress(unique_id)
        choice = _resolve_model_choice(model)

        decoding = {
            "prompt": prompt,
            "resolution": resolution,
            "duration": int(duration),
            "seed": int(seed),
            "greedy": greedy,
            "max_new_tokens": int(settings["max_new_tokens"]),
            "temperature": float(settings["temperature"]),
            "top_p": float(settings["top_p"]),
            "top_k": int(settings["top_k"]),
            "repetition_penalty": float(settings["repetition_penalty"]),
        }

        if choice.fmt == FORMAT_GGUF:
            if choice.local:
                model_path = choice.reference
            else:
                model_path = _ensure_file(
                    choice.reference, choice.file, "Base model", settings["auto_download"], progress
                )
            adapter_path = None
            if settings["use_lora"]:
                adapter_path = _resolve_adapter(
                    FORMAT_GGUF, settings["adapter"], settings["auto_download"], progress
                )
            common = dict(
                model_path=model_path,
                adapter_path=adapter_path,
                gpu_layers=int(settings["gpu_layers"]),
                n_ctx=int(settings["n_ctx"]),
                keep_loaded=keep_model_loaded,
                progress=progress,
                **decoding,
            )
            # llama-cpp-python when it is installed, official binaries when it is
            # not. The wheel is faster to start and can hold the model between
            # runs; the binaries need nothing installed and cannot be broken by
            # a CPU without AVX-512 or a driver older than the wheel's PTX.
            if gguf_engine.available():
                text = gguf_engine.rewrite(**common)
            else:
                text = cli_engine.rewrite(
                    backend=settings["llama_backend"],
                    auto_download=settings["auto_download"],
                    **common,
                )
        else:
            _verify_base_model(choice.reference, progress)
            base_dir = _ensure_present(choice.reference, BASE_SPEC, settings["auto_download"], progress)
            adapter_dir = None
            if settings["use_lora"]:
                adapter_dir = _resolve_adapter(
                    FORMAT_TRANSFORMERS, settings["adapter"], settings["auto_download"], progress
                )
            text = engine.rewrite(
                base_dir=base_dir,
                adapter_dir=adapter_dir,
                quantization=quantization,
                attn_implementation=settings["attn_implementation"],
                keep_loaded=keep_model_loaded,
                progress=progress,
                **decoding,
            )

        fields = split_fields(text)
        progress.text(text[-2000:] if text else "(empty rewrite)", force=True)
        return (text,) + tuple(fields[name] for name in OUTPUT_FIELDS)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptRewriter": MiniMaxH3PromptRewriter,
    "MiniMaxH3RewriterOptions": MiniMaxH3RewriterOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptRewriter": "MiniMax-H3 Prompt Rewriter",
    "MiniMaxH3RewriterOptions": "MiniMax-H3 Rewriter Options",
}
