"""The ComfyUI nodes exposed by this package."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from . import (
    catalog,
    cli_engine,
    discovery,
    download,
    engine,
    gguf_engine,
    guide_prompt,
    guides,
    llamacpp,
)
from .catalog import FORMAT_GGUF, FORMAT_TRANSFORMERS
from .constants import (
    ADAPTER_FILES,
    ADAPTER_REPO,
    ATTN_IMPLEMENTATIONS,
    BASE_MODEL_REPO,
    BASE_SKIP_SUFFIXES,
    DURATION_MAX,
    DURATION_MIN,
    GGUF_RUNTIMES,
    OUTPUT_FIELDS,
    QUANTIZATIONS,
    REF_OUTPUT_FIELDS,
    RESOLUTIONS,
    RUNTIME_AUTO,
    RUNTIME_BINARY,
    RUNTIME_WHEEL,
)
from .fields import missing, split_fields, split_sections
from .paths import adapter_is_complete, base_model_is_complete, models_root, resolve_source
from .progress import NodeProgress, TransferReporter
from .prompt_template import build_messages

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
    "gguf_runtime": RUNTIME_AUTO,
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


_WRITER_MAP: dict[str, Choice] = {}


def _build_writer_map() -> dict[str, Choice]:
    """The guided writers' model list: any GGUF, not just the adapter's base.

    Nothing here is verified against an architecture, because there is nothing to
    match: the format lives in the system prompt, so the only requirement is that
    the file is a GGUF language model with a chat template. That is what makes a
    4B on an 8 GB card a real answer rather than a consolation prize.
    """
    mapping: dict[str, Choice] = {}
    try:
        for entry in catalog.writers():
            mapping[entry.label] = Choice(reference=entry.repo, fmt=FORMAT_GGUF, file=entry.file)
    except Exception:
        log.warning("[minimax_h3_rewriter._build_writer_map] catalog unreadable", exc_info=True)
    try:
        for label, path in discovery.scan_writer_gguf():
            mapping[f"{LOCAL_PREFIX}{label}"] = Choice(reference=path, fmt=FORMAT_GGUF, local=True)
    except Exception:
        log.warning("[minimax_h3_rewriter._build_writer_map] gguf scan failed", exc_info=True)

    _WRITER_MAP.clear()
    _WRITER_MAP.update(mapping)
    return mapping


def writer_choices() -> list[str]:
    choices = list(_build_writer_map())
    return choices or ["(no GGUF model found — see the model list)"]


def _resolve_writer_choice(choice: str) -> Choice:
    found = _WRITER_MAP.get(choice)
    if found is None:
        found = _build_writer_map().get(choice)
    if found is not None:
        return found
    if choice and choice.lower().endswith(".gguf") and os.path.isfile(choice):
        return Choice(reference=choice, fmt=FORMAT_GGUF, local=True)
    raise RuntimeError(
        f"'{choice}' is not in the writer model list any more. Pick another entry, drop a "
        f"'.gguf' into ComfyUI's models/LLM folder, or add it under \"writers\" in "
        f"{catalog.user_file()}."
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
                "gguf_runtime": (
                    list(GGUF_RUNTIMES),
                    {
                        "default": RUNTIME_AUTO,
                        "tooltip": (
                            "GGUF only: what runs the model. 'auto' uses llama-cpp-python when "
                            "it is importable and the official llama.cpp binaries otherwise. "
                            "Force 'llama.cpp' if an installed wheel is broken; force "
                            "'llama-cpp-python' to keep the model resident between runs, which "
                            "the binaries cannot do."
                        ),
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
            "messages": build_messages(prompt, resolution, int(duration)),
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
                problem = discovery.gguf_problem(model_path)
                if problem:
                    raise RuntimeError(
                        "This GGUF cannot run the prompt-rewriter LoRA.\n  - "
                        + problem
                        + "\nTurn 'use_lora' off to run it as a plain model anyway."
                    )
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
            runtime = settings.get("gguf_runtime", RUNTIME_AUTO)
            if runtime == RUNTIME_AUTO:
                runtime = RUNTIME_WHEEL if gguf_engine.available() else RUNTIME_BINARY

            if runtime == RUNTIME_WHEEL:
                if not gguf_engine.available():
                    raise RuntimeError(
                        f"gguf_runtime is set to '{RUNTIME_WHEEL}', but it is not importable "
                        f"here. Install it, or set gguf_runtime to '{RUNTIME_AUTO}' or "
                        f"'{RUNTIME_BINARY}' to run the official binaries instead.\n\n"
                        + gguf_engine.INSTALL_HINT
                    )
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


REFERENCE_PLACEHOLDER = (
    "Describe what the reference frames show, one per line:\n"
    "Picture 1: ...\n"
)

REFERENCE_ASSETS_PLACEHOLDER = (
    "List every reference asset, one per line:\n"
    "Picture 1: young woman, long dark hair, blue cardigan, seated by a window\n"
    "Video 1: source clip being edited — handheld walk down a night street\n"
    "Audio 1: voice-timbre reference for the woman\n"
)


def _guided_text(
    mode: str,
    model: str,
    prompt: str,
    resolution: str,
    duration: int,
    references: str,
    greedy: bool,
    seed: int,
    keep_model_loaded: bool,
    settings: dict,
    progress: NodeProgress,
) -> str:
    """Run one guided rewrite and return the model's raw answer."""
    choice = _resolve_writer_choice(model)
    if choice.local:
        model_path = choice.reference
    else:
        model_path = _ensure_file(
            choice.reference, choice.file, "Writer model", settings["auto_download"], progress
        )

    if discovery.gguf_header(model_path)["kind"] == "adapter":
        raise RuntimeError(
            f"'{os.path.basename(model_path)}' is a LoRA adapter, not a model that can be run "
            f"on its own. Pick a base model from the list."
        )

    guide = guides.text(
        guide_prompt.GUIDE_FOR_MODE[mode], settings["auto_download"], progress
    )
    messages = guide_prompt.build_messages(
        guide, mode, prompt, resolution, int(duration), references
    )

    max_new_tokens = int(settings["max_new_tokens"])
    n_ctx = int(settings["n_ctx"])
    needed = guide_prompt.context_needed(messages, max_new_tokens)
    if needed > n_ctx:
        log.info(
            "[minimax_h3_rewriter._guided_text] raising n_ctx from %d to %d for the %s guide",
            n_ctx, needed, mode,
        )
        progress.text(
            f"{mode}: the guide needs a {needed}-token context, raising n_ctx from {n_ctx}",
            force=True,
        )
        n_ctx = needed

    common = dict(
        model_path=model_path,
        adapter_path=None,
        gpu_layers=int(settings["gpu_layers"]),
        n_ctx=n_ctx,
        keep_loaded=keep_model_loaded,
        progress=progress,
        messages=messages,
        seed=int(seed),
        greedy=greedy,
        max_new_tokens=max_new_tokens,
        temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]),
        top_k=int(settings["top_k"]),
        repetition_penalty=float(settings["repetition_penalty"]),
    )

    runtime = settings.get("gguf_runtime", RUNTIME_AUTO)
    if runtime == RUNTIME_AUTO:
        runtime = RUNTIME_WHEEL if gguf_engine.available() else RUNTIME_BINARY

    if runtime == RUNTIME_WHEEL:
        if not gguf_engine.available():
            raise RuntimeError(
                f"gguf_runtime is set to '{RUNTIME_WHEEL}', but it is not importable here. "
                f"Install it, or set gguf_runtime to '{RUNTIME_AUTO}' or '{RUNTIME_BINARY}' to "
                f"run the official binaries instead.\n\n" + gguf_engine.INSTALL_HINT
            )
        return gguf_engine.rewrite(**common)
    return cli_engine.rewrite(
        backend=settings["llama_backend"],
        auto_download=settings["auto_download"],
        **common,
    )


def _report(progress: NodeProgress, text: str, sections: dict, names: tuple[str, ...]) -> None:
    """Leave the answer on the node, and say so when it is not the right shape."""
    absent = missing(sections, names)
    note = ""
    if absent:
        note = (
            f"⚠ {len(absent)} field(s) not found in the answer: {', '.join(absent)}\n"
            f"Lower the temperature, or try a larger writer model.\n\n"
        )
        log.warning("[minimax_h3_rewriter._report] fields missing from the rewrite: %s", absent)
    progress.text(note + (text[-2000:] if text else "(empty rewrite)"), force=True)


class MiniMaxH3GuidedWriter:
    """Write a T2VA/I2VA/FL2VA/L2VA prompt with any model, guided rather than trained."""

    DESCRIPTION = (
        "Writes a MiniMax-H3 audio-video description from MiniMax's own writing guide, which "
        "goes into the system prompt. No LoRA, so any instruction-following GGUF can run it — "
        "a 4B on an 8 GB card instead of Qwen3.6-27B. The guide is fetched from "
        "MiniMaxAI/MiniMax-H3 on first use. Outputs match the LoRA rewriter node, so the two "
        "are interchangeable in a workflow."
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
                    writer_choices(),
                    {
                        "tooltip": (
                            "Any GGUF language model. Entries prefixed 'on disk:' are already "
                            "in your ComfyUI model folders; the rest are fetched on first use. "
                            "Nothing has to be installed: without llama-cpp-python the node "
                            "runs the official llama.cpp binaries."
                        ),
                    },
                ),
                "task": (
                    list(guide_prompt.BASE_MODES),
                    {
                        "default": "T2VA",
                        "tooltip": (
                            "T2VA: text only. I2VA: the reference image is the first frame. "
                            "FL2VA: first and last frame. L2VA: the reference image is the last "
                            "frame. Everything but T2VA also writes the alignment instruction "
                            "line, with the duration already filled in."
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
                "greedy": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Deterministic decoding. Worth keeping on for small models, which "
                            "drift out of the format when they sample."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
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
                            "Keep the writer in VRAM after the rewrite. Leave off when the same "
                            "GPU has to run MiniMax-H3 video generation afterwards."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_material": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "This node reads text, not pixels. For I2VA, FL2VA and L2VA, "
                            "describe what the reference frames show — by hand, or from a "
                            "captioner node — so the rewrite is anchored to them.\n\n"
                            + REFERENCE_PLACEHOLDER
                        ),
                    },
                ),
                "options": (OPTIONS_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",) * (1 + len(OUTPUT_FIELDS))
    RETURN_NAMES = ("rewritten_prompt",) + OUTPUT_FIELDS
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(
        self,
        prompt,
        model,
        task,
        resolution,
        duration,
        greedy,
        seed,
        keep_model_loaded,
        reference_material="",
        options=None,
        unique_id=None,
    ):
        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)
        progress = NodeProgress(unique_id)

        text = _guided_text(
            task, model, prompt, resolution, duration, reference_material,
            greedy, seed, keep_model_loaded, settings, progress,
        )
        _head, sections = split_sections(text, OUTPUT_FIELDS)
        _report(progress, text, sections, OUTPUT_FIELDS)
        return (text,) + tuple(sections[name] for name in OUTPUT_FIELDS)


class MiniMaxH3GuidedWriterRef:
    """Write a full-reference (Ref2VA) prompt with any model."""

    DESCRIPTION = (
        "Writes a MiniMax-H3 full-reference (Ref2VA) description — six sections, with "
        "<Subject>/<Picture>/<Video>/<Audio> labels and a retention analysis — from MiniMax's "
        "own full-reference guide, which goes into the system prompt. No LoRA, so any "
        "instruction-following GGUF can run it. The guide is fetched from MiniMaxAI/MiniMax-H3 "
        "on first use."
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
                        "tooltip": "What the target video should show, and how it uses the references.",
                    },
                ),
                "reference_assets": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Required. One asset per line — the node reads text, not pixels, so "
                            "this is all the model knows about them. Label them Picture N, "
                            "Video N or Audio N and say what each one is for.\n\n"
                            + REFERENCE_ASSETS_PLACEHOLDER
                        ),
                    },
                ),
                "model": (
                    writer_choices(),
                    {
                        "tooltip": (
                            "Any GGUF language model. The full-reference guide is the longer of "
                            "the two, so a 4B will hold the format but a 9B keeps the labels "
                            "consistent across all six sections."
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
                "greedy": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Deterministic decoding. Keep it on unless the result is too plain.",
                    },
                ),
                "seed": (
                    "INT",
                    {
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
                        "tooltip": "Keep the writer in VRAM after the rewrite.",
                    },
                ),
            },
            "optional": {
                "options": (OPTIONS_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",) * (1 + len(REF_OUTPUT_FIELDS))
    RETURN_NAMES = ("rewritten_prompt",) + REF_OUTPUT_FIELDS
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(
        self,
        prompt,
        reference_assets,
        model,
        resolution,
        duration,
        greedy,
        seed,
        keep_model_loaded,
        options=None,
        unique_id=None,
    ):
        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)
        progress = NodeProgress(unique_id)

        text = _guided_text(
            guide_prompt.REF_MODE, model, prompt, resolution, duration, reference_assets,
            greedy, seed, keep_model_loaded, settings, progress,
        )
        _head, sections = split_sections(text, REF_OUTPUT_FIELDS, fallback="detailed_description")
        _report(progress, text, sections, REF_OUTPUT_FIELDS)
        return (text,) + tuple(sections[name] for name in REF_OUTPUT_FIELDS)


class MiniMaxH3GuidePrompt:
    """The guide-based system and user messages, for any other LLM node to run."""

    DESCRIPTION = (
        "Builds the system and user prompt from MiniMax's writing guide and returns them as "
        "strings, without running anything. Wire them into whatever LLM node you already use — "
        "local, API, or remote — when you would rather not run the model here. Costs no VRAM "
        "and no time."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "task": (
                    list(guide_prompt.ALL_MODES),
                    {
                        "default": "T2VA",
                        "tooltip": "Ref2VA uses the full-reference guide and its six output sections.",
                    },
                ),
                "resolution": (list(RESOLUTIONS), {"default": "16:9"}),
                "duration": (
                    "INT",
                    {"default": 10, "min": DURATION_MIN, "max": DURATION_MAX, "step": 1},
                ),
            },
            "optional": {
                "reference_material": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "What the reference frames or assets show. Required for Ref2VA.\n\n"
                            + REFERENCE_ASSETS_PLACEHOLDER
                        ),
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Fetch the guide from MiniMaxAI/MiniMax-H3 if it is not already in "
                            "the ComfyUI user directory."
                        ),
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "user_prompt")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        prompt,
        task,
        resolution,
        duration,
        reference_material="",
        auto_download=True,
        unique_id=None,
    ):
        progress = NodeProgress(unique_id)
        guide = guides.text(guide_prompt.GUIDE_FOR_MODE[task], auto_download, progress)
        messages = guide_prompt.build_messages(
            guide, task, prompt, resolution, int(duration), reference_material
        )
        system, user = messages[0]["content"], messages[1]["content"]
        progress.finish(
            f"{task} · system {len(system)} chars · user {len(user)} chars\n"
            f"about {guide_prompt.context_needed(messages, 0)} tokens of context before the answer"
        )
        return (system, user)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptRewriter": MiniMaxH3PromptRewriter,
    "MiniMaxH3RewriterOptions": MiniMaxH3RewriterOptions,
    "MiniMaxH3GuidedWriter": MiniMaxH3GuidedWriter,
    "MiniMaxH3GuidedWriterRef": MiniMaxH3GuidedWriterRef,
    "MiniMaxH3GuidePrompt": MiniMaxH3GuidePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptRewriter": "MiniMax-H3 Prompt Rewriter",
    "MiniMaxH3RewriterOptions": "MiniMax-H3 Rewriter Options",
    "MiniMaxH3GuidedWriter": "MiniMax-H3 Prompt Writer (T2VA/I2VA/FL2VA/L2VA)",
    "MiniMaxH3GuidedWriterRef": "MiniMax-H3 Prompt Writer (Ref2VA)",
    "MiniMaxH3GuidePrompt": "MiniMax-H3 Guide Prompt (any LLM)",
}
