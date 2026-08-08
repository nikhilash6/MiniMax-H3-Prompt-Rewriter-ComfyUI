"""Running the rewriter from GGUF weights through llama-cpp-python.

The same adapter, converted to GGUF, attaches to a quantised base under
llama.cpp: a ``Q4_K_M`` build of Qwen3.6-27B is 15.7 GB instead of 52 GB, and
smaller quants go lower still. llama-cpp-python is an optional dependency —
absent it, this backend is simply unavailable and the node says so.

The prompt is rendered from the GGUF's own chat template rather than through
llama-cpp-python's chat formatter, because the reference implementation passes
``enable_thinking=False`` and the formatter has no way to forward it.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading

from . import chat_template
from .constants import install_command, normalize_seed
from .progress import NodeProgress

log = logging.getLogger(__name__)

_STATE: dict = {"key": None, "llama": None}
_LOCK = threading.RLock()

PREVIEW_TAIL = 280
RELEASES_URL = "https://github.com/abetlen/llama-cpp-python/releases"
WHEEL_RELEASE = "v0.3.34-vulkan"
WHEEL_FILES = {
    "win32": "llama_cpp_python-0.3.34-py3-none-win_amd64.whl",
    "linux": "llama_cpp_python-0.3.34-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
}


def wheel_url() -> str:
    """The prebuilt Vulkan wheel for this platform, or "" if there is none."""
    name = WHEEL_FILES.get(sys.platform)
    return f"{RELEASES_URL}/download/{WHEEL_RELEASE}/{name}" if name else ""


def _install_hint() -> str:
    head = (
        "GGUF models need llama-cpp-python, which is not installed in this Python "
        "environment.\n"
    )
    tail = (
        "\nInstalling it is optional: without it the node runs GGUF models through the "
        "official llama.cpp binaries instead, which it fetches on first use. The wheel is "
        "worth having only if you want the model to stay resident between runs."
    )
    url = wheel_url()
    if not url:
        return (
            head
            + f"    {install_command('llama-cpp-python')}\n"
            + f"Prebuilt wheels for other platforms are at {RELEASES_URL}."
            + tail
        )
    return (
        head
        + "Use the Vulkan build. It runs on NVIDIA, AMD and Intel GPUs alike, and "
        "needs no match between the wheel's CUDA version and your driver:\n"
        f"    {install_command(url)}\n"
        "Then restart ComfyUI. The CUDA wheels at that same releases page are "
        "faster when they fit, but cu130 and earlier are compiled with AVX-512 and "
        "die with 0xC000001D on consumer Intel 12th-14th generation CPUs, while "
        "cu132 ships PTX that a driver older than CUDA 13.2 refuses to compile."
        + tail
    )


INSTALL_HINT = _install_hint()


def available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        return False


def _llama_cpp():
    try:
        import llama_cpp
    except ImportError as error:
        raise RuntimeError(INSTALL_HINT) from error
    return llama_cpp


def _free_comfy_vram() -> None:
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception:
        log.debug("[minimax_h3_rewriter.gguf._free_comfy_vram] skipped", exc_info=True)


def _interrupted() -> bool:
    try:
        import comfy.model_management as mm

        return bool(mm.processing_interrupted())
    except Exception:
        return False


def load(
    model_path: str,
    adapter_path: str | None,
    gpu_layers: int,
    n_ctx: int,
    progress: NodeProgress | None = None,
):
    """Return a cached ``Llama`` for this combination of files and placement."""
    key = (
        os.path.normcase(model_path),
        os.path.normcase(adapter_path or ""),
        int(gpu_layers),
        int(n_ctx),
    )

    with _LOCK:
        if _STATE["key"] == key and _STATE["llama"] is not None:
            return _STATE["llama"]

        unload()
        _free_comfy_vram()

        llama_cpp = _llama_cpp()

        if progress is not None:
            progress.set_total(1000)
            name = os.path.basename(model_path)
            adapter_note = f" + {os.path.basename(adapter_path)}" if adapter_path else " (no adapter)"
            progress.ratio(0.05, f"Loading {name}{adapter_note}\nllama.cpp, {gpu_layers} GPU layers")

        kwargs = {
            "model_path": model_path,
            "n_gpu_layers": int(gpu_layers),
            "n_ctx": int(n_ctx),
            "verbose": False,
        }
        if adapter_path:
            kwargs["lora_path"] = adapter_path

        try:
            llama = llama_cpp.Llama(**kwargs)
        except TypeError as error:
            # Older builds spell the adapter argument differently; retry without
            # it rather than silently dropping the rewriter.
            if adapter_path:
                raise RuntimeError(
                    f"This llama-cpp-python build did not accept 'lora_path' ({error}). "
                    "Upgrade it, or run the transformers backend instead."
                ) from error
            raise

        _STATE.update(key=key, llama=llama)
        if progress is not None:
            progress.ratio(1.0, "Model ready")
        return llama


def unload() -> None:
    with _LOCK:
        llama = _STATE.get("llama")
        _STATE.update(key=None, llama=None)
    if llama is not None:
        close = getattr(llama, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.debug("[minimax_h3_rewriter.gguf.unload] close failed", exc_info=True)
        del llama
    gc.collect()


def is_loaded() -> bool:
    return _STATE["llama"] is not None


def _render(llama, messages: list[dict[str, str]]) -> str:
    metadata = dict(getattr(llama, "metadata", {}) or {})
    return chat_template.from_metadata(metadata, messages, enable_thinking=False)


def generate(
    llama,
    messages: list[dict[str, str]],
    seed: int,
    greedy: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    progress: NodeProgress | None = None,
) -> str:
    rendered = _render(llama, messages)

    call_kwargs = {
        "max_tokens": int(max_new_tokens),
        "repeat_penalty": float(repetition_penalty),
        "stream": True,
        "seed": normalize_seed(seed),
    }
    if greedy:
        call_kwargs["temperature"] = 0.0
    else:
        call_kwargs.update(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )

    if progress is not None:
        progress.set_total(max(int(max_new_tokens), 1))
        progress.update(0, "Generating\n0 tokens")

    try:
        stream = llama(rendered, **call_kwargs)
    except TypeError:
        call_kwargs.pop("seed", None)
        stream = llama(rendered, **call_kwargs)

    pieces: list[str] = []
    produced = 0
    interrupted = False
    for chunk in stream:
        try:
            piece = chunk["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            continue
        if not piece:
            continue
        pieces.append(piece)
        produced += 1
        if progress is not None:
            tail = "".join(pieces)[-PREVIEW_TAIL:]
            progress.update(produced, f"Generating · {produced}/{max_new_tokens} tokens\n{tail}")
        if _interrupted():
            interrupted = True
            break

    if interrupted:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.debug("[minimax_h3_rewriter.gguf.generate] stream close failed", exc_info=True)
        import comfy.model_management as mm

        raise mm.InterruptProcessingException()

    if progress is not None:
        progress.finish(f"Done · {produced} tokens")
    return "".join(pieces).strip()


def rewrite(
    model_path: str,
    adapter_path: str | None,
    gpu_layers: int,
    n_ctx: int,
    keep_loaded: bool,
    progress: NodeProgress | None = None,
    **generation,
) -> str:
    """Load (or reuse) the GGUF rewriter, generate once, release unless kept."""
    llama = load(model_path, adapter_path, gpu_layers, n_ctx, progress)
    try:
        return generate(llama, progress=progress, **generation)
    finally:
        del llama
        if not keep_loaded:
            unload()
