"""Running the rewriter through the ``llama-cli`` binary, in a subprocess.

The backend of last resort, and a surprisingly good one. It needs nothing
installed into ComfyUI's Python: see ``llamacpp.py`` for why the wheel is worth
avoiding when it is not already there.

A subprocess reloads the model on every run, which sounds expensive and is not.
The node's own default is ``keep_model_loaded = False``, because the card is
needed for video generation the moment the rewrite finishes -- and in that mode
the in-process backend already unloads after every run. Reloading a 15.7 GB
Q4_K_M from the page cache takes about 8 seconds, which is what the in-process
backend spends too. What this backend genuinely cannot do is honour
``keep_model_loaded = True``; that is reported rather than silently ignored.

Two things come free with the process boundary: VRAM is returned by the
operating system rather than by hoping a deallocator ran, and a llama.cpp crash
takes down a child process instead of ComfyUI and its queue.

The prompt travels through ``--file`` rather than the command line, so a
multi-line template full of quotes needs no shell escaping on any platform.
"""

from __future__ import annotations

import atexit
import codecs
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

from . import chat_template, llamacpp
from .constants import normalize_seed
from .progress import NodeProgress
from .prompt_template import build_messages

log = logging.getLogger(__name__)

PREVIEW_TAIL = 280
READ_CHUNK = 4096
POLL_SECONDS = 0.25
STDERR_TAIL = 40

#: How long silence is allowed to last before the child is declared hung.
#:
#: Generous before the first byte, because loading 15-50 GB from a cold disk
#: legitimately takes minutes; short afterwards, because a model that has begun
#: emitting tokens and then stops for three minutes is not going to resume.
#: Either way the node fails with a message instead of wedging the queue, which
#: is what a subprocess that never exits does to ComfyUI.
FIRST_BYTE_SECONDS = 900.0
STALL_SECONDS = 180.0

#: Every child ever started, so none can outlive the interpreter holding VRAM.
_LIVE: set = set()


def _kill_all() -> None:
    for process in list(_LIVE):
        if process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass


atexit.register(_kill_all)

#: llama.cpp spells "all layers" as a large number, not -1.
ALL_LAYERS = 999

#: Rough characters per token, used only to advance the progress bar.
CHARS_PER_TOKEN = 4.0

_PERF = re.compile(r"eval time =.*?\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
_METADATA_CACHE: dict[tuple[str, int, int], dict] = {}


def available() -> bool:
    return llamacpp.available()


def _free_comfy_vram() -> None:
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception:
        log.debug("[minimax_h3_rewriter.cli._free_comfy_vram] skipped", exc_info=True)


def _interrupted() -> bool:
    try:
        import comfy.model_management as mm

        return bool(mm.processing_interrupted())
    except Exception:
        return False


def gguf_metadata(model_path: str) -> dict:
    """The GGUF key/value header, read straight from the file.

    Only the header is touched, and the answer is cached per file identity, so
    this costs a stat on every run after the first even for a 15.7 GB model.
    """
    try:
        stat = os.stat(model_path)
    except OSError as error:
        raise RuntimeError(f"Cannot read '{model_path}': {error}") from error

    key = (os.path.normcase(model_path), stat.st_size, int(stat.st_mtime))
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from gguf import GGUFReader
    except ImportError as error:  # pragma: no cover - gguf ships with ComfyUI
        raise RuntimeError(
            "Reading the GGUF chat template needs the 'gguf' package, which ComfyUI "
            "normally provides."
        ) from error

    reader = GGUFReader(model_path, "r")
    metadata = {}
    for name, field in reader.fields.items():
        try:
            metadata[name] = field.contents()
        except Exception:
            continue
    _METADATA_CACHE[key] = metadata
    return metadata


def render_prompt(model_path: str, prompt: str, resolution: str, duration: int) -> str:
    messages = build_messages(prompt, resolution, duration)
    return chat_template.from_metadata(gguf_metadata(model_path), messages, enable_thinking=False)


def build_command(
    binary: str,
    model_path: str,
    adapter_path: str | None,
    prompt_file: str,
    gpu_layers: int,
    n_ctx: int,
    seed: int,
    greedy: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> list[str]:
    layers = ALL_LAYERS if int(gpu_layers) < 0 else int(gpu_layers)
    command = [
        binary,
        "--model", model_path,
        "--file", prompt_file,
        "--n-gpu-layers", str(layers),
        "--ctx-size", str(int(n_ctx)),
        "--predict", str(int(max_new_tokens)),
        "--seed", str(normalize_seed(seed)),
        "--repeat-penalty", f"{float(repetition_penalty):g}",
        "-no-cnv",
        "--no-display-prompt",
        "--no-warmup",
        "--simple-io",
    ]
    if adapter_path:
        command += ["--lora", adapter_path]
    if greedy:
        command += ["--temp", "0"]
    else:
        command += [
            "--temp", f"{float(temperature):g}",
            "--top-p", f"{float(top_p):g}",
            "--top-k", str(int(top_k)),
        ]
    return command


def _spawn(command: list[str], binary: str) -> subprocess.Popen:
    environment = dict(os.environ)
    directory = os.path.dirname(os.path.abspath(binary))
    if sys.platform != "win32":
        # The tar releases put the shared libraries beside the executable.
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = f"{directory}{os.pathsep}{existing}" if existing else directory

    creation = 0
    if sys.platform == "win32":
        # Otherwise a console window flashes over the ComfyUI browser tab.
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    process = subprocess.Popen(
        command,
        # Closed, not inherited: a child that can never block waiting for a key
        # is one failure mode fewer, and ComfyUI's own stdin is not ours to read.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=directory,
        env=environment,
        creationflags=creation,
        bufsize=0,
    )
    _LIVE.add(process)
    return process


def _pump(stream, sink: queue.Queue) -> None:
    try:
        while True:
            chunk = stream.read(READ_CHUNK)
            if not chunk:
                break
            sink.put(chunk)
    except Exception:
        log.debug("[minimax_h3_rewriter.cli._pump] reader stopped", exc_info=True)
    finally:
        sink.put(None)


def generate(
    binary: str,
    model_path: str,
    adapter_path: str | None,
    prompt: str,
    resolution: str,
    duration: int,
    gpu_layers: int,
    n_ctx: int,
    seed: int,
    greedy: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    progress: NodeProgress | None = None,
) -> str:
    rendered = render_prompt(model_path, prompt, resolution, duration)

    handle, prompt_file = tempfile.mkstemp(prefix="minimax_h3_", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
        file.write(rendered)

    command = build_command(
        binary, model_path, adapter_path, prompt_file, gpu_layers, n_ctx, seed,
        greedy, max_new_tokens, temperature, top_p, top_k, repetition_penalty,
    )
    log.info("[minimax_h3_rewriter.cli.generate] %s", " ".join(command))

    _free_comfy_vram()
    if progress is not None:
        name = os.path.basename(model_path)
        note = f" + {os.path.basename(adapter_path)}" if adapter_path else " (no adapter)"
        progress.set_total(max(int(max_new_tokens), 1))
        progress.text(f"Loading {name}{note}\nllama.cpp binary, {gpu_layers} GPU layers", force=True)

    try:
        process = _spawn(command, binary)
    except OSError as error:
        os.unlink(prompt_file)
        raise RuntimeError(f"Could not start '{binary}': {error}") from error

    output: queue.Queue = queue.Queue()
    errors: queue.Queue = queue.Queue()
    threading.Thread(target=_pump, args=(process.stdout, output), daemon=True).start()
    threading.Thread(target=_pump, args=(process.stderr, errors), daemon=True).start()

    # Token pieces arrive as raw bytes and a multi-byte character can straddle
    # two reads, so decoding has to carry state rather than run per chunk.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pieces: list[str] = []
    interrupted = False
    finished = False
    stalled = ""
    last_output = time.monotonic()

    try:
        while not finished:
            if _interrupted():
                interrupted = True
                break
            try:
                chunk = output.get(timeout=POLL_SECONDS)
            except queue.Empty:
                limit = STALL_SECONDS if pieces else FIRST_BYTE_SECONDS
                waited = time.monotonic() - last_output
                if waited > limit:
                    stalled = (
                        f"produced nothing for {waited:.0f} s"
                        if not pieces
                        else f"stopped mid-generation for {waited:.0f} s"
                    )
                    break
                continue
            if chunk is None:
                finished = True
                break
            last_output = time.monotonic()
            text = decoder.decode(chunk)
            if not text:
                continue
            pieces.append(text)
            if progress is not None:
                whole = "".join(pieces)
                # llama-completion streams text, not token boundaries, so the bar
                # is driven by an estimate. Four characters per token is close
                # enough for English prose and never overshoots the cap, and the
                # run almost always ends at EOS well before it anyway.
                progress.update(
                    min(len(whole) / CHARS_PER_TOKEN, float(max_new_tokens)),
                    f"Generating · {len(whole)} chars\n{whole[-PREVIEW_TAIL:]}",
                )
        pieces.append(decoder.decode(b"", final=True))
    finally:
        if interrupted or stalled or process.poll() is None:
            process.kill()
        process.wait()
        _LIVE.discard(process)
        try:
            os.unlink(prompt_file)
        except OSError:
            log.debug("[minimax_h3_rewriter.cli.generate] could not remove %s", prompt_file)

    stderr_text = _drain(errors)

    if interrupted:
        import comfy.model_management as mm

        raise mm.InterruptProcessingException()

    if stalled:
        tail = "\n".join(stderr_text.splitlines()[-STDERR_TAIL:])
        raise RuntimeError(
            f"{os.path.basename(binary)} {stalled} and was stopped, so it could not wedge "
            f"the queue. Last output from it:\n{tail}"
        )

    if process.returncode != 0:
        tail = "\n".join(stderr_text.splitlines()[-STDERR_TAIL:])
        raise RuntimeError(
            f"llama-cli exited with code {process.returncode}.\n{tail}"
        )

    text = "".join(pieces).strip()
    if progress is not None:
        progress.finish(f"Done · {len(text)} chars{_speed(stderr_text)}")
    return text


def _drain(sink: queue.Queue) -> str:
    chunks = []
    while True:
        try:
            chunk = sink.get_nowait()
        except queue.Empty:
            break
        if chunk is None:
            continue
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _speed(stderr_text: str) -> str:
    match = _PERF.search(stderr_text)
    return f" · {match.group(2)} tok/s" if match else ""


def unload() -> None:
    """Nothing to unload: the model left with the process that held it."""


def is_loaded() -> bool:
    return False


def rewrite(
    model_path: str,
    adapter_path: str | None,
    gpu_layers: int,
    n_ctx: int,
    keep_loaded: bool,
    backend: str = "auto",
    auto_download: bool = True,
    progress: NodeProgress | None = None,
    **generation,
) -> str:
    """Fetch the runtime if needed, generate once, and let the process go."""
    binary = llamacpp.ensure(backend, auto_download, progress)
    if keep_loaded:
        log.info(
            "[minimax_h3_rewriter.cli.rewrite] keep_model_loaded has no effect on the "
            "llama.cpp binary backend: the model leaves with the subprocess"
        )
    return generate(
        binary=binary,
        model_path=model_path,
        adapter_path=adapter_path,
        progress=progress,
        gpu_layers=gpu_layers,
        n_ctx=n_ctx,
        **generation,
    )
