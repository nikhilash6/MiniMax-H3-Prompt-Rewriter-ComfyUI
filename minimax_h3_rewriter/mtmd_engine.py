"""Describing a reference asset with ``llama-mtmd-cli``.

The writer nodes read text, not pixels. This is where the text comes from: an
image, an audio clip or a video goes to a multimodal model in a subprocess and
comes back as one or two sentences, which then become a line of
``reference_assets``.

It reuses the runtime the rewriter already fetches -- ``llama-mtmd-cli`` ships in
the same release archive as ``llama-completion``, so a machine that has run one
rewrite already has everything this needs.

Two differences from ``cli_engine`` are worth knowing:

- **The chat template is applied by the binary**, not here. ``-p`` is a plain
  instruction and mtmd wraps it, which is also what lets it splice the media
  tokens into the right place in the turn. Rendering the template ourselves, as
  the rewriter does to force ``enable_thinking=False``, would put the image
  markers in the wrong position.
- **The prompt goes on the command line**, not through ``--file``. There is no
  shell in the way -- the command is a list -- so quotes and newlines in an
  instruction need no escaping, and ``-f`` is not honoured alongside media.

Which models actually work here is a much shorter list than the set of
multimodal GGUFs on the Hub: the projector format has to be one ``mtmd``
understands. See ``models.json`` for the ones this pack has been run against.
"""

from __future__ import annotations

import logging
import os

from . import llamacpp, media, runner
from .constants import normalize_seed
from .progress import NodeProgress

log = logging.getLogger(__name__)

PREVIEW_TAIL = 280
CHARS_PER_TOKEN = 4.0

ALL_LAYERS = 999

DEFAULT_MAX_TOKENS = 256

FLAG_FOR_KIND = {"image": "--image", "audio": "--audio", "video": "--video"}

CONTEXT_FROM_MODEL = 0


def build_command(
    binary: str,
    model_path: str,
    mmproj_path: str,
    instruction: str,
    attachments: list[tuple[str, str]],
    gpu_layers: int,
    n_ctx: int = CONTEXT_FROM_MODEL,
    seed: int = 42,
    greedy: bool = True,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> list[str]:
    layers = ALL_LAYERS if int(gpu_layers) < 0 else int(gpu_layers)
    command = [
        binary,
        "--model", model_path,
        "--mmproj", mmproj_path,
        "--n-gpu-layers", str(layers),
        "--ctx-size", str(int(n_ctx)),
        "--predict", str(int(max_new_tokens)),
        "--seed", str(normalize_seed(seed)),
    ]
    for kind, path in attachments:
        flag = FLAG_FOR_KIND.get(kind)
        if flag is None:
            raise ValueError(f"unknown attachment kind '{kind}'")
        command += [flag, path]
    if greedy:
        command += ["--temp", "0"]
    else:
        command += [
            "--temp", f"{float(temperature):g}",
            "--top-p", f"{float(top_p):g}",
            "--top-k", str(int(top_k)),
        ]
    # Last, so the instruction is never mistaken for a value of another flag.
    command += ["--prompt", instruction]
    return command


def attachments_from(
    workspace: media.Workspace,
    image=None,
    audio=None,
    video=None,
    max_frames: int = media.DEFAULT_MAX_FRAMES,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Write the connected inputs to disk. Returns ``(attachments, notes)``."""
    attachments: list[tuple[str, str]] = []
    notes: list[str] = []

    if video is not None:
        attachments.append(("video", media.video_file(video, workspace)))
        notes.append("video")

    if image is not None:
        paths = media.image_files(image, workspace, max_frames)
        attachments.extend(("image", path) for path in paths)
        total = int(getattr(image, "shape", [len(paths)])[0])
        notes.append(
            f"{len(paths)} of {total} frames" if total > len(paths) else f"{len(paths)} image(s)"
        )

    if audio is not None:
        attachments.append(("audio", media.audio_file(audio, workspace)))
        notes.append("audio")

    return attachments, notes


def describe(
    model_path: str,
    mmproj_path: str,
    instruction: str,
    image=None,
    audio=None,
    video=None,
    max_frames: int = media.DEFAULT_MAX_FRAMES,
    gpu_layers: int = -1,
    n_ctx: int = CONTEXT_FROM_MODEL,
    seed: int = 42,
    greedy: bool = True,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    backend: str = "auto",
    auto_download: bool = True,
    progress: NodeProgress | None = None,
) -> str:
    """Fetch the runtime if needed, describe the attachments once, and return the text."""
    if image is None and audio is None and video is None:
        raise ValueError(
            "nothing to describe: connect an image, an audio clip or a video, or type the "
            "description in by hand."
        )

    binary = llamacpp.ensure_mtmd(backend, auto_download, progress)

    with media.Workspace() as workspace:
        attachments, notes = attachments_from(workspace, image, audio, video, max_frames)
        command = build_command(
            binary, model_path, mmproj_path, instruction, attachments,
            gpu_layers, n_ctx, seed, greedy, max_new_tokens, temperature, top_p, top_k,
        )

        runner.free_comfy_vram()
        if progress is not None:
            progress.set_total(max(int(max_new_tokens), 1))
            progress.text(
                f"Describing {' + '.join(notes)}\n{os.path.basename(model_path)}",
                force=True,
            )

        def report(whole: str) -> None:
            if progress is not None:
                progress.update(
                    min(len(whole) / CHARS_PER_TOKEN, float(max_new_tokens)),
                    f"Describing · {len(whole)} chars\n{whole[-PREVIEW_TAIL:]}",
                )

        try:
            text, stderr_text = runner.run(command, binary, report)
        except runner.ChildFailed as error:
            raise RuntimeError(
                f"{error}\n\nNot every multimodal GGUF works here: llama.cpp's mtmd has to "
                f"understand the projector format, and several current models abort while "
                f"loading it. Pick an entry from the captioner list, which names the ones this "
                f"pack has been run against."
            ) from error

    if progress is not None:
        progress.finish(f"Done · {len(text)} chars{runner.speed(stderr_text)}")
    return text
