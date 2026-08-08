"""Turning ComfyUI's in-memory media into files a subprocess can open.

``llama-mtmd-cli`` takes paths, not tensors, so an IMAGE has to become a PNG, an
AUDIO a WAV and a VIDEO a container on disk before any of it can be described.
Everything lands in one temporary directory that is removed on the way out, so a
workflow run leaves nothing behind even when the child crashes.

Two deliberate choices:

- **WAV is written with the standard library**, not torchaudio or soundfile.
  Both are usually present in a ComfyUI install and neither is guaranteed, and a
  captioner that fails to import is worse than one that writes 16-bit PCM by
  hand -- which is eleven lines and exactly what llama.cpp wants anyway.
- **Frames are sampled, not dumped.** An IMAGE batch out of a video loader is
  hundreds of frames; passing all of them would blow the context window and the
  wall clock. A handful spread evenly across the batch describes the clip about
  as well, and the caller is told how many were used.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import wave

log = logging.getLogger(__name__)

#: Frames taken from an IMAGE batch when it is longer than one.
DEFAULT_MAX_FRAMES = 8

VIDEO_SUFFIX = ".mp4"


class Workspace:
    """A temporary directory that cleans up after itself."""

    def __init__(self, prefix: str = "minimax_h3_media_"):
        self.path = tempfile.mkdtemp(prefix=prefix)

    def file(self, name: str) -> str:
        return os.path.join(self.path, name)

    def close(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _numpy():
    import numpy

    return numpy


def frame_indices(count: int, limit: int) -> list[int]:
    """Evenly spread ``limit`` indices across ``count`` frames, endpoints included."""
    if count <= 0:
        return []
    if count <= limit or limit <= 1:
        return list(range(min(count, max(limit, 1))))
    step = (count - 1) / (limit - 1)
    return sorted({int(round(index * step)) for index in range(limit)})


def image_files(image, workspace: Workspace, max_frames: int = DEFAULT_MAX_FRAMES) -> list[str]:
    """Write an IMAGE batch out as PNGs. Returns the paths, in order."""
    from PIL import Image

    numpy = _numpy()

    array = image.detach().cpu().numpy() if hasattr(image, "detach") else numpy.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"expected an IMAGE tensor of shape (batch, height, width, channels), got {array.shape}")

    paths = []
    for position, index in enumerate(frame_indices(array.shape[0], max_frames)):
        frame = numpy.clip(array[index] * 255.0 + 0.5, 0, 255).astype(numpy.uint8)
        if frame.shape[-1] == 1:
            frame = frame[..., 0]
        path = workspace.file(f"frame_{position:03d}.png")
        Image.fromarray(frame).save(path, format="PNG")
        paths.append(path)
    return paths


def audio_file(audio, workspace: Workspace, name: str = "audio.wav") -> str:
    """Write a ComfyUI AUDIO dict out as 16-bit PCM WAV. Returns the path."""
    numpy = _numpy()

    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("expected a ComfyUI AUDIO input with a 'waveform' and a 'sample_rate'")

    waveform = audio["waveform"]
    rate = int(audio.get("sample_rate") or 44100)

    array = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else numpy.asarray(waveform)
    if array.ndim == 3:  # (batch, channels, samples) -- only the first clip is described
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"expected an AUDIO waveform of shape (channels, samples), got {array.shape}")

    # (channels, samples) -> interleaved (samples, channels), which is what a WAV
    # frame actually is.
    interleaved = numpy.ascontiguousarray(array.T)
    clipped = numpy.clip(interleaved, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    path = workspace.file(name)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(int(array.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def video_file(video, workspace: Workspace, name: str = "video" + VIDEO_SUFFIX) -> str:
    """Materialise a ComfyUI VIDEO input as a file. Returns the path.

    A VIDEO loaded from disk can often hand over its own source path, which
    saves a full re-encode of something that is already a file; anything else is
    written out through the object's own ``save_to``.
    """
    source = getattr(video, "_VideoFromFile__file", None)
    if isinstance(source, str) and os.path.isfile(source):
        return source

    path = workspace.file(name)
    save_to = getattr(video, "save_to", None)
    if callable(save_to):
        save_to(path)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
        raise RuntimeError("the VIDEO input produced an empty file")

    raise ValueError(
        "this VIDEO input cannot be written to disk: it has no 'save_to'. Feed the frames "
        "into the 'image' input instead."
    )
