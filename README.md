# MiniMax-H3 Prompt Rewriter for ComfyUI

ComfyUI nodes for the [LightX2V MiniMax-H3 T2VA Prompt Rewriter LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA).
A short prompt goes in; a structured, production-ready audio-video description
for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) comes out — entirely locally.

[Русская версия](README_RU.md)

![The rewriter node in ComfyUI: a short prompt on the left, the structured shot-by-shot description, soundscape and music fields on the right](docs/node_preview.png)

The prompt may be in any language the base model reads; the rewrite comes back in
English, which is what MiniMax-H3 expects.

```text
"A red fox walks through a snowy forest at dawn."  +  16:9  +  15s
                              │
                              ▼
              Qwen3.6-27B + Prompt Rewriter LoRA
                              │
                              ▼
   integrated_multimodal_description: [Shot 1] ... [Shot 2] 0:06 ...
   overall_soundscape: ...
   non_diegetic_music: ...
                              │
                              ▼
              MiniMax-H3 video + synchronized audio
```

## What you need before installing

This is a 27-billion-parameter language model, not a small helper. There is no
way around the following, because the LoRA is bound to one specific base model.

| Resource | Requirement |
|---|---|
| Disk | **~52 GB** for `Qwen/Qwen3.6-27B` + **~3.5 GB** for the adapter — or **~10–16 GB** total on the GGUF route |
| VRAM (`nf4`, default) | **~16 GB** |
| VRAM (`int8`) | ~28 GB |
| VRAM (`bfloat16`) | ~54 GB, spills into system RAM via accelerate |
| VRAM (GGUF) | ~13–19 GB depending on the quant, lower still with fewer offloaded layers |
| Packages | `transformers`, `peft`, `accelerate`, and `bitsandbytes` for `nf4`/`int8`; `llama-cpp-python` for GGUF |

> **The MiniMax-H3 text encoder cannot be reused for this.** It is a different
> model (Qwen3-VL-32B, vocabulary 151936) from the LoRA's base (Qwen3.6-27B,
> vocabulary 248320); it contains none of the linear-attention `in_proj_*`
> modules the adapter targets; it is truncated to the first 50 of 64 layers; and
> it ships without `lm_head` or a final norm, so it cannot generate text at all.
> It only produces hidden states for the DiT.

## Install

Clone into `ComfyUI/custom_nodes/` and install the requirements into the same
Python environment ComfyUI runs on:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI
```

For ComfyUI portable on Windows:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\MiniMax-H3-Prompt-Rewriter-ComfyUI\requirements.txt
```

Or install it from the Comfy registry through ComfyUI-Manager.

## Nodes

### MiniMax-H3 Prompt Rewriter

The main node. It downloads whatever is missing, loads the model, generates, and
releases the VRAM again.

**Outputs**

| Name | Contents |
|---|---|
| `rewritten_prompt` | The full rewrite, ready to paste into a MiniMax-H3 text input |
| `integrated_multimodal_description` | Just the shot-by-shot visual section |
| `overall_soundscape` | Just the diegetic audio section |
| `non_diegetic_music` | Just the score section |

**Inputs**

- `prompt` — the short prompt to expand.
- `model` — the base model. The list holds every entry from your model list plus
  every Qwen3.6-27B already on disk (prefixed `on disk:`). Anything not present
  is downloaded on first use, resuming if interrupted. The **Open model list**
  button edits the list — see below.
- `resolution` / `duration` — conditions the rewrite is composed for. Keep them
  equal to what you pass to MiniMax-H3, or the shot pacing will not match.
- `quantization` — how to load an *unquantized* checkpoint: `nf4` (default,
  ~16 GB VRAM), `int8` (~28 GB), `bfloat16` / `float16` (~54 GB). Ignored when
  the checkpoint brings its own quantization.
- `greedy` — on by default for deterministic output. Turn it off to sample.
- `seed`
- `keep_model_loaded` — **off by default.** The 27B model is released as soon as
  the rewrite finishes, so the same GPU can run H3 video generation next. Turn it
  on only when iterating on prompts back-to-back.
- `options` — optional; connect a **MiniMax-H3 Rewriter Options** node.

### MiniMax-H3 Rewriter Options

Everything you rarely touch, kept off the main node. Leave it unconnected and the
rewriter uses the decoding parameters the adapter was published with.

| Input | Default | Purpose |
|---|---|---|
| `max_new_tokens` | 2048 | Generation cap |
| `temperature` / `top_p` / `top_k` | 0.7 / 0.8 / 20 | Sampling, used only when `greedy` is off |
| `repetition_penalty` | 1.05 | |
| `attn_implementation` | `sdpa` | `eager` or `flash_attention_2` if you have it |
| `adapter` | the LightX2V repo | Repository id or local folder of the LoRA |
| `use_lora` | on | Turn off for the plain Qwen3.6-27B baseline |
| `auto_download` | on | Turn off to fail loudly instead of fetching 52 GB |

### The model list

The **Open model list** button on the rewriter node opens

```text
ComfyUI/user/minimax_h3_rewriter/models.json
```

in your desktop's JSON editor — on the machine running ComfyUI, which is not
necessarily the one looking at the browser tab. It is seeded from the packaged
copy on first use, so updating the node pack never overwrites your edits.

```json
{
  "name": "Qwen3.6-27B FP8",
  "repo": "Qwen/Qwen3.6-27B-FP8",
  "download_gb": 28.8,
  "vram": "~29 GB, no extra package needed"
}
```

`repo` may equally be an absolute path to a folder you already have. Add an entry,
refresh the browser, and it is in the dropdown. Keep `name` stable — saved
workflows remember the label, and a node whose stored choice has vanished says so
by name instead of silently picking something else.

## Where the weights go

```text
ComfyUI/models/LLM/
├── Qwen3.6-27B/                          # base model, ~52 GB
└── MiniMax-H3-Prompt-Rewriter-LoRA/      # adapter, ~3.5 GB
```

**Already downloaded the LoRA?** Point the `adapter` widget at that folder
directly (an absolute path is accepted) and nothing is fetched again.

## Using a model you already have

52 GB is a lot to ask, so the node looks for the base model before offering to
download it. The `model` dropdown lists every Qwen3.6-27B found in

- all directories ComfyUI registers for `LLM`, `transformers` and `diffusers`
  — including anything mapped in through `extra_model_paths.yaml`, and
- the Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE` / `~/.cache/huggingface/hub`),

so a copy pulled earlier by any other tool is reused instead of downloaded twice.
Only directories whose weights are actually present are listed; a cache entry
holding nothing but `config.json` is not offered.

### Smaller repackings

Any repository you put in the model list is checked from a 4 KB `config.json`
**before** a single weight moves. Every repacking keeps
the same fingerprint (`model_type qwen3_5`, hidden 5120, 64 layers, vocab 248320),
so what actually decides the outcome is the quantization runtime:

| Build | Download | Runtime package | LoRA attaches? | Verdict |
|---|---|---|---|---|
| `quant_method: bitsandbytes` (nf4 repack) | **~17 GB** | `bitsandbytes`, already required | yes | **works as shipped — the low-resource route** |
| `Qwen/Qwen3.6-27B` (bf16) | 52 GB | none; `bitsandbytes` for `nf4`/`int8` | yes | **works as shipped** |
| `Qwen/Qwen3.6-27B-FP8` | 29 GB | `kernels`, plus a kernel fetched from the Hub | yes | advanced only; needs >29 GB of VRAM |
| `quant_method: awq` | ~19 GB | `autoawq` | yes | works after installing it |
| `quant_method: gptq` | ~19 GB | `gptqmodel` | yes | works after installing it |
| `quant_method: compressed-tensors` | ~19 GB | `compressed-tensors` | **no** | not supported |
| `quant_method: modelopt` (NVFP4) | ~20 GB | `nvidia-modelopt` | **no** | not supported |

PEFT ships LoRA dispatchers for bitsandbytes, AWQ, GPTQ, HQQ, EETQ, AQLM and
torchao layers, and plain `nn.Linear` covers bf16/fp16 and FP8. For
compressed-tensors and modelopt there is no dispatcher, so the adapter cannot be
attached at all — no package will fix that.

> A repository's **name** is not its format: `cyankiwi/Qwen3.6-27B-AWQ-INT4` and
> `unsloth/Qwen3.6-27B-NVFP4` are both `compressed-tensors`, so neither can take
> the LoRA. The node reports the real `quant_method` and the exact reason before
> downloading anything.

### If the node says a package is missing

Nothing in the recommended routes needs one. bf16, the nf4 repacks and the
`nf4`/`int8` options all run on packages this pack already declares, so a
missing-package message only appears for a build you went looking for.

When it does appear it names the package **and the command for the interpreter
that is actually running ComfyUI** — `pip install kernels` typed into an
ordinary terminal installs into whichever Python is on PATH, which on a portable
install is never the right one, and the node goes on refusing:

```
This base model cannot run the prompt-rewriter LoRA.
  - the 'fp8' checkpoint needs the 'kernels' package, which is not installed in
    this Python environment. Install it with:
      "…\python_embeded\python.exe" -m pip install kernels
    Note: installing it is only half of it: the FP8 matmul is a Triton kernel
    that transformers then downloads from 'kernels-community/finegrained-fp8'
    on the first generation, and that needs a build matching this torch and
    CUDA version.
```

Run the line as printed and restart ComfyUI. The pack installs nothing into your
environment on its own, and never will: a node that silently pip-installs is a
node that can break an unrelated part of ComfyUI while you watch a progress bar.

## Smallest download without any extra install

A `bitsandbytes` nf4 repack of Qwen3.6-27B is **~17 GB to download and ~16 GB of
VRAM**, and it needs nothing this node does not already require. That is the
route to point people at: same code path as the official checkpoint, no new
dependency, a third of the download.

Those repacks are third-party — the node verifies the architecture from
`config.json` before fetching anything, which proves the shape is right, not that
the uploader is trustworthy. Judge that yourself, or make your own repack once
from the official weights.

## GGUF — smaller still, and nothing to install

Pick a `[gguf]` entry from the model list and the rewriter runs under llama.cpp
instead of Transformers. **No pip install is involved.** If `llama-cpp-python`
happens to be in ComfyUI's environment the node uses it; if it is not, the node
fetches the official llama.cpp binaries (~34 MB) into
`ComfyUI/user/minimax_h3_rewriter/runtime/` and runs `llama-cli` as a
subprocess. Same download switch as the weights: `auto_download`.

The subprocess reloads the model on every run, which costs nothing in practice —
the node's default is `keep_model_loaded = False`, because the card is needed
for video generation the moment the rewrite finishes, and the in-process backend
already unloads after every run too. What the binary backend genuinely cannot do
is honour `keep_model_loaded = True`. Two things come free with it: VRAM is
returned by the operating system rather than by a deallocator, and a llama.cpp
crash takes down a child process instead of ComfyUI and its queue.

Two options in the options node control this, and they answer different
questions. **`gguf_runtime`** picks *what runs the model*:

| `gguf_runtime` | Meaning |
|---|---|
| `auto` | llama-cpp-python if it is importable, the binaries otherwise |
| `llama-cpp-python` | force the wheel; fails with a clear message if it is absent, rather than quietly using something else |
| `llama.cpp` | force the binaries, even when a wheel is installed — the way out when the installed wheel is broken |

Only `llama-cpp-python` can honour `keep_model_loaded`; the binaries hand the
model back to the operating system when the subprocess exits.

**`llama_backend`** then picks *which official build to fetch*, and applies only
when the binaries are in use:

| `llama_backend` | Download | Notes |
|---|---|---|
| `auto` → `vulkan` | 34 MB | NVIDIA, AMD and Intel alike; about half the CUDA throughput |
| `cuda` | 511 MB | ~2× faster on NVIDIA; **Windows only** — upstream publishes no Linux CUDA build |
| `cpu` | 17 MB | no GPU at all |

> **Why not the `llama-cpp-python` CUDA wheels.** Both current ones fail on
> ordinary consumer hardware, in two unrelated ways:
>
> | Wheel | Build flags | What happens |
> |---|---|---|
> | `v0.3.34-cu130` | `AVX512 = 1`, `ARCHS = 750..900` | weights load, then `llama_init_from_model` dies with `0xC000001D` — no consumer Intel 12th–14th gen chip has AVX-512 |
> | `v0.3.34-cu132` | `AVX512 = 0`, `ARCHS = 750..900` | reaches the first kernel, then `the provided PTX was compiled with an unsupported toolchain` — no `sm_120` in the list, so an RTX 50-series card falls back to JIT, which a driver older than the build's toolkit refuses |
> | `v0.3.34-vulkan` | `AVX512 = 0`, no arch list | works, and picks up `NV_coopmat2` where the driver offers it |
>
> The official release archives have neither problem. They carry **14** CPU
> backend variants and choose one at run time, which is why the same model runs
> fine under `llama-cli` on the machine where the cu130 wheel dies. And their
> CUDA archive carries native SASS with no PTX at all —
> `cuobjdump --list-elf` reports `sm_86 sm_89 sm_120a sm_121a` — so the driver
> is never asked to compile anything.
>
> If you want the in-process backend anyway, the Vulkan wheel is the one that
> works: `pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-vulkan/llama_cpp_python-0.3.34-py3-none-win_amd64.whl`

| Base quant | Download | VRAM with the adapter |
|---|---|---|
| `Q4_K_M` | 15.7 GB | ~19 GB |
| `IQ4_XS` | 14.4 GB | ~18 GB |
| `UD-Q3_K_XL` | 13.5 GB | ~17 GB |
| `UD-IQ2_M` | 10.1 GB | ~13 GB, noticeably lower fidelity |

Lower `gpu_layers` in the options node to fit a smaller card, at the cost of
speed. With `Q4_K_M` fully offloaded to a high-end consumer NVIDIA card, CUDA
generates at roughly **50 tok/s** with the adapter against 78 tok/s without it —
that ~35% is llama.cpp doing the adapter's matmuls — and Vulkan at roughly half
the CUDA figure.

> **A smaller Qwen3.5 is not a substitute.** Qwen3.5-9B carries the same
> `general.architecture = qwen35` in its header, so it looks like a match and
> the model list will show it — but it has 32 blocks of width 4096 where the
> adapter needs 64 of 5120. llama.cpp refuses to attach the LoRA
> (`tensor 'blk.0.attn_gate.weight' has incorrect shape`) and the run fails.
> The node checks those two header numbers first and says so before anything is
> downloaded. If you see a 9B producing a plausible-looking rewrite, it is
> running **without** the adapter: the format comes from the system prompt, not
> from the LoRA.

The GGUF route uses a **converted** adapter, not the PEFT one. Point the options
node's `adapter` at a local `.gguf`, or set `adapters.gguf.repo` in the model
list. The prompt is built from the GGUF's own chat template with
`enable_thinking=False`, and the result is byte-identical to what
`transformers.apply_chat_template` produces — the model sees exactly the text the
LoRA was trained on.

When the checkpoint carries its own quantization, the `quantization` widget is
ignored — bitsandbytes is not stacked on top of AWQ or FP8.

### Progress on the node

Downloads, weight loading, and token generation all report onto the node itself
through ComfyUI's own progress channels — a bar plus a caption with the current
file, transferred size, speed and ETA. No custom frontend extension is involved,
so nothing breaks when the ComfyUI frontend updates.

### Environment variables

| Variable | Effect |
|---|---|
| `HF_TOKEN` | Access token for gated or private repositories |
| `HF_ENDPOINT` | Mirror to download from instead of `huggingface.co` |

## Notes

- **Speed.** Qwen3.6-27B is a hybrid model: 48 of its 64 layers use linear
  attention. Without `flash-linear-attention` and `causal-conv1d` installed,
  Transformers falls back to a slower pure-PyTorch path and says so in the
  console. The fallback is correct, just slower; both packages are optional and
  awkward to build on Windows.
- **Determinism.** With `greedy` on, the same prompt, resolution, duration and
  seed produce the same rewrite, and ComfyUI caches the node accordingly.
- **Interruption.** Cancelling a run stops both a download and a generation in
  progress; a partial download resumes on the next run.
- The rewrite may add details a short prompt never stated. Review it before
  generating when identity, dialogue, timing or composition must be exact.

## Credits

The model work is entirely [LightX2V](https://github.com/ModelTC/LightX2V)'s —
this repository only wires their adapter into ComfyUI. If you find it useful,
star **[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)**, where the
MiniMax-H3 inference support and future rewriter tasks (FL2VA, Ref2VA) are
maintained.

| Component | Source |
|---|---|
| LoRA adapter | [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA) |
| Base language model | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Video/audio generator | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| Inference framework | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) |

The prompt template in `minimax_h3_rewriter/prompt_template.py` is reproduced
byte-for-byte from the adapter repository; changing it degrades the rewrite.

Use of MiniMax-H3 is governed by the licence and acceptable-use terms in the
[official MiniMax-H3 repository](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## Licence

MIT — see [LICENSE](LICENSE). This covers the ComfyUI integration code only; the
model weights carry their own licences.
