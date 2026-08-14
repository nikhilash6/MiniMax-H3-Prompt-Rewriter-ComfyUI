# Changelog

[Русская версия](CHANGELOG_RU.md)

The version in `pyproject.toml`, the git tag and the release on GitHub always say
the same thing; the release workflow refuses a tag that disagrees with
`pyproject.toml`, or one that neither changelog has a section for.

## 0.10.0 - 2026-08-15

### Added

- **`MiniMax-H3 Multi Reference Caption`, a whole shot's references in one
  node.** A chain of caption nodes is exact but it grows: five references are
  five nodes, five wires and five chances to leave the wrong role on one of
  them. This node has no `role` widget at all - the group an asset is plugged
  into is its label. That is the guide's own vocabulary made structural: Ref2VA
  defines exactly four reference labels and forbids inventing more, so
  `subjects`, `pictures`, `videos` and `audios` cover the format completely, and
  describing an image as audio stops being possible rather than merely
  discouraged. Slots grow as they are filled, one spare always waiting, which is
  `io.Autogrow` from ComfyUI's v3 node API.

  The block comes out in the guide's order - subjects, pictures, videos, audio -
  rather than in wiring order, and each label is still numbered within its own
  category, continuing from whatever arrives on `previous`, so the node sits in
  a chain with single caption nodes on either side. `model`, `length`, `seed`,
  `max_frames`, `context_size` and `bypass` are shared by every asset in it.
  `description` and `instruction` are not carried over: text written by hand
  belongs to one asset at a time, and the single node still has them.

- **A checkbox on each reference slot, on the slot's own row.** A caption costs
  a model load and seconds to minutes, which makes "everything except this one"
  the ordinary thing to want, and pulling the wire out to get it throws away the
  wiring that was the point. A dropdown of names would have been the cheap
  answer and the wrong one: the whole value is being able to hit the switch
  belonging to the input you are looking at. ComfyUI will not lay a widget out
  there - the frontend sorts every plain socket above every widget, whatever
  order the schema asks for - so the box is drawn onto the node at the row's own
  height and the click is picked up from the canvas. The state itself is an
  ordinary hidden widget holding JSON, which is what makes it survive a save and
  reach the backend through the API like any other value; only switched-off
  slots are written down, so an untouched node stays empty in the saved
  workflow. A frontend that never runs the script leaves the JSON field visible
  and everything still works.

- **A video slot takes a `VIDEO` or an `IMAGE` batch.** Video loaders disagree
  about which they hand out - VideoHelperSuite's `Load Video (Upload)` gives
  frames, not a `VIDEO` - and both are the same reference, so the slot accepts
  either and the run sorts out which one arrived. Frames are sampled evenly up
  to `max_frames` in both cases, so the cost of a clip stays independent of its
  length.

### Changed

- **The pack no longer registers all or nothing.** The new node needs a recent
  ComfyUI for its growing inputs, so it is registered on its own and an install
  too old for the v3 node API loses that one node instead of every node in the
  pack to a single failed import.

## 0.9.5 - 2026-08-14

### Added

- **`bypass`, on the four nodes that run a model.** The LoRA rewriter, both
  guided writers and the reference caption node each grew a switch that skips
  the model outright: nothing is downloaded, nothing is loaded, no VRAM is
  touched. The writers hand `prompt` straight to `rewritten_prompt`, which is
  the cheap way to hold a written prompt against the raw one without unwiring
  anything; the caption node passes `previous` through unchanged, which drops a
  single asset from the chain while leaving the chain wired. Numbering survives
  it, because every node numbers what it receives, so the assets after the
  bypassed one close the gap. The section outputs come back empty.

  ComfyUI's own bypass, Ctrl+B, cannot do this here: it forwards a connected
  link and nothing else, and every input these nodes write from is a widget. Hit
  Ctrl+B on the rewriter and there is no link to forward, so the nodes
  downstream are handed nothing at all. On the caption node it half works - the
  `previous` link is forwarded - and then fails on the first node of a chain,
  which has no `previous` to forward. The switch is the last widget on each
  node, so a workflow saved before this release keeps every value it had.

- **A badge above the node title, and a violet node while bypassed.** Collapsed,
  a node draws its title bar and nothing else, which is exactly the state in
  which the switch is out of reach and someone wants to fold a workflow up and
  turn one heavy step off. LiteGraph draws badges above the title whatever the
  node's state, and its hit test walks the badges carrying a click handler
  before it gives up, so the badge is clickable with the node collapsed and no
  canvas handler had to be patched to get there. The colour is what reads at the
  zoom where a whole workflow is on screen; it is swapped underneath the node's
  own colour getter rather than written to the node, which is how ComfyUI's
  native bypass does it too, so it never reaches the saved workflow and a node
  coloured by hand keeps the colour it was given. A frontend too old for badges
  loses the badge and keeps the switch, which is where the feature lives.

## 0.9.4 - 2026-08-10

### Added

- **`trust_remote_code`, off.** A Transformers checkpoint can carry its own
  modelling code, named by `auto_map` in its `config.json`, and loading such a
  model imports and runs that Python with your user's rights. Nothing in the
  shipped list does this - every `transformers` entry is a Qwen3.6-27B variant,
  and the GGUF entries never reach Transformers at all - so the switch changes
  nothing for the models the node offers. It exists for a model you added to
  `models.json` yourself: the node stops and says so instead of running the
  code, and turning the switch on is you saying which model you trust.

### Changed

- **`adapter` refuses a network path.** Every other model is picked from a
  dropdown, so a saved workflow carries the *name* of an entry and never a path;
  `adapter` is a text field because pointing it at a `.gguf` LoRA you converted
  yourself is a thing people do, which also means a workflow you downloaded gets
  to fill it in. `\\host\share\...` and `//host/share/...` are rejected before
  anything reads them, because merely looking at one is an authentication
  attempt against whatever host is named. A share of your own is reachable by
  drive letter as usual, and a path in `models.json` is not restricted at all.
- **The adapter that was applied is logged**, every run, at warning level when
  it is not the configured one. A swapped LoRA is otherwise invisible: the node
  still runs and still fills every field, it just writes something else.

## 0.9.3 - 2026-08-09

### Added

- **An entry in `models.json` can point at a file you already have.** `repo` may
  be a folder on this machine instead of a Hugging Face id, or the whole path
  may go in `file` with `repo` left out - both forms work, in all three
  sections, and the file is read where it lies with nothing downloaded and
  nothing copied. A path that does not exist is named and refused rather than
  downloaded around.

### Fixed

- **A broken `models.json` was the quietest failure in the pack.** The parse
  threw, the packaged defaults were served instead, and the dropdown looked
  ordinary - an edit that never took was indistinguishable from an edit that did
  nothing. The first entry of every model dropdown now carries the parse error
  with its line and column; picking it and hitting Run repeats the message and
  names the file to fix. The rest of the list is still there and ComfyUI still
  runs.

## 0.9.2 - 2026-08-08

### Fixed

- **An audio track from a video loader was rejected.** ComfyUI's own AUDIO is a
  plain `dict`, but nothing enforces that and the common loaders do not oblige:
  VideoHelperSuite hands over a `LazyAudioMap`, a `Mapping` that runs ffmpeg the
  first time a key is read, and `isinstance(audio, dict)` says no to it. The two
  keys are asked for instead of the container's type - and reading them is what
  makes a lazy input decode, so it has to happen there rather than in a
  membership test.
- **A VIDEO could hang the run for good.** `llama-mtmd-cli --video` feeds the
  file to `ffprobe` through *stdin*, and when the MP4 carries its `moov` atom at
  the front - which is what "faststart" means, and what ComfyUI, phones and most
  of the web produce - ffprobe has what it needs after a few kilobytes and exits
  without reading the rest. llama.cpp is still writing the remaining megabytes
  into that pipe and blocks there for good: no output, no error, no end. The
  same clip with `moov` at the end runs in six seconds. Frames are now decoded
  in-process instead - by seeking straight to each one on long clips, and to the
  frame rather than to its keyframe, so eight samples over a 250-frame GOP do
  not collapse onto four.
- **`max_frames` now means something for a VIDEO**, which is what makes the cost
  of describing a clip independent of its length: two seconds at 25 fps is 56
  images through the vision tower, and thirty seconds is 750.

### Added

- Screenshots of the T2VA writer, the Ref2VA writer, the Reference Caption chain
  and the options node in both READMEs, with the measured timings beside them.

## 0.9.1 - 2026-08-08

### Added

- **`device` - `auto`, `cpu`, or one `cuda:N` per GPU ComfyUI can see**, in one
  spelling across all three backends (`--device CUDA1` for the llama.cpp
  binaries, `main_gpu` for llama-cpp-python, `device_map` for Transformers). The
  important part is not the placement: **on another card ComfyUI's own models
  are no longer evicted first.** Every backend unloaded them unconditionally,
  which is right when both want the same VRAM and pure waste when they do not -
  it cost a full reload of the diffusion model after every rewrite. Pick a
  second card and `keep_model_loaded` becomes worth switching on. A device the
  machine does not have is refused, not quietly demoted.

### Fixed

- **Listing your GGUF models cost 31 seconds for ten files.** Building the
  dropdown needs six values from the first few kilobytes of each file, but
  `gguf.GGUFReader` materialises the whole header the moment it opens one -
  including `tokenizer.ggml.tokens`, a quarter of a million strings - and the
  bill arrived the first time ComfyUI answered `/object_info`. The header is now
  walked directly and skipped past: the same six values in **0.4 s**. A
  half-downloaded file is still refused, by checking its tensor offsets against
  its size rather than by failing to map them.

## 0.8.1 - 2026-08-08

### Changed

- **Models added to the pack after you installed are merged into your list.**
  Your copy is still never overwritten, but "we will not touch your list"
  quietly became "you will never see anything new", with nothing anywhere to say
  the node knew about more. The rule is set algebra, not a version check: beside
  the lists your file records `seed_offered`, every name the packaged list has
  ever put in front of this installation, and an update adds only the names that
  are in the pack, not in your file, and not already offered. An entry you
  deleted stays deleted, one you renamed is not duplicated, a genuinely new one
  arrives. One exception, once: a file written before this existed has no record
  of what it was offered, so on the first update everything missing comes back,
  and the previous file is kept beside it as `models.json.bak`. A file the node
  cannot parse is left exactly as it is.

## 0.8.0 - 2026-08-08

### Added

- **MiniMax-H3 Reference Caption.** The writer nodes read text, not pixels; this
  is where the text comes from. Connect an image, an audio clip or a video and a
  small multimodal model describes it into one labelled line of
  `reference_assets` - 3 s for a frame, 2 s for an audio clip, 5 s for a video on
  a 3.4 GB Qwen2.5-Omni-3B. It runs through the same llama.cpp binaries as
  everything else, so a machine that has run one rewrite downloads no runtime at
  all.
- **Chaining by wiring**: `reference_assets` into the next node's `previous`.
  Each label is numbered within its own category, which is the guide's own rule,
  so four assets come out as `Picture 1`, `Picture 2`, `Video 1`, `Audio 1` -
  not 1 through 4.
- **`role` picks both the label and the question asked.** The `Audio` question is
  the one that matters: `<Audio N>` is usually a *timbre* reference and a
  transcript throws away exactly the part that is needed, so the instruction says
  "do not transcribe" outright and asks for voice, delivery, instrumentation and
  ambience instead.
- **A `captioners` list in `models.json`**, with `mmproj` beside `file` - a
  multimodal model is two files from the same conversion. Only pairs that have
  actually been run are listed: llama.cpp's `mtmd` has to understand the
  projector format, and Gemma 4's aborts the process outright while
  `llama-completion` runs the same model as text with no trouble.

## 0.7.0 - 2026-08-08

### Added

- **The writer nodes - T2VA/I2VA/FL2VA/L2VA and Ref2VA.** The same output fields
  without the LoRA and without the 27B: MiniMax's own prompt-writing guide goes
  into the system prompt and any instruction-following GGUF writes to it. The
  smallest working setup drops from ~10 GB and ~13 GB of VRAM to **2.6 GB and
  ~5 GB**, and the four tasks the LoRA cannot do come with it. Ref2VA writes six
  sections instead of three, subject definitions and retention analysis included.
- **MiniMax-H3 Guide Prompt**, which hands the same system prompt to any LLM you
  already have, for people who would rather run the model themselves.
- **The guides are fetched from MiniMax's repository rather than bundled**, so an
  update to the guide is an update to the output without a release here.

## 0.6.2 - 2026-08-08

### Added

- **The base model's shape is checked before anything is downloaded.**
  Qwen3.5-9B carries the same `general.architecture = qwen35` in its header, so
  it looks like a match and the model list showed it - but it has 32 blocks of
  width 4096 where the adapter needs 64 of 5120, and llama.cpp refuses to attach
  the LoRA. The node reads those two header numbers first and says so. A 9B
  producing a plausible-looking rewrite is running **without** the adapter: the
  format is coming from the system prompt, not from the LoRA.
- **`gguf_runtime` - `auto`, `llama-cpp-python`, `llama.cpp`** - separating *what
  runs the model* from `llama_backend`, which picks *which official build to
  fetch*. Forcing the wheel now fails with a clear message when it is absent
  rather than quietly using something else, and forcing the binaries is the way
  out when the installed wheel is broken. Only the wheel can honour
  `keep_model_loaded`; the binaries hand the model back to the operating system
  when the subprocess exits.

## 0.6.0 - 2026-08-07

First public release, shared without a tag.

- The rewriter node: a short prompt in any language in, and H3's three fields -
  `integrated_multimodal_description`, `overall_soundscape`,
  `non_diegetic_music` - out, entirely locally.
- The LightX2V Prompt Rewriter LoRA on Qwen3.6-27B, in `nf4`, `int8` or
  `bfloat16`, or on a GGUF quant through llama.cpp with the binaries fetched on
  first use.
- Weights, adapter and runtime downloaded on demand into ComfyUI's own model
  folders, with progress on the node.
- A `models.json` of your own, copied on first use and never overwritten by an
  update.
