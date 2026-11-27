# CherryColaLLM
Local model runtime for GraceOS: Ollama, llama.cpp or LM Studio underneath.

The app, command and package are called CherryColaLLM (`cherrycolallm`,
`cherrycolallm-chat`). It was CherryCola until 2026-11-23 and CherrySoda
before that, and the old `cherrycola` and `cherrycola-chat` commands still
work as aliases. The Python package, config file
(`~/.config/grace/cherrycola.conf`) and data folders keep the `cherrycola`
name, so nothing moves on upgrade. [GraceOS](https://github.com/itsHarshDabas/GraceOS)
builds its `cherrycolallm` package from the commit pinned in its
`iso/cherrycolallm.ref`. This code lived in GraceOS under
`services/cherrycola` until 2026-10-21; its history came along.

Local-model CLI for GraceOS. Runs inference through **Ollama** (default),
**llama.cpp** (`llama-server`, every load option, requantizing, streaming
big models from disk; it comes with the packages), **LM Studio's local
server**, or **AirLLM** (optional: big Hugging Face models layer by layer
on a small graphics card). The GraceOS desktop calls it; users can too.

- Stdlib only (Python 3.12+). `pytest` is a test-only dependency.
- No AI, provider, or network needed to *run the tool itself* — but `ask`
  and `chat` need a local model server (Ollama or LM Studio) with a model.
- The Ollama server starts on demand (`ask`, `chat`, `pull`, `models`,
  `start`); never at boot.

## Install

Copy the package to `/usr/lib/cherrycola/`, the launcher to
`/usr/bin/cherrycolallm`, and link the old name to it:

```sh
sudo mkdir -p /usr/lib/cherrycola /usr/bin
sudo cp -r cherrycola /usr/lib/cherrycola/
sudo install -m755 bin/cherrycolallm /usr/bin/cherrycolallm
sudo ln -sf cherrycolallm /usr/bin/cherrycola
```

### Windows

Download `cherrycolallm-<version>-windows-x64-setup.exe` from the
[releases](https://github.com/itsHarshDabas/CherryColaLLM/releases) (per-user, no
admin, adds itself to PATH; `cherrycola` works as the old name), or use the
`.zip`, or Scoop. llama.cpp comes with it (see below), so a `.gguf` file
runs with nothing else installed; Ollama (`winget install Ollama.Ollama`)
and LM Studio are optional. Engines installed to their usual folders are
found even when they aren't on PATH. Same commands as
on Linux (`cherrycolallm recommend`, `pull`, `ask`); settings are in
`%APPDATA%\CherryCola\grace\`, models, chats and logs in
`%LOCALAPPDATA%\CherryCola\`. The setup installs **CherryColaLLM**, a native window (Rust) with chats, model
switching, a Models page (get by name, Hugging Face repo, link or file) and
attachments: drop in images, documents (text, Word, Excel, PowerPoint, PDF),
audio or video. **Model settings** (chat header or Ctrl+,) has the same settings as the
Linux app: context length, GPU layers, sampling and the rest as sliders with safe limits
and hover help, for one model or all of them, plus model folders, model tools (info,
GPU/RAM plan, requantize, Hugging Face download, Ollama import) and every llama-server
option, searchable. It drives the same `cherrycolallm` command, so it works with
Ollama, llama.cpp and LM Studio. Images need a model that can see (gemma3,
llava...); video needs `ffmpeg` and audio a transcriber (`whisper`), see
`cherrycola/attach.py`. The GTK chat app stays for Linux.

Or run from this directory with [uv](https://docs.astral.sh/uv/):

```sh
uv run cherrycolallm ask "hello"
uv run python -m cherrycola status
./bin/cherrycolallm status
```

## Config

`~/.config/grace/cherrycola.conf` (respects `$XDG_CONFIG_HOME`), simple
`key = value` lines with `#` comments. All optional:

| key | default | notes |
|---|---|---|
| `backend` | `auto` | `auto` = Ollama if the `ollama` command exists or its server answers, else LM Studio if its server answers |
| `model` | RAM-based | under 6 GB RAM → `qwen3:1.7b`, under 12 GB → `qwen3:4b`, else `qwen3:8b`; LM Studio defaults to its first listed model |
| `ollama_url` | `http://127.0.0.1:11434` | |
| `lmstudio_url` | `http://127.0.0.1:1234` | |
| `search` | `duckduckgo` | `duckduckgo` (default), `searxng`, or `off` |
| `searxng_url` | `http://127.0.0.1:8888` | base URL of a local SearXNG instance |
| `duckduckgo_url` | `https://html.duckduckgo.com/html/` | override for tests |
| `web_pages` | `3` | pages to read for `ask --web` (clamped 0..5) |
| `llamacpp_url` | `http://127.0.0.1:11436` | where CherryColaLLM runs its own `llama-server` |
| `models_dir` | `~/.local/share/cherrycola/models` | GGUF files for the llama.cpp engine |
| `llama_server`, `llama_quantize`, `llama_imatrix` | the llama.cpp command names | set a full path for a custom build |
| `model_dirs` | none | more folders of GGUF models, separated by `|` (`cherrycolallm folder add` writes it) |
| `hf_url` | `https://huggingface.co` | override for tests |

`cherrycolallm use MODEL` and `cherrycolallm backend NAME` rewrite only that one
line (comments and other lines kept; appended if absent).

## Usage

```sh
cherrycolallm ask "what is ext4?"          # one question, streamed
echo "summarize this" | cherrycolallm ask -   # question from stdin
cherrycolallm ask --show-thinking "why?"   # keep <think> blocks (stripped by default)
cherrycolallm ask --web "latest Arch news?"  # search the web, cite sources
cherrycolallm search "latest Arch news" [--json]  # web results, no model
cherrycolallm chat                         # interactive (/exit, /clear, /web, Ctrl+D)
cherrycolallm status [--json]
cherrycolallm models
cherrycolallm recommend [--json]           # models that fit this PC, and the Ollama build for its GPU
cherrycolallm pull [MODEL] [--use]         # through the selected engine; see "Getting models"
cherrycolallm unload [-m MODEL | --all]    # free the model's memory now
cherrycolallm ask --temporary "fix: teh wether"  # load only for this answer, then unload
cherrycolallm ask --pull --temporary -m qwen3:1.7b "hi"  # get the model first if it's missing
cherrycolallm use qwen3:4b
cherrycolallm backend lmstudio             # or ollama, llamacpp, auto
cherrycolallm start                        # Ollama, or load the model in llama.cpp
cherrycolallm stop                         # stops only the server CherryColaLLM started
```

### Bundled llama.cpp

Every package except the wheel carries llama.cpp's own release build
(`packaging/llamacpp.env` pins it; `packaging/fetch-llamacpp.sh` fetches and
checks it): `llama-server`, `llama-quantize` and `llama-imatrix` with their
libraries, about 90 MB. It goes in `<install folder>\llama.cpp` on Windows
and `/usr/lib/cherrycola/llama.cpp` on Linux. It is the **Vulkan** build:
one build that uses NVIDIA, AMD and Intel graphics through the normal
graphics driver, and falls back to the CPU (picking the best CPU code for
the processor) when there is no usable GPU. CUDA builds are NVIDIA-only and
add about 400 MB, so they aren't bundled; a `llama-server` you install
yourself (on PATH, or `llama_server` in the config) wins over the bundled
one. With nothing else installed, `backend = auto` picks llama.cpp.

## Getting models

**Model files and folders on this PC are used where they are, never
copied.** A `.gguf` file or a folder of them (LM Studio's models folder, for
example) runs with llama.cpp, whatever engine is selected; `pull --use`
(the app's File… and Folder… buttons) also switches the engine to
llama.cpp. A Hugging Face model folder (`config.json` + `.safetensors`)
runs with AirLLM. Ollama and LM Studio can only run models in their own
stores, so CherryColaLLM never imports a file into them (that copy took as
long as a download, which is what the app used to show as "Downloading").

`pull` takes a model in any of these forms:

| You give | Ollama | llama.cpp | LM Studio | AirLLM |
|---|---|---|---|---|
| Ollama tag: `qwen3:4b` | `ollama pull` | links Ollama's copy if it has one, else downloads the same model's GGUF from Hugging Face (the recommended ones) | `lms get` | no |
| Hugging Face GGUF: `Qwen/Qwen3-4B-GGUF:Q4_K_M`, `hf.co/...`, `https://huggingface.co/...` | pulls `hf.co/...` | downloads the GGUF (resumes) | `lms get` | no |
| Hugging Face model: `Qwen/Qwen2.5-7B-Instruct` | | | | downloads the safetensors |
| `https://` link to a `.gguf` | llama.cpp downloads it into `models/downloads/` | same | same | no |
| a `.gguf` file, or a folder of them | runs with llama.cpp, in place | in place | runs with llama.cpp | runs with llama.cpp |
| a model folder (safetensors) | runs with AirLLM, in place | | | in place |

Links must be `https://`, and a download is kept only if it really is a
GGUF file. On Ollama, `owner/name` is an Ollama namespace unless the name
says GGUF; write `hf.co/owner/name` to be sure.

### Loading a model only for a moment

`ask --temporary` loads the model for one answer and unloads it as soon
as the answer is done, so a quick job (CottonCandy's Enhance search, for
example) doesn't keep gigabytes of RAM taken. Ollama gets
`keep_alive: 0`; llama.cpp's server is stopped; LM Studio gets
`lms unload`. If the model was already loaded before (you are chatting
with it), it stays loaded. `--pull` gets a missing model through the
route above first, with progress on stderr.

## Model settings

Every model has its own load and sampling settings, saved in
`~/.config/grace/cherrycola-models.json`, plus defaults for all models.
`cherrycolallm settings --list` explains each one and which engines use it;
the chat app has the same settings in its side panel (the sliders button,
Ctrl+,), and hovering any row explains it.

```sh
cherrycolallm settings                        # what applies to the current model
cherrycolallm set context_length 16384        # this model
cherrycolallm set --all kv_cache_type q8_0    # every model
cherrycolallm preset code                     # precise | balanced | creative | code
cherrycolallm ask -o temperature=1.2 --preset creative "a poem"   # one run only
cherrycolallm unset gpu_layers
```

| group | settings |
|---|---|
| Loading | `context_length`, `threads`, `batch_size`, `ubatch_size`, `use_mmap`, `use_mlock`, `disk_stream`, `keep_alive` |
| GPU and memory | `gpu_layers`, `main_gpu`, `tensor_split`, `cpu_moe_layers`, `flash_attention`, `kv_cache_type` |
| Long context | `rope_scaling`, `rope_freq_base`, `rope_freq_scale`, `yarn_orig_ctx` |
| Sampling | `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `max_tokens`, `seed`, `stop` |
| Behaviour | `system_prompt`, `preset`, `draft_model` |

Ollama takes most load settings per request; `kv_cache_type` and
`flash_attention` are server-wide there, so set them with `--all` and they
apply when CherryColaLLM starts `ollama serve`. LM Studio only takes sampling
settings (loading is set in its app). `settings` marks anything the current
engine doesn't use.

**Every other llama.cpp option** is available too, read from the installed
`llama-server --help`, so new llama.cpp options show up without a CherryColaLLM
update. Options that would open the server to the network, run tools or
download things (`--host`, `--tools`, `--hf-*`, ...) are left out.

```sh
cherrycolallm options mirostat                # search them, with their help
cherrycolallm flag --mirostat 2               # set one for this model
cherrycolallm flag --all --override-tensor "exps=CPU"
cherrycolallm flag --remove --mirostat
```

## The llama.cpp engine

`cherrycolallm backend llamacpp` switches to it. CherryColaLLM starts one
`llama-server` per user on demand (never at boot), with the model's
settings as flags, and restarts it when the model or a load setting
changes. Its log is `~/.local/state/cherrycola/llama-server.log`.

```sh
cherrycolallm get Qwen/Qwen3-4B-GGUF:Q4_K_M   # GGUF from Hugging Face (resumes)
cherrycolallm folder add ~/.lmstudio/models   # use GGUFs from any folder
cherrycolallm folder list
cherrycolallm import-ollama qwen3:4b          # reuse an Ollama download (a link, no copy)
cherrycolallm use Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M
cherrycolallm info                            # architecture, quantization, size
cherrycolallm quantize MODEL Q5_K_M           # requantize with llama-quantize
cherrycolallm imatrix MODEL calibration.txt   # importance matrix for IQ types
cherrycolallm quantize MODEL IQ3_M --imatrix FILE
cherrycolallm plan                            # how it splits over GPU, RAM, disk
cherrycolallm stop
```

Requantizing an already quantized model needs `--allow-requantize`
(it loses more quality than starting from F16/BF16), and the very low-bit
IQ types refuse to run without an importance matrix.

### Models bigger than RAM: streaming from disk

`cherrycolallm set disk_stream on` runs a model that doesn't fit in graphics
memory and RAM together, the way AirLLM does: what fits goes on the GPU,
and the remaining layers are read from disk for each token. CherryColaLLM
memory-maps the file, turns off weight repacking (which would copy the
weights into RAM), keeps one context cache, and picks the GPU layers from
`cherrycolallm plan`. It works, but it is slow: the whole streamed part is
read once per token, so expect seconds per token on an NVMe SSD and far
worse on a hard disk. Smaller quantizations stream faster.

AirLLM's other ideas map onto llama.cpp like this: its 4/8-bit
"compression" is `quantize`, its layer sharding is the GGUF file itself
(read through mmap, no separate shard files), and prefetching is the
kernel's readahead.

## The AirLLM engine (optional)

[AirLLM](https://github.com/lyogavin/airllm) runs a model one transformer
layer at a time, reading each layer from disk just before it is needed, so
a 70B model runs in about 4 GB of graphics memory. It is slow (every token
reads the whole model from disk) and best with an NVIDIA card, but it runs
models nothing else here can.

```sh
cherrycolallm airllm install      # its own Python env: PyTorch (CUDA build with NVIDIA) + airllm
cherrycolallm backend airllm
cherrycolallm pull --use Qwen/Qwen2.5-7B-Instruct    # or a model folder on this PC
cherrycolallm ask "hello"
cherrycolallm airllm              # status;  airllm uninstall removes the env, keeps models
```

It needs PyTorch (about 1 GB, 3 GB for the CUDA build), so it isn't
bundled: `airllm install` makes a Python environment just for it in
CherryColaLLM's data folder, using a Python 3.9+ on the PC (the Windows
installer has none of its own; `winget install Python.Python.3.12`), or set
`airllm_python` to a Python that already has `airllm`. The Windows app's
Models page has an Install AirLLM button when AirLLM is the engine.
`cherrycola/airllm_worker.py` serves the model on `airllm_url` (default
`http://127.0.0.1:11437`) with an OpenAI-style chat endpoint, started on
demand like llama-server. Settings: context length, sampling (temperature,
top K/P, min P, repeat penalty, max tokens, seed, stop strings), system
prompt, plus `airllm_compression` (4bit/8bit layers, NVIDIA only, about 3x
faster) and `airllm_prefetch`.

The first load splits the model into one file per layer (AirLLM's own
format) under its data folder: that takes a while and as much disk again as
the model. The model folder itself stays where it is.

Earlier, only AirLLM's *idea* was here: `disk_stream` on llama.cpp (below)
streams the layers that don't fit from disk the same way, for GGUF files.

## `status --json` shape (stable)

```json
{"backend": "ollama", "configured_backend": "auto", "reachable": true, "model": "qwen3:4b",
 "installed": true, "running": "qwen3:4b", "models": ["qwen3:4b"], "search": "duckduckgo"}
```

`running` is `null` when no model is loaded. Exit 0 even when the server is
down (that is a status, not an error). Exit codes everywhere: 0 ok,
1 usage/other error, 2 backend unreachable, 3 model missing
(`ask`/`chat` when the model isn't downloaded — nothing is downloaded
without `pull` or `ask --pull`).

## How web search works

`cherrycolallm search` queries DuckDuckGo's HTML results (no key) or a
local SearXNG instance (`search = searxng`), then `ask --web` reads the
top `web_pages` pages locally and asks the model to answer from them,
with `[n]` citations and a `Sources:` list. Results and page text are
untrusted: the model only writes text, it never runs anything.
In `chat`, `/web` toggles web mode; history keeps only the question
and the answer, never the page text.

## Chat app

`cherrycolallm-chat` (CherryColaLLM in the app grid) is a GTK4 / libadwaita window
on the same engine: pick the model in the header, turn on web search with the
globe, and old chats stay in the sidebar. A missing model shows a Download
banner. The sliders button (Ctrl+,) opens Model settings: the engine, every
setting above with an explanation on hover (numbers are sliders that
stay within a safe range, and context length stops at 4x what the model
was trained for), the model folders, and with llama.cpp the model
tools (plan, requantize, download, import from Ollama) and every
llama-server option, searchable. Enter sends, Shift+Enter adds a line;
each answer has a copy button and its speed. Chats are JSON files in `~/.local/share/cherrycola/chats/`; delete one
with the trash button. Needs `python-gobject` and `libadwaita`.

## Run the tests

Requires [uv](https://docs.astral.sh/uv/). From this directory:

```sh
uv run pytest -q
```

Tests spin up fake Ollama/LM Studio servers on localhost (port 0) — no
network, no real `ollama` process. The spawn function is injectable
(`start_server(cfg, spawn=...)`) and monkeypatched in tests.

## Attachments from the command line

```sh
cherrycolallm ask --attach report.docx --attach chart.png "Summarise this"
cherrycolallm ask --history earlier.json "and the next step?"
```
