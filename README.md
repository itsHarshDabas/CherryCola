# CherryCola
Local model runtime for GraceOS: Ollama, llama.cpp or LM Studio underneath.

The repo, command, app and package are all called CherryCola (the repo was
called CherrySoda until 2026-11-23). [GraceOS](https://github.com/itsHarshDabas/GraceOS) builds its `cherrycola` package from the commit pinned in its `iso/cherrysoda.ref`. This code lived in GraceOS under `services/cherrycola` until 2026-10-21; its history came along.

Local-model CLI for GraceOS. Runs inference through **Ollama** (default),
**llama.cpp** (`llama-server`, every load option, requantizing, streaming
big models from disk) or **LM Studio's local server**. The GraceOS desktop
calls it; users can too.

- Stdlib only (Python 3.12+). `pytest` is a test-only dependency.
- No AI, provider, or network needed to *run the tool itself* — but `ask`
  and `chat` need a local model server (Ollama or LM Studio) with a model.
- The Ollama server starts on demand (`ask`, `chat`, `pull`, `models`,
  `start`); never at boot.

## Install

Copy the package to `/usr/lib/cherrycola/` and the launcher to
`/usr/bin/cherrycola`:

```sh
sudo mkdir -p /usr/lib/cherrycola /usr/bin
sudo cp -r cherrycola /usr/lib/cherrycola/
sudo install -m755 bin/cherrycola /usr/bin/cherrycola
```

Or run from this directory with [uv](https://docs.astral.sh/uv/):

```sh
uv run cherrycola ask "hello"
uv run python -m cherrycola status
./bin/cherrycola status
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
| `llamacpp_url` | `http://127.0.0.1:11436` | where CherryCola runs its own `llama-server` |
| `models_dir` | `~/.local/share/cherrycola/models` | GGUF files for the llama.cpp engine |
| `llama_server`, `llama_quantize`, `llama_imatrix` | the llama.cpp command names | set a full path for a custom build |
| `hf_url` | `https://huggingface.co` | override for tests |

`cherrycola use MODEL` and `cherrycola backend NAME` rewrite only that one
line (comments and other lines kept; appended if absent).

## Usage

```sh
cherrycola ask "what is ext4?"          # one question, streamed
echo "summarize this" | cherrycola ask -   # question from stdin
cherrycola ask --show-thinking "why?"   # keep <think> blocks (stripped by default)
cherrycola ask --web "latest Arch news?"  # search the web, cite sources
cherrycola search "latest Arch news" [--json]  # web results, no model
cherrycola chat                         # interactive (/exit, /clear, /web, Ctrl+D)
cherrycola status [--json]
cherrycola models
cherrycola recommend [--json]           # models that fit this PC, and the Ollama build for its GPU
cherrycola pull [MODEL]                 # Ollama only; downloads from the internet
cherrycola use qwen3:4b
cherrycola backend lmstudio             # or ollama, llamacpp, auto
cherrycola start                        # Ollama, or load the model in llama.cpp
cherrycola stop                         # stops only the server cherrycola started
```

## Model settings

Every model has its own load and sampling settings, saved in
`~/.config/grace/cherrycola-models.json`, plus defaults for all models.
`cherrycola settings --list` explains each one and which engines use it;
the chat app has the same settings in its side panel (the sliders button,
Ctrl+,), and hovering any row explains it.

```sh
cherrycola settings                        # what applies to the current model
cherrycola set context_length 16384        # this model
cherrycola set --all kv_cache_type q8_0    # every model
cherrycola preset code                     # precise | balanced | creative | code
cherrycola ask -o temperature=1.2 --preset creative "a poem"   # one run only
cherrycola unset gpu_layers
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
apply when CherryCola starts `ollama serve`. LM Studio only takes sampling
settings (loading is set in its app). `settings` marks anything the current
engine doesn't use.

**Every other llama.cpp option** is available too, read from the installed
`llama-server --help`, so new llama.cpp options show up without a CherryCola
update. Options that would open the server to the network, run tools or
download things (`--host`, `--tools`, `--hf-*`, ...) are left out.

```sh
cherrycola options mirostat                # search them, with their help
cherrycola flag --mirostat 2               # set one for this model
cherrycola flag --all --override-tensor "exps=CPU"
cherrycola flag --remove --mirostat
```

## The llama.cpp engine

`cherrycola backend llamacpp` switches to it. CherryCola starts one
`llama-server` per user on demand (never at boot), with the model's
settings as flags, and restarts it when the model or a load setting
changes. Its log is `~/.local/state/cherrycola/llama-server.log`.

```sh
cherrycola get Qwen/Qwen3-4B-GGUF:Q4_K_M   # GGUF from Hugging Face (resumes)
cherrycola import-ollama qwen3:4b          # reuse an Ollama download (a link, no copy)
cherrycola use Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M
cherrycola info                            # architecture, quantization, size
cherrycola quantize MODEL Q5_K_M           # requantize with llama-quantize
cherrycola imatrix MODEL calibration.txt   # importance matrix for IQ types
cherrycola quantize MODEL IQ3_M --imatrix FILE
cherrycola plan                            # how it splits over GPU, RAM, disk
cherrycola stop
```

Requantizing an already quantized model needs `--allow-requantize`
(it loses more quality than starting from F16/BF16), and the very low-bit
IQ types refuse to run without an importance matrix.

### Models bigger than RAM: streaming from disk

`cherrycola set disk_stream on` runs a model that doesn't fit in graphics
memory and RAM together, the way AirLLM does: what fits goes on the GPU,
and the remaining layers are read from disk for each token. CherryCola
memory-maps the file, turns off weight repacking (which would copy the
weights into RAM), keeps one context cache, and picks the GPU layers from
`cherrycola plan`. It works, but it is slow: the whole streamed part is
read once per token, so expect seconds per token on an NVMe SSD and far
worse on a hard disk. Smaller quantizations stream faster.

AirLLM's other ideas map onto llama.cpp like this: its 4/8-bit
"compression" is `quantize`, its layer sharding is the GGUF file itself
(read through mmap, no separate shard files), and prefetching is the
kernel's readahead.

## `status --json` shape (stable)

```json
{"backend": "ollama", "configured_backend": "auto", "reachable": true, "model": "qwen3:4b",
 "installed": true, "running": "qwen3:4b", "models": ["qwen3:4b"], "search": "duckduckgo"}
```

`running` is `null` when no model is loaded. Exit 0 even when the server is
down (that is a status, not an error). Exit codes everywhere: 0 ok,
1 usage/other error, 2 backend unreachable, 3 model missing
(`ask`/`chat` when the model isn't downloaded — nothing is downloaded
without `pull`).

## How web search works

`cherrycola search` queries DuckDuckGo's HTML results (no key) or a
local SearXNG instance (`search = searxng`), then `ask --web` reads the
top `web_pages` pages locally and asks the model to answer from them,
with `[n]` citations and a `Sources:` list. Results and page text are
untrusted: the model only writes text, it never runs anything.
In `chat`, `/web` toggles web mode; history keeps only the question
and the answer, never the page text.

## Chat app

`cherrycola-chat` (CherryCola in the app grid) is a GTK4 / libadwaita window
on the same engine: pick the model in the header, turn on web search with the
globe, and old chats stay in the sidebar. A missing model shows a Download
banner. The sliders button (Ctrl+,) opens Model settings: the engine, every
setting above with an explanation on hover, and with llama.cpp the model
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
