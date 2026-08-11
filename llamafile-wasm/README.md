# llamafile in the browser (WebAssembly)

Run a GGUF model entirely inside a web page. No server, no install, no
network round-trip for inference — llama.cpp is compiled to WebAssembly,
the model is fetched (or picked from disk) into the tab, and every token is
generated on the user's own machine.

This is the browser counterpart to a llamafile: instead of one executable
that runs on six operating systems, it is one page that runs in any browser
with WebAssembly SIMD.

```
llamafile-wasm/
├── API.md                  the contract between the wasm module and the front-end
├── build.sh                builds llama.cpp + the binding into dist/
├── src/llamafile-wasm.cpp  embind binding — the lf_* API
├── web/                    the demo page, the Llamafile JS class and its Web Worker
├── serve.py                dependency-free static dev server
└── tests/                  node smoke test + Playwright browser spec
```

## Build

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)
(tested with 4.0.14) and the `llama.cpp` submodule checked out with llamafile's
patches applied:

```sh
git submodule update --init llama.cpp
./llama.cpp.patches/apply-patches.sh

source /path/to/emsdk/emsdk_env.sh    # build.sh will also find an EMSDK env var
bash llamafile-wasm/build.sh          # add --clean for a full rebuild, -jN for parallelism
```

This produces `dist/llamafile.mjs` (~88 KB) and `dist/llamafile.wasm` (~3.8 MB).
Both are build artefacts and are not checked in.

## Run

```sh
python3 llamafile-wasm/serve.py           # http://127.0.0.1:8080/
```

Open the page, keep the pre-filled model URL (a 135M-parameter SmolLM2) or
point it at any GGUF you like, press **Load model**, and chat. A local `.gguf`
can be picked with the file input instead — it never leaves the machine.

`serve.py --models <dir>` exposes a directory of GGUFs at `/models/`, which is
handy for testing without re-downloading. The server sets the cross-origin
isolation headers and honours range requests so large models stream properly.

A plain static host works too; the only requirement is that `.wasm` is served
as `application/wasm`.

## Test

```sh
# native: exercises the whole lf_* contract under node
node llamafile-wasm/tests/smoke.mjs /path/to/model.gguf

# browsers: starts serve.py itself, then drives the real page
cd /path/to/playwright/install
node /path/to/llamafile-wasm/tests/browser.spec.mjs --browser=chromium
node /path/to/llamafile-wasm/tests/browser.spec.mjs --browser=firefox
```

The browser spec covers page load, the DOM contract, loading a model from a
URL and from the file picker, streamed generation, cancellation, and the
degraded state when `dist/` has not been built yet. `LF_MODEL=big` switches it
to the larger model.

The Playwright MCP server is a convenient way to drive the page by hand while
developing:

```sh
claude mcp add playwright -- npx -y @playwright/mcp@latest --browser firefox
```

## How it works

`src/llamafile-wasm.cpp` links against llama.cpp's static libraries and exposes
a small embind surface — load a model, apply its chat template, start a
generation, pull one token at a time. Pulling token-by-token is what makes
streaming possible: `web/llamafile-worker.js` runs the module in a Web Worker
and posts each piece back as it appears, so a long generation never freezes the
page, and a **Stop** click is processed between tokens.

Model bytes are written into the Emscripten filesystem and loaded with mmap
disabled, since a browser has no file to map.

The module is compiled with WebAssembly SIMD (`-msimd128`), which selects
ggml's wasm quant kernels rather than the scalar fallback.

## Speed

On a 4-core x86-64 container, SmolLM2-135M-Instruct Q8_0 runs at roughly
42 tok/s prompt eval and 25 tok/s generation in Chromium 141 — single-threaded,
on one core.

Firefox works but was about 6x slower in that same container. That gap is not
in this port: it reproduces on a 921-byte wasm module containing none of this
code, and it survives every build-flag change (exception encoding, memory
growth, SIMD). The cause is that the Firefox build there executed all
WebAssembly in its baseline tier and never tiered up to the optimizing
compiler. Whether that is specific to that container or that build is unknown,
so do not read it as a property of Firefox generally — measure on your own
machine before drawing conclusions.

## Limits, and where to go next

- **Single-threaded.** The build deliberately does not use pthreads yet.
  Threads need `SharedArrayBuffer`, which needs cross-origin isolation — the
  headers `serve.py` already sends. This is the biggest available speedup on a
  multi-core machine and the obvious next step.
- **No GPU.** WebGPU is the natural follow-up; ggml has a WebGPU backend
  in progress upstream.
- **Memory is 32-bit.** wasm32 caps usable memory below 4 GB, so this suits
  small models — a few hundred million parameters, quantized. A 135M Q8_0
  model needs about 140 MB.
- **No prefix reuse.** Each turn re-evaluates the whole conversation, because
  the KV cache is cleared at the start of every generation. Multi-turn chats
  get slower than they need to be.
- The cosmopolitan-specific parts of llamafile — tinyBLAS, the CPU dispatch in
  `llamafile/`, the single-file executable packaging — are not part of this
  build. `GGML_LLAMAFILE` is off and the plain ggml CPU backend is used.
