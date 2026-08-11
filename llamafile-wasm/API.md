# llamafile-wasm internal API contract

This file is the contract between the WebAssembly module (`src/`, `build.sh`)
and the browser front-end (`web/`). Both sides are developed against it, so
neither side may change it unilaterally.

## Artifacts produced by `build.sh`

* `dist/llamafile.mjs`  — Emscripten ES6 module factory (`MODULARIZE`, `EXPORT_ES6`),
  default export is `async function createLlamafileModule(opts?) -> Module`.
* `dist/llamafile.wasm` — the WebAssembly binary, loaded relative to the `.mjs`.

The module is built for `ENVIRONMENT=web,worker,node` so the same artifact runs
in a browser main thread, in a Web Worker, and under Node for smoke tests.

## Low-level module functions (embind, on the `Module` object)

All strings are UTF-8 `std::string`.

| function | signature | notes |
| --- | --- | --- |
| `lf_load(path, n_ctx, n_threads)` | `(string, int, int) -> int` | 0 on success, non-zero on failure. `path` is a path inside the Emscripten FS. |
| `lf_model_info()` | `() -> string` | JSON: `{"n_params":…, "n_ctx_train":…, "n_ctx":…, "n_vocab":…, "desc":"…", "has_chat_template":bool, "size_bytes":…}` |
| `lf_format_chat(messages_json, add_assistant)` | `(string, bool) -> string` | Applies the model's chat template to `[{"role":…,"content":…}, …]`. Falls back to a generic template when the model has none. |
| `lf_gen_begin(prompt, n_predict, temp, top_p, seed)` | `(string, int, float, float, int) -> int` | 0 on success. Tokenises + evaluates the prompt. `seed < 0` means random. |
| `lf_gen_next()` | `() -> string` | Next token piece, or `""` when generation is finished. |
| `lf_gen_done()` | `() -> bool` | True once EOG/`n_predict` was reached. |
| `lf_gen_end()` | `() -> void` | Aborts/finishes the current generation. |
| `lf_perf()` | `() -> string` | JSON: `{"prompt_tokens":…, "predicted_tokens":…, "prompt_ms":…, "predict_ms":…}` |
| `lf_last_error()` | `() -> string` | Human-readable message for the last failed call. |
| `lf_unload()` | `() -> void` | Frees model + context. |

Also exported at runtime: `FS`, `ccall`, `cwrap`, `HEAPU8`.

## High-level browser API — `web/llamafile.js`

```js
import { Llamafile } from './llamafile.js';

const lf = new Llamafile({ wasmUrl: './dist/llamafile.mjs' });

await lf.load(source, {
  nCtx: 2048,
  onProgress: ({ stage, loaded, total }) => {},   // stage: 'download' | 'fs' | 'model'
});
// `source` is a URL string, a File/Blob (from <input type=file>), or an ArrayBuffer.

const info = await lf.info();                      // parsed lf_model_info()
for await (const piece of lf.chat(messages, { nPredict: 256, temp: 0.7, topP: 0.9, seed: -1 })) {
  // `piece` is a token string, already detokenised
}
await lf.stop();       // cancel an in-flight generation
await lf.perf();       // parsed lf_perf()
await lf.unload();
```

The work happens in a Web Worker (`web/llamafile-worker.js`) so generation never
blocks the UI thread. The wrapper falls back to running the module on the main
thread if `Worker` is unavailable.

## DOM contract for the demo page — `web/index.html`

Stable IDs / attributes the Playwright tests rely on:

| selector | meaning |
| --- | --- |
| `#model-url` | text input holding the GGUF URL to fetch |
| `#model-file` | `<input type=file>` for a local GGUF |
| `#load-model` | button that starts loading the model in `#model-url` |
| `#status` | status line; `data-state` is one of `idle`, `loading`, `ready`, `generating`, `error` |
| `#model-info` | populated with the model description once ready |
| `#prompt` | textarea for the user message |
| `#send` | button that sends the message |
| `#stop` | button that aborts generation |
| `#messages` | chat log container |
| `.message[data-role=user\|assistant]` | one chat message; `.message-content` holds the text |
| `#perf` | timing summary shown after each generation |

`window.llamafileReady` resolves once the page's JS is wired up, and
`window.__llamafileState` mirrors `#status`'s `data-state` for test polling.
