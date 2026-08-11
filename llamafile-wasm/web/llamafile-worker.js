// llamafile-worker.js — runs the llamafile WebAssembly module off the main
// thread. Started as a module worker: `new Worker(url, { type: 'module' })`.
//
// Protocol (every request carries an `id`; every reply echoes it):
//
//   main -> worker   { id, type: 'ping' }
//                    { id, type: 'init',        wasmUrl? }
//                    { id, type: 'load',        source, options }
//                    { id, type: 'info' }
//                    { id, type: 'format-chat', messages, addAssistant }
//                    { id, type: 'chat-start',  options }
//                    { id, type: 'chat-cancel' }
//                    { id, type: 'perf' }
//                    { id, type: 'unload' }
//
//   worker -> main   { type: 'ready' }                       (once, on startup)
//                    { id, type: 'progress', stage, loaded, total }
//                    { id, type: 'piece',    text }
//                    { id, type: 'result',   result }
//                    { id, type: 'error',    error: { message, stack } }
//
// `source` is { kind:'buffer', buffer, name } (the ArrayBuffer is transferred),
// { kind:'file', file, name } (a File/Blob the worker reads itself) or
// { kind:'url', url, name }.
//
// chat-cancel is answered out of band: it is handled the moment it arrives,
// even while a chat-start request is still streaming, because the generation
// loop in llamafile-engine.js yields to the event loop between tokens.

import { LlamafileEngine, handleRequest } from './llamafile-engine.js';

// Resolved against this file, i.e. llamafile-wasm/dist/llamafile.mjs. The main
// thread normally overrides it so both sides agree on one absolute URL.
const DEFAULT_WASM_URL = new URL('../dist/llamafile.mjs', import.meta.url).href;

const engine = new LlamafileEngine({ wasmUrl: DEFAULT_WASM_URL });

let queue = Promise.resolve();

function post(message, transfer) {
  self.postMessage(message, transfer || []);
}

function fail(id, err) {
  post({
    id,
    type: 'error',
    error: {
      message: err && err.message ? err.message : String(err),
      stack: err && err.stack ? err.stack : null,
    },
  });
}

self.onmessage = (event) => {
  const msg = event.data;
  if (!msg || typeof msg.type !== 'string') return;

  if (msg.wasmUrl) engine.wasmUrl = msg.wasmUrl;

  // Handled inline so it can interrupt an in-flight generation.
  if (msg.type === 'chat-cancel') {
    try {
      post({ id: msg.id, type: 'result', result: { cancelled: engine.cancel() } });
    } catch (err) {
      fail(msg.id, err);
    }
    return;
  }

  queue = queue.then(async () => {
    try {
      const result = await handleRequest(engine, msg, (event_) => post({ id: msg.id, ...event_ }));
      post({ id: msg.id, type: 'result', result });
    } catch (err) {
      fail(msg.id, err);
    }
  });
};

self.onerror = (event) => {
  post({
    type: 'error',
    id: null,
    error: { message: (event && event.message) || 'Worker error', stack: null },
  });
};

post({ type: 'ready' });
