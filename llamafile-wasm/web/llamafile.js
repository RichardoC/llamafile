// llamafile.js — the browser-facing API described in llamafile-wasm/API.md.
//
//   import { Llamafile } from './llamafile.js';
//
//   const lf = new Llamafile();
//   await lf.load(urlOrFileOrArrayBuffer, { nCtx: 2048, onProgress });
//   const info = await lf.info();
//   for await (const piece of lf.chat(messages, { nPredict: 256 })) { ... }
//   await lf.stop(); await lf.perf(); await lf.unload();
//
// All of the heavy lifting happens in `llamafile-worker.js`; if Workers (or
// module workers) are unavailable we transparently fall back to running the
// same engine on the main thread.

const DEFAULT_WASM_URL = new URL('../dist/llamafile.mjs', import.meta.url).href;
const DEFAULT_WORKER_URL = new URL('./llamafile-worker.js', import.meta.url).href;
const WORKER_START_TIMEOUT_MS = 15000;

let nextId = 1;

export class LlamafileError extends Error {
  constructor(message, stack) {
    super(message);
    this.name = 'LlamafileError';
    if (stack) this.workerStack = stack;
  }
}

/* ------------------------------------------------------------------ backends */

class WorkerBackend {
  constructor({ workerUrl, wasmUrl }) {
    this.workerUrl = workerUrl;
    this.wasmUrl = wasmUrl;
    this.worker = null;
    this.pending = new Map();
  }

  start() {
    if (this.startPromise) return this.startPromise;
    this.startPromise = new Promise((resolve, reject) => {
      let worker;
      try {
        worker = new Worker(this.workerUrl, { type: 'module', name: 'llamafile' });
      } catch (err) {
        reject(err);
        return;
      }
      this.worker = worker;

      const timer = setTimeout(
        () => reject(new Error('The llamafile worker did not start in time')),
        WORKER_START_TIMEOUT_MS
      );
      const settleFail = (err) => {
        clearTimeout(timer);
        reject(err);
      };

      worker.onerror = (event) => {
        const message =
          (event && (event.message || (event.error && event.error.message))) || 'llamafile worker failed to start';
        settleFail(new Error(message));
        this._rejectAll(new LlamafileError(message));
      };
      worker.onmessageerror = () => settleFail(new Error('llamafile worker sent an uncloneable message'));
      worker.onmessage = (event) => {
        const msg = event.data;
        if (!msg) return;
        if (msg.type === 'ready') {
          clearTimeout(timer);
          resolve(this);
          return;
        }
        this._dispatch(msg);
      };
    });
    return this.startPromise;
  }

  _dispatch(msg) {
    const entry = this.pending.get(msg.id);
    if (!entry) return;
    switch (msg.type) {
      case 'progress':
        if (entry.handlers.onProgress) {
          entry.handlers.onProgress({ stage: msg.stage, loaded: msg.loaded, total: msg.total });
        }
        break;
      case 'piece':
        if (entry.handlers.onPiece) entry.handlers.onPiece(msg.text);
        break;
      case 'result':
        this.pending.delete(msg.id);
        entry.resolve(msg.result);
        break;
      case 'error':
        this.pending.delete(msg.id);
        entry.reject(new LlamafileError(msg.error ? msg.error.message : 'Unknown worker error', msg.error && msg.error.stack));
        break;
      default:
        break;
    }
  }

  _rejectAll(err) {
    for (const [, entry] of this.pending) entry.reject(err);
    this.pending.clear();
  }

  send(msg, handlers = {}, transfer = []) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, handlers });
      try {
        this.worker.postMessage({ ...msg, id, wasmUrl: this.wasmUrl }, transfer);
      } catch (err) {
        this.pending.delete(id);
        reject(err);
      }
    });
  }

  dispose() {
    this._rejectAll(new LlamafileError('llamafile worker terminated'));
    if (this.worker) this.worker.terminate();
    this.worker = null;
  }
}

class MainThreadBackend {
  constructor({ wasmUrl }) {
    this.wasmUrl = wasmUrl;
    this.queue = Promise.resolve();
  }

  async start() {
    const mod = await import('./llamafile-engine.js');
    this.engine = new mod.LlamafileEngine({ wasmUrl: this.wasmUrl });
    this.handleRequest = mod.handleRequest;
    return this;
  }

  send(msg, handlers = {}) {
    // Cancels must not queue behind the generation they are cancelling.
    if (msg.type === 'chat-cancel') {
      return Promise.resolve({ cancelled: this.engine.cancel() });
    }
    const run = async () => {
      try {
        return await this.handleRequest(this.engine, msg, (event) => {
          if (event.type === 'progress' && handlers.onProgress) {
            handlers.onProgress({ stage: event.stage, loaded: event.loaded, total: event.total });
          } else if (event.type === 'piece' && handlers.onPiece) {
            handlers.onPiece(event.text);
          }
        });
      } catch (err) {
        throw err instanceof LlamafileError ? err : new LlamafileError(err.message, err.stack);
      }
    };
    const result = this.queue.then(run, run);
    this.queue = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  dispose() {
    try {
      if (this.engine) this.engine.unload();
    } catch (_) {
      /* ignore */
    }
  }
}

/* -------------------------------------------------------------- public API */

export class Llamafile {
  /**
   * @param {object} [opts]
   * @param {string} [opts.wasmUrl]   URL of dist/llamafile.mjs
   * @param {string} [opts.workerUrl] URL of llamafile-worker.js
   * @param {boolean} [opts.useWorker=true] set false to force main-thread mode
   */
  constructor(opts = {}) {
    this.wasmUrl = new URL(opts.wasmUrl || DEFAULT_WASM_URL, document.baseURI || location.href).href;
    this.workerUrl = new URL(opts.workerUrl || DEFAULT_WORKER_URL, document.baseURI || location.href).href;
    this.useWorker = opts.useWorker !== false && typeof Worker === 'function';
    this.usingWorker = false;
    this.loaded = false;
    this.generating = false;
    this.modelInfo = null;
    this._backend = null;
    this._backendPromise = null;
  }

  /** Resolve the backend, preferring a Worker and falling back to this thread. */
  async _ensureBackend() {
    if (this._backend) return this._backend;
    if (!this._backendPromise) {
      this._backendPromise = (async () => {
        if (this.useWorker) {
          try {
            const backend = new WorkerBackend({ workerUrl: this.workerUrl, wasmUrl: this.wasmUrl });
            await backend.start();
            this.usingWorker = true;
            return backend;
          } catch (err) {
            console.warn('[llamafile] worker unavailable, falling back to the main thread:', err.message);
          }
        }
        const backend = new MainThreadBackend({ wasmUrl: this.wasmUrl });
        await backend.start();
        this.usingWorker = false;
        return backend;
      })().then(
        (backend) => {
          this._backend = backend;
          return backend;
        },
        (err) => {
          this._backendPromise = null;
          throw err;
        }
      );
    }
    return this._backendPromise;
  }

  /** Instantiate the wasm module without loading a model (useful for probing). */
  async init() {
    const backend = await this._ensureBackend();
    return backend.send({ type: 'init' });
  }

  /**
   * @param source URL string, File/Blob, or ArrayBuffer/TypedArray
   * @param {object} [opts] { nCtx, nThreads, name, onProgress({stage,loaded,total}) }
   */
  async load(source, opts = {}) {
    const backend = await this._ensureBackend();
    const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : () => {};
    const options = {
      nCtx: opts.nCtx || 2048,
      nThreads: opts.nThreads || 1,
      name: opts.name,
    };

    const { payload, transfer } = await this._prepareSource(source, opts, onProgress);
    const result = await backend.send({ type: 'load', source: payload, options }, { onProgress }, transfer);
    this.loaded = true;
    this.modelInfo = result && result.info ? result.info : null;
    return result;
  }

  async _prepareSource(source, opts, onProgress) {
    if (typeof source === 'string') {
      // Stream the download on this thread so onProgress is byte-accurate.
      const buffer = await downloadWithProgress(source, onProgress);
      return {
        payload: { kind: 'buffer', buffer, name: opts.name || fileNameFromUrl(source) },
        transfer: [buffer],
      };
    }
    if (typeof File !== 'undefined' && source instanceof File) {
      return { payload: { kind: 'file', file: source, name: opts.name || source.name }, transfer: [] };
    }
    if (typeof Blob !== 'undefined' && source instanceof Blob) {
      return { payload: { kind: 'file', file: source, name: opts.name || 'model.gguf' }, transfer: [] };
    }
    if (source instanceof ArrayBuffer) {
      return { payload: { kind: 'buffer', buffer: source, name: opts.name || 'model.gguf' }, transfer: [source] };
    }
    if (ArrayBuffer.isView(source)) {
      const buffer = source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
      return { payload: { kind: 'buffer', buffer, name: opts.name || 'model.gguf' }, transfer: [buffer] };
    }
    throw new LlamafileError('load() expects a URL string, a File/Blob, or an ArrayBuffer');
  }

  async info() {
    const backend = await this._ensureBackend();
    this.modelInfo = await backend.send({ type: 'info' });
    return this.modelInfo;
  }

  async formatChat(messages, addAssistant = true) {
    const backend = await this._ensureBackend();
    const { prompt } = await backend.send({ type: 'format-chat', messages, addAssistant });
    return prompt;
  }

  /**
   * Stream a chat completion.
   * @param messages [{role, content}, ...]
   * @param {object} [opts] { nPredict, temp, topP, seed, prompt }
   * @returns {AsyncGenerator<string>} yields detokenised pieces
   */
  chat(messages, opts = {}) {
    const self = this;
    const queue = [];
    let finished = false;
    let failure = null;
    let notify = null;
    const wake = () => {
      const fn = notify;
      notify = null;
      if (fn) fn();
    };

    const started = this._ensureBackend().then((backend) => {
      self.generating = true;
      return backend.send(
        {
          type: 'chat-start',
          options: {
            messages,
            prompt: opts.prompt,
            nPredict: opts.nPredict,
            temp: opts.temp,
            topP: opts.topP,
            seed: opts.seed,
          },
        },
        {
          onPiece: (text) => {
            queue.push(text);
            wake();
          },
        }
      );
    });

    const settled = started.then(
      (result) => {
        self.generating = false;
        self.lastGeneration = result;
        finished = true;
        wake();
      },
      (err) => {
        self.generating = false;
        failure = err instanceof Error ? err : new LlamafileError(String(err));
        finished = true;
        wake();
      }
    );

    const iterator = (async function* () {
      try {
        for (;;) {
          if (queue.length) {
            yield queue.shift();
            continue;
          }
          if (finished) break;
          await new Promise((resolve) => {
            notify = resolve;
          });
        }
        if (failure) throw failure;
      } finally {
        if (!finished) {
          // Consumer broke out of the loop early: stop the generation.
          await self.stop().catch(() => {});
          await settled.catch(() => {});
        }
      }
    })();
    iterator.result = settled.then(() => self.lastGeneration);
    return iterator;
  }

  /** Cancel an in-flight generation. Safe to call when nothing is running. */
  async stop() {
    if (!this._backend) return { cancelled: false };
    return this._backend.send({ type: 'chat-cancel' });
  }

  async perf() {
    const backend = await this._ensureBackend();
    return backend.send({ type: 'perf' });
  }

  async unload() {
    if (!this._backend) return;
    try {
      await this._backend.send({ type: 'unload' });
    } finally {
      this.loaded = false;
      this.modelInfo = null;
    }
  }

  /** Tear down the worker entirely. */
  dispose() {
    if (this._backend) this._backend.dispose();
    this._backend = null;
    this._backendPromise = null;
    this.loaded = false;
  }
}

/* -------------------------------------------------------------- utilities */

/** Fetch `url` into an ArrayBuffer, reporting real byte progress as it streams. */
export async function downloadWithProgress(url, onProgress = () => {}, signal) {
  let res;
  try {
    res = await fetch(url, { signal, cache: 'no-store' });
  } catch (err) {
    throw new LlamafileError(`Could not fetch ${url}: ${err.message}`);
  }
  if (!res.ok) {
    throw new LlamafileError(`Could not fetch ${url}: HTTP ${res.status} ${res.statusText}`);
  }
  const total = Number(res.headers.get('content-length')) || 0;
  onProgress({ stage: 'download', loaded: 0, total });

  if (!res.body || typeof res.body.getReader !== 'function') {
    const buffer = await res.arrayBuffer();
    onProgress({ stage: 'download', loaded: buffer.byteLength, total: buffer.byteLength });
    return buffer;
  }

  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress({ stage: 'download', loaded, total });
  }
  if (!loaded) throw new LlamafileError(`Downloaded 0 bytes from ${url}`);

  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  chunks.length = 0;
  onProgress({ stage: 'download', loaded, total: loaded });
  return out.buffer;
}

export function fileNameFromUrl(url) {
  try {
    const parsed = new URL(url, location.href);
    const name = parsed.pathname.split('/').filter(Boolean).pop();
    return name || 'model.gguf';
  } catch (_) {
    return 'model.gguf';
  }
}

export default Llamafile;
