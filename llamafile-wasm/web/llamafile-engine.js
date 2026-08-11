// llamafile-engine.js — the part that actually talks to the WebAssembly module.
//
// It is deliberately transport-agnostic: `llamafile-worker.js` runs it inside a
// Web Worker and pipes `handleRequest()` events out over postMessage, while
// `llamafile.js` runs the exact same code on the main thread when Workers are
// unavailable. Only the low-level functions listed in API.md are used.

const FS_ROOT = '/models';
const FS_CHUNK = 4 * 1024 * 1024; // bytes written into the Emscripten FS at a time

/** Yield to the event loop so queued messages (e.g. chat-cancel) get a turn. */
const macrotask =
  typeof MessageChannel === 'function'
    ? () =>
        new Promise((resolve) => {
          const channel = new MessageChannel();
          channel.port1.onmessage = () => {
            channel.port1.close();
            resolve();
          };
          channel.port2.postMessage(0);
        })
    : () => new Promise((resolve) => setTimeout(resolve, 0));

const timeout = () => new Promise((resolve) => setTimeout(resolve, 0));

function now() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/**
 * Turn a failed `import('.../llamafile.mjs')` into something a human can act on.
 * The module is a build artifact, so "you have not run build.sh" is by far the
 * most likely cause and deserves to be said out loud rather than surfaced as
 * "Failed to fetch dynamically imported module".
 */
async function explainImportFailure(url, err) {
  let status = null;
  try {
    const res = await fetch(url, { method: 'GET', cache: 'no-store' });
    status = res.status;
    if (res.body && typeof res.body.cancel === 'function') res.body.cancel();
  } catch (_) {
    /* network-level failure; status stays null */
  }
  if (status === 404 || status === 403 || status === null) {
    return (
      `WebAssembly module not built yet: ${url} could not be loaded` +
      (status ? ` (HTTP ${status})` : '') +
      '. Build it first with `llamafile-wasm/build.sh`, then reload this page.'
    );
  }
  return `Failed to load the WebAssembly module from ${url}: ${err && err.message ? err.message : err}`;
}

export class LlamafileEngine {
  constructor({ wasmUrl } = {}) {
    this.wasmUrl = wasmUrl;
    this.module = null;
    this.modelPath = null;
    this.loaded = false;
    this.generating = false;
    this._cancelled = false;
    this._instantiating = null;
  }

  // ------------------------------------------------------------ module init

  async instantiate() {
    if (this.module) return this.module;
    if (!this._instantiating) {
      this._instantiating = this._instantiate().catch((err) => {
        this._instantiating = null;
        throw err;
      });
    }
    return this._instantiating;
  }

  async _instantiate() {
    if (!this.wasmUrl) throw new Error('No wasmUrl configured for the llamafile module');
    let namespace;
    try {
      namespace = await import(/* @vite-ignore */ this.wasmUrl);
    } catch (err) {
      throw new Error(await explainImportFailure(this.wasmUrl, err));
    }
    const factory = namespace.default || namespace.createLlamafileModule;
    if (typeof factory !== 'function') {
      throw new Error(
        `${this.wasmUrl} did not default-export a module factory ` +
          '(expected `async function createLlamafileModule(opts?) -> Module`).'
      );
    }
    const module = await factory({
      print: (text) => console.log('[llamafile]', text),
      printErr: (text) => console.warn('[llamafile]', text),
    });
    for (const fn of ['lf_load', 'lf_gen_begin', 'lf_gen_next', 'lf_unload']) {
      if (typeof module[fn] !== 'function') {
        throw new Error(`The wasm module is missing \`${fn}\` — is dist/ from an older build?`);
      }
    }
    this.module = module;
    return module;
  }

  _lastError(fallback) {
    try {
      const msg = this.module && this.module.lf_last_error ? this.module.lf_last_error() : '';
      return msg || fallback;
    } catch (_) {
      return fallback;
    }
  }

  // ----------------------------------------------------------------- loading

  /**
   * @param source {kind:'buffer',buffer}|{kind:'file',file}|{kind:'url',url}
   * @param opts   {nCtx, nThreads, name}
   * @param emit   (event) => void
   */
  async load(source, opts, emit) {
    const module = await this.instantiate();
    if (this.loaded) {
      try {
        module.lf_unload();
      } catch (_) {
        /* ignore */
      }
      this.loaded = false;
    }

    const name = sanitiseName(source.name || opts.name || 'model.gguf');
    const path = `${FS_ROOT}/${name}`;
    mkdirp(module.FS, FS_ROOT);
    try {
      module.FS.unlink(path);
    } catch (_) {
      /* not there yet */
    }

    const bytes = await this._writeToFs(module, path, source, emit);
    this.modelPath = path;

    emit({ type: 'progress', stage: 'model', loaded: 0, total: 0 });
    const nCtx = Math.max(0, opts.nCtx | 0) || 2048;
    const nThreads = Math.max(1, opts.nThreads | 0) || 1;
    const rc = module.lf_load(path, nCtx, nThreads);
    if (rc !== 0) {
      throw new Error(this._lastError(`lf_load() failed with code ${rc}`));
    }
    this.loaded = true;
    emit({ type: 'progress', stage: 'model', loaded: 1, total: 1 });
    return { path, bytes, name };
  }

  async _writeToFs(module, path, source, emit) {
    const FS = module.FS;
    const stream = FS.open(path, 'w');
    let written = 0;
    const report = (total) => emit({ type: 'progress', stage: 'fs', loaded: written, total });
    try {
      if (source.kind === 'buffer') {
        const view = new Uint8Array(source.buffer);
        const total = view.length;
        if (!total) throw new Error('Model data is empty (0 bytes)');
        for (let off = 0; off < total; off += FS_CHUNK) {
          const chunk = view.subarray(off, Math.min(off + FS_CHUNK, total));
          FS.write(stream, chunk, 0, chunk.length, written);
          written += chunk.length;
          report(total);
          await macrotask();
        }
      } else if (source.kind === 'file') {
        const file = source.file;
        const total = file.size;
        if (!total) throw new Error('Model file is empty (0 bytes)');
        if (typeof file.stream === 'function') {
          const reader = file.stream().getReader();
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = value instanceof Uint8Array ? value : new Uint8Array(value);
            FS.write(stream, chunk, 0, chunk.length, written);
            written += chunk.length;
            report(total);
          }
        } else {
          for (let off = 0; off < total; off += FS_CHUNK) {
            const slice = file.slice(off, Math.min(off + FS_CHUNK, total));
            const chunk = new Uint8Array(await slice.arrayBuffer());
            FS.write(stream, chunk, 0, chunk.length, written);
            written += chunk.length;
            report(total);
          }
        }
      } else if (source.kind === 'url') {
        written = await this._fetchToFs(FS, stream, source.url, emit);
      } else {
        throw new Error(`Unsupported model source: ${source.kind}`);
      }
    } finally {
      FS.close(stream);
    }
    if (!written) throw new Error('Nothing was written to the virtual filesystem');
    return written;
  }

  async _fetchToFs(FS, stream, url, emit) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`Download failed: HTTP ${res.status} ${res.statusText} for ${url}`);
    const total = Number(res.headers.get('content-length')) || 0;
    let written = 0;
    if (!res.body || typeof res.body.getReader !== 'function') {
      const chunk = new Uint8Array(await res.arrayBuffer());
      FS.write(stream, chunk, 0, chunk.length, 0);
      emit({ type: 'progress', stage: 'download', loaded: chunk.length, total: chunk.length });
      return chunk.length;
    }
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      FS.write(stream, value, 0, value.length, written);
      written += value.length;
      emit({ type: 'progress', stage: 'download', loaded: written, total });
    }
    return written;
  }

  // -------------------------------------------------------------- inference

  info() {
    this._requireModel();
    return JSON.parse(this.module.lf_model_info());
  }

  perf() {
    this._requireModel();
    return JSON.parse(this.module.lf_perf());
  }

  formatChat(messages, addAssistant = true) {
    this._requireModel();
    return this.module.lf_format_chat(JSON.stringify(messages), !!addAssistant);
  }

  cancel() {
    this._cancelled = true;
    return this.generating;
  }

  /**
   * Drive lf_gen_next() to completion, handing each piece to `onPiece` and
   * yielding to the event loop between tokens so a cancel message can land.
   */
  async generate(opts, onPiece) {
    this._requireModel();
    if (this.generating) throw new Error('A generation is already in flight');
    const module = this.module;

    const prompt =
      typeof opts.prompt === 'string'
        ? opts.prompt
        : this.formatChat(opts.messages || [], opts.addAssistant !== false);
    if (!prompt) throw new Error('Refusing to generate from an empty prompt');

    const nPredict = Number.isFinite(opts.nPredict) ? opts.nPredict | 0 : 256;
    const temp = Number.isFinite(opts.temp) ? +opts.temp : 0.7;
    const topP = Number.isFinite(opts.topP) ? +opts.topP : 0.9;
    const seed = Number.isFinite(opts.seed) ? opts.seed | 0 : -1;

    this.generating = true;
    this._cancelled = false;
    const started = now();
    let tokens = 0;
    let text = '';
    let stopReason = 'eog';

    try {
      const rc = module.lf_gen_begin(prompt, nPredict, temp, topP, seed);
      if (rc !== 0) throw new Error(this._lastError(`lf_gen_begin() failed with code ${rc}`));

      let emptyStreak = 0;
      for (;;) {
        if (this._cancelled) {
          stopReason = 'cancelled';
          break;
        }
        const piece = module.lf_gen_next();
        if (piece) {
          tokens += 1;
          text += piece;
          emptyStreak = 0;
          onPiece(piece);
        } else {
          emptyStreak += 1;
        }
        if (module.lf_gen_done()) break;
        if (!piece && emptyStreak > 32) {
          // Defensive: never spin forever if the module stops making progress.
          stopReason = 'stalled';
          break;
        }
        // Every token gets a macrotask; every 16th also gets a timer turn, so a
        // postMessage can never be starved by an over-eager microtask queue.
        await (tokens % 16 === 0 ? timeout() : macrotask());
      }
      if (this._cancelled) stopReason = 'cancelled';
    } finally {
      this.generating = false;
      this._cancelled = false;
      try {
        module.lf_gen_end();
      } catch (_) {
        /* ignore */
      }
    }

    const elapsedMs = now() - started;
    let perf = null;
    try {
      perf = JSON.parse(module.lf_perf());
    } catch (_) {
      /* perf is best-effort */
    }
    return {
      tokens,
      text,
      stopReason,
      elapsedMs,
      tokensPerSecond: elapsedMs > 0 ? (tokens * 1000) / elapsedMs : 0,
      perf,
    };
  }

  unload() {
    if (this.module && this.loaded) {
      this.module.lf_unload();
      this.loaded = false;
    }
    if (this.module && this.modelPath) {
      try {
        this.module.FS.unlink(this.modelPath);
      } catch (_) {
        /* ignore */
      }
      this.modelPath = null;
    }
  }

  _requireModel() {
    if (!this.module) throw new Error('The WebAssembly module has not been instantiated yet');
    if (!this.loaded) throw new Error('No model is loaded — call load() first');
  }
}

function sanitiseName(name) {
  const base = String(name).split(/[\\/]/).pop() || 'model.gguf';
  const clean = base.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 96);
  return clean || 'model.gguf';
}

function mkdirp(FS, dir) {
  if (typeof FS.mkdirTree === 'function') {
    try {
      FS.mkdirTree(dir);
      return;
    } catch (_) {
      /* fall through */
    }
  }
  const parts = dir.split('/').filter(Boolean);
  let path = '';
  for (const part of parts) {
    path += `/${part}`;
    try {
      FS.mkdir(path);
    } catch (_) {
      /* already exists */
    }
  }
}

/**
 * The request/response protocol shared by the worker and the main-thread
 * fallback. `emit` receives streaming events ({type:'progress'|'piece'}); the
 * returned value is the request's result.
 */
export async function handleRequest(engine, msg, emit) {
  switch (msg.type) {
    case 'ping':
      return { pong: true };

    case 'init':
      await engine.instantiate();
      return { ok: true };

    case 'load': {
      const result = await engine.load(msg.source, msg.options || {}, emit);
      let info = null;
      try {
        info = engine.info();
      } catch (_) {
        /* info is a bonus here; the caller can ask again */
      }
      return { ...result, info };
    }

    case 'info':
      return engine.info();

    case 'format-chat':
      return { prompt: engine.formatChat(msg.messages || [], msg.addAssistant !== false) };

    case 'chat-start':
      return engine.generate(msg.options || {}, (piece) => emit({ type: 'piece', text: piece }));

    case 'chat-cancel':
      return { cancelled: engine.cancel() };

    case 'perf':
      return engine.perf();

    case 'unload':
      engine.unload();
      return { ok: true };

    default:
      throw new Error(`Unknown request: ${msg && msg.type}`);
  }
}
