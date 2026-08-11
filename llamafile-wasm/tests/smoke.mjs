#!/usr/bin/env node
//
// Node smoke test for the llamafile-wasm core.
//
//   node llamafile-wasm/tests/smoke.mjs /path/to/model.gguf [--n-predict 64]
//
// Exercises the whole lf_* contract from API.md: FS write -> lf_load ->
// lf_model_info -> lf_format_chat -> lf_gen_begin/lf_gen_next -> lf_perf ->
// lf_unload. Exits non-zero on any failure.

import { readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve, basename } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// --------------------------------------------------------------------------
// args
// --------------------------------------------------------------------------
const argv = process.argv.slice(2);
const positional = [];
const opts = { nPredict: 64, temp: 0.7, topP: 0.9, seed: 42 };

for (let i = 0; i < argv.length; i++) {
  switch (argv[i]) {
    case '--n-predict': opts.nPredict = Number(argv[++i]); break;
    case '--temp':      opts.temp     = Number(argv[++i]); break;
    case '--top-p':     opts.topP     = Number(argv[++i]); break;
    case '--seed':      opts.seed     = Number(argv[++i]); break;
    default:            positional.push(argv[i]);
  }
}

const modelPath = positional[0];
if (!modelPath) {
  console.error('usage: node smoke.mjs <model.gguf> [--n-predict N] [--temp T] [--top-p P] [--seed S]');
  process.exit(2);
}

function fail(msg, Module) {
  console.error(`\nFAIL: ${msg}`);
  if (Module) {
    try {
      const err = Module.lf_last_error();
      if (err) console.error(`lf_last_error(): ${err}`);
    } catch { /* ignore */ }
  }
  process.exit(1);
}

// --------------------------------------------------------------------------
// run
// --------------------------------------------------------------------------
const t0 = Date.now();

let createLlamafileModule;
try {
  ({ default: createLlamafileModule } = await import(resolve(__dirname, '../dist/llamafile.mjs')));
} catch (e) {
  console.error(e);
  fail('could not import ../dist/llamafile.mjs — run llamafile-wasm/build.sh first');
}

console.log('loading wasm module ...');
const Module = await createLlamafileModule();

for (const fn of ['lf_load', 'lf_model_info', 'lf_format_chat', 'lf_gen_begin',
                  'lf_gen_next', 'lf_gen_done', 'lf_gen_end', 'lf_perf',
                  'lf_last_error', 'lf_unload']) {
  if (typeof Module[fn] !== 'function') fail(`Module.${fn} is not exported`);
}
if (!Module.FS) fail('Module.FS is not exported');

// ---- copy the gguf into the Emscripten FS ---------------------------------
const bytes = readFileSync(modelPath);
const sizeMB = (statSync(modelPath).size / 1048576).toFixed(1);
const fsPath = `/models/${basename(modelPath)}`;

console.log(`writing ${sizeMB} MB into the Emscripten FS at ${fsPath} ...`);
try {
  Module.FS.mkdir('/models');
} catch { /* already exists */ }
Module.FS.writeFile(fsPath, bytes);

// ---- lf_load --------------------------------------------------------------
const tLoad = Date.now();
const rc = Module.lf_load(fsPath, 2048, 1);
if (rc !== 0) fail(`lf_load returned ${rc}`, Module);
console.log(`lf_load ok in ${((Date.now() - tLoad) / 1000).toFixed(2)}s`);

// ---- lf_model_info --------------------------------------------------------
let info;
try {
  info = JSON.parse(Module.lf_model_info());
} catch (e) {
  fail(`lf_model_info did not return valid JSON: ${e.message}`, Module);
}
for (const key of ['n_params', 'n_ctx_train', 'n_ctx', 'n_vocab', 'desc',
                   'has_chat_template', 'size_bytes']) {
  if (!(key in info)) fail(`lf_model_info missing key "${key}"`, Module);
}
console.log('lf_model_info:', JSON.stringify(info));

// ---- lf_format_chat -------------------------------------------------------
const messages = [
  { role: 'system',  content: 'You are a helpful assistant. Answer briefly.' },
  { role: 'user',    content: 'What is the capital of France?' },
];

const formatted = Module.lf_format_chat(JSON.stringify(messages), true);
if (typeof formatted !== 'string' || formatted.length === 0) {
  fail('lf_format_chat returned an empty string', Module);
}
console.log('\n--- lf_format_chat output ---');
console.log(JSON.stringify(formatted));
console.log('-----------------------------\n');

if (info.has_chat_template && !formatted.includes('capital of France')) {
  fail('lf_format_chat output does not contain the user message', Module);
}

// ---- generation -----------------------------------------------------------
// Models with a real chat template get the templated prompt; a bare
// completion model (e.g. stories260K) gets a plain story prefix.
const prompt = info.has_chat_template ? formatted : 'Once upon a time';
if (!info.has_chat_template) {
  console.log(`model has no chat template; generating a completion from ${JSON.stringify(prompt)}`);
}

const rcGen = Module.lf_gen_begin(prompt, opts.nPredict, opts.temp, opts.topP, opts.seed);
if (rcGen !== 0) fail(`lf_gen_begin returned ${rcGen}`, Module);

console.log('--- generated ---');
if (!info.has_chat_template) process.stdout.write(prompt);

let out = '';
let steps = 0;
const tGen = Date.now();
while (!Module.lf_gen_done()) {
  const piece = Module.lf_gen_next();
  if (piece === '') break;
  out += piece;
  process.stdout.write(piece);
  if (++steps > opts.nPredict + 8) fail('lf_gen_next never terminated', Module);
}
const genSec = (Date.now() - tGen) / 1000;
console.log('\n-----------------\n');

if (!Module.lf_gen_done()) fail('lf_gen_done() is still false after the loop', Module);
if (Module.lf_gen_next() !== '') fail('lf_gen_next() returned a piece after completion', Module);
if (out.trim().length === 0) fail('generation produced no text', Module);

Module.lf_gen_end();

// ---- lf_perf --------------------------------------------------------------
let perf;
try {
  perf = JSON.parse(Module.lf_perf());
} catch (e) {
  fail(`lf_perf did not return valid JSON: ${e.message}`, Module);
}
for (const key of ['prompt_tokens', 'predicted_tokens', 'prompt_ms', 'predict_ms']) {
  if (!(key in perf)) fail(`lf_perf missing key "${key}"`, Module);
}
if (perf.predicted_tokens < 1) fail('lf_perf reports zero predicted tokens', Module);

const ppTps = perf.prompt_ms  > 0 ? perf.prompt_tokens    / (perf.prompt_ms  / 1000) : 0;
const tgTps = perf.predict_ms > 0 ? perf.predicted_tokens / (perf.predict_ms / 1000) : 0;

console.log('lf_perf:', JSON.stringify(perf));
console.log(`prompt : ${perf.prompt_tokens} tok in ${perf.prompt_ms.toFixed(1)} ms  (${ppTps.toFixed(2)} tok/s)`);
console.log(`predict: ${perf.predicted_tokens} tok in ${perf.predict_ms.toFixed(1)} ms  (${tgTps.toFixed(2)} tok/s)`);
console.log(`wall   : ${genSec.toFixed(2)}s for the generation loop`);

// ---- teardown -------------------------------------------------------------
Module.lf_unload();
if (Module.lf_model_info() !== '{}') fail('lf_model_info should be empty after lf_unload', Module);

console.log(`\nOK — smoke test passed in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
process.exit(0);
