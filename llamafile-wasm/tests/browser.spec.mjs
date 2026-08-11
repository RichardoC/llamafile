#!/usr/bin/env node
// Browser end-to-end test for llamafile-wasm's demo page.
//
// Plain `playwright` (no @playwright/test) so it runs with bare node:
//
//   NODE_PATH=/home/user/pwtest/node_modules node llamafile-wasm/tests/browser.spec.mjs
//   node llamafile-wasm/tests/browser.spec.mjs --browser=firefox
//
// It starts serve.py on a free port (with --models pointed at a directory of
// local .gguf files so the test never touches the network), opens web/index.html,
// loads a model, sends a prompt and asserts that streamed tokens show up in the
// transcript. Browser console errors and uncaught page errors fail the run.
//
// Until `build.sh` has produced dist/llamafile.mjs the test runs in *degraded
// mode*: it asserts the page comes up, reports a clear "not built yet" error and
// stays responsive, and skips the inference assertions.
//
// Environment:
//   LF_BROWSERS=chromium,firefox   which browsers to run (also --browser=)
//   LF_MODEL=tiny|big|<filename>   model served from LF_MODELS_DIR (default: tiny)
//   LF_MODELS_DIR=/path            directory exposed at /models/
//   LF_CHROMIUM=/path/to/chrome    chromium binary (this box needs an explicit one)
//   LF_SERVE_ROOT=/path            directory to serve instead of llamafile-wasm/
//   LF_VERBOSE=1                   echo browser console output
//   LF_HEADED=1                    run with a visible browser
//   LF_KEEP_OPEN=1                 pause before closing the browser

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

// `playwright` may live outside this repo (there is no package.json here), and
// ESM ignores NODE_PATH — so resolve it by hand as well.
const playwright = await (async () => {
  const candidates = ['playwright'];
  if (process.env.LF_PLAYWRIGHT) candidates.unshift(process.env.LF_PLAYWRIGHT);
  for (const dir of (process.env.NODE_PATH || '').split(path.delimiter).filter(Boolean)) {
    candidates.push(pathToFileURL(path.join(dir, 'playwright', 'index.js')).href);
  }
  const problems = [];
  for (const specifier of candidates) {
    try {
      const ns = await import(specifier);
      // playwright is CJS: named exports are sometimes only on `default`.
      return ns && ns.chromium ? ns : ns.default || ns;
    } catch (err) {
      problems.push(`  ${specifier}: ${err.message.split('\n')[0]}`);
    }
  }
  console.error(
    'Could not load the `playwright` package. Run this spec from a directory that has it\n' +
      'installed, or point NODE_PATH / LF_PLAYWRIGHT at it, e.g.\n' +
      '  NODE_PATH=/home/user/pwtest/node_modules node llamafile-wasm/tests/browser.spec.mjs\n' +
      `tried:\n${problems.join('\n')}`
  );
  process.exit(1);
})();

const HERE = path.dirname(fileURLToPath(import.meta.url));
// The directory serve.py exposes. Override with LF_SERVE_ROOT to test a staging
// copy (e.g. a tree without dist/, to exercise the "not built yet" path).
const ROOT = process.env.LF_SERVE_ROOT ? path.resolve(process.env.LF_SERVE_ROOT) : path.resolve(HERE, '..');
const DIST_MODULE = path.join(ROOT, 'dist', 'llamafile.mjs');

const argv = process.argv.slice(2);
const argOf = (name) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : null;
};

const MODELS_DIR =
  process.env.LF_MODELS_DIR ||
  argOf('models') ||
  '/tmp/claude-0/-home-user-llamafile/a16930dc-9430-57cc-916b-41e5b51d79fb/scratchpad';

const MODEL_ALIASES = { tiny: 'stories260K.gguf', big: 'smollm2-135m-q8.gguf' };
const MODEL_ARG = process.env.LF_MODEL || argOf('model') || 'tiny';
const MODEL_FILE = MODEL_ALIASES[MODEL_ARG] || MODEL_ARG;

const BROWSERS = (process.env.LF_BROWSERS || argOf('browser') || 'chromium,firefox')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

const CHROMIUM_PATH = process.env.LF_CHROMIUM || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const HEADLESS = !process.env.LF_HEADED;
const DEGRADED = !fs.existsSync(DIST_MODULE);

const LOAD_TIMEOUT = Number(process.env.LF_LOAD_TIMEOUT || (MODEL_FILE.includes('smollm') ? 300000 : 90000));
const GEN_TIMEOUT = Number(process.env.LF_GEN_TIMEOUT || 180000);
const PROMPT = process.env.LF_PROMPT || 'Write one short sentence about a llama.';

/* --------------------------------------------------------------- utilities */

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const DIM = '\x1b[2m';
const YELLOW = '\x1b[33m';
const OFF = '\x1b[0m';

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

class Runner {
  constructor(label) {
    this.label = label;
    this.passed = [];
    this.failed = [];
    this.skipped = [];
  }

  async step(name, fn) {
    if (this.failed.length) {
      this.skipped.push(name);
      console.log(`  ${DIM}- ${name} (skipped after failure)${OFF}`);
      return;
    }
    try {
      await fn();
      this.passed.push(name);
      console.log(`  ${GREEN}✓${OFF} ${name}`);
    } catch (err) {
      this.failed.push({ name, err });
      console.log(`  ${RED}✗ ${name}${OFF}\n      ${String(err && err.message).split('\n').join('\n      ')}`);
    }
  }

  skip(name, why) {
    this.skipped.push(name);
    console.log(`  ${YELLOW}~${OFF} ${name} ${DIM}(${why})${OFF}`);
  }
}

/* ------------------------------------------------------------------ server */

function startServer() {
  const args = [path.resolve(HERE, '..', 'serve.py'), '--port', '0', '--dir', ROOT];
  if (fs.existsSync(MODELS_DIR)) args.push('--models', MODELS_DIR);
  const proc = spawn('python3', args, { stdio: ['ignore', 'pipe', 'pipe'] });
  const stderr = [];
  proc.stderr.on('data', (d) => {
    stderr.push(d.toString());
    if (stderr.length > 200) stderr.shift();
  });

  return new Promise((resolve, reject) => {
    let out = '';
    const onData = (chunk) => {
      out += chunk.toString();
      const match = out.match(/http:\/\/[^\s]+/);
      if (match) {
        proc.stdout.off('data', onData);
        clearTimeout(timer);
        resolve({ baseUrl: match[0], proc, stderr });
      }
    };
    const timer = setTimeout(() => {
      proc.kill('SIGKILL');
      reject(new Error(`serve.py did not report a URL within 10s\n${stderr.join('')}`));
    }, 10000);
    proc.stdout.on('data', onData);
    proc.on('error', reject);
    proc.on('exit', (code) => {
      if (!out.match(/http:\/\//)) {
        clearTimeout(timer);
        reject(new Error(`serve.py exited with code ${code}\n${stderr.join('')}`));
      }
    });
  });
}

/* ---------------------------------------------------------------- browsers */

async function launch(name) {
  const type = playwright[name];
  if (!type) throw new Error(`Unknown browser "${name}" (expected chromium, firefox or webkit)`);
  const opts = { headless: HEADLESS };
  if (name === 'chromium' && fs.existsSync(CHROMIUM_PATH)) opts.executablePath = CHROMIUM_PATH;
  return type.launch(opts);
}

// Errors we tolerate *only* while dist/llamafile.mjs is missing: the page
// deliberately probes for the module and reports its absence.
const DEGRADED_PATTERNS = [
  /llamafile\.mjs/i,
  /not built yet/i,
  /404/,
  /failed to load resource/i,
  /NS_ERROR_/i,
];

function isTolerated(text) {
  return DEGRADED && DEGRADED_PATTERNS.some((re) => re.test(text));
}

/* ------------------------------------------------------------------- suite */

async function runBrowser(name, baseUrl) {
  const runner = new Runner(name);
  console.log(`\n${name} ${DIM}${baseUrl}${OFF}`);

  const browser = await launch(name);
  const context = await browser.newContext();
  const page = await context.newPage();

  const consoleErrors = [];
  const pageErrors = [];
  page.on('console', (msg) => {
    if (msg.type() !== 'error' && msg.type() !== 'warning') return;
    const text = `[${msg.type()}] ${msg.text()}`;
    if (msg.type() === 'error') consoleErrors.push(text);
    if (process.env.LF_VERBOSE) console.log(`      ${DIM}console${OFF} ${text}`);
  });
  page.on('pageerror', (err) => pageErrors.push(String(err && err.stack ? err.stack : err)));
  page.on('requestfailed', (req) => {
    if (process.env.LF_VERBOSE) console.log(`      ${DIM}requestfailed${OFF} ${req.url()}`);
  });

  const state = () => page.evaluate(() => window.__llamafileState);
  const statusText = () => page.$eval('#status', (el) => el.textContent.trim());
  const errorText = () => page.$eval('#error', (el) => (el.hidden ? '' : el.textContent.trim()));
  const waitForState = (want, timeout) =>
    page.waitForFunction(
      (wanted) => (Array.isArray(wanted) ? wanted : [wanted]).includes(window.__llamafileState),
      want,
      { timeout }
    );

  try {
    await runner.step('page loads', async () => {
      const res = await page.goto(baseUrl, { waitUntil: 'load', timeout: 30000 });
      assert(res, 'no response from the dev server');
      assert(res.status() < 400, `server returned HTTP ${res.status()}`);
      assert(/\/web\/?$/.test(page.url()) || page.url().endsWith('index.html'), `unexpected URL ${page.url()}`);
    });

    await runner.step('window.llamafileReady resolves', async () => {
      await page.waitForFunction(() => typeof window.llamafileReady !== 'undefined', null, { timeout: 10000 });
      await page.evaluate(
        () =>
          Promise.race([
            window.llamafileReady,
            new Promise((_, rej) => setTimeout(() => rej(new Error('llamafileReady never resolved')), 15000)),
          ])
      );
    });

    await runner.step('ES modules parsed without errors', async () => {
      const fatal = pageErrors.filter((e) => !isTolerated(e));
      assert(fatal.length === 0, `uncaught page errors:\n${fatal.join('\n')}`);
      const syntax = consoleErrors.filter((e) => /SyntaxError|Unexpected token|is not defined|import/i.test(e));
      assert(syntax.length === 0, `module errors in console:\n${syntax.join('\n')}`);
    });

    await runner.step('DOM contract from API.md is present', async () => {
      const missing = await page.evaluate(() => {
        const ids = [
          'model-url', 'model-file', 'load-model', 'status',
          'model-info', 'prompt', 'send', 'stop', 'messages', 'perf',
        ];
        return ids.filter((id) => !document.getElementById(id));
      });
      assert(missing.length === 0, `missing elements: ${missing.join(', ')}`);
      assert(typeof (await state()) === 'string', 'window.__llamafileState is not a string');
      const url = await page.$eval('#model-url', (el) => el.value);
      assert(/\.gguf(\?|$)/.test(url), `#model-url is not pre-filled with a GGUF URL (got "${url}")`);
    });

    if (DEGRADED) {
      await runner.step('reports a clear error when dist/llamafile.mjs is missing', async () => {
        await waitForState('error', 10000);
        const text = await errorText();
        assert(text, '#error is empty — the missing module was not surfaced on the page');
        assert(
          /not built yet/i.test(text) && /llamafile\.mjs/i.test(text),
          `#error does not explain the missing build:\n${text}`
        );
        const status = await statusText();
        assert(/not built/i.test(status), `#status does not mention the missing build (got "${status}")`);
      });

      await runner.step('clicking Load fails fast instead of hanging', async () => {
        await page.fill('#model-url', new URL(`models/${MODEL_FILE}`, baseUrl).href);
        await page.click('#load-model');
        await waitForState('error', 30000);
        assert(await errorText(), 'no error shown after a failed load');
        assert(await page.evaluate(() => 1 + 1) === 2, 'the page stopped responding');
        assert(await page.isEnabled('#load-model'), 'the Load button was left disabled');
      });

      runner.skip('loads a model and streams tokens', 'dist/llamafile.mjs not built yet');
      runner.skip('Stop cancels generation', 'dist/llamafile.mjs not built yet');
    } else {
      const modelUrl = new URL(`models/${MODEL_FILE}`, baseUrl).href;

      await runner.step(`loads ${MODEL_FILE}`, async () => {
        const head = await page.evaluate((u) => fetch(u, { method: 'HEAD' }).then((r) => r.status), modelUrl);
        assert(head === 200, `the dev server does not serve ${modelUrl} (HTTP ${head}) — check --models`);
        await page.fill('#model-url', modelUrl);
        await page.click('#load-model');
        await waitForState(['ready', 'error'], LOAD_TIMEOUT);
        assert((await state()) === 'ready', `load failed: ${await errorText()}`);
        const info = await page.$eval('#model-info', (el) => el.textContent.trim());
        assert(info.length > 0, '#model-info was not populated');
        assert(await page.isEnabled('#send'), 'Send stayed disabled after the model became ready');
      });

      await runner.step('streams tokens into the transcript', async () => {
        await page.fill('#prompt', PROMPT);
        await page.click('#send');
        await page.waitForSelector('.message[data-role=user] .message-content', { timeout: 10000 });
        const user = await page.$eval('.message[data-role=user] .message-content', (el) => el.textContent);
        assert(user.includes(PROMPT.slice(0, 20)), 'the user message was not echoed into #messages');

        // First piece must appear well before the whole generation finishes.
        await page.waitForFunction(
          () => {
            const nodes = document.querySelectorAll('.message[data-role=assistant] .message-content');
            const last = nodes[nodes.length - 1];
            return !!last && last.textContent.trim().length > 0;
          },
          null,
          { timeout: GEN_TIMEOUT }
        );
        const partial = await page.$$eval('.message[data-role=assistant] .message-content', (n) =>
          n[n.length - 1].textContent
        );

        await waitForState(['ready', 'error'], GEN_TIMEOUT);
        assert((await state()) === 'ready', `generation failed: ${await errorText()}`);

        const final = await page.$$eval('.message[data-role=assistant] .message-content', (n) =>
          n[n.length - 1].textContent
        );
        assert(final.trim().length > 0, 'the assistant message is empty');
        assert(final.startsWith(partial), 'the streamed text was rewritten rather than appended');
        const perf = await page.$eval('#perf', (el) => el.textContent);
        assert(/tok/.test(perf), `#perf does not show a token/timing summary (got "${perf}")`);
      });

      await runner.step('Stop cancels generation', async () => {
        await page.fill('#prompt', 'Count slowly from one to two hundred, one number per line.');
        await page.click('#send');
        await waitForState('generating', 20000);
        await page.waitForFunction(
          () => {
            const nodes = document.querySelectorAll('.message[data-role=assistant] .message-content');
            const last = nodes[nodes.length - 1];
            return !!last && last.textContent.length > 0;
          },
          null,
          { timeout: GEN_TIMEOUT }
        );
        await page.click('#stop');
        await waitForState(['ready', 'error'], 30000);
        assert((await state()) === 'ready', `stop left the page in an error state: ${await errorText()}`);
      });

      await runner.step('loads a model from the local file picker', async () => {
        const local = path.join(MODELS_DIR, MODEL_FILE);
        assert(fs.existsSync(local), `${local} is missing`);
        await page.setInputFiles('#model-file', local);
        await page.click('#load-model');
        await waitForState(['ready', 'error'], LOAD_TIMEOUT);
        assert((await state()) === 'ready', `file-picker load failed: ${await errorText()}`);
        const info = await page.$eval('#model-info', (el) => el.textContent);
        assert(info.includes(MODEL_FILE), `#model-info does not name the picked file (got "${info}")`);
      });
    }

    await runner.step('no unexpected console errors', async () => {
      const fatal = consoleErrors.filter((e) => !isTolerated(e));
      assert(fatal.length === 0, `console errors:\n${fatal.join('\n')}`);
      const crashes = pageErrors.filter((e) => !isTolerated(e));
      assert(crashes.length === 0, `uncaught page errors:\n${crashes.join('\n')}`);
    });
  } finally {
    if (process.env.LF_KEEP_OPEN) await page.waitForTimeout(60000);
    await context.close().catch(() => {});
    await browser.close().catch(() => {});
  }

  return runner;
}

/* -------------------------------------------------------------------- main */

async function main() {
  console.log(`llamafile-wasm browser tests`);
  console.log(`  root      ${ROOT}`);
  console.log(`  browsers  ${BROWSERS.join(', ')}`);
  console.log(`  models    ${MODELS_DIR} ${fs.existsSync(MODELS_DIR) ? '' : '(missing!)'}`);
  console.log(`  model     ${MODEL_FILE}`);
  if (DEGRADED) {
    console.log(
      `  ${YELLOW}dist/llamafile.mjs is missing — running in degraded mode ` +
        `(page + error-state assertions only)${OFF}`
    );
  }

  const { baseUrl, proc, stderr } = await startServer();
  const runners = [];
  try {
    for (const name of BROWSERS) {
      try {
        runners.push(await runBrowser(name, baseUrl));
      } catch (err) {
        const runner = new Runner(name);
        runner.failed.push({ name: 'launch', err });
        console.log(`  ${RED}✗ could not run ${name}: ${err.message}${OFF}`);
        runners.push(runner);
      }
    }
  } finally {
    proc.kill('SIGTERM');
  }

  let failures = 0;
  console.log('\nsummary');
  for (const r of runners) {
    failures += r.failed.length;
    const bits = [`${r.passed.length} passed`];
    if (r.failed.length) bits.push(`${RED}${r.failed.length} failed${OFF}`);
    if (r.skipped.length) bits.push(`${r.skipped.length} skipped`);
    console.log(`  ${r.label.padEnd(10)} ${bits.join(', ')}`);
  }
  if (failures) {
    console.log(`\n${DIM}server log tail:${OFF}\n${stderr.slice(-20).join('')}`);
  }
  process.exit(failures ? 1 : 0);
}

main().catch((err) => {
  console.error(`${RED}fatal:${OFF}`, err);
  process.exit(1);
});
