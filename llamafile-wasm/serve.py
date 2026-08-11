#!/usr/bin/env python3
"""Dependency-free static dev server for llamafile-wasm.

Serves the llamafile-wasm/ directory (so the page at web/index.html can reach
../dist/llamafile.mjs) with the MIME types and headers a WebAssembly build
needs:

  * application/wasm for .wasm, text/javascript for .js/.mjs
  * Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp
      -> makes the page cross-origin isolated, which the future pthread build
         needs for SharedArrayBuffer. Harmless for the single-threaded build.
  * Cross-Origin-Resource-Policy: same-origin
      -> COEP would otherwise block the page's *own* subresources; every asset
         we serve is same-origin, so this keeps them loadable.
  * HTTP range requests, so multi-hundred-megabyte .gguf files can be fetched
    with a real progress bar and resumed by the browser.

Usage:
    python3 serve.py                       # http://127.0.0.1:8080/
    python3 serve.py --port 0              # pick a free port, print it
    python3 serve.py --models ~/models     # also expose ~/models at /models/
"""

import argparse
import os
import posixpath
import re
import shutil
import socket
import sys
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))

MIME_TYPES = {
    ".wasm": "application/wasm",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".cjs": "text/javascript",
    ".json": "application/json",
    ".map": "application/json",
    ".css": "text/css",
    ".html": "text/html",
    ".htm": "text/html",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".gguf": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".data": "application/octet-stream",
}

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
COPY_CHUNK = 1 << 20  # 1 MiB


class LlamafileHandler(SimpleHTTPRequestHandler):
    """Static handler with wasm-friendly headers and range support."""

    protocol_version = "HTTP/1.1"
    server_version = "llamafile-wasm-dev"
    sys_version = ""

    root = HERE
    models_dir = None
    models_mount = "models"

    # ---------------------------------------------------------------- paths

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        trailing_slash = path.endswith("/")
        try:
            path = urllib.parse.unquote(path, errors="surrogatepass")
        except UnicodeDecodeError:
            path = urllib.parse.unquote(path)
        path = posixpath.normpath(path)
        words = [w for w in path.split("/") if w and w not in (os.curdir, os.pardir)]

        base = self.root
        if self.models_dir and words and words[0] == self.models_mount:
            base = self.models_dir
            words = words[1:]

        resolved = base
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            resolved = os.path.join(resolved, word)
        if trailing_slash:
            resolved += "/"
        return resolved

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in MIME_TYPES:
            mime = MIME_TYPES[ext]
            if mime.startswith(("text/", "application/json", "image/svg")):
                return mime + "; charset=utf-8"
            return mime
        return super().guess_type(path)

    # -------------------------------------------------------------- headers

    def end_headers(self):
        # Cross-origin isolation (needed later for SharedArrayBuffer/pthreads).
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # ...but COEP blocks our own subresources unless they opt in. They are
        # all same-origin, so same-origin is the tightest policy that works.
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        # Dev server: never let a stale llamafile.wasm hang around.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---------------------------------------------------------------- verbs

    def do_GET(self):
        handle = self._open_request()
        if handle is None:
            return
        stream, start, length = handle
        try:
            if start:
                stream.seek(start)
            self._copy(stream, length)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            stream.close()

    def do_HEAD(self):
        handle = self._open_request()
        if handle is not None:
            handle[0].close()

    def _copy(self, stream, length):
        remaining = length
        while remaining > 0:
            chunk = stream.read(min(COPY_CHUNK, remaining))
            if not chunk:
                break
            self.wfile.write(chunk)
            remaining -= len(chunk)

    # ------------------------------------------------------------ internals

    def _open_request(self):
        """Resolve the request, send headers, return (file, start, length)."""
        path = self.translate_path(self.path)

        if os.path.isdir(path):
            parts = urllib.parse.urlsplit(self.path)
            if not parts.path.endswith("/"):
                new = urllib.parse.urlunsplit(
                    (parts.scheme, parts.netloc, parts.path + "/", parts.query, parts.fragment)
                )
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", new)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            index = os.path.join(path, "index.html")
            if os.path.isfile(index):
                path = index
            elif os.path.isfile(os.path.join(path, "web", "index.html")):
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", parts.path + "web/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            else:
                return self._send_listing(path)

        try:
            stream = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            size = os.fstat(stream.fileno()).st_size
            ctype = self.guess_type(path)
            rng = self._parse_range(self.headers.get("Range"), size)

            if rng == "invalid":
                stream.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None

            if rng is None:
                self.send_response(HTTPStatus.OK)
                start, length = 0, size
            else:
                start, end = rng
                length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))

            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(os.fstat(stream.fileno()).st_mtime))
            self.end_headers()
            return stream, start, length
        except Exception:
            stream.close()
            raise

    @staticmethod
    def _parse_range(header, size):
        """None = whole file, 'invalid' = 416, else (start, end) inclusive."""
        if not header:
            return None
        match = RANGE_RE.match(header.strip())
        if not match:
            return None  # multi-range and other exotica: just send everything
        first, last = match.group(1), match.group(2)
        if first == "" and last == "":
            return "invalid"
        if first == "":
            length = int(last)
            if length == 0:
                return "invalid"
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
            end = min(end, size - 1)
        if size == 0 or start >= size or start > end:
            return "invalid"
        return start, end

    def _send_listing(self, path):
        listing = self.list_directory(path)
        if listing is None:
            return None
        try:
            shutil.copyfileobj(listing, self.wfile)
        finally:
            listing.close()
        return None

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (0 = pick a free one)")
    parser.add_argument("--dir", default=HERE, help="directory to serve (default: llamafile-wasm/)")
    parser.add_argument("--models", default=None, help="extra directory exposed at /models/ (e.g. a GGUF cache)")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default: 127.0.0.1)")
    args = parser.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(root):
        parser.error("--dir %s is not a directory" % root)
    models = None
    if args.models:
        models = os.path.abspath(os.path.expanduser(args.models))
        if not os.path.isdir(models):
            parser.error("--models %s is not a directory" % models)

    class Handler(LlamafileHandler):
        pass

    Handler.root = root
    Handler.models_dir = models

    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.address_family = socket.AF_INET
    ThreadingHTTPServer.daemon_threads = True
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    port = httpd.server_address[1]

    print("llamafile-wasm dev server: http://%s:%d/" % (args.host, port), flush=True)
    print("  serving   %s" % root, file=sys.stderr, flush=True)
    if models:
        print("  models    %s -> /models/" % models, file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
