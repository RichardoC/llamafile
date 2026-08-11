#!/usr/bin/env bash
#
# Build the llamafile WebAssembly core.
#
#   bash llamafile-wasm/build.sh            # incremental (re-runnable)
#   bash llamafile-wasm/build.sh --clean    # wipe build/ and dist/ first
#
# Produces:
#   llamafile-wasm/dist/llamafile.mjs
#   llamafile-wasm/dist/llamafile.wasm
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LLAMA_CPP="$ROOT/llama.cpp"
BUILD_DIR="$HERE/build/llama.cpp"
OBJ_DIR="$HERE/build/obj"
DIST_DIR="$HERE/dist"
PATCH_DIR="$HERE/patches"

JOBS="${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) )}"
CLEAN=0

for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        -j*)     JOBS="${arg#-j}" ;;
        -h|--help)
            sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Emscripten
# ---------------------------------------------------------------------------
if ! command -v emcc >/dev/null 2>&1; then
    for candidate in "${EMSDK:-}" /home/user/emsdk "$HOME/emsdk" /opt/emsdk; do
        if [ -n "$candidate" ] && [ -f "$candidate/emsdk_env.sh" ]; then
            say "activating emsdk at $candidate"
            # shellcheck disable=SC1091
            source "$candidate/emsdk_env.sh" >/dev/null 2>&1 || true
            break
        fi
    done
fi
if ! command -v emcc >/dev/null 2>&1; then
    echo "error: emcc not found. Install emsdk and 'source emsdk_env.sh', or set \$EMSDK." >&2
    exit 1
fi
say "emcc: $(emcc --version | head -1)"

if [ ! -f "$LLAMA_CPP/CMakeLists.txt" ]; then
    echo "error: llama.cpp submodule not checked out at $LLAMA_CPP" >&2
    echo "       run: git submodule update --init --recursive llama.cpp" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Optional submodule patches (emscripten-only fixes that cannot be committed
# because llama.cpp is a submodule). Applied idempotently.
# ---------------------------------------------------------------------------
if [ -d "$PATCH_DIR" ] && compgen -G "$PATCH_DIR/*.patch" >/dev/null; then
    for p in "$PATCH_DIR"/*.patch; do
        if git -C "$LLAMA_CPP" apply --check --reverse "$p" >/dev/null 2>&1; then
            say "patch already applied: $(basename "$p")"
        else
            say "applying patch: $(basename "$p")"
            git -C "$LLAMA_CPP" apply "$p"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
if [ "$CLEAN" = "1" ]; then
    say "cleaning $HERE/build and $DIST_DIR"
    rm -rf "$HERE/build" "$DIST_DIR/llamafile.mjs" "$DIST_DIR/llamafile.wasm"
fi

mkdir -p "$BUILD_DIR" "$OBJ_DIR" "$DIST_DIR"

# ---------------------------------------------------------------------------
# Flags
#
#  -msimd128        wasm SIMD (big speedup; Chrome 91+/Firefox 89+)
#  -fwasm-exceptions native wasm exception handling. Must be used identically
#                   on compile *and* link. Emscripten still emits the legacy
#                   (phase-3) encoding by default, which is what Chrome 95+,
#                   Firefox 131+ and Node 22 all execute. Add
#                   -sWASM_LEGACY_EXCEPTIONS=0 to emit the standardised exnref
#                   encoding once the runtime floor is Chrome 137+/Node 24+.
#
# GGML_LLAMAFILE=OFF is required: llamafile's ggml-cpu patches (guarded by
# GGML_USE_LLAMAFILE) call into the cosmopolitan tinyBLAS kernels, which are
# not part of this build.
#  single threaded: no -pthread anywhere, so no SharedArrayBuffer/COOP-COEP
#                   requirement on the serving side.
# ---------------------------------------------------------------------------
ARCH_FLAGS="-msimd128 -fwasm-exceptions"
C_FLAGS="$ARCH_FLAGS"
CXX_FLAGS="$ARCH_FLAGS"

# ---------------------------------------------------------------------------
# Configure + build llama.cpp static libs
# ---------------------------------------------------------------------------
say "configuring llama.cpp (emcmake cmake) -> $BUILD_DIR"
emcmake cmake -S "$LLAMA_CPP" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm \
    -DCMAKE_C_FLAGS="$C_FLAGS" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_WASM_SINGLE_FILE=OFF \
    -DGGML_BUILD_TESTS=OFF \
    -DGGML_BUILD_EXAMPLES=OFF \
    -DLLAMA_WASM_MEM64=OFF \
    -DLLAMA_BUILD_HTML=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    > "$HERE/build/configure.log" 2>&1 || {
        echo "cmake configure failed; tail of $HERE/build/configure.log:" >&2
        tail -40 "$HERE/build/configure.log" >&2
        exit 1
    }
grep -E "GGML_SYSTEM_ARCH|Wasm detected|Adding CPU backend variant" "$HERE/build/configure.log" || true

say "building targets: ggml llama llama-common (-j$JOBS)"
cmake --build "$BUILD_DIR" --target ggml llama llama-common -j "$JOBS"

# ---------------------------------------------------------------------------
# Collect the static libraries (link order matters for wasm-ld archives, so we
# also wrap them in a --start-group).
# ---------------------------------------------------------------------------
LIBS=()
for lib in \
    common/libllama-common.a \
    common/libllama-common-base.a \
    vendor/cpp-httplib/libcpp-httplib.a \
    src/libllama.a \
    ggml/src/libggml.a \
    ggml/src/libggml-cpu.a \
    ggml/src/libggml-base.a
do
    if [ -f "$BUILD_DIR/$lib" ]; then
        LIBS+=("$BUILD_DIR/$lib")
    fi
done
# Anything else that got built (e.g. extra ggml helper archives).
while IFS= read -r extra; do
    for known in "${LIBS[@]}"; do
        [ "$known" = "$extra" ] && continue 2
    done
    LIBS+=("$extra")
done < <(find "$BUILD_DIR" -name '*.a' | sort)

if [ "${#LIBS[@]}" -eq 0 ]; then
    echo "error: no static libraries produced under $BUILD_DIR" >&2
    exit 1
fi
say "linking against ${#LIBS[@]} static archives"

# ---------------------------------------------------------------------------
# Compile the embind binding
# ---------------------------------------------------------------------------
INCLUDES=(
    -I"$LLAMA_CPP/include"
    -I"$LLAMA_CPP/ggml/include"
    -I"$LLAMA_CPP/common"
    -I"$LLAMA_CPP/vendor"
    -I"$BUILD_DIR/common"
)

say "compiling src/llamafile-wasm.cpp"
em++ -std=c++17 -O3 $ARCH_FLAGS "${INCLUDES[@]}" \
    -c "$HERE/src/llamafile-wasm.cpp" -o "$OBJ_DIR/llamafile-wasm.o"

# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------
LINK_FLAGS=(
    -O3
    $ARCH_FLAGS
    -lembind
    -lworkerfs.js
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sENVIRONMENT=web,worker,node
    -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=4gb
    -sINITIAL_MEMORY=64mb
    -sSTACK_SIZE=5mb
    -sFORCE_FILESYSTEM=1
    -sEXPORTED_RUNTIME_METHODS=FS,ccall,cwrap,HEAPU8
    -sALLOW_TABLE_GROWTH
    -sEXIT_RUNTIME=0
    -sASSERTIONS=0
)

say "linking -> $DIST_DIR/llamafile.mjs"
em++ "${LINK_FLAGS[@]}" \
    "$OBJ_DIR/llamafile-wasm.o" \
    -Wl,--start-group "${LIBS[@]}" -Wl,--end-group \
    -o "$DIST_DIR/llamafile.mjs"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
say "done"
for f in "$DIST_DIR/llamafile.mjs" "$DIST_DIR/llamafile.wasm"; do
    if [ -f "$f" ]; then
        size=$(wc -c < "$f")
        printf '  %-40s %10s bytes (%.1f MB)\n' \
            "${f#"$ROOT"/}" "$size" "$(echo "$size" | awk '{print $1/1048576}')"
    else
        echo "error: expected output missing: $f" >&2
        exit 1
    fi
done
