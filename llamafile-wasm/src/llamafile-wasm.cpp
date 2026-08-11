// llamafile-wasm — WebAssembly core for "llamafile in the browser".
//
// Implements the low-level `lf_*` contract described in ../API.md, exposed to
// JavaScript through embind. Everything is single-threaded: the module is built
// without pthreads so it can be served without COOP/COEP headers.
//
// Rules of the house:
//   * No exception may ever escape into JS. Every exported function catches
//     everything and stashes a message retrievable through lf_last_error().
//   * All state lives in one process-global `g` struct.

#include "llama.h"
#include "ggml.h"

#include "common.h"
#include "chat.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <emscripten/bind.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// global state
// ---------------------------------------------------------------------------

namespace {

struct lf_context {
    llama_model   *      model = nullptr;
    llama_context *      ctx   = nullptr;
    const llama_vocab *  vocab = nullptr;

    common_chat_templates_ptr tmpls;

    // sampler chain for the in-flight generation
    llama_sampler * smpl = nullptr;

    int  n_ctx     = 0;
    int  n_batch   = 0;
    bool has_chat_template = false;

    // generation state
    bool gen_active    = false;   // lf_gen_begin succeeded and lf_gen_end not called
    bool gen_finished  = true;    // EOG / n_predict / context full reached
    int  n_predict     = 0;
    int  n_decoded     = 0;       // tokens produced by lf_gen_next
    int  n_prompt      = 0;       // tokens in the evaluated prompt
    int  n_past        = 0;       // tokens currently in the KV cache

    double prompt_ms  = 0.0;
    double predict_ms = 0.0;

    std::string last_error;
};

lf_context g;
bool g_backend_ready = false;

double now_ms() {
    return (double) ggml_time_us() / 1000.0;
}

void lf_log_callback(ggml_log_level level, const char * text, void * /*user*/) {
    // llama.cpp is chatty; only surface warnings and above to the JS console.
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

void ensure_backend() {
    if (!g_backend_ready) {
        llama_log_set(lf_log_callback, nullptr);
        ggml_log_set(lf_log_callback, nullptr);
        llama_backend_init();
        g_backend_ready = true;
    }
}

void free_sampler() {
    if (g.smpl) {
        llama_sampler_free(g.smpl);
        g.smpl = nullptr;
    }
}

void reset_generation() {
    free_sampler();
    g.gen_active   = false;
    g.gen_finished = true;
    g.n_predict    = 0;
    g.n_decoded    = 0;
    g.n_past       = 0;
}

void do_unload() {
    reset_generation();
    g.tmpls.reset();
    if (g.ctx) {
        llama_free(g.ctx);
        g.ctx = nullptr;
    }
    if (g.model) {
        llama_model_free(g.model);
        g.model = nullptr;
    }
    g.vocab   = nullptr;
    g.n_ctx   = 0;
    g.n_batch = 0;
    g.has_chat_template = false;
    g.n_prompt   = 0;
    g.prompt_ms  = 0.0;
    g.predict_ms = 0.0;
}

// Evaluate `tokens` in chunks of n_batch. Returns false on decode failure.
bool decode_chunked(const std::vector<llama_token> & tokens) {
    const int n_batch = g.n_batch > 0 ? g.n_batch : 512;
    for (size_t i = 0; i < tokens.size(); i += (size_t) n_batch) {
        const int n = (int) std::min((size_t) n_batch, tokens.size() - i);
        // llama_batch_get_one wants a non-const pointer but does not write.
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + i), n);
        if (llama_decode(g.ctx, batch) != 0) {
            return false;
        }
        g.n_past += n;
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// exported API
//
// The LF_GUARD macros keep every entry point exception-free: anything thrown
// (std::exception, llama.cpp's runtime_errors, jinja parse errors, ...) is
// converted into lf_last_error() plus a caller-visible failure value.
// ---------------------------------------------------------------------------

#define LF_TRY  try {
#define LF_CATCH(failure_value)                                              \
    } catch (const std::exception & e) {                                     \
        g.last_error = e.what();                                             \
        return failure_value;                                                \
    } catch (...) {                                                          \
        g.last_error = "unknown error";                                      \
        return failure_value;                                                \
    }

#define LF_CATCH_VOID                                                        \
    } catch (const std::exception & e) {                                     \
        g.last_error = e.what();                                             \
    } catch (...) {                                                          \
        g.last_error = "unknown error";                                      \
    }

// lf_load(path, n_ctx, n_threads) -> int   (0 = ok)
int lf_load(const std::string & path, int n_ctx, int n_threads) {
LF_TRY
    g.last_error.clear();
    ensure_backend();
    do_unload();

    if (path.empty()) {
        g.last_error = "lf_load: empty model path";
        return 1;
    }

    // The whole model is read through the Emscripten FS into the wasm heap;
    // mmap/mlock are meaningless (and broken) here.
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap  = false;
    mparams.use_mlock = false;
    mparams.n_gpu_layers = 0;
    mparams.check_tensors = false;

    g.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g.model) {
        g.last_error = "lf_load: failed to load model from '" + path + "'";
        return 2;
    }

    g.vocab = llama_model_get_vocab(g.model);

    const int n_ctx_train = llama_model_n_ctx_train(g.model);
    int n_ctx_req = n_ctx > 0 ? n_ctx : n_ctx_train;
    if (n_ctx_train > 0) {
        n_ctx_req = std::min(n_ctx_req, n_ctx_train);
    }
    n_ctx_req = std::max(n_ctx_req, 32);

    // Single-threaded build: n_threads is accepted for API compatibility but
    // clamped to 1 (no pthreads / SharedArrayBuffer in this artifact).
    (void) n_threads;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) n_ctx_req;
    cparams.n_batch         = (uint32_t) std::min(n_ctx_req, 512);
    cparams.n_ubatch        = (uint32_t) std::min(n_ctx_req, 512);
    cparams.n_seq_max       = 1;
    cparams.n_threads       = 1;
    cparams.n_threads_batch = 1;
    cparams.no_perf         = false;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    g.ctx = llama_init_from_model(g.model, cparams);
    if (!g.ctx) {
        llama_model_free(g.model);
        g.model = nullptr;
        g.vocab = nullptr;
        g.last_error = "lf_load: failed to create llama context";
        return 3;
    }

    g.n_ctx   = (int) llama_n_ctx(g.ctx);
    g.n_batch = (int) cparams.n_batch;

    const char * tmpl = llama_model_chat_template(g.model, /*name*/ nullptr);
    g.has_chat_template = tmpl != nullptr && *tmpl != '\0';

    // common_chat_templates_init falls back to ChatML when the GGUF carries no
    // template of its own, which is exactly the behaviour API.md specifies.
    try {
        g.tmpls = common_chat_templates_init(g.model, /*chat_template_override*/ "");
    } catch (const std::exception & e) {
        g.tmpls.reset();
        g.last_error = std::string("lf_load: chat template init failed: ") + e.what();
        // Not fatal — plain completion still works.
    }

    return 0;
LF_CATCH(9)
}

// lf_model_info() -> JSON string
std::string lf_model_info() {
LF_TRY
    if (!g.model || !g.ctx) {
        g.last_error = "lf_model_info: no model loaded";
        return std::string("{}");
    }

    char desc[512] = {0};
    llama_model_desc(g.model, desc, sizeof(desc));

    json j;
    j["n_params"]          = (uint64_t) llama_model_n_params(g.model);
    j["n_ctx_train"]       = (int32_t)  llama_model_n_ctx_train(g.model);
    j["n_ctx"]             = (int32_t)  llama_n_ctx(g.ctx);
    j["n_vocab"]           = (int32_t)  llama_vocab_n_tokens(g.vocab);
    j["desc"]              = std::string(desc);
    j["has_chat_template"] = g.has_chat_template;
    j["size_bytes"]        = (uint64_t) llama_model_size(g.model);

    return j.dump();
LF_CATCH(std::string("{}"))
}

// lf_format_chat(messages_json, add_assistant) -> prompt string
std::string lf_format_chat(const std::string & messages_json, bool add_assistant) {
LF_TRY
    g.last_error.clear();
    if (!g.model) {
        g.last_error = "lf_format_chat: no model loaded";
        return std::string();
    }
    if (!g.tmpls) {
        g.last_error = "lf_format_chat: no chat template available";
        return std::string();
    }

    const json msgs = json::parse(messages_json);
    if (!msgs.is_array()) {
        g.last_error = "lf_format_chat: messages must be a JSON array";
        return std::string();
    }

    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.add_generation_prompt = add_assistant;
    inputs.enable_thinking       = false;

    for (const auto & m : msgs) {
        common_chat_msg msg;
        msg.role = m.contains("role") && m.at("role").is_string()
                     ? m.at("role").get<std::string>() : std::string("user");
        if (m.contains("content")) {
            const auto & c = m.at("content");
            if (c.is_string()) {
                msg.content = c.get<std::string>();
            } else if (c.is_array()) {
                // OpenAI-style content parts: concatenate the text pieces.
                for (const auto & part : c) {
                    if (part.contains("text") && part.at("text").is_string()) {
                        msg.content += part.at("text").get<std::string>();
                    }
                }
            } else if (!c.is_null()) {
                msg.content = c.dump();
            }
        }
        inputs.messages.push_back(std::move(msg));
    }

    const common_chat_params params = common_chat_templates_apply(g.tmpls.get(), inputs);
    return params.prompt;
LF_CATCH(std::string())
}

// lf_gen_begin(prompt, n_predict, temp, top_p, seed) -> int  (0 = ok)
int lf_gen_begin(const std::string & prompt, int n_predict, float temp, float top_p, int seed) {
LF_TRY
    g.last_error.clear();
    if (!g.ctx || !g.model) {
        g.last_error = "lf_gen_begin: no model loaded";
        return 1;
    }

    reset_generation();

    // Fresh KV cache for every generation. (Prefix reuse is a future
    // optimisation; correctness first.)
    llama_memory_clear(llama_get_memory(g.ctx), true);
    llama_perf_context_reset(g.ctx);

    // Sampler chain: top_p -> temp -> dist, i.e. the classic temperature
    // sampling setup. temp <= 0 degenerates to greedy decoding.
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    g.smpl = llama_sampler_chain_init(sparams);
    if (!g.smpl) {
        g.last_error = "lf_gen_begin: failed to create sampler chain";
        return 2;
    }

    if (temp <= 0.0f) {
        llama_sampler_chain_add(g.smpl, llama_sampler_init_greedy());
    } else {
        if (top_p > 0.0f && top_p < 1.0f) {
            llama_sampler_chain_add(g.smpl, llama_sampler_init_top_p(top_p, /*min_keep*/ 1));
        }
        llama_sampler_chain_add(g.smpl, llama_sampler_init_temp(temp));
        const uint32_t s = seed < 0 ? LLAMA_DEFAULT_SEED : (uint32_t) seed;
        llama_sampler_chain_add(g.smpl, llama_sampler_init_dist(s));
    }

    std::vector<llama_token> tokens = common_tokenize(g.vocab, prompt,
                                                      /*add_special*/ true,
                                                      /*parse_special*/ true);
    if (tokens.empty()) {
        // Nothing to condition on — start from BOS so llama_decode has input.
        const llama_token bos = llama_vocab_bos(g.vocab);
        if (bos != LLAMA_TOKEN_NULL) {
            tokens.push_back(bos);
        } else {
            g.last_error = "lf_gen_begin: prompt tokenised to zero tokens";
            free_sampler();
            return 3;
        }
    }

    // Leave room for at least a few generated tokens.
    const size_t max_prompt = (size_t) std::max(g.n_ctx - 4, 1);
    if (tokens.size() > max_prompt) {
        // Keep the tail: the most recent context matters most for chat.
        tokens.erase(tokens.begin(), tokens.end() - (long) max_prompt);
    }

    g.n_predict = n_predict > 0 ? n_predict : (g.n_ctx - (int) tokens.size());
    g.n_predict = std::max(g.n_predict, 0);

    const double t0 = now_ms();
    if (!decode_chunked(tokens)) {
        g.last_error = "lf_gen_begin: llama_decode failed on the prompt";
        free_sampler();
        return 4;
    }
    g.prompt_ms  = now_ms() - t0;
    g.predict_ms = 0.0;
    g.n_prompt   = (int) tokens.size();
    g.n_decoded  = 0;

    g.gen_active   = true;
    g.gen_finished = g.n_predict == 0;

    return 0;
LF_CATCH(9)
}

// lf_gen_next() -> next piece, or "" when generation is finished
std::string lf_gen_next() {
LF_TRY
    if (!g.gen_active || g.gen_finished || !g.ctx || !g.smpl) {
        return std::string();
    }

    const double t0 = now_ms();

    const llama_token id = llama_sampler_sample(g.smpl, g.ctx, -1);

    if (llama_vocab_is_eog(g.vocab, id)) {
        g.gen_finished = true;
        g.predict_ms += now_ms() - t0;
        return std::string();
    }

    llama_sampler_accept(g.smpl, id);

    // Detokenise before decoding so the caller gets the piece even if the
    // follow-up decode is the one that trips the context limit.
    std::string piece = common_token_to_piece(g.ctx, id, /*special*/ false);

    g.n_decoded++;

    if (g.n_decoded >= g.n_predict || g.n_past + 1 >= g.n_ctx) {
        g.gen_finished = true;
        g.predict_ms += now_ms() - t0;
        return piece;
    }

    llama_token tok = id;
    llama_batch batch = llama_batch_get_one(&tok, 1);
    if (llama_decode(g.ctx, batch) != 0) {
        g.last_error = "lf_gen_next: llama_decode failed";
        g.gen_finished = true;
        g.predict_ms += now_ms() - t0;
        return piece;
    }
    g.n_past++;

    g.predict_ms += now_ms() - t0;
    return piece;
LF_CATCH(std::string())
}

// lf_gen_done() -> bool
bool lf_gen_done() {
LF_TRY
    return !g.gen_active || g.gen_finished;
LF_CATCH(true)
}

// lf_gen_end() -> void
void lf_gen_end() {
LF_TRY
    g.gen_finished = true;
    g.gen_active   = false;
    free_sampler();
LF_CATCH_VOID
}

// lf_perf() -> JSON string
std::string lf_perf() {
LF_TRY
    json j;
    j["prompt_tokens"]    = g.n_prompt;
    j["predicted_tokens"] = g.n_decoded;
    j["prompt_ms"]        = g.prompt_ms;
    j["predict_ms"]       = g.predict_ms;
    return j.dump();
LF_CATCH(std::string("{}"))
}

// lf_last_error() -> string
std::string lf_last_error() {
    return g.last_error;
}

// lf_unload() -> void
void lf_unload() {
LF_TRY
    do_unload();
LF_CATCH_VOID
}

// ---------------------------------------------------------------------------
// embind
// ---------------------------------------------------------------------------

EMSCRIPTEN_BINDINGS(llamafile_wasm) {
    emscripten::function("lf_load",        &lf_load);
    emscripten::function("lf_model_info",  &lf_model_info);
    emscripten::function("lf_format_chat", &lf_format_chat);
    emscripten::function("lf_gen_begin",   &lf_gen_begin);
    emscripten::function("lf_gen_next",    &lf_gen_next);
    emscripten::function("lf_gen_done",    &lf_gen_done);
    emscripten::function("lf_gen_end",     &lf_gen_end);
    emscripten::function("lf_perf",        &lf_perf);
    emscripten::function("lf_last_error",  &lf_last_error);
    emscripten::function("lf_unload",      &lf_unload);
}
