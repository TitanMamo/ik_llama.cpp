// kvarn-eval — in-process KVarN pseudo-decode A/B evaluator.
//
// Evaluates the same token chunk several times in one process with fresh
// contexts: f16 baseline, full VarN, Hadamard-only (alpha=0), partial alpha.
// Every phase calls llama_kvarn_debug_quantize with the same code path; only
// the `enabled`/codec parameters differ. This removes cross-process and
// code-path confounds that made the perplexity-tool route unusable.
//
// Usage:
//   kvarn-eval -m model.gguf -f corpus.txt [-c 2048] [-t threads] [-s skip]
//              [--phases f16,varn,a0.75,a0.5,...]
//   kvarn-eval -m model.gguf -f corpus.txt --gen 256 [-p prompt1.txt -p ...]
//              [--chat] [--out DIR] [--phases ...]
//
// -p FILE   generation prompt (repeatable). Without -p the first 768 tokens of
//           the corpus are used (legacy behavior).
// --chat    apply the model's chat template to -p prompts.
// --out DIR dump generated text (gen_<label>_<phase>.txt) and token ids
//           (.ids) plus prompt_<label>.txt.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static std::string read_file(const char * path, size_t max_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (s.size() > max_bytes) {
        s.resize(max_bytes);
    }
    return s;
}

static double token_nll(const float * logits, int n_vocab, llama_token target) {
    float mx = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        mx = std::max(mx, logits[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum += std::exp((double) logits[i] - (double) mx);
    }
    return -((double) logits[target] - (double) mx - std::log(sum));
}

struct phase_result {
    double ppl;
    int    n_eval;
};

struct codec {
    const char * name;
    bool  quantize;
    float alpha;
    int   kb;
    int   vb;
};

static const codec ppl_codecs[] = {
    { "f16#1", false, 1.00f, 4, 2 },
    { "k8v8",  true,  1.00f, 8, 8 },
    { "k8v2",  true,  1.00f, 8, 2 },
    { "k4v8",  true,  1.00f, 4, 8 },
    { "varn",  true,  1.00f, 4, 2 },
    { "a0.75", true,  0.75f, 4, 2 },
    { "a0.6",  true,  0.60f, 4, 2 },
    { "a0.5",  true,  0.50f, 4, 2 },
    { "a0.25", true,  0.25f, 4, 2 },
    { "had",   true,  0.00f, 4, 2 },
    { "k4v4",  true,  1.00f, 4, 4 },
    { "k4v3",  true,  1.00f, 4, 3 },
    { "k5v3",  true,  1.00f, 5, 3 },
    { "k6v3",  true,  1.00f, 6, 3 },
    { "k5v4",  true,  1.00f, 5, 4 },
    { "k6v4",  true,  1.00f, 6, 4 },
    { "f16#2", false, 1.00f, 4, 2 },
};

static const codec gen_codecs[] = {
    { "f16",   false, 1.00f, 4, 2 },
    { "varn",  true,  1.00f, 4, 2 },
    { "a0.75", true,  0.75f, 4, 2 },
    { "a0.6",  true,  0.60f, 4, 2 },
    { "a0.5",  true,  0.50f, 4, 2 },
    { "a0.25", true,  0.25f, 4, 2 },
    { "had",   true,  0.00f, 4, 2 },
    { "k4v4",  true,  1.00f, 4, 4 },
    { "k4v3",  true,  1.00f, 4, 3 },
    { "k5v3",  true,  1.00f, 5, 3 },
    { "k6v3",  true,  1.00f, 6, 3 },
    { "k5v4",  true,  1.00f, 5, 4 },
    { "k6v4",  true,  1.00f, 6, 4 },
    { "k8v8",  true,  1.00f, 8, 8 },
};

static bool phase_selected(const char * filter, const char * name) {
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    const std::string f(filter);
    size_t pos = 0;
    while (pos <= f.size()) {
        const size_t c = f.find(',', pos);
        const std::string tok = f.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
        if (!tok.empty() && tok == name) {
            return true;
        }
        if (c == std::string::npos) {
            break;
        }
        pos = c + 1;
    }
    return false;
}

static std::string apply_chat(const struct llama_model * model, const std::string & user) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl == nullptr) {
        fprintf(stderr, "note: model has no chat template; using raw prompt\n");
        return user;
    }
    std::vector<char> buf(user.size() * 2 + 4096);
    const llama_chat_message msg = { "user", user.c_str() };
    int32_t n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    if (n > (int32_t) buf.size()) {
        buf.resize(size_t(n) + 1);
        n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    }
    if (n <= 0) {
        return user;
    }
    return std::string(buf.data(), size_t(n));
}

// Optional runtime KVarN path: mirrors common.cpp's -ctk/-ctv interception.
// When set, phases use the production store/materialize graph instead of the
// post-decode debug hook (pass "f16" phases to avoid double quantization).
static std::string g_cache_type_k;
static std::string g_cache_type_v;
static int         g_ubatch = 128;

static void apply_kvarn_ctx_params(llama_context_params & cparams) {
    if (g_cache_type_k.empty() && g_cache_type_v.empty()) {
        return;
    }
    const char * ck = g_cache_type_k.empty() ? "f16" : g_cache_type_k.c_str();
    const char * cv = g_cache_type_v.empty() ? "f16" : g_cache_type_v.c_str();
    const char * error = nullptr;
    if (!llama_kvarn_params_from_cache_types(&cparams.kvarn, ck, cv, &error)) {
        fprintf(stderr, "invalid cache types: %s\n", error != nullptr ? error : "?");
        exit(1);
    }
    if (cparams.kvarn.enabled) {
        cparams.type_k     = GGML_TYPE_F16;
        cparams.type_v     = GGML_TYPE_F16;
        cparams.flash_attn = true; // the materialized cache is consumed by FA
    }
}

static phase_result run_phase(
        struct llama_model * model,
        const struct llama_vocab * vocab,
        const std::vector<llama_token> & toks,
        bool  quantize,
        float alpha,
        int   k_bits,
        int   v_bits,
        int   iters,
        int   n_threads) {
    llama_context_params cparams = llama_context_default_params();
    apply_kvarn_ctx_params(cparams);
    cparams.n_ctx           = (uint32_t) toks.size();
    cparams.n_batch         = g_ubatch;
    cparams.n_ubatch        = g_ubatch;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        exit(1);
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_init(std::max(128, g_ubatch), 0, 1);

    uint32_t done = 0;
    double nll = 0.0;
    int n_eval = 0;

    for (int i0 = 0; i0 < (int) toks.size(); i0 += g_ubatch) {
        const int n = std::min(g_ubatch, (int) toks.size() - i0);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = toks[i0 + i];
            batch.pos[i]       = i0 + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at %d\n", i0);
            exit(1);
        }
        for (int i = 0; i + 1 < n; ++i) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (logits == nullptr) {
                continue;
            }
            nll += token_nll(logits, n_vocab, toks[i0 + i + 1]);
            ++n_eval;
        }
        llama_kvarn_debug_quantize(ctx, alpha, k_bits, v_bits, iters, quantize, &done);
    }

    llama_batch_free(batch);
    llama_free(ctx);

    return { std::exp(nll / std::max(1, n_eval)), n_eval };
}

static std::vector<llama_token> generate(
        struct llama_model * model,
        const struct llama_vocab * vocab,
        const std::vector<llama_token> & prompt,
        int n_gen, bool quantize, float alpha, int k_bits, int v_bits, int n_threads,
        std::string & text) {
    llama_context_params cparams = llama_context_default_params();
    apply_kvarn_ctx_params(cparams);
    cparams.n_ctx           = (uint32_t) (prompt.size() + n_gen + 128);
    cparams.n_batch         = g_ubatch;
    cparams.n_ubatch        = g_ubatch;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        exit(1);
    }
    uint32_t done = 0;
    llama_batch batch = llama_batch_init(std::max(128, g_ubatch), 0, 1);

    for (size_t i0 = 0; i0 < prompt.size(); i0 += 128) {
        const int n = std::min<size_t>(128, prompt.size() - i0);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = prompt[i0 + i];
            batch.pos[i]       = (int) (i0 + i);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i0 + i + 1 == prompt.size()) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
        llama_kvarn_debug_quantize(ctx, alpha, k_bits, v_bits, 16, quantize, &done);
    }

    std::vector<llama_token> out;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int pos = (int) prompt.size();
    for (int g = 0; g < n_gen; ++g) {
        const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) { best = i; }
        }
        out.push_back(best);
        if (llama_vocab_is_eog(vocab, best)) { break; }
        batch.n_tokens = 1;
        batch.token[0]     = best;
        batch.pos[0]       = pos++;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
        llama_kvarn_debug_quantize(ctx, alpha, k_bits, v_bits, 16, quantize, &done);
    }

    text.clear();
    for (llama_token t : out) {
        char buf[256];
        const int n = llama_token_to_piece_vocab(vocab, t, buf, sizeof(buf), 0, true);
        if (n > 0) { text.append(buf, n); }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    return out;
}

static void print_stats(const std::vector<llama_token> & out) {
    if (out.empty()) { printf("  (no output)\n"); return; }
    std::unordered_map<llama_token, int> cnt;
    for (llama_token t : out) { cnt[t]++; }
    const int n = (int) out.size();
    int maxrep = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int l = 0;
            while (i + l < j && j + l < n && out[i + l] == out[j + l]) { ++l; }
            if (l > maxrep) { maxrep = l; }
        }
    }
    uint64_t h = 1469598103934665603ull;
    for (llama_token t : out) { h ^= (uint64_t) (uint32_t) t; h *= 1099511628211ull; }
    printf("  gen=%d unique=%zu ratio=%.2f max_repeat=%d hash=%016llx\n", n, cnt.size(), (double) cnt.size() / n, maxrep, (unsigned long long) h);
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * corpus     = nullptr;
    int n_ctx  = 512;
    int n_thr  = 6;
    int skip   = 0;
    int n_gen  = 0;
    int n_gpu_layers = 0;
    const char * out_dir = nullptr;
    const char * phases  = nullptr;
    bool chat = false;
    std::vector<std::string> prompt_files;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "-m") && i + 1 < argc) { model_path = argv[++i]; }
        else if (!std::strcmp(argv[i], "-f") && i + 1 < argc) { corpus     = argv[++i]; }
        else if (!std::strcmp(argv[i], "-c") && i + 1 < argc) { n_ctx      = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) { n_thr      = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) { skip       = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--gen") && i + 1 < argc) { n_gen   = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-ngl") && i + 1 < argc) { n_gpu_layers = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) { prompt_files.push_back(argv[++i]); }
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) { out_dir = argv[++i]; }
        else if (!std::strcmp(argv[i], "--phases") && i + 1 < argc) { phases = argv[++i]; }
        else if (!std::strcmp(argv[i], "--chat")) { chat = true; }
        else if (!std::strcmp(argv[i], "--ctk") && i + 1 < argc) { g_cache_type_k = argv[++i]; }
        else if (!std::strcmp(argv[i], "--ctv") && i + 1 < argc) { g_cache_type_v = argv[++i]; }
        else if (!std::strcmp(argv[i], "--ub") && i + 1 < argc) { g_ubatch = std::max(1, std::atoi(argv[++i])); }
        else {
            fprintf(stderr, "usage: %s -m model.gguf [-f corpus.txt] [--gen N [-p prompt.txt] [--chat]] [--out DIR] [--phases a,b] [-c n] [-t threads] [-s skip]\n", argv[0]);
            return 1;
        }
    }
    if (model_path == nullptr || (corpus == nullptr && prompt_files.empty())) {
        fprintf(stderr, "usage: %s -m model.gguf [-f corpus.txt] [--gen N [-p prompt.txt] [--chat]] [--out DIR] [--phases a,b] [-c n] [-t threads] [-s skip]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers; // 0 = CPU-only (build-kvarn); 99 on the CUDA build
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks;
    if (corpus != nullptr) {
        const std::string text = read_file(corpus, 64 * 1024);
        std::vector<llama_token> all(256 * 1024);
        int n = llama_tokenize(model, text.data(), (int) text.size(), all.data(), (int) all.size(), true, true);
        if (n < 0) {
            fprintf(stderr, "tokenize failed (need %d tokens)\n", -n);
            return 1;
        }
        if (skip < 0) {
            skip = 0;
        }
        if (skip > n) {
            skip = n;
        }
        if (n - skip > n_ctx) {
            n = skip + n_ctx;
        }
        toks.assign(all.begin() + skip, all.begin() + n);
        printf("tokens: %d (chunk %d, skip %d)\n", n - skip, n_ctx, skip);
    }

    if (n_gen > 0) {
        // build the prompt list
        std::vector<std::pair<std::string, std::vector<llama_token>>> prompts;
        if (prompt_files.empty()) {
            const int n_prompt = std::min(768, (int) toks.size());
            prompts.push_back({ "corpus", std::vector<llama_token>(toks.begin(), toks.begin() + n_prompt) });
        } else {
            for (size_t pi = 0; pi < prompt_files.size(); ++pi) {
                std::string text = read_file(prompt_files[pi].c_str(), 64 * 1024);
                if (chat) {
                    text = apply_chat(model, text);
                }
                std::vector<llama_token> ptoks(256 * 1024);
                const int n = llama_tokenize(model, text.data(), (int) text.size(),
                        ptoks.data(), (int) ptoks.size(), !chat, true);
                if (n < 0) {
                    fprintf(stderr, "tokenize failed for %s (need %d tokens)\n", prompt_files[pi].c_str(), -n);
                    return 1;
                }
                ptoks.resize(n);
                char label[64];
                std::snprintf(label, sizeof(label), "p%zu", pi);
                printf("prompt %s: %d tokens (%s)\n", label, n, prompt_files[pi].c_str());
                prompts.push_back({ label, ptoks });
                if (out_dir != nullptr) {
                    char path[1024];
                    std::snprintf(path, sizeof(path), "%s/prompt_%s.txt", out_dir, label);
                    FILE * f = std::fopen(path, "w");
                    if (f != nullptr) { std::fwrite(text.data(), 1, text.size(), f); std::fclose(f); }
                }
            }
        }

        for (const auto & pr : prompts) {
            for (const codec & c : gen_codecs) {
                if (!phase_selected(phases, c.name)) {
                    continue;
                }
                std::string gtext;
                const clock_t t0 = clock();
                const std::vector<llama_token> out = generate(model, vocab, pr.second, n_gen,
                        c.quantize, c.alpha, c.kb, c.vb, n_thr, gtext);
                const double secs = double(clock() - t0) / CLOCKS_PER_SEC;
                printf("=== %s / %s (%.0fs) ===\n", pr.first.c_str(), c.name, secs);
                print_stats(out);
                const size_t tail = gtext.size() > 400 ? gtext.size() - 400 : 0;
                printf("  ...%s\n", gtext.substr(tail).c_str());
                if (out_dir != nullptr) {
                    char path[1024];
                    std::snprintf(path, sizeof(path), "%s/gen_%s_%s.txt", out_dir, pr.first.c_str(), c.name);
                    FILE * f = std::fopen(path, "w");
                    if (f != nullptr) { std::fwrite(gtext.data(), 1, gtext.size(), f); std::fclose(f); }
                    std::snprintf(path, sizeof(path), "%s/gen_%s_%s.ids", out_dir, pr.first.c_str(), c.name);
                    f = std::fopen(path, "w");
                    if (f != nullptr) {
                        for (llama_token t : out) { std::fprintf(f, "%d\n", t); }
                        std::fclose(f);
                    }
                    printf("  -> %s/gen_%s_%s.txt\n", out_dir, pr.first.c_str(), c.name);
                }
                std::fflush(stdout);
            }
        }

        llama_free_model(model);
        llama_backend_free();
        return 0;
    }

    if (toks.empty()) {
        fprintf(stderr, "no corpus tokens\n");
        return 1;
    }

    for (const codec & c : ppl_codecs) {
        if (!phase_selected(phases, c.name)) {
            continue;
        }
        const clock_t t0 = clock();
        const phase_result r = run_phase(model, vocab, toks, c.quantize, c.alpha, c.kb, c.vb, 16, n_thr);
        printf("%s : PPL %.4f (n=%d) [%.0fs]\n", c.name, r.ppl, r.n_eval, double(clock() - t0) / CLOCKS_PER_SEC);
        std::fflush(stdout);
    }

    llama_free_model(model);
    llama_backend_free();
    return 0;
}
