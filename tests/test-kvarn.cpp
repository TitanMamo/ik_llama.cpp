// test-kvarn.cpp — CPU smoke tests and golden-vector dump for the KVarN ops.
//
// Exercises GGML_OP_KVARN_STORE / GGML_OP_KVARN_MATERIALIZE through the public
// API. With --dump-dir DIR the raw tensors of one deterministic run are
// written to disk so bench_results/kvarn_p0/kvarn_ref.py can re-quantize the
// same input in numpy and cross-check the produced records.

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            ++n_fail; \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
        } \
    } while (0)

// deterministic RNG
static uint32_t g_rng = 0x9e3779b9u;
static uint32_t rng_next(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return g_rng = x;
}
static float rng_float(void) { // [-1, 1)
    return float(int32_t(rng_next() >> 8)) * (1.0f / 8388608.0f) - 1.0f;
}

static const int64_t D     = 128;
static const int64_t H     = 2;
static const int64_t G     = 128;
static const int64_t STAGE = 2 * G;
static const int64_t NG    = 8;
static const int     NTH   = 4;

struct tensors {
    struct ggml_tensor * idx;
    struct ggml_tensor * idx_full;
    struct ggml_tensor * k_cur;
    struct ggml_tensor * v_cur;
    struct ggml_tensor * k_stage;
    struct ggml_tensor * v_stage;
    struct ggml_tensor * k_rec;
    struct ggml_tensor * v_rec;
    int64_t T;
};

static struct tensors make_tensors(struct ggml_context * ctx, int64_t T, int kbits, int vbits, int S = 2) {
    struct tensors ts;
    const struct ggml_kvarn_layout l = ggml_kvarn_layout_make(D, G, kbits, vbits);
    ts.T       = T;
    ts.idx     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ts.idx_full = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NG * G);
    for (int64_t i = 0; i < NG * G; ++i) {
        ((int32_t *) ts.idx_full->data)[i] = int32_t(i);
    }
    ts.k_cur   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
    ts.v_cur   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
    ts.k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, int64_t(S) * G, H);
    ts.v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, int64_t(S) * G, H);
    ts.k_rec   = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, int64_t(l.record_bytes), NG * H);
    ts.v_rec   = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, int64_t(l.record_bytes), NG * H);
    return ts;
}

static void reset_stage_records(const struct tensors & ts) {
    std::memset(ts.k_stage->data, 0, ggml_nbytes(ts.k_stage));
    std::memset(ts.v_stage->data, 0, ggml_nbytes(ts.v_stage));
    std::memset(ts.k_rec->data,   0, ggml_nbytes(ts.k_rec));
    std::memset(ts.v_rec->data,   0, ggml_nbytes(ts.v_rec));
}

// originals kept outside ggml, layout [D, H, N]
static void fill_originals(std::vector<float> & k, std::vector<float> & v, int64_t n) {
    k.assign(size_t(D * H * n), 0.0f);
    v.assign(size_t(D * H * n), 0.0f);
    for (int64_t p = 0; p < n; ++p) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t d = 0; d < D; ++d) {
                float x = rng_float() * 0.7f;
                const int64_t key = d * 7 + p * 3 + h * 5;
                if (key % 37 == 0) {
                    x *= 9.0f; // scattered outliers
                }
                if (d == 0 && p % 16 == 0) {
                    x += 12.0f; // persistent channel outlier
                }
                k[size_t(d + h * D + p * D * H)] = x;
                v[size_t(d + h * D + p * D * H)] = rng_float() * 0.7f;
            }
        }
    }
}

static void set_batch(
        const struct tensors & ts,
        const std::vector<float> & orig_k,
        const std::vector<float> & orig_v,
        int64_t pos0,
        int64_t n) {
    GGML_ASSERT(n <= ts.T);
    int32_t * idx = (int32_t *) ts.idx->data;
    float * kc = (float *) ts.k_cur->data;
    float * vc = (float *) ts.v_cur->data;
    for (int64_t t = 0; t < n; ++t) {
        idx[t] = int32_t(pos0 + t);
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t d = 0; d < D; ++d) {
                kc[d + h * D + t * D * H] = orig_k[size_t(d + h * D + (pos0 + t) * D * H)];
                vc[d + h * D + t * D * H] = orig_v[size_t(d + h * D + (pos0 + t) * D * H)];
            }
        }
    }
}

static void compute_store(
        struct ggml_context * ctx,
        const struct tensors & ts,
        int kbits, int vbits, int iters, int64_t owner, int64_t sealed, int S = 2) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * sk = ggml_kvarn_store(ctx, ts.k_cur, ts.idx, ts.k_stage, ts.k_rec,
            kbits, vbits, iters, false, S, 1, int(owner), int(sealed), -1, 1.0f);
    struct ggml_tensor * sv = ggml_kvarn_store(ctx, ts.v_cur, ts.idx, ts.v_stage, ts.v_rec,
            kbits, vbits, iters, true, S, 1, int(owner), int(sealed), -1, 1.0f);
    ggml_build_forward_expand(gf, sk);
    ggml_build_forward_expand(gf, sv);
    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, NTH);
    CHECK(st == GGML_STATUS_SUCCESS, "store graph compute failed: %d", int(st));
}

static struct ggml_tensor * compute_materialize(
        struct ggml_context * ctx,
        const struct tensors & ts,
        int64_t n_kv, int kbits, int vbits, int64_t sealed, bool value, int S = 2) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * out = ggml_kvarn_materialize(ctx,
            value ? ts.v_rec : ts.k_rec,
            value ? ts.v_stage : ts.k_stage,
            ts.idx_full, int(n_kv), kbits, vbits, value, S, 1, int(sealed));
    ggml_build_forward_expand(gf, out);
    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, NTH);
    CHECK(st == GGML_STATUS_SUCCESS, "materialize graph compute failed: %d", int(st));
    return out;
}

// NMSE of the materialized range [p0, p1) against the originals.
static double nmse_range(
        const std::vector<float> & orig,
        const struct ggml_tensor * mat,
        int64_t p0, int64_t p1) {
    const ggml_fp16_t * md = (const ggml_fp16_t *) mat->data;
    double se = 0.0;
    double sr = 0.0;
    for (int64_t p = p0; p < p1; ++p) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t d = 0; d < D; ++d) {
                const float x = orig[size_t(d + h * D + p * D * H)];
                const float y = ggml_fp16_to_fp32(md[(h * D + d) + p * D * H]);
                se += double(x - y) * double(x - y);
                sr += double(x) * double(x);
            }
        }
    }
    return sr > 0.0 ? se / sr : 0.0;
}

// checks that the materialized range is a bit-exact F16 copy of the originals
// (used for sink rows and the incomplete live group, which live in the stage).
static bool exact_f16_range(
        const std::vector<float> & orig,
        const struct ggml_tensor * mat,
        int64_t p0, int64_t p1) {
    const ggml_fp16_t * md = (const ggml_fp16_t *) mat->data;
    for (int64_t p = p0; p < p1; ++p) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t d = 0; d < D; ++d) {
                const ggml_fp16_t want = ggml_fp32_to_fp16(orig[size_t(d + h * D + p * D * H)]);
                if (md[(h * D + d) + p * D * H] != want) {
                    return false;
                }
            }
        }
    }
    return true;
}

static void test_layout(void) {
    const struct ggml_kvarn_layout l = ggml_kvarn_layout_make(D, G, 4, 2);
    CHECK(l.k_payload_off   == 0,     "k_payload_off %zu", l.k_payload_off);
    CHECK(l.k_payload_bytes == 8192,  "k_payload_bytes %zu", l.k_payload_bytes);
    CHECK(l.k_scale_off     == 8192,  "k_scale_off %zu", l.k_scale_off);
    CHECK(l.k_zp_off        == 8448,  "k_zp_off %zu", l.k_zp_off);
    CHECK(l.k_other_off     == 8704,  "k_other_off %zu", l.k_other_off);
    CHECK(l.v_payload_off   == 8960,  "v_payload_off %zu", l.v_payload_off);
    CHECK(l.v_payload_bytes == 4096,  "v_payload_bytes %zu", l.v_payload_bytes);
    CHECK(l.v_other_off     == 13056, "v_other_off %zu", l.v_other_off);
    CHECK(l.v_scale_off     == 13312, "v_scale_off %zu", l.v_scale_off);
    CHECK(l.v_zp_off        == 13568, "v_zp_off %zu", l.v_zp_off);
    CHECK(l.record_bytes    == 13824, "record_bytes %zu", l.record_bytes);
}

static void test_sink(void) {
    struct ggml_init_params p = { 64 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(p);

    const int kbits = 4, vbits = 2, iters = 16;
    struct tensors ts = make_tensors(ctx, 128, kbits, vbits);
    std::vector<float> ok, ov;
    fill_originals(ok, ov, 128);

    set_batch(ts, ok, ov, 0, 128);
    compute_store(ctx, ts, kbits, vbits, iters, -1, 0);

    struct ggml_tensor * mk = compute_materialize(ctx, ts, 128, kbits, vbits, 0, false);
    struct ggml_tensor * mv = compute_materialize(ctx, ts, 128, kbits, vbits, 0, true);

    CHECK(exact_f16_range(ok, mk, 0, 128), "K sink rows are not exact");
    CHECK(exact_f16_range(ov, mv, 0, 128), "V sink rows are not exact");

    ggml_free(ctx);
}

static void test_one_group(void) {
    struct ggml_init_params p = { 64 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(p);

    const int kbits = 4, vbits = 2, iters = 16;
    struct tensors ts = make_tensors(ctx, 128, kbits, vbits);
    std::vector<float> ok, ov;
    fill_originals(ok, ov, 384);

    // sink first
    set_batch(ts, ok, ov, 0, 128);
    compute_store(ctx, ts, kbits, vbits, iters, -1, 0);
    // then one full group (positions 128..255)
    set_batch(ts, ok, ov, 128, 128);
    compute_store(ctx, ts, kbits, vbits, iters, -1, 0);
    // ring stage: processing group 2 evicts group 1 -> it gets sealed
    set_batch(ts, ok, ov, 256, 128);
    compute_store(ctx, ts, kbits, vbits, iters, 1, 0);

    struct ggml_tensor * mk = compute_materialize(ctx, ts, 256, kbits, vbits, 1, false);
    struct ggml_tensor * mv = compute_materialize(ctx, ts, 256, kbits, vbits, 1, true);

    CHECK(exact_f16_range(ok, mk, 0, 128), "K sink disturbed by group seal");
    const double nk = nmse_range(ok, mk, 128, 256);
    const double nv = nmse_range(ov, mv, 128, 256);
    printf("one-group NMSE: K(%d-bit)=%.5f V(%d-bit)=%.5f\n", kbits, nk, vbits, nv);
    CHECK(nk < 0.10, "K 4-bit NMSE too high: %.5f", nk);
    CHECK(nv < 0.40, "V 2-bit NMSE too high: %.5f", nv);

    ggml_free(ctx);
}

// Ring stage: with S slots, a group stays F16 (exact) until S-1 newer groups
// arrive; only then is it sealed and quantized.
static void test_ring(void) {
    struct ggml_init_params p = { 256 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(p);

    const int kbits = 4, vbits = 2, iters = 16, S = 4;
    const int64_t CHUNK = 4;
    const int64_t N_TOTAL = 768;   // 0..511 in small chunks, then one 256-token call
    const int64_t BIG0 = 512;

    g_rng = 0xb0a71001u;
    // two batch shapes sharing one cache state (the store processes the tensor's
    // full T, so the small chunks need a T=CHUNK tensor set)
    struct tensors ts_small = make_tensors(ctx, CHUNK, kbits, vbits, S);
    struct tensors ts       = make_tensors(ctx, 256,   kbits, vbits, S);
    ts.k_stage  = ts_small.k_stage;
    ts.v_stage  = ts_small.v_stage;
    ts.k_rec    = ts_small.k_rec;
    ts.v_rec    = ts_small.v_rec;
    ts.idx_full = ts_small.idx_full;
    std::vector<float> ok, ov;
    fill_originals(ok, ov, N_TOTAL);

    int32_t owner  = -1;
    int32_t sealed = 0;
    auto mirror = [&](int64_t pos, int64_t n) {
        int32_t g_first = -1;
        for (int64_t k = 0; k < n; ++k) {
            if (pos + k >= G) { g_first = int32_t((pos + k) / G); break; }
        }
        if (g_first >= 1) {
            sealed = std::max(sealed, g_first - 1);
        }
        if (pos + n - 1 >= G) {
            owner = int32_t((pos + n - 1) / G);
        }
    };

    for (int64_t pos = 0; pos < BIG0; pos += CHUNK) {
        set_batch(ts_small, ok, ov, pos, CHUNK);
        compute_store(ctx, ts_small, kbits, vbits, iters, owner, sealed, S);
        mirror(pos, CHUNK);
    }
    // one larger call: groups 4 and 5 (pos 512..767); at its start groups 1..3
    // are sealed, the call's own groups stay in the ring (S=4 slots)
    set_batch(ts, ok, ov, BIG0, 256);
    compute_store(ctx, ts, kbits, vbits, iters, owner, sealed, S);
    mirror(BIG0, 256);

    struct ggml_tensor * mk = compute_materialize(ctx, ts, N_TOTAL, kbits, vbits, sealed, false, S);
    struct ggml_tensor * mv = compute_materialize(ctx, ts, N_TOTAL, kbits, vbits, sealed, true, S);

    // sealed = 3: groups 1..3 quantized, groups 4..5 (the current call) in the ring
    CHECK(exact_f16_range(ok, mk, 0, 128), "ring: K sink not exact");
    CHECK(exact_f16_range(ov, mv, 0, 128), "ring: V sink not exact");
    CHECK(exact_f16_range(ok, mk, 512, 768), "ring: K current-call groups not exact");
    CHECK(exact_f16_range(ov, mv, 512, 768), "ring: V current-call groups not exact");
    const double nk = nmse_range(ok, mk, 128, 256);
    const double nv = nmse_range(ov, mv, 128, 256);
    const double nk3 = nmse_range(ok, mk, 384, 512);
    const double nv3 = nmse_range(ov, mv, 384, 512);
    printf("ring NMSE: K4 g1=%.5f g3=%.5f | V2 g1=%.5f g3=%.5f (sealed=%d, S=%d)\n",
            nk, nk3, nv, nv3, sealed, S);
    CHECK(nk < 0.10 && nk3 < 0.10, "ring: K NMSE too high: %.5f %.5f", nk, nk3);
    CHECK(nv < 0.40 && nv3 < 0.40, "ring: V NMSE too high: %.5f %.5f", nv, nv3);

    ggml_free(ctx);
}

static void test_bits_ordering(void) {
    struct ggml_init_params p = { 64 * 1024 * 1024, nullptr, false };

    double nk2 = 0.0, nv2 = 0.0, nk6 = 0.0, nv6 = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const int bits = pass == 0 ? 2 : 6;
        g_rng = 0xdeadbeefu; // same data both passes
        struct ggml_context * ctx = ggml_init(p);
        struct tensors ts = make_tensors(ctx, 128, bits, bits);
        std::vector<float> ok, ov;
        fill_originals(ok, ov, 384);

        set_batch(ts, ok, ov, 128, 128);
        compute_store(ctx, ts, bits, bits, 16, -1, 0);
        set_batch(ts, ok, ov, 256, 128);
        compute_store(ctx, ts, bits, bits, 16, 1, 0);
        struct ggml_tensor * mk = compute_materialize(ctx, ts, 256, bits, bits, 1, false);
        struct ggml_tensor * mv = compute_materialize(ctx, ts, 256, bits, bits, 1, true);
        const double nk = nmse_range(ok, mk, 128, 256);
        const double nv = nmse_range(ov, mv, 128, 256);
        printf("bits-ordering %d-bit NMSE: K=%.5f V=%.5f\n", bits, nk, nv);
        if (pass == 0) { nk2 = nk; nv2 = nv; } else { nk6 = nk; nv6 = nv; }
        ggml_free(ctx);
    }
    CHECK(nk6 <= nk2, "6-bit K NMSE (%.5f) worse than 2-bit (%.5f)", nk6, nk2);
    CHECK(nv6 <= nv2, "6-bit V NMSE (%.5f) worse than 2-bit (%.5f)", nv6, nv2);
}

static bool dump_tensor(const char * dir, const char * name, const struct ggml_tensor * t) {
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE * f = std::fopen(path, "wb");
    if (f == nullptr) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    const size_t n = ggml_nbytes(t);
    const size_t w = std::fwrite(t->data, 1, n, f);
    std::fclose(f);
    return w == n;
}

static bool dump_buffer(const char * dir, const char * name, const void * data, size_t n) {
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE * f = std::fopen(path, "wb");
    if (f == nullptr) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    const size_t w = std::fwrite(data, 1, n, f);
    std::fclose(f);
    return w == n;
}

static void test_decode_sequence(const char * dump_dir) {
    struct ggml_init_params p = { 128 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(p);

    const int kbits = 4, vbits = 2, iters = 16;
    const int64_t CHUNK = 4;
    const int64_t N_TOTAL = 500;

    g_rng = 0xc0ffee01u;
    struct tensors ts = make_tensors(ctx, CHUNK, kbits, vbits);
    std::vector<float> ok, ov;
    fill_originals(ok, ov, N_TOTAL);

    int32_t owner  = -1;
    int32_t sealed = 0;
    for (int64_t pos = 0; pos < N_TOTAL; pos += CHUNK) {
        set_batch(ts, ok, ov, pos, CHUNK);
        compute_store(ctx, ts, kbits, vbits, iters, owner, sealed);
        // mirror the runtime's post-call state: groups older than this call's
        // first group are sealed at the start of the call (one-call deferral)
        int32_t g_first = -1;
        for (int64_t k = 0; k < CHUNK; ++k) {
            if (pos + k >= G) { g_first = int32_t((pos + k) / G); break; }
        }
        if (g_first >= 1) {
            sealed = std::max(sealed, g_first - 1);
        }
        const int64_t pos_last = pos + CHUNK - 1;
        if (pos_last >= G) {
            owner = int32_t(pos_last / G);
        }
    }

    struct ggml_tensor * mk = compute_materialize(ctx, ts, N_TOTAL, kbits, vbits, sealed, false);
    struct ggml_tensor * mv = compute_materialize(ctx, ts, N_TOTAL, kbits, vbits, sealed, true);

    CHECK(exact_f16_range(ok, mk, 0, 128), "decode: K sink not exact");
    CHECK(exact_f16_range(ov, mv, 0, 128), "decode: V sink not exact");
    CHECK(exact_f16_range(ok, mk, 384, N_TOTAL), "decode: K live group not exact");
    CHECK(exact_f16_range(ov, mv, 384, N_TOTAL), "decode: V live group not exact");

    const double nk1 = nmse_range(ok, mk, 128, 256);
    const double nv1 = nmse_range(ov, mv, 128, 256);
    const double nk2 = nmse_range(ok, mk, 256, 384);
    const double nv2 = nmse_range(ov, mv, 256, 384);
    printf("decode NMSE: K4 g1=%.5f g2=%.5f | V2 g1=%.5f g2=%.5f\n", nk1, nk2, nv1, nv2);
    CHECK(nk1 < 0.10 && nk2 < 0.10, "decode K NMSE too high: %.5f %.5f", nk1, nk2);
    CHECK(nv1 < 0.40 && nv2 < 0.40, "decode V NMSE too high: %.5f %.5f", nv1, nv2);

    // determinism
    struct ggml_tensor * mk2 = compute_materialize(ctx, ts, N_TOTAL, kbits, vbits, sealed, false);
    CHECK(memcmp(mk->data, mk2->data, ggml_nbytes(mk)) == 0, "materialize not deterministic");

    if (dump_dir != nullptr) {
        bool ok_dump = true;
        ok_dump &= dump_buffer(dump_dir, "orig_k.f32", ok.data(), ok.size() * sizeof(float));
        ok_dump &= dump_buffer(dump_dir, "orig_v.f32", ov.data(), ov.size() * sizeof(float));
        ok_dump &= dump_tensor(dump_dir, "k_stage.f16", ts.k_stage);
        ok_dump &= dump_tensor(dump_dir, "v_stage.f16", ts.v_stage);
        ok_dump &= dump_tensor(dump_dir, "k_records.i8", ts.k_rec);
        ok_dump &= dump_tensor(dump_dir, "v_records.i8", ts.v_rec);
        ok_dump &= dump_tensor(dump_dir, "k_mat.f16", mk);
        ok_dump &= dump_tensor(dump_dir, "v_mat.f16", mv);
        char manifest[512];
        const int n = std::snprintf(manifest, sizeof(manifest),
                "D=%lld H=%lld n_kv=%lld chunk=%lld k_bits=%d v_bits=%d iters=%d stage_groups=2 owner=%lld alpha=1.00\n",
                (long long) D, (long long) H, (long long) N_TOTAL, (long long) CHUNK,
                kbits, vbits, iters, (long long) owner);
        ok_dump &= n > 0 && dump_buffer(dump_dir, "manifest.txt", manifest, size_t(n));
        CHECK(ok_dump, "dump to %s failed", dump_dir);
    }

    ggml_free(ctx);
}

// 128-wide channel slices: a head_dim > 128 model is stored as independent
// 128-wide slices. Verify the full-width materialized span against per-slice
// roundtrips (the semantics the hook harness and the quality tables used).
static void test_slices(void) {
    struct ggml_init_params p = { 256 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(p);

    const int64_t Dfull = 256, H = 2, G = 128;
    const int SLICES = 2, Ds = 128;
    const int kbits = 4, vbits = 4, iters = 16, S = 4;
    const int64_t NG = 4;
    const int64_t N = 512;
    const struct ggml_kvarn_layout l = ggml_kvarn_layout_make(Ds, G, kbits, vbits);

    struct ggml_tensor * idx_a   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 384);
    struct ggml_tensor * idx_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 128);
    struct ggml_tensor * idx512  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    struct ggml_tensor * k_cur_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dfull, H, 384);
    struct ggml_tensor * v_cur_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dfull, H, 384);
    struct ggml_tensor * k_cur_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dfull, H, 128);
    struct ggml_tensor * v_cur_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dfull, H, 128);
    struct ggml_tensor * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, Ds, int64_t(S)*G, H*SLICES);
    struct ggml_tensor * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, Ds, int64_t(S)*G, H*SLICES);
    struct ggml_tensor * k_rec  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, int64_t(l.record_bytes), NG*H*SLICES);
    struct ggml_tensor * v_rec  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, int64_t(l.record_bytes), NG*H*SLICES);
    for (int64_t i = 0; i < N; ++i) { ((int32_t *) idx512->data)[i] = int32_t(i); }

    // f16-rounded originals [Dfull, H, N] (the stage stores F16)
    std::vector<float> ok(size_t(Dfull)*H*N), ov(size_t(Dfull)*H*N);
    g_rng = 0x51ce5001u;
    for (size_t i = 0; i < ok.size(); ++i) {
        float x = rng_float() * 0.7f;
        if ((i * 7) % 37 == 0) { x *= 9.0f; }
        ok[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
        ov[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(rng_float() * 0.7f));
    }

    auto set_batch = [&](int64_t pos0, int64_t n, struct ggml_tensor * idx_t,
                         struct ggml_tensor * kc, struct ggml_tensor * vc) {
        for (int64_t t = 0; t < n; ++t) {
            ((int32_t *) idx_t->data)[t] = int32_t(pos0 + t);
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t d = 0; d < Dfull; ++d) {
                    ((float *) kc->data)[d + h*Dfull + t*Dfull*H] = ok[size_t(d + h*Dfull + (pos0+t)*Dfull*H)];
                    ((float *) vc->data)[d + h*Dfull + t*Dfull*H] = ov[size_t(d + h*Dfull + (pos0+t)*Dfull*H)];
                }
            }
        }
        struct ggml_cgraph * gfc = ggml_new_graph(ctx);
        struct ggml_tensor * sk = ggml_kvarn_store(ctx, kc, idx_t, k_stage, k_rec,
                kbits, vbits, iters, false, S, SLICES, -1, 0, -1, 1.0f);
        struct ggml_tensor * sv = ggml_kvarn_store(ctx, vc, idx_t, v_stage, v_rec,
                kbits, vbits, iters, true, S, SLICES, -1, 0, -1, 1.0f);
        ggml_build_forward_expand(gfc, sk);
        ggml_build_forward_expand(gfc, sv);
        CHECK(ggml_graph_compute_with_ctx(ctx, gfc, NTH) == GGML_STATUS_SUCCESS, "slices: store compute failed");
    };

    // call A: sink + groups 1..2 (pos 0..383); call B: group 3 -> seals 1..2
    set_batch(0,   384, idx_a, k_cur_a, v_cur_a);
    set_batch(384, 128, idx_b, k_cur_b, v_cur_b);

    struct ggml_tensor * mk;
    struct ggml_tensor * mv;
    {
        struct ggml_cgraph * gfm = ggml_new_graph(ctx);
        mk = ggml_kvarn_materialize(ctx, k_rec, k_stage, idx512, N, kbits, vbits, false, S, SLICES, 2);
        mv = ggml_kvarn_materialize(ctx, v_rec, v_stage, idx512, N, kbits, vbits, true,  S, SLICES, 2);
        ggml_build_forward_expand(gfm, mk);
        ggml_build_forward_expand(gfm, mv);
        CHECK(ggml_graph_compute_with_ctx(ctx, gfm, NTH) == GGML_STATUS_SUCCESS, "slices: materialize compute failed");
    }

    auto mat_at = [&](const struct ggml_tensor * t, int64_t d_full, int64_t h, int64_t i) {
        // native layout: element (d,i,h) at (h*Dfull + d) + i*Dfull*H
        return ((const ggml_fp16_t *) t->data)[(h*Dfull + d_full) + i*Dfull*H];
    };

    // sink and the live group (3) must be exact F16 copies
    int64_t n_exact = 0;
    for (int64_t pos = 0; pos < 128; ++pos) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t d = 0; d < Dfull; ++d) {
                const ggml_fp16_t wantk = ggml_fp32_to_fp16(ok[size_t(d + h*Dfull + pos*Dfull*H)]);
                const ggml_fp16_t wantv = ggml_fp32_to_fp16(ov[size_t(d + h*Dfull + pos*Dfull*H)]);
                if (mat_at(mk, d, h, pos) != wantk) { ++n_exact; }
                if (mat_at(mv, d, h, pos) != wantv) { ++n_exact; }
            }
        }
    }
    CHECK(n_exact == 0, "slices: sink not exact (%lld mismatches)", (long long) n_exact);
    n_exact = 0;
    for (int64_t pos = 384; pos < 512; ++pos) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t d = 0; d < Dfull; ++d) {
                const ggml_fp16_t wantk = ggml_fp32_to_fp16(ok[size_t(d + h*Dfull + pos*Dfull*H)]);
                const ggml_fp16_t wantv = ggml_fp32_to_fp16(ov[size_t(d + h*Dfull + pos*Dfull*H)]);
                if (mat_at(mk, d, h, pos) != wantk) { ++n_exact; }
                if (mat_at(mv, d, h, pos) != wantv) { ++n_exact; }
            }
        }
    }
    CHECK(n_exact == 0, "slices: live group not exact (%lld mismatches)", (long long) n_exact);

    // sealed groups 1..2: compare against per-slice roundtrips of the same input
    int n_mism = 0;
    double nk = 0.0, nv = 0.0, rk = 0.0, rv = 0.0;
    for (int64_t g = 1; g <= 2; ++g) {
        for (int64_t h = 0; h < H; ++h) {
            for (int s = 0; s < SLICES; ++s) {
                { // K slice: tile [Ds, G]
                    std::vector<float> tile(size_t(Ds)*G), back(size_t(Ds)*G);
                    for (int64_t r = 0; r < Ds; ++r) {
                        for (int64_t c = 0; c < G; ++c) {
                            const size_t src_i = size_t(h*Dfull + s*Ds + r) + size_t(g*G + c)*Dfull*H;
                            tile[r*G + c] = ok[src_i];
                            nk += double(ok[src_i]) * double(ok[src_i]);
                        }
                    }
                    ggml_kvarn_roundtrip_tile(tile.data(), Ds, G, kbits, iters, 1.0f, false, back.data());
                    for (int64_t r = 0; r < Ds; ++r) {
                        for (int64_t c = 0; c < G; ++c) {
                            const ggml_fp16_t want = ggml_fp32_to_fp16(back[r*G + c]);
                            if (mat_at(mk, s*Ds + r, h, g*G + c) != want) { ++n_mism; }
                            const double d = double(tile[r*G + c]) - double(back[r*G + c]);
                            rk += d * d;
                        }
                    }
                }
                { // V slice: tile [G, Ds]
                    std::vector<float> tile(size_t(G)*Ds), back(size_t(G)*Ds);
                    for (int64_t t = 0; t < G; ++t) {
                        for (int64_t c = 0; c < Ds; ++c) {
                            const size_t src_i = size_t(h*Dfull + s*Ds + c) + size_t(g*G + t)*Dfull*H;
                            tile[t*Ds + c] = ov[src_i];
                            nv += double(ov[src_i]) * double(ov[src_i]);
                        }
                    }
                    ggml_kvarn_roundtrip_tile(tile.data(), G, Ds, vbits, iters, 1.0f, true, back.data());
                    for (int64_t t = 0; t < G; ++t) {
                        for (int64_t c = 0; c < Ds; ++c) {
                            const ggml_fp16_t want = ggml_fp32_to_fp16(back[t*Ds + c]);
                            if (mat_at(mv, s*Ds + c, h, g*G + t) != want) { ++n_mism; }
                            const double d = double(tile[t*Ds + c]) - double(back[t*Ds + c]);
                            rv += d * d;
                        }
                    }
                }
            }
        }
    }
    printf("slices: sealed NMSE K=%.5f V=%.5f, mismatches=%d\n",
            rk / nk, rv / std::max(nv, 1e-30), n_mism);
    CHECK(n_mism == 0, "slices: sealed groups differ from per-slice roundtrip (%d)", n_mism);
    CHECK(rk / nk < 0.10, "slices: K NMSE too high: %.5f", rk / nk);
    CHECK(rv / std::max(nv, 1e-30) < 0.40, "slices: V NMSE too high: %.5f", rv / std::max(nv, 1e-30));

    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * dump_dir = nullptr;
    const char * only = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            only = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--dump-dir DIR] [--only layout|sink|group|ring|bits|decode|slices]\n", argv[0]);
            return 1;
        }
    }

    auto want = [&](const char * name) { return only == nullptr || std::strcmp(only, name) == 0; };

    if (want("layout")) { printf("== layout\n"); fflush(stdout); test_layout(); }
    if (want("sink"))   { printf("== sink\n");   fflush(stdout); test_sink(); }
    if (want("group"))  { printf("== group\n");  fflush(stdout); test_one_group(); }
    if (want("ring"))   { printf("== ring\n");   fflush(stdout); test_ring(); }
    if (want("bits"))   { printf("== bits\n");   fflush(stdout); test_bits_ordering(); }
    if (want("decode")) { printf("== decode\n"); fflush(stdout); test_decode_sequence(dump_dir); }
    if (want("slices")) { printf("== slices\n"); fflush(stdout); test_slices(); }

    if (n_fail == 0) {
        printf("test-kvarn: all tests passed\n");
        return 0;
    }
    fprintf(stderr, "test-kvarn: %d failure(s)\n", n_fail);
    return 1;
}
