// KVarN: variance-normalized KV-cache quantization — CPU reference kernels.
//
// This file implements the structured KVarN cache format and the CPU kernels
// backing GGML_OP_KVARN_STORE and GGML_OP_KVARN_MATERIALIZE. The format follows
// the Huawei reference implementation (github.com/huawei-csl/KVarN) and the
// llama.cpp-family port (Anbeeld/beellama.cpp), with two deliberate choices:
//
//   * the per-head Hadamard rotation is NOT applied here. ik_llama.cpp already
//     applies the head-wise Hadamard to K (together with Q) and to V (with the
//     post-attention inverse) via -khad/-vhad, so records hold rotated values
//     and attention consumes them in the rotated domain.
//
//   * one record covers one 128-token group of one KV head, for K and V
//     together. Record layout (all side metadata F16), 8-byte aligned:
//
//       [ K payload | k_s_col[D] | k_zp[D] | k_s_row[G] |
//         V payload | v_s_col[D] | v_s_row[G] | v_zp[G] ]
//
//     (reference KVarN cache layout; v_s_col is the raw per-channel VarN
//     scale, v_s_row/v_zp are the absorbed per-token scale/zero point.)
//     Dequantization:
//       K[r,c] = (code * k_s_col[r] + k_zp[r]) * k_s_row[c]   tile [D, G]
//       V[r,c] = (code * v_s_row[r] + v_zp[r]) * v_s_col[c]   tile [G, D]
//
//     with r = channel, c = token for K, and r = token, c = channel for V.
//     k_s_col/k_zp and v_s_row/v_zp absorb the VarN row scale; the remaining
//     VarN axis (token scale for K, channel scale for V) is stored raw.
//
//   * positions < 128 are the permanent sink: they live in stage block 0 in
//     F16 and are never quantized. The currently incomplete group lives in
//     stage block 1; a group is sealed into `records` when its last token
//     arrives. This matches the reference's sink + in-progress tile policy.
//
// v1 constraints (enforced by callers, asserted here):
//   * one contiguous token stream per cache (no gaps, no rewinds),
//   * stage_groups == 2 (sink block + live block),
//   * head_dim in {64, 128}.

#include "ggml-kvarn.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// The op_params layout (GGML_KVARN_OP_*) lives in ggml-kvarn.h so the CUDA
// dispatch can share it.

// Fail closed when a tensor does not live in host memory: with a mixed
// offload the scheduler can hand this CPU op a device-resident cache tensor,
// and silently writing the host-side copy would lose the cache update.
static void kvarn_require_host(const struct ggml_tensor * t, const char * what) {
    if (t->buffer != nullptr && !ggml_backend_buffer_is_host(t->buffer)) {
        GGML_ABORT("KVarN: %s is not host-resident (mixed CPU/GPU KV cache offload is unsupported)", what);
    }
}

static size_t kvarn_packed_bytes(int64_t n, int bits) {
    GGML_ASSERT(n >= 0);
    GGML_ASSERT(bits >= 2 && bits <= 8);
    return (size_t(n) * size_t(bits) + 7) / 8;
}

static void kvarn_pack_bits(const uint8_t * src, int64_t n, int bits, uint8_t * dst) {
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(dst != nullptr);

    std::memset(dst, 0, kvarn_packed_bytes(n, bits));

    const uint8_t mask = uint8_t((1u << bits) - 1u);
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t value = uint8_t(src[i]) & mask;
        const size_t bit_offset = size_t(i) * size_t(bits);
        for (int b = 0; b < bits; ++b) {
            const size_t dst_bit = bit_offset + size_t(b);
            dst[dst_bit / 8] |= uint8_t(((value >> b) & 1u) << (dst_bit % 8));
        }
    }
}

static uint8_t kvarn_unpack_bits(const uint8_t * src, int64_t index, int bits) {
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(index >= 0);

    uint8_t value = 0;
    const size_t bit_offset = size_t(index) * size_t(bits);
    for (int b = 0; b < bits; ++b) {
        const size_t src_bit = bit_offset + size_t(b);
        value |= uint8_t(((src[src_bit / 8] >> (src_bit % 8)) & 1u) << b);
    }
    return value;
}

static void kvarn_f16_store(uint8_t * record, size_t off, int64_t index, float value) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(value);
    std::memcpy(record + off + size_t(index) * sizeof(h), &h, sizeof(h));
}

static float kvarn_f16_load(const uint8_t * record, size_t off, int64_t index) {
    ggml_fp16_t h;
    std::memcpy(&h, record + off + size_t(index) * sizeof(h), sizeof(h));
    return ggml_fp16_to_fp32(h);
}

// ---------------------------------------------------------------------------
// variance normalization (Sinkhorn-like, log domain, best-so-far)
// ---------------------------------------------------------------------------

static float kvarn_std_stride(const float * values, int64_t n, int64_t stride) {
    double sum    = 0.0;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double x = values[i * stride];
        sum    += x;
        sum_sq += x * x;
    }
    const double mean = sum / double(n);
    const double var  = std::max(0.0, (sum_sq - double(n) * mean * mean) / double(std::max<int64_t>(n - 1, 1)));
    return float(std::sqrt(var));
}

// per-column std of a rows x cols tile, accumulated row-major (cache friendly)
static void kvarn_col_stds(const float * cur, int64_t rows, int64_t cols, float * out) {
    std::vector<double> sum(cols, 0.0);
    std::vector<double> sum_sq(cols, 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        const float * row = cur + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            const double x = row[c];
            sum[c]    += x;
            sum_sq[c] += x * x;
        }
    }
    for (int64_t c = 0; c < cols; ++c) {
        const double mean = sum[c] / double(rows);
        const double var  = std::max(0.0, (sum_sq[c] - double(rows) * mean * mean) / double(std::max<int64_t>(rows - 1, 1)));
        out[c] = float(std::sqrt(var));
    }
}

// imbalance of the current balanced tile; also fills col_std/row_std
static float kvarn_imbalance(const float * cur, int64_t rows, int64_t cols,
        std::vector<float> & col_std, std::vector<float> & row_std) {
    kvarn_col_stds(cur, rows, cols, col_std.data());

    float col_min = std::numeric_limits<float>::infinity();
    float col_max = 0.0f;
    for (int64_t c = 0; c < cols; ++c) {
        col_min = std::min(col_min, col_std[c]);
        col_max = std::max(col_max, col_std[c]);
    }

    float row_min = std::numeric_limits<float>::infinity();
    float row_max = 0.0f;
    for (int64_t r = 0; r < rows; ++r) {
        const float s = kvarn_std_stride(cur + r * cols, cols, 1);
        row_std[r] = s;
        row_min = std::min(row_min, s);
        row_max = std::max(row_max, s);
    }
    return col_max / std::max(col_min, 1e-8f) + row_max / std::max(row_min, 1e-8f);
}

static void kvarn_variance_normalize(
        const float * tile, int64_t rows, int64_t cols, int iters,
        std::vector<float> & balanced,
        std::vector<float> & s_row_best,
        std::vector<float> & s_col_best) {
    GGML_ASSERT(tile != nullptr);
    GGML_ASSERT(iters > 0);
    GGML_ASSERT(rows > 1 && cols > 1);

    std::vector<float> log_s_row(rows, 0.0f);
    std::vector<float> log_s_col(cols, 0.0f);
    std::vector<float> cur(tile, tile + rows * cols);
    std::vector<float> col_std(cols);
    std::vector<float> row_std(rows);
    std::vector<float> s_row_exp(rows);
    std::vector<float> s_col_exp(cols);

    s_row_best.assign(rows, 1.0f);
    s_col_best.assign(cols, 1.0f);
    float imbalance_best = kvarn_imbalance(cur.data(), rows, cols, col_std, row_std);

    // cur = tile / (s_col * s_row); the per-axis exps are precomputed once per
    // rebuild (the original recomputed exp(log_s_col[c]) for every element),
    // keeping the exact same arithmetic per element
    auto rebuild = [&]() {
        for (int64_t r = 0; r < rows; ++r) {
            s_row_exp[r] = std::exp(log_s_row[r]);
        }
        for (int64_t c = 0; c < cols; ++c) {
            s_col_exp[c] = std::exp(log_s_col[c]);
        }
        for (int64_t r = 0; r < rows; ++r) {
            const float sr = s_row_exp[r];
            const float * src = tile + r * cols;
            float * dst = cur.data() + r * cols;
            for (int64_t c = 0; c < cols; ++c) {
                dst[c] = src[c] / (s_col_exp[c] * sr);
            }
        }
    };

    for (int iter = 0; iter < iters; ++iter) {
        // column pass (row-major accumulation)
        kvarn_col_stds(cur.data(), rows, cols, col_std.data());
        for (int64_t c = 0; c < cols; ++c) {
            const float std = std::clamp(col_std[c], 1e-3f, 1e3f);
            log_s_col[c] = std::clamp(log_s_col[c] + std::log(std), -0.3f, 10.0f);
        }
        rebuild();

        // row pass
        for (int64_t r = 0; r < rows; ++r) {
            const float std = std::clamp(kvarn_std_stride(cur.data() + r * cols, cols, 1), 1e-3f, 1e3f);
            log_s_row[r] = std::clamp(log_s_row[r] + std::log(std), -0.3f, 10.0f);
        }
        rebuild();

        const float imbalance = kvarn_imbalance(cur.data(), rows, cols, col_std, row_std);
        if (imbalance <= imbalance_best) {
            imbalance_best = imbalance;
            for (int64_t r = 0; r < rows; ++r) {
                s_row_best[r] = s_row_exp[r]; // == exp(log_s_row[r]) after the rebuild
            }
            for (int64_t c = 0; c < cols; ++c) {
                s_col_best[c] = s_col_exp[c]; // == exp(log_s_col[c]) after the rebuild
            }
        }
    }

    balanced.resize(size_t(rows) * size_t(cols));
    for (int64_t r = 0; r < rows; ++r) {
        const float sr = s_row_best[r];
        const float * src = tile + r * cols;
        float * dst = balanced.data() + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            dst[c] = src[c] / (s_col_best[c] * sr);
        }
    }
}

// ---------------------------------------------------------------------------
// tile quantization / dequantization
// ---------------------------------------------------------------------------

static void kvarn_quantize_tile(
        const float * tile, int64_t rows, int64_t cols, int bits, int iters, float alpha,
        const struct ggml_kvarn_layout & layout, bool value, uint8_t * record) {
    GGML_ASSERT(tile != nullptr);
    GGML_ASSERT(record != nullptr);

    std::vector<float> balanced;
    std::vector<float> s_row;
    std::vector<float> s_col;
    kvarn_variance_normalize(tile, rows, cols, iters, balanced, s_row, s_col);

    // Experimental VarN damping: stored scales are s**alpha. alpha=1 is the
    // reference KVarN (default); alpha=0 collapses to Hadamard-only RTN.
    // Intermediate values balance uniform absolute error (good for
    // high-magnitude tokens) against uniform relative error (token scales);
    // see the KVarN P0 evaluation notes for the measured tradeoff.
    if (alpha != 1.0f) {
        for (int64_t r = 0; r < rows; ++r) {
            s_row[r] = std::pow(s_row[r], alpha);
        }
        for (int64_t c = 0; c < cols; ++c) {
            s_col[c] = std::pow(s_col[c], alpha);
        }
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                balanced[r * cols + c] = tile[r * cols + c] / (s_col[c] * s_row[r]);
            }
        }
    }

    std::vector<uint8_t> codes(size_t(rows) * size_t(cols));
    const float qmax = float((1 << bits) - 1);

    const size_t payload_off = value ? layout.v_payload_off : layout.k_payload_off;
    const size_t scale_off   = value ? layout.v_scale_off   : layout.k_scale_off;
    const size_t zp_off      = value ? layout.v_zp_off      : layout.k_zp_off;
    const size_t other_off   = value ? layout.v_other_off   : layout.k_other_off;

    for (int64_t r = 0; r < rows; ++r) {
        const float * beg = balanced.data() + r * cols;
        const float lo = *std::min_element(beg, beg + cols);
        const float hi = *std::max_element(beg, beg + cols);
        const float scale = std::max((hi - lo) / qmax, 1e-10f);

        for (int64_t c = 0; c < cols; ++c) {
            const float v = std::round((beg[c] - lo) / scale);
            codes[r * cols + c] = uint8_t(std::clamp(v, 0.0f, qmax));
        }

        // absorb the row VarN scale into the RTN scale / zero point
        const float absorb = s_row[r];
        kvarn_f16_store(record, scale_off, r, absorb * scale);
        kvarn_f16_store(record, zp_off,    r, absorb * lo);
    }

    for (int64_t i = 0; i < cols; ++i) {
        kvarn_f16_store(record, other_off, i, s_col[i]);
    }

    GGML_ASSERT(kvarn_packed_bytes(rows * cols, bits) <=
            (value ? layout.v_payload_bytes : layout.k_payload_bytes));
    kvarn_pack_bits(codes.data(), rows * cols, bits, record + payload_off);
}

static void kvarn_dequantize_tile(
        const uint8_t * record, int64_t rows, int64_t cols, int bits,
        const struct ggml_kvarn_layout & layout, bool value, float * tile) {
    GGML_ASSERT(record != nullptr);
    GGML_ASSERT(tile != nullptr);

    const size_t payload_off = value ? layout.v_payload_off : layout.k_payload_off;
    const size_t scale_off   = value ? layout.v_scale_off   : layout.k_scale_off;
    const size_t zp_off      = value ? layout.v_zp_off      : layout.k_zp_off;
    const size_t other_off   = value ? layout.v_other_off   : layout.k_other_off;

    for (int64_t r = 0; r < rows; ++r) {
        const float scale = kvarn_f16_load(record, scale_off, r);
        const float zp    = kvarn_f16_load(record, zp_off,    r);
        for (int64_t c = 0; c < cols; ++c) {
            const float other = kvarn_f16_load(record, other_off, c);
            const uint8_t q = kvarn_unpack_bits(record + payload_off, r * cols + c, bits);
            tile[r * cols + c] = (float(q) * scale + zp) * other;
        }
    }
}

// ---------------------------------------------------------------------------
// layout (public)
// ---------------------------------------------------------------------------

struct ggml_kvarn_layout ggml_kvarn_layout_make(int64_t head_dim, int64_t group, int64_t k_bits, int64_t v_bits) {
    GGML_ASSERT(head_dim > 1 && head_dim <= GGML_KVARN_MAX_DIM);
    GGML_ASSERT(group > 1 && group <= GGML_KVARN_GROUP);
    GGML_ASSERT(k_bits >= 2 && k_bits <= 8);
    GGML_ASSERT(v_bits >= 2 && v_bits <= 8);

    struct ggml_kvarn_layout layout = {};
    layout.group    = int32_t(group);
    layout.head_dim = int32_t(head_dim);

    size_t off = 0;
    layout.k_payload_off   = off; layout.k_payload_bytes = kvarn_packed_bytes(head_dim * group, int(k_bits)); off += layout.k_payload_bytes;
    layout.k_scale_off     = off; off += size_t(head_dim) * sizeof(uint16_t);
    layout.k_zp_off        = off; off += size_t(head_dim) * sizeof(uint16_t);
    layout.k_other_off     = off; off += size_t(group) * sizeof(uint16_t);

    layout.v_payload_off   = off; layout.v_payload_bytes = kvarn_packed_bytes(group * head_dim, int(v_bits)); off += layout.v_payload_bytes;
    layout.v_other_off     = off; off += size_t(head_dim) * sizeof(uint16_t); // raw per-channel VarN scale (reference: s_col_V)
    layout.v_scale_off     = off; off += size_t(group) * sizeof(uint16_t);    // absorbed per-token scale (s_row_V)
    layout.v_zp_off        = off; off += size_t(group) * sizeof(uint16_t);    // absorbed per-token zero point (zp_V)

    layout.record_bytes = (off + 7) & ~size_t(7);
    return layout;
}

// ---------------------------------------------------------------------------
// host-side helpers (offline analysis / debug hooks)
// ---------------------------------------------------------------------------

void ggml_kvarn_wht(float * values, int64_t n) {
    GGML_ASSERT(values != nullptr);
    GGML_ASSERT(n > 0 && (n & (n - 1)) == 0);

    for (int64_t stride = 1; stride < n; stride *= 2) {
        for (int64_t base = 0; base < n; base += 2 * stride) {
            for (int64_t i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i]          = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }

    const float inv = 1.0f / std::sqrt(float(n));
    for (int64_t i = 0; i < n; ++i) {
        values[i] *= inv;
    }
}

void ggml_kvarn_roundtrip_tile(
        const float * tile_in, int64_t rows, int64_t cols, int bits, int iters, float alpha,
        bool value, float * tile_out) {
    const int64_t head_dim = value ? cols : rows;
    const int64_t group    = value ? rows : cols;
    const struct ggml_kvarn_layout layout = ggml_kvarn_layout_make(head_dim, group, bits, bits);
    std::vector<uint8_t> record(layout.record_bytes);
    kvarn_quantize_tile(tile_in, rows, cols, bits, iters, alpha, layout, value, record.data());
    kvarn_dequantize_tile(record.data(), rows, cols, bits, layout, value, tile_out);
}

// ---------------------------------------------------------------------------
// op compute
// ---------------------------------------------------------------------------

static int32_t kvarn_op_int(const struct ggml_tensor * dst, int index) {
    int32_t value;
    std::memcpy(&value, (const char *) dst->op_params + index * sizeof(int32_t), sizeof(value));
    return value;
}

static float kvarn_op_float(const struct ggml_tensor * dst, int index) {
    float value;
    std::memcpy(&value, (const char *) dst->op_params + index * sizeof(int32_t), sizeof(value));
    return value;
}

static void kvarn_stage_write(
        ggml_fp16_t * stage, int64_t D, int64_t stage_rows, int64_t h,
        int64_t block, int64_t slot, const float * row) {
    ggml_fp16_t * dst = stage + (block * GGML_KVARN_GROUP + slot) * D + h * D * stage_rows;
    for (int64_t d = 0; d < D; ++d) {
        dst[d] = ggml_fp32_to_fp16(row[d]);
    }
}

// Ring stage: block 0 is the permanent sink, blocks 1..stage_slots hold the
// newest groups (group g lives in block 1 + (g-1) % stage_slots).
static int64_t kvarn_ring_block(int64_t g, int64_t stage_slots) {
    GGML_ASSERT(g >= 1 && stage_slots >= 1);
    return 1 + (g - 1) % stage_slots;
}

// Gather the tile of one group out of stage block `block` for head `h`.
// K tile: [D channels, G tokens]; V tile: [G tokens, D channels].
static void kvarn_gather_stage_tile(
        const ggml_fp16_t * stage, int64_t D, int64_t stage_rows, int64_t h,
        int64_t block, bool value, std::vector<float> & tile) {
    const int64_t rows = value ? GGML_KVARN_GROUP : D;
    const int64_t cols = value ? D : GGML_KVARN_GROUP;
    tile.resize(size_t(rows) * size_t(cols));

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const int64_t chan = value ? c : r; // K rows are channels, V columns are channels
            const int64_t slot = value ? r : c; // K columns are tokens, V rows are tokens
            const ggml_fp16_t * src = stage + (block * GGML_KVARN_GROUP + slot) * D + h * D * stage_rows;
            tile[r * cols + c] = ggml_fp16_to_fp32(src[chan]);
        }
    }
}

// One side (K or V) of the store: seals completed ring groups into records and
// stages the current rows. Shared by the single-side op and the combined K+V op.
static void kvarn_store_side(
        struct ggml_tensor * dst,
        const struct ggml_tensor * current, const struct ggml_tensor * indices,
        const struct ggml_tensor * stage, const struct ggml_tensor * records,
        bool value, int bits, int other_bits, int iters, float alpha,
        int64_t owner0, int64_t sealed0, int64_t g_first_arg, int64_t stage_groups, int n_slices,
        int ith, int nth) {
    GGML_ASSERT(current->type == GGML_TYPE_F32);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);
    GGML_ASSERT(stage->type   == GGML_TYPE_F16);
    GGML_ASSERT(records->type == GGML_TYPE_I8);
    GGML_ASSERT(indices->ne[0] == current->ne[2]);

    kvarn_require_host(current, "store current");
    kvarn_require_host(indices, "store indices");
    kvarn_require_host(stage,   "store stage");
    kvarn_require_host(records, "store records");

    const int64_t D = current->ne[0]; // full head dim
    const int64_t H = current->ne[1];
    const int64_t T = current->ne[2];
    const int64_t G = GGML_KVARN_GROUP;

    GGML_ASSERT(n_slices >= 1 && n_slices <= 4);
    const int64_t D_slice = stage->ne[0];
    const int64_t H_s = H * n_slices;
    GGML_ASSERT(D == D_slice * n_slices);
    GGML_ASSERT(stage->ne[2] == H_s);
    GGML_ASSERT(D_slice % 64 == 0 && D_slice <= GGML_KVARN_MAX_DIM);
    GGML_ASSERT(stage_groups >= 2);
    GGML_ASSERT(stage->ne[1] >= stage_groups * G);

    const int64_t stage_slots = stage_groups - 1;

    const struct ggml_kvarn_layout layout = ggml_kvarn_layout_make(
            D_slice, G, value ? other_bits : bits, value ? bits : other_bits);
    GGML_ASSERT(size_t(records->ne[0]) == layout.record_bytes);

    const int64_t stage_rows = stage->ne[1];
    ggml_fp16_t * stage_data = (ggml_fp16_t *) stage->data;
    const float * cur_data = (const float *) current->data;
    const int32_t * idx = (const int32_t *) indices->data;
    uint8_t * rec_data = (uint8_t *) records->data;

    const int64_t n_records = records->ne[1];

    std::vector<float> tile;

    // First group touched by this call (>= 1), or -1 when the call is
    // sink-only. Callers that know it pass it in op_params (G_FIRST) because
    // the CUDA dispatch cannot scan a device-resident indices tensor; -1
    // falls back to the scan below.
    int64_t g_first = g_first_arg;
    if (g_first < 0) {
        g_first = -1;
        for (int64_t t = 0; t < T; ++t) {
            if (idx[t] >= G) {
                g_first = idx[t] / G;
                break;
            }
        }
    }

    for (int64_t hh = ith; hh < H_s; hh += nth) {
        const int64_t h     = hh / n_slices;         // attention head
        const int64_t s     = hh % n_slices;         // 128-wide channel slice
        const int64_t chan0 = h * D + s * D_slice;   // channel offset within `current`

        int64_t owner  = owner0;
        int64_t sealed = sealed0;
        int64_t last_pos = -2; // no contiguity information for the first token

        auto seal = [&](int64_t g) {
            if (g < 1) {
                return; // the sink group is never sealed
            }
            const int64_t ridx = g * H_s + hh;
            GGML_ASSERT(ridx < n_records);
            kvarn_gather_stage_tile(stage_data, D_slice, stage_rows, hh, kvarn_ring_block(g, stage_slots), value, tile);
            kvarn_quantize_tile(tile.data(), value ? G : D_slice, value ? D_slice : G, bits, iters, alpha, layout, value,
                    rec_data + size_t(ridx) * layout.record_bytes);

            // debug: in-process NMSE of the just-sealed tile
            if (getenv("IK_KVARN_DEBUG") != nullptr && hh == 0) {
                const int64_t rows_d = value ? G : D_slice;
                const int64_t cols_d = value ? D_slice : G;
                std::vector<float> back(size_t(rows_d) * size_t(cols_d));
                kvarn_dequantize_tile(rec_data + size_t(ridx) * layout.record_bytes, rows_d, cols_d, bits, layout, value, back.data());
                double se = 0.0, sr = 0.0;
                for (size_t z = 0; z < back.size(); ++z) {
                    const double d = double(tile[z]) - double(back[z]);
                    se += d * d; sr += double(tile[z]) * double(tile[z]);
                }
                fprintf(stderr, "[store %s nmse] g=%lld %d-bit %.5f\n", value ? "V" : "K", (long long) g, bits, se / sr);
            }

            // debug: dump the tile and its record for offline NMSE analysis
            if (getenv("IK_KVARN_DUMP_TILE") != nullptr && hh == 0) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/%c_tile_last.f32", getenv("IK_KVARN_DUMP_TILE"), value ? 'v' : 'k');
                FILE * ft = fopen(path, "wb");
                if (ft != nullptr) { fwrite(tile.data(), sizeof(float), tile.size(), ft); fclose(ft); }
                snprintf(path, sizeof(path), "%s/%c_rec_last.bin", getenv("IK_KVARN_DUMP_TILE"), value ? 'v' : 'k');
                FILE * fr = fopen(path, "wb");
                if (fr != nullptr) { fwrite(rec_data + size_t(ridx) * layout.record_bytes, 1, layout.record_bytes, fr); fclose(fr); }
            }
        };

        // One-call deferral (streaming semantics): groups completed in PREVIOUS
        // calls are sealed now, before the ring blocks are reused. A group
        // completed in the CURRENT call keeps its block in F16 for this call's
        // attention; it is sealed at the start of the next call.
        if (g_first >= 1) {
            while (sealed + 1 < g_first) {
                seal(sealed + 1);
                ++sealed;
            }
        }

        for (int64_t t = 0; t < T; ++t) {
            const int64_t pos = idx[t];
            GGML_ASSERT(pos >= 0);

            if (last_pos >= 0 && pos != last_pos + 1) {
                GGML_ABORT("KVarN store: non-contiguous token stream (pos %lld after %lld)\n",
                        (long long) pos, (long long) last_pos);
            }
            last_pos = pos;

            if (pos < G) {
                // permanent sink: F16 stage block 0
                kvarn_stage_write(stage_data, D_slice, stage_rows, hh, /*block=*/0, pos,
                        cur_data + chan0 + t * D * H);
                continue;
            }

            const int64_t g = pos / G;
            const int64_t slot = pos % G;

            if (owner < 0) {
                owner = g;
            } else if (g != owner) {
                if (g < owner) {
                    GGML_ABORT("KVarN store: token stream went back from group %lld to group %lld\n",
                            (long long) owner, (long long) g);
                }
                owner = g;
            }

            kvarn_stage_write(stage_data, D_slice, stage_rows, hh, kvarn_ring_block(g, stage_slots), slot,
                    cur_data + chan0 + t * D * H);
        }

        if (getenv("IK_KVARN_DEBUG_VERBOSE") != nullptr && hh == 0) {
            fprintf(stderr, "[store] %s T=%lld owner0=%lld sealed0=%lld -> owner=%lld sealed=%lld pos=[%lld..%lld]\n",
                    value ? "V" : "K", (long long) T, (long long) owner0, (long long) sealed0,
                    (long long) owner, (long long) sealed,
                    (long long) (T > 0 ? idx[0] : -1), (long long) (T > 0 ? idx[T - 1] : -1));
        }
    }

    if (ith == 0) {
        ((float *) dst->data)[0] = 1.0f;
    }
}

void ggml_kvarn_compute_store(struct ggml_tensor * dst, int ith, int nth) {
    if (kvarn_op_int(dst, GGML_KVARN_OP_COMBINED) != 0) {
        // src[0]=k_cur src[1]=v_cur src[2]=indices
        // src[3]=k_stage src[4]=v_stage src[5]=k_records src[6]=v_records
        kvarn_store_side(dst, dst->src[0], dst->src[2], dst->src[3], dst->src[5],
                /*value=*/false, kvarn_op_int(dst, GGML_KVARN_OP_BITS), kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS),
                kvarn_op_int(dst, GGML_KVARN_OP_ITERS), kvarn_op_float(dst, GGML_KVARN_OP_ALPHA),
                kvarn_op_int(dst, GGML_KVARN_OP_OWNER), kvarn_op_int(dst, GGML_KVARN_OP_SEALED),
                kvarn_op_int(dst, GGML_KVARN_OP_G_FIRST), kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP),
                (int) kvarn_op_int(dst, GGML_KVARN_OP_SLICES),
                ith, nth);
        if (dst->src[1] != nullptr) {
            kvarn_store_side(dst, dst->src[1], dst->src[2], dst->src[4], dst->src[6],
                    /*value=*/true, kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS), kvarn_op_int(dst, GGML_KVARN_OP_BITS),
                    kvarn_op_int(dst, GGML_KVARN_OP_ITERS), kvarn_op_float(dst, GGML_KVARN_OP_ALPHA),
                    kvarn_op_int(dst, GGML_KVARN_OP_OWNER), kvarn_op_int(dst, GGML_KVARN_OP_SEALED),
                    kvarn_op_int(dst, GGML_KVARN_OP_G_FIRST), kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP),
                    (int) kvarn_op_int(dst, GGML_KVARN_OP_SLICES),
                    ith, nth);
        }
        return;
    }
    kvarn_store_side(dst, dst->src[0], dst->src[1], dst->src[2], dst->src[3],
            kvarn_op_int(dst, GGML_KVARN_OP_VALUE) != 0,
            kvarn_op_int(dst, GGML_KVARN_OP_BITS), kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS),
            kvarn_op_int(dst, GGML_KVARN_OP_ITERS), kvarn_op_float(dst, GGML_KVARN_OP_ALPHA),
            kvarn_op_int(dst, GGML_KVARN_OP_OWNER), kvarn_op_int(dst, GGML_KVARN_OP_SEALED),
            kvarn_op_int(dst, GGML_KVARN_OP_G_FIRST), kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP),
            (int) kvarn_op_int(dst, GGML_KVARN_OP_SLICES),
            ith, nth);
}

// One side (K or V) of the materialize: reconstructs the dense F16 span.
static void kvarn_materialize_side(
        struct ggml_tensor * out_dst,
        const struct ggml_tensor * records, const struct ggml_tensor * stage,
        const struct ggml_tensor * indices,
        bool value, int bits, int other_bits, int64_t sealed, int64_t stage_groups, int64_t n_kv, int n_slices,
        int ith, int nth) {
    GGML_ASSERT(records->type == GGML_TYPE_I8);
    GGML_ASSERT(stage->type   == GGML_TYPE_F16);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);
    GGML_ASSERT(out_dst->type == GGML_TYPE_F16);

    kvarn_require_host(records, "materialize records");
    kvarn_require_host(stage,   "materialize stage");
    kvarn_require_host(indices, "materialize indices");

    // Output layout mirrors the native F16 cache: one row per cell, holding
    // [head 0 channels | head 1 channels | ...] (element (d,t,h) at
    // d + D*h + D*H*t). The attention views it exactly like the cache, so the
    // FA kernel sees the same strides on both paths.
    const int64_t D_slice = stage->ne[0];
    const int64_t H_s = stage->ne[2];
    const int64_t G = GGML_KVARN_GROUP;

    GGML_ASSERT(n_slices >= 1 && n_slices <= 4);
    GGML_ASSERT(H_s % n_slices == 0);
    const int64_t H = H_s / n_slices;
    const int64_t D = D_slice * n_slices;

    GGML_ASSERT(D_slice % 64 == 0 && D_slice <= GGML_KVARN_MAX_DIM);
    GGML_ASSERT(indices->ne[0] >= n_kv);
    GGML_ASSERT(out_dst->ne[0] == D * H);
    GGML_ASSERT(stage_groups >= 2);
    GGML_ASSERT(stage->ne[1] >= stage_groups * G);

    const int64_t stage_slots = stage_groups - 1;

    const struct ggml_kvarn_layout layout = ggml_kvarn_layout_make(
            D_slice, G, value ? other_bits : bits, value ? bits : other_bits);
    GGML_ASSERT(size_t(records->ne[0]) == layout.record_bytes);

    const int64_t stage_rows = stage->ne[1];
    const ggml_fp16_t * stage_data = (const ggml_fp16_t *) stage->data;
    const int32_t * idx = (const int32_t *) indices->data;
    const uint8_t * rec_data = (const uint8_t *) records->data;
    ggml_fp16_t * out = (ggml_fp16_t *) out_dst->data;

    const int64_t rows = value ? G : D_slice;
    const int64_t cols = value ? D_slice : G;

    std::vector<float> tile;

    for (int64_t hh = ith; hh < H_s; hh += nth) {
        const int64_t h = hh / n_slices; // attention head
        const int64_t s = hh % n_slices; // 128-wide channel slice

        int64_t dequant_group = -1;
        bool tile_valid = false;

        for (int64_t i = 0; i < n_kv; ++i) {
            const int64_t pos = idx[i];
            GGML_ASSERT(pos >= 0);

            if (pos < G) {
                // sink: stage block 0
                const ggml_fp16_t * src = stage_data + pos * D_slice + hh * D_slice * stage_rows;
                for (int64_t d = 0; d < D_slice; ++d) {
                    out[(h * D + s * D_slice + d) + i * D * H] = src[d];
                }
                dequant_group = -1;
                tile_valid = false;
                continue;
            }

            const int64_t g = pos / G;
            const int64_t slot = pos % G;

            if (g > sealed) {
                // live group in the ring stage (F16, unquantized)
                const ggml_fp16_t * src = stage_data + (kvarn_ring_block(g, stage_slots) * G + slot) * D_slice + hh * D_slice * stage_rows;
                for (int64_t d = 0; d < D_slice; ++d) {
                    out[(h * D + s * D_slice + d) + i * D * H] = src[d];
                }
                dequant_group = -1;
                tile_valid = false;
                continue;
            }

            if (!tile_valid || dequant_group != g) {
                const int64_t ridx = g * H_s + hh;
                GGML_ASSERT(ridx < records->ne[1]);
                tile.resize(size_t(rows) * size_t(cols));
                kvarn_dequantize_tile(rec_data + size_t(ridx) * layout.record_bytes,
                        rows, cols, bits, layout, value, tile.data());
                dequant_group = g;
                tile_valid = true;
            }

            for (int64_t d = 0; d < D_slice; ++d) {
                const float v = value ? tile[slot * cols + d] : tile[d * cols + slot];
                out[(h * D + s * D_slice + d) + i * D * H] = ggml_fp32_to_fp16(v);
            }
        }

        if (getenv("IK_KVARN_DEBUG_VERBOSE") != nullptr && hh == 0) {
            fprintf(stderr, "[mat ] %s n_kv=%lld sealed=%lld\n", value ? "V" : "K", (long long) n_kv, (long long) sealed);
        }
    }
}

void ggml_kvarn_compute_materialize(struct ggml_tensor * dst, int ith, int nth) {
    if (kvarn_op_int(dst, GGML_KVARN_OP_COMBINED) != 0) {
        // src[0]=k_records src[1]=k_stage src[2]=indices
        // src[3]=v_records src[4]=v_stage src[5]=v_flat (side-effect output)
        const int64_t n_kv = dst->ne[1];
        kvarn_materialize_side(dst, dst->src[0], dst->src[1], dst->src[2],
                /*value=*/false, kvarn_op_int(dst, GGML_KVARN_OP_BITS), kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS),
                kvarn_op_int(dst, GGML_KVARN_OP_SEALED), kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP), n_kv,
                (int) kvarn_op_int(dst, GGML_KVARN_OP_SLICES),
                ith, nth);
        if (dst->src[5] != nullptr) {
            kvarn_materialize_side(dst->src[5], dst->src[3], dst->src[4], dst->src[2],
                    /*value=*/true, kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS), kvarn_op_int(dst, GGML_KVARN_OP_BITS),
                    kvarn_op_int(dst, GGML_KVARN_OP_SEALED), kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP), n_kv,
                    (int) kvarn_op_int(dst, GGML_KVARN_OP_SLICES),
                    ith, nth);
        }
        return;
    }
    kvarn_materialize_side(dst, dst->src[0], dst->src[1], dst->src[2],
            kvarn_op_int(dst, GGML_KVARN_OP_VALUE) != 0,
            kvarn_op_int(dst, GGML_KVARN_OP_BITS), kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS),
            kvarn_op_int(dst, GGML_KVARN_OP_SEALED), kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP),
            dst->ne[1], (int) kvarn_op_int(dst, GGML_KVARN_OP_SLICES), ith, nth);
}

// ---------------------------------------------------------------------------
// KVarN decode attention (GGML_OP_KVARN_ATTN)
//
// Single query token, no sinks/SWA/softcap (gated by the caller). Scores and
// PV read the sink/live rows from the F16 stage and the sealed groups from
// their records; the math mirrors the materialize + FA path exactly:
//   s_i   = kq_scale * dot(q, K_i) + mask_i
//   out_d = sum_i softmax(s)_i * V_i[d]
// op_params: [0]=n_kv [1]=sealed [2]=stage_groups [3]=k_bits [4]=v_bits
//            [5]=kq_scale (float bits)
// ---------------------------------------------------------------------------
void ggml_kvarn_compute_attn(struct ggml_tensor * dst, int ith, int nth) {
    const struct ggml_tensor * q       = dst->src[0]; // F32 [D_k, 1, n_head]
    const struct ggml_tensor * k_rec   = dst->src[1]; // I8
    const struct ggml_tensor * k_stage = dst->src[2]; // F16 [D_k, rows, n_head_kv]
    const struct ggml_tensor * v_rec   = dst->src[3]; // I8
    const struct ggml_tensor * v_stage = dst->src[4]; // F16 [D_v, rows, n_head_kv]
    const struct ggml_tensor * indices = dst->src[5]; // I32 [n_kv]
    const struct ggml_tensor * mask    = dst->src[6]; // F16 [n_kv, 1]

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k_rec->type == GGML_TYPE_I8 && v_rec->type == GGML_TYPE_I8);
    GGML_ASSERT(k_stage->type == GGML_TYPE_F16 && v_stage->type == GGML_TYPE_F16);
    GGML_ASSERT(indices->type == GGML_TYPE_I32 && mask->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(q->ne[1] == 1);

    kvarn_require_host(q, "attn q");
    kvarn_require_host(k_rec, "attn k_records");
    kvarn_require_host(k_stage, "attn k_stage");
    kvarn_require_host(v_rec, "attn v_records");
    kvarn_require_host(v_stage, "attn v_stage");
    kvarn_require_host(indices, "attn indices");
    kvarn_require_host(mask, "attn mask");

    const int64_t n_kv    = kvarn_op_int(dst, 0);
    const int64_t sealed  = kvarn_op_int(dst, 1);
    const int64_t stage_groups = kvarn_op_int(dst, 2);
    const int     k_bits  = kvarn_op_int(dst, 3);
    const int     v_bits  = kvarn_op_int(dst, 4);
    const float   kq_scale = kvarn_op_float(dst, 5);
    const int     n_slices = kvarn_op_int(dst, 6);

    const int64_t Dk = q->ne[0];            // full K head dim
    const int64_t Dk_slice = k_stage->ne[0];
    const int64_t Dv_slice = v_stage->ne[0];
    const int64_t Dv = Dv_slice * n_slices; // full V head dim
    const int64_t Hq = q->ne[2];
    GGML_ASSERT(n_slices >= 1 && n_slices <= 4);
    GGML_ASSERT(k_stage->ne[2] % n_slices == 0);
    const int64_t Hkv = k_stage->ne[2] / n_slices;
    const int64_t G = GGML_KVARN_GROUP;

    GGML_ASSERT(Hq % Hkv == 0);
    GGML_ASSERT(Dk == Dk_slice * n_slices);
    GGML_ASSERT(k_stage->ne[2] == v_stage->ne[2]);
    GGML_ASSERT(stage_groups >= 2 && indices->ne[0] >= n_kv);

    const int64_t slots = stage_groups - 1;

    const struct ggml_kvarn_layout k_layout = ggml_kvarn_layout_make(Dk_slice, G, k_bits, v_bits);
    const struct ggml_kvarn_layout v_layout = ggml_kvarn_layout_make(Dv_slice, G, k_bits, v_bits);
    GGML_ASSERT(size_t(k_rec->ne[0]) == k_layout.record_bytes);
    GGML_ASSERT(size_t(v_rec->ne[0]) == v_layout.record_bytes);

    const ggml_fp16_t * k_stage_data = (const ggml_fp16_t *) k_stage->data;
    const ggml_fp16_t * v_stage_data = (const ggml_fp16_t *) v_stage->data;
    const int64_t k_rows = k_stage->ne[1];
    const int64_t v_rows = v_stage->ne[1];
    const int32_t * idx = (const int32_t *) indices->data;
    const ggml_fp16_t * mask_data = (const ggml_fp16_t *) mask->data;
    const uint8_t * k_rec_data = (const uint8_t *) k_rec->data;
    const uint8_t * v_rec_data = (const uint8_t *) v_rec->data;
    float * out = (float *) dst->data;

    const int64_t hq_per_kv = Hq / Hkv;

    std::vector<float> kt, vt;
    std::vector<float> scores;
    std::vector<float> acc;

    for (int64_t hq = ith; hq < Hq; hq += nth) {
        const int64_t h = hq / hq_per_kv;
        const float * qh = (const float *) ((const char *) q->data + hq * q->nb[2]);

        scores.assign(n_kv, 0.0f);
        int64_t cur_group = -1;

        auto ensure_tiles = [&](int64_t g) {
            if (g == cur_group) {
                return;
            }
            kt.resize(size_t(n_slices) * size_t(Dk_slice) * size_t(G));
            vt.resize(size_t(n_slices) * size_t(G) * size_t(Dv_slice));
            for (int sl = 0; sl < n_slices; ++sl) {
                const size_t ridx = (size_t(g) * Hkv + h) * n_slices + sl;
                kvarn_dequantize_tile(k_rec_data + ridx * k_layout.record_bytes,
                        Dk_slice, G, k_bits, k_layout, false, kt.data() + size_t(sl) * Dk_slice * G);
                kvarn_dequantize_tile(v_rec_data + ridx * v_layout.record_bytes,
                        G, Dv_slice, v_bits, v_layout, true, vt.data() + size_t(sl) * G * Dv_slice);
            }
            cur_group = g;
        };

        float mask_max = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < n_kv; ++i) {
            const int64_t pos = idx[i];
            GGML_ASSERT(pos >= 0);
            float s = 0.0f;
            for (int sl = 0; sl < n_slices; ++sl) {
                const int64_t hs = h * n_slices + sl;
                const float * qs = qh + sl * Dk_slice;
                if (pos < G) {
                    const ggml_fp16_t * row = k_stage_data + pos * Dk_slice + size_t(hs) * Dk_slice * k_rows;
                    for (int64_t d = 0; d < Dk_slice; ++d) {
                        s += qs[d] * ggml_fp16_to_fp32(row[d]);
                    }
                } else {
                    const int64_t g = pos / G;
                    const int64_t slot = pos % G;
                    if (g > sealed) {
                        const ggml_fp16_t * row = k_stage_data + (kvarn_ring_block(g, slots) * G + slot) * Dk_slice
                                                + size_t(hs) * Dk_slice * k_rows;
                        for (int64_t d = 0; d < Dk_slice; ++d) {
                            s += qs[d] * ggml_fp16_to_fp32(row[d]);
                        }
                    } else {
                        ensure_tiles(g);
                        const float * row = kt.data() + size_t(sl) * Dk_slice * G + slot; // kt[d*G + slot]
                        for (int64_t d = 0; d < Dk_slice; ++d) {
                            s += qs[d] * row[size_t(d) * G];
                        }
                    }
                }
            }
            const float mv = ggml_fp16_to_fp32(mask_data[i]);
            scores[i] = s * kq_scale + mv;
            if (scores[i] > mask_max) {
                mask_max = scores[i];
            }
        }

        double sum = 0.0;
        for (int64_t i = 0; i < n_kv; ++i) {
            if (scores[i] == -std::numeric_limits<float>::infinity()) {
                scores[i] = 0.0f; // masked
                continue;
            }
            scores[i] = std::exp(scores[i] - mask_max);
            sum += scores[i];
        }
        const float inv_sum = sum > 0.0 ? float(1.0 / sum) : 0.0f;

        acc.assign(Dv, 0.0f);
        cur_group = -1;
        for (int64_t i = 0; i < n_kv; ++i) {
            const float p = scores[i] * inv_sum;
            if (p == 0.0f) {
                continue;
            }
            const int64_t pos = idx[i];
            for (int sl = 0; sl < n_slices; ++sl) {
                const int64_t hs = h * n_slices + sl;
                float * arow = acc.data() + sl * Dv_slice;
                if (pos < G) {
                    const ggml_fp16_t * row = v_stage_data + pos * Dv_slice + size_t(hs) * Dv_slice * v_rows;
                    for (int64_t d = 0; d < Dv_slice; ++d) {
                        arow[d] += p * ggml_fp16_to_fp32(row[d]);
                    }
                } else {
                    const int64_t g = pos / G;
                    const int64_t slot = pos % G;
                    if (g > sealed) {
                        const ggml_fp16_t * row = v_stage_data + (kvarn_ring_block(g, slots) * G + slot) * Dv_slice
                                                + size_t(hs) * Dv_slice * v_rows;
                        for (int64_t d = 0; d < Dv_slice; ++d) {
                            arow[d] += p * ggml_fp16_to_fp32(row[d]);
                        }
                    } else {
                        ensure_tiles(g);
                        const float * row = vt.data() + size_t(sl) * G * Dv_slice + size_t(slot) * Dv_slice; // vt[slot*Dv + d]
                        for (int64_t d = 0; d < Dv_slice; ++d) {
                            arow[d] += p * row[d];
                        }
                    }
                }
            }
        }

        float * oh = (float *) ((char *) dst->data + hq * dst->nb[2]);
        for (int64_t d = 0; d < Dv; ++d) {
            oh[d] = acc[d];
        }
    }
}
