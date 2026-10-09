#include "kvarn.cuh"
#include "kvarn-op.cuh"
#include "ggml-kvarn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// bit packing (must match kvarn_unpack_bits in ggml/src/ggml-kvarn.cpp)
// ---------------------------------------------------------------------------

template <int bits>
static __device__ __forceinline__ uint8_t kvarn_unpack(const uint8_t * __restrict__ src, int64_t index) {
    uint8_t value = 0;
    const size_t bit_offset = size_t(index) * bits;
#pragma unroll
    for (int b = 0; b < bits; ++b) {
        const size_t src_bit = bit_offset + b;
        value |= uint8_t(((src[src_bit / 8] >> (src_bit % 8)) & 1u) << b);
    }
    return value;
}

// ---------------------------------------------------------------------------
// K: one record, one query slice -> scores[128]
//   scores[t] = k_other[t] * sum_d q[d]*(code(d,t)*k_scale[d] + k_zp[d])
// ---------------------------------------------------------------------------

template <int k_bits>
static __global__ void kvarn_kq_record_kernel(
        const uint8_t * __restrict__ record, const float * __restrict__ q,
        const int head_dim, float * __restrict__ scores,
        const size_t k_payload_off, const size_t k_scale_off,
        const size_t k_zp_off, const size_t k_other_off) {
    const int t = threadIdx.x; // token slot in the record, [0, 128)
    if (t >= 128) {
        return;
    }

    const half * __restrict__ k_scale = (const half *) (record + k_scale_off);
    const half * __restrict__ k_zp    = (const half *) (record + k_zp_off);
    const half * __restrict__ k_other = (const half *) (record + k_other_off);

    float acc = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        const uint8_t code = kvarn_unpack<k_bits>(record + k_payload_off, int64_t(d) * 128 + t);
        const float val = float(code) * __half2float(k_scale[d]) + __half2float(k_zp[d]);
        acc = fmaf(q[d], val, acc);
    }
    scores[t] = acc * __half2float(k_other[t]);
}

// ---------------------------------------------------------------------------
// V: one record -> tile [128, head_dim] f32 (token-major)
//   tile[t*head_dim + c] = (code(t,c)*v_scale[t] + v_zp[t]) * v_other[c]
// ---------------------------------------------------------------------------

template <int v_bits>
static __global__ void kvarn_v_record_kernel(
        const uint8_t * __restrict__ record, const int head_dim, float * __restrict__ tile,
        const size_t v_payload_off, const size_t v_scale_off,
        const size_t v_zp_off, const size_t v_other_off) {
    const int c = threadIdx.x; // channel, [0, head_dim)
    if (c >= head_dim) {
        return;
    }

    const half * __restrict__ v_scale = (const half *) (record + v_scale_off);
    const half * __restrict__ v_zp    = (const half *) (record + v_zp_off);
    const half * __restrict__ v_other = (const half *) (record + v_other_off);

    const float other = __half2float(v_other[c]);
    for (int t = 0; t < 128; ++t) {
        const uint8_t code = kvarn_unpack<v_bits>(record + v_payload_off, int64_t(t) * head_dim + c);
        tile[t * head_dim + c] = (float(code) * __half2float(v_scale[t]) + __half2float(v_zp[t])) * other;
    }
}

// ---------------------------------------------------------------------------
// batched variants: one block per record (MMVQ decode shape)
// ---------------------------------------------------------------------------

template <int k_bits>
static __global__ void kvarn_kq_records_kernel(
        const uint8_t * __restrict__ records, const float * __restrict__ q,
        const int head_dim, const size_t record_bytes, float * __restrict__ scores,
        const size_t k_payload_off, const size_t k_scale_off,
        const size_t k_zp_off, const size_t k_other_off) {
    __shared__ float ksf[128];
    __shared__ float kzf[128];
    __shared__ float kof[128];

    const uint8_t * __restrict__ record = records + size_t(blockIdx.x) * record_bytes;
    const int t  = threadIdx.x; // 0..63, handles tokens 2t and 2t+1
    const int t0 = 2 * t;
    const half * __restrict__ k_scale = (const half *) (record + k_scale_off);
    const half * __restrict__ k_zp    = (const half *) (record + k_zp_off);
    const half * __restrict__ k_other = (const half *) (record + k_other_off);

    for (int i = t; i < head_dim; i += blockDim.x) {
        ksf[i] = __half2float(k_scale[i]);
        kzf[i] = __half2float(k_zp[i]);
    }
    for (int i = t; i < 128; i += blockDim.x) {
        kof[i] = __half2float(k_other[i]);
    }
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint8_t * __restrict__ payload = record + k_payload_off;
    if constexpr (k_bits == 4) {
        // two adjacent tokens share one payload byte in a 128-wide row
        for (int d = 0; d < head_dim; ++d) {
            const uint8_t b = payload[size_t(d) * 64 + t];
            const float s = ksf[d];
            const float z = kzf[d];
            acc0 = fmaf(q[d], float(b & 0xF) * s + z, acc0);
            acc1 = fmaf(q[d], float(b >> 4)  * s + z, acc1);
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            const float s = ksf[d];
            const float z = kzf[d];
            const uint8_t c0 = kvarn_unpack<k_bits>(payload, size_t(d) * 128 + t0);
            const uint8_t c1 = kvarn_unpack<k_bits>(payload, size_t(d) * 128 + t0 + 1);
            acc0 = fmaf(q[d], float(c0) * s + z, acc0);
            acc1 = fmaf(q[d], float(c1) * s + z, acc1);
        }
    }

    scores[size_t(blockIdx.x) * 128 + t0]     = acc0 * kof[t0];
    scores[size_t(blockIdx.x) * 128 + t0 + 1] = acc1 * kof[t0 + 1];
}

template <int v_bits>
static __global__ void kvarn_pv_records_kernel(
        const uint8_t * __restrict__ records, const float * __restrict__ weights,
        const int head_dim, const size_t record_bytes, float * __restrict__ out,
        const size_t v_payload_off, const size_t v_scale_off,
        const size_t v_zp_off, const size_t v_other_off) {
    __shared__ float vsf[128];
    __shared__ float vzf[128];
    __shared__ float vof[128];
    __shared__ float wf[128];

    const uint8_t * __restrict__ record = records + size_t(blockIdx.x) * record_bytes;
    const int c = threadIdx.x; // 0..head_dim/4-1, owns channels 4c..4c+3
    const half * __restrict__ v_scale = (const half *) (record + v_scale_off);
    const half * __restrict__ v_zp    = (const half *) (record + v_zp_off);
    const half * __restrict__ v_other = (const half *) (record + v_other_off);

    for (int i = c; i < 128; i += blockDim.x) {
        const float s = __half2float(v_scale[i]);
        const float z = __half2float(v_zp[i]);
        const float w = weights[size_t(blockIdx.x) * 128 + i];
        vsf[i] = w * s; // folded: (code*w*v_scale + w*v_zp)
        vzf[i] = w * z;
        wf[i]  = w;
    }
    for (int i = c; i < head_dim; i += blockDim.x) {
        vof[i] = __half2float(v_other[i]);
    }
    __syncthreads();

    __shared__ float csum_s;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int t = 0; t < 128; ++t) { s += vzf[t]; }
        csum_s = s;
    }
    __syncthreads();

    const uint8_t * __restrict__ payload = record + v_payload_off;

    if constexpr (v_bits == 2 && true) {
        // four adjacent channels share one payload byte (head_dim % 4 == 0)
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int t = 0; t < 128; ++t) {
            const uint8_t b = payload[t * (head_dim / 4) + c];
            const float ws = vsf[t]; // = w * v_scale
            a0 = fmaf(float(b & 0x3),        ws, a0);
            a1 = fmaf(float((b >> 2) & 0x3), ws, a1);
            a2 = fmaf(float((b >> 4) & 0x3), ws, a2);
            a3 = fmaf(float((b >> 6) & 0x3), ws, a3);
        }
        const float o0 = vof[4 * c], o1 = vof[4 * c + 1], o2 = vof[4 * c + 2], o3 = vof[4 * c + 3];
        const float cs = csum_s; // sum_t w*v_zp, channel independent
        atomicAdd(&out[4 * c],     (a0 + cs) * o0);
        atomicAdd(&out[4 * c + 1], (a1 + cs) * o1);
        atomicAdd(&out[4 * c + 2], (a2 + cs) * o2);
        atomicAdd(&out[4 * c + 3], (a3 + cs) * o3);
    } else {
        float acc = 0.0f;
        for (int t = 0; t < 128; ++t) {
            const uint8_t code = kvarn_unpack<v_bits>(payload, int64_t(t) * head_dim + c);
            const float v = (float(code) * vsf[t] + vzf[t]) * vof[c];
            acc = fmaf(wf[t], v, acc);
        }
        atomicAdd(&out[c], acc);
    }
}

// ---------------------------------------------------------------------------
// store: VarN (Sinkhorn-style) + RTN quantization of one 128-token tile.
//   tile_in: [rows, cols] f16 row-major (K: [head_dim, 128], V: [128, head_dim])
// Mirrors kvarn_variance_normalize/kvarn_quantize_tile of the CPU reference.
// The original tile stays in shared memory; the balanced view is recomputed
// implicitly from the log-domain scales (same formulas, double reductions), so
// results can differ from the CPU only at RTN rounding boundaries.
// ---------------------------------------------------------------------------

static __device__ float kvarn_std(const float * __restrict__ v, const int n, const int stride) {
    float sum    = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float x = v[size_t(i) * stride];
        sum    += x;
        sum_sq += x * x;
    }
    const float mean = sum / n;
    const float var  = fmaxf(0.0f, (sum_sq - float(n) * mean * mean) / float(n > 1 ? n - 1 : 1));
    return sqrtf(var);
}

static __device__ __forceinline__ void kvarn_put_f16(uint8_t * __restrict__ rec, size_t off, int i, float v) {
    const uint16_t h = __half_as_ushort(__float2half_rn(v));
    *reinterpret_cast<uint16_t *>(rec + off + size_t(i) * 2) = h;
}

static __device__ __forceinline__ void kvarn_imb_reduce(
        const float * __restrict__ s_col, const float * __restrict__ s_row,
        const int rows, const int cols, float * __restrict__ s_imb) {
    __shared__ float wr[4][4]; // [warp][cmin,cmax,rmin,rmax]
    const int tid = threadIdx.x;
    float cmin = 1e30f, cmax = 0.0f, rmin = 1e30f, rmax = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) { cmin = fminf(cmin, s_col[c]); cmax = fmaxf(cmax, s_col[c]); }
    for (int r = tid; r < rows; r += blockDim.x) { rmin = fminf(rmin, s_row[r]); rmax = fmaxf(rmax, s_row[r]); }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmin = fminf(cmin, __shfl_xor_sync(0xffffffffu, cmin, off));
        cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, off));
        rmin = fminf(rmin, __shfl_xor_sync(0xffffffffu, rmin, off));
        rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, off));
    }
    if ((tid & 31) == 0) {
        wr[tid >> 5][0] = cmin; wr[tid >> 5][1] = cmax;
        wr[tid >> 5][2] = rmin; wr[tid >> 5][3] = rmax;
    }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w) {
            wr[0][0] = fminf(wr[0][0], wr[w][0]);
            wr[0][1] = fmaxf(wr[0][1], wr[w][1]);
            wr[0][2] = fminf(wr[0][2], wr[w][2]);
            wr[0][3] = fmaxf(wr[0][3], wr[w][3]);
        }
        *s_imb = wr[0][1] / fmaxf(wr[0][0], 1e-8f) + wr[0][3] / fmaxf(wr[0][2], 1e-8f);
    }
    __syncthreads();
}

// Shared VarN + RTN + packing math for one 128-token tile. `smem` must hold
// the tile already loaded as f32 [rows*cols] row-major, followed by the
// scratch regions this function uses:
//   tile[rows*cols] | e_row[rows] | e_col[cols] | l_row[rows] | l_col[cols] |
//   b_row[rows] | b_col[cols] | s_col[128] | s_row[128] | s_imb[1] | spare[1]
// Mirrors kvarn_variance_normalize/kvarn_quantize_tile of the CPU reference.
// The original tile stays in shared memory; the balanced view is recomputed
// implicitly from the log-domain scales (same formulas, double reductions), so
// results can differ from the CPU only at RTN rounding boundaries.
template <int bits>
static __device__ void kvarn_store_tile_quantize(
        float * __restrict__ smem, uint8_t * __restrict__ record,
        const int rows, const int cols, const int iters, const float alpha,
        const size_t payload_off, const size_t payload_bytes,
        const size_t scale_off, const size_t zp_off, const size_t other_off) {
    float * tile  = smem;                 // [rows*cols] original tile
    float * e_row = tile + size_t(rows) * cols;
    float * e_col = e_row + rows;
    float * l_row = e_col + cols;
    float * l_col = l_row + rows;
    float * b_row = l_col + cols;
    float * b_col = b_row + rows;
    float * s_col = b_col + cols;         // scratch: column stds [128]
    float * s_row = s_col + 128;          // scratch: row stds    [128]
    float * s_imb = s_row + 128;          // scratch: current imbalance

    const int tid = threadIdx.x;
    const long n  = long(rows) * cols;

    if (tid < rows) { e_row[tid] = 1.0f; l_row[tid] = 0.0f; b_row[tid] = 1.0f; }
    if (tid < cols) { e_col[tid] = 1.0f; l_col[tid] = 0.0f; b_col[tid] = 1.0f; }
    __syncthreads();

    // initial imbalance of the raw tile (scales = 1)
    if (tid < cols) { s_col[tid] = kvarn_std(tile + tid, rows, cols); }
    if (tid < rows) { s_row[tid] = kvarn_std(tile + size_t(tid) * cols, cols, 1); }
    __syncthreads();
    kvarn_imb_reduce(s_col, s_row, rows, cols, s_imb);
    // every thread reads the same s_imb (written by thread 0 before the
    // reduce's trailing barrier) and keeps its own register copy of the best
    float imb_best = *s_imb;

    for (int it = 0; it < iters; ++it) {
        // column pass: std over rows of cur[r][c] = tile[r][c]/(e_row[r]*e_col[c])
        if (tid < cols) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int r = 0; r < rows; ++r) {
                const float x = tile[size_t(r) * cols + tid] / e_row[r];
                sum += x; sum_sq += x * x;
            }
            const float mean = sum / rows;
            const float var  = fmaxf(0.0f, (sum_sq - float(rows) * mean * mean) / float(rows > 1 ? rows - 1 : 1));
            const float std  = sqrtf(var) / e_col[tid];
            const float s    = fminf(fmaxf(std, 1e-3f), 1e3f);
            l_col[tid] = fminf(fmaxf(l_col[tid] + logf(s), -0.3f), 10.0f);
        }
        __syncthreads();
        if (tid < cols) { e_col[tid] = expf(l_col[tid]); }
        __syncthreads();

        // row pass
        if (tid < rows) {
            const float * row = tile + size_t(tid) * cols;
            float sum = 0.0f, sum_sq = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const float x = row[c] / e_col[c];
                sum += x; sum_sq += x * x;
            }
            const float mean = sum / cols;
            const float var  = fmaxf(0.0f, (sum_sq - float(cols) * mean * mean) / float(cols > 1 ? cols - 1 : 1));
            const float std  = sqrtf(var) / e_row[tid];
            const float s    = fminf(fmaxf(std, 1e-3f), 1e3f);
            l_row[tid] = fminf(fmaxf(l_row[tid] + logf(s), -0.3f), 10.0f);
        }
        __syncthreads();
        if (tid < rows) { e_row[tid] = expf(l_row[tid]); }
        __syncthreads();

        // imbalance of the current normalization; keep best-so-far scales
        if (tid < cols) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int r = 0; r < rows; ++r) {
                const float x = tile[size_t(r) * cols + tid] / e_row[r];
                sum += x; sum_sq += x * x;
            }
            const float mean = sum / rows;
            const float var  = fmaxf(0.0f, (sum_sq - float(rows) * mean * mean) / float(rows > 1 ? rows - 1 : 1));
            s_col[tid] = sqrtf(var) / e_col[tid];
        }
        if (tid < rows) {
            const float * row = tile + size_t(tid) * cols;
            float sum = 0.0f, sum_sq = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const float x = row[c] / e_col[c];
                sum += x; sum_sq += x * x;
            }
            const float mean = sum / cols;
            const float var  = fmaxf(0.0f, (sum_sq - float(cols) * mean * mean) / float(cols > 1 ? cols - 1 : 1));
            s_row[tid] = sqrtf(var) / e_row[tid];
        }
        __syncthreads();
        kvarn_imb_reduce(s_col, s_row, rows, cols, s_imb);
        // All threads derive the same decision from the same s_imb and their
        // own (identically updated) register copy of the best imbalance, so no
        // shared flag is handed across a barrier. The previous shared-flag
        // version was miscompiled by nvcc 12.4: non-zero threads' flag read was
        // scheduled into the same barrier interval as thread 0's write, before
        // it, making the seal nondeterministic.
        const bool keep = (*s_imb <= imb_best);
        if (keep) { imb_best = *s_imb; }
        __syncthreads();
        if (keep) {
            if (tid < rows) { b_row[tid] = e_row[tid]; }
            if (tid < cols) { b_col[tid] = e_col[tid]; }
        }
        __syncthreads();
    }

    // optional VarN damping: stored scales are s**alpha (alpha=1 = reference)
    if (alpha != 1.0f) {
        if (tid < rows) { b_row[tid] = powf(b_row[tid], alpha); }
        if (tid < cols) { b_col[tid] = powf(b_col[tid], alpha); }
        __syncthreads();
    }

    // final balanced view, in place
    for (long i = tid; i < n; i += blockDim.x) {
        const int r = int(i / cols);
        const int c = int(i % cols);
        tile[i] = tile[i] / (b_col[c] * b_row[r]);
    }
    for (size_t i = tid; i < payload_bytes; i += blockDim.x) {
        record[payload_off + i] = 0;
    }
    __syncthreads();

    // per-row RTN, absorbed scales, bit packing (row bit ranges are disjoint)
    if (tid < rows) {
        const float * row = tile + size_t(tid) * cols;
        float lo = row[0], hi = row[0];
        for (int c = 1; c < cols; ++c) { lo = fminf(lo, row[c]); hi = fmaxf(hi, row[c]); }
        const float qmax  = float((1u << bits) - 1u);
        const float scale = fmaxf((hi - lo) / qmax, 1e-10f);

        kvarn_put_f16(record, scale_off, tid, b_row[tid] * scale);
        kvarn_put_f16(record, zp_off,    tid, b_row[tid] * lo);

        const size_t row_bit0 = size_t(tid) * cols * bits;
        for (int c = 0; c < cols; ++c) {
            float v = roundf((row[c] - lo) / scale);
            v = fminf(fmaxf(v, 0.0f), qmax);
            const uint8_t code = (uint8_t) v;
            for (int b = 0; b < bits; ++b) {
                const size_t bit = row_bit0 + size_t(c) * bits + b;
                record[payload_off + bit / 8] |= uint8_t(((code >> b) & 1u) << (bit % 8));
            }
        }
    }
    if (tid < cols) {
        kvarn_put_f16(record, other_off, tid, b_col[tid]);
    }
}

// Standalone tile producer: tile_in f16 [rows, cols] row-major -> record.
template <int bits>
static __global__ void kvarn_store_tile_kernel(
        const half * __restrict__ tile_in, uint8_t * __restrict__ record,
        const int rows, const int cols, const int iters, const float alpha,
        const size_t payload_off, const size_t payload_bytes,
        const size_t scale_off, const size_t zp_off, const size_t other_off) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const long n  = long(rows) * cols;
    for (long i = tid; i < n; i += blockDim.x) {
        smem[i] = __half2float(tile_in[i]);
    }
    kvarn_store_tile_quantize<bits>(smem, record, rows, cols, iters, alpha,
            payload_off, payload_bytes, scale_off, zp_off, other_off);
}

// Fused ring seal: one block per (ring slot, head-slice); reads the slice's
// tile from its stage ring block (K is channel-major in the stage, V
// token-major) and quantizes it into the record. The block grid covers the
// whole ring (stage_slots blocks); each block skips itself unless its group
// lies in (sealed_pre, g_first) as read from the device-side state, so the
// launch geometry is call-independent (captured CUDA graphs stay valid).
template <int bits>
static __global__ void kvarn_seal_kernel(
        const half * __restrict__ stage, uint8_t * __restrict__ records,
        const int32_t * __restrict__ state, // [0]=g_first, [1]=sealed (before this call)
        const int head_dim, const int n_heads_s, const int stage_rows, const int stage_slots,
        const int value, const int iters, const float alpha,
        const size_t record_bytes, const size_t payload_off, const size_t payload_bytes,
        const size_t scale_off, const size_t zp_off, const size_t other_off) {
    extern __shared__ float smem[];
    const int g  = 1 + blockIdx.x; // ring group for this slot
    const int hh = blockIdx.y; // head-slice index (h*n_slices + s)
    if (g <= state[1] || g >= state[0]) {
        return; // this slot's group is not being sealed in this call
    }
    const int rows = value ? 128 : head_dim;
    const int cols = value ? head_dim : 128;
    const int block = 1 + (g - 1) % stage_slots;
    const int n = rows * cols;
    const int tid = threadIdx.x;

    for (int e = tid; e < n; e += blockDim.x) {
        const int slot = e / head_dim; // 128 slots per stage block
        const int chan = e % head_dim;
        const half * src = stage + (size_t(block) * 128 + slot) * head_dim
                                 + size_t(hh) * head_dim * stage_rows;
        smem[value ? e : (chan * cols + slot)] = __half2float(src[chan]);
    }

    uint8_t * record = records + (size_t(g) * n_heads_s + hh) * record_bytes;
    kvarn_store_tile_quantize<bits>(smem, record, rows, cols, iters, alpha,
            payload_off, payload_bytes, scale_off, zp_off, other_off);
}

// ---------------------------------------------------------------------------
// materialize: sink + live ring groups -> native cell-major output rows.
// Grid (n_kv, n_heads); one thread per channel. (A fused single-launch variant
// was measured: both a z-grid with a shared record tile and a per-element
// decode version came out slower than these two kernels - the stage copy
// loses occupancy with dynamic smem and the per-element decode wastes L2
// bandwidth -, so the two-kernel split is kept.)
// ---------------------------------------------------------------------------
__global__ void kvarn_materialize_stage_kernel(
        const int32_t * __restrict__ indices, const half * __restrict__ stage,
        half * __restrict__ out, const int32_t * __restrict__ state,
        const int head_dim, const int head_dim_s, const int n_heads, const int n_slices,
        const int stage_rows, const int stage_slots) {
    const int i = blockIdx.x;
    const int hh = blockIdx.y; // head-slice index (h*n_slices + s)
    const int h = hh / n_slices;
    const int s = hh % n_slices;
    const int d = threadIdx.x;
    if (d >= head_dim_s) {
        return;
    }
    const int sealed = state[2]; // groups sealed by this call
    const int pos = indices[i];
    int block;
    if (pos < 128) {
        block = 0; // permanent sink
    } else {
        const int g = pos / 128;
        if (g <= sealed) {
            return; // sealed groups are written by the record kernel
        }
        block = 1 + (g - 1) % stage_slots;
    }
    const int slot = pos & 127;
    out[size_t(h) * head_dim + s * head_dim_s + d + size_t(i) * head_dim * n_heads] =
            stage[size_t(block * 128 + slot) * head_dim_s + size_t(hh) * head_dim_s * stage_rows + d];
}

// One block per (sealed group, head): decode the record tile into shared
// halves once, then write every output row belonging to the group. `indices`
// must be non-decreasing (v1: one contiguous token stream); lower/upper bounds
// handle duplicate positions.
template <int bits>
static __global__ void kvarn_materialize_records_kernel(
        const int32_t * __restrict__ indices, const uint8_t * __restrict__ records,
        const int32_t * __restrict__ state,
        half * __restrict__ out, const int n_kv, const int head_dim, const int n_heads_s, const int n_slices,
        const int value, const size_t record_bytes,
        const size_t payload_off, const size_t scale_off,
        const size_t zp_off, const size_t other_off) {
    __shared__ int s_lo;
    __shared__ int s_hi;
    const int g  = 1 + blockIdx.x;
    const int hh = blockIdx.y; // head-slice index (h*n_slices + s)
    const int tid = threadIdx.x;
    if (g > state[2]) {
        return; // this group is not sealed yet (grid covers the whole record set)
    }

    if (tid == 0) {
        const int pos0 = g * 128;
        const int pos1 = pos0 + 128;
        int lo = 0, hi = n_kv;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (indices[mid] < pos0) { lo = mid + 1; } else { hi = mid; }
        }
        int up = lo, hi2 = n_kv;
        while (up < hi2) {
            const int mid = (up + hi2) >> 1;
            if (indices[mid] < pos1) { up = mid + 1; } else { hi2 = mid; }
        }
        s_lo = lo;
        s_hi = up;
    }
    __syncthreads();

    const int lo = s_lo;
    const int n = s_hi - lo;
    if (n <= 0) {
        return;
    }

    extern __shared__ half tile[]; // rows*cols halves: K [D_slice, 128], V [128, D_slice]
    const int rows = value ? 128 : head_dim;
    const int cols = value ? head_dim : 128;
    const uint8_t * record = records + (size_t(g) * n_heads_s + hh) * record_bytes;
    const half * scale_s = (const half *) (record + scale_off);
    const half * zp_s    = (const half *) (record + zp_off);
    const half * other_s = (const half *) (record + other_off);

    const int nn = rows * cols;
    for (int e = tid; e < nn; e += blockDim.x) {
        const int r = e / cols;
        const int c = e % cols;
        const uint8_t code = kvarn_unpack<bits>(record + payload_off, size_t(r) * cols + c);
        tile[e] = __float2half_rn((float(code) * __half2float(scale_s[r]) + __half2float(zp_s[r]))
                * __half2float(other_s[c]));
    }
    __syncthreads();

    const int h = hh / n_slices;
    const int s = hh % n_slices;
    const int dim_full = head_dim * n_slices;
    const int n_heads = n_heads_s / n_slices;
    for (int e = tid; e < n * head_dim; e += blockDim.x) {
        const int j = e / head_dim;
        const int d = e % head_dim;
        const int slot = indices[lo + j] & 127;
        const half v = value ? tile[slot * head_dim + d] : tile[d * 128 + slot];
        out[size_t(h) * dim_full + s * head_dim + d + size_t(lo + j) * dim_full * n_heads] = v;
    }
}

// ---------------------------------------------------------------------------
// K dequant: one record -> tile [head_dim, 128] f32 (channel-major)
//   tile[d*128 + t] = (code(d,t)*k_scale[d] + k_zp[d]) * k_other[t]
// ---------------------------------------------------------------------------

template <int k_bits>
static __global__ void kvarn_k_record_kernel(
        const uint8_t * __restrict__ record, const int head_dim, float * __restrict__ tile,
        const size_t k_payload_off, const size_t k_scale_off,
        const size_t k_zp_off, const size_t k_other_off) {
    const int d = threadIdx.x; // channel
    if (d >= head_dim) {
        return;
    }
    const half * __restrict__ k_scale = (const half *) (record + k_scale_off);
    const half * __restrict__ k_zp    = (const half *) (record + k_zp_off);
    const half * __restrict__ k_other = (const half *) (record + k_other_off);

    const float scale = __half2float(k_scale[d]);
    const float zp    = __half2float(k_zp[d]);
    for (int t = 0; t < 128; ++t) {
        const uint8_t code = kvarn_unpack<k_bits>(record + k_payload_off, int64_t(d) * 128 + t);
        tile[d * 128 + t] = (float(code) * scale + zp) * __half2float(k_other[t]);
    }
}

// ---------------------------------------------------------------------------
// host entry points
// ---------------------------------------------------------------------------

static bool kvarn_launch_ok(const char * what) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kvarn: %s failed: %s\n", what, cudaGetErrorString(err));
        return false;
    }
    return true;
}

void ggml_cuda_kvarn_kq_record(
        const struct ggml_kvarn_layout * layout, const void * record,
        const float * q, int head_dim, float * scores) {
    kvarn_kq_record_kernel<4><<<1, 128>>>(
            (const uint8_t *) record, q, head_dim, scores,
            layout->k_payload_off, layout->k_scale_off, layout->k_zp_off, layout->k_other_off);
    kvarn_launch_ok(__func__);
}

void ggml_cuda_kvarn_v_record(
        const struct ggml_kvarn_layout * layout, const void * record, int head_dim, float * tile) {
    kvarn_v_record_kernel<2><<<1, head_dim>>>(
            (const uint8_t *) record, head_dim, tile,
            layout->v_payload_off, layout->v_scale_off, layout->v_zp_off, layout->v_other_off);
    kvarn_launch_ok(__func__);
}

void ggml_cuda_kvarn_kq_records(
        const struct ggml_kvarn_layout * layout, const void * records, const float * q,
        int head_dim, int n_records, size_t record_bytes, float * scores) {
    kvarn_kq_records_kernel<4><<<n_records, 64>>>(
            (const uint8_t *) records, q, head_dim, record_bytes, scores,
            layout->k_payload_off, layout->k_scale_off, layout->k_zp_off, layout->k_other_off);
    kvarn_launch_ok(__func__);
}

void ggml_cuda_kvarn_k_record(
        const struct ggml_kvarn_layout * layout, const void * record, int head_dim, float * tile) {
    kvarn_k_record_kernel<4><<<1, head_dim>>>(
            (const uint8_t *) record, head_dim, tile,
            layout->k_payload_off, layout->k_scale_off, layout->k_zp_off, layout->k_other_off);
    kvarn_launch_ok(__func__);
}

void ggml_cuda_kvarn_pv_records(
        const struct ggml_kvarn_layout * layout, const void * records, const float * weights,
        int head_dim, int n_records, size_t record_bytes, float * out) {
    if (head_dim % 4 != 0) {
        fprintf(stderr, "kvarn: pv_records: head_dim %d not a multiple of 4\n", head_dim);
        return;
    }
    kvarn_pv_records_kernel<2><<<n_records, head_dim / 4>>>(
            (const uint8_t *) records, weights, head_dim, record_bytes, out,
            layout->v_payload_off, layout->v_scale_off, layout->v_zp_off, layout->v_other_off);
    kvarn_launch_ok(__func__);
}

void ggml_cuda_kvarn_store_tile(
        const struct ggml_kvarn_layout * layout, const void * tile_f16,
        int head_dim, int group, bool value, int bits, int iters, float alpha, void * record) {    const int rows = value ? group : head_dim;
    const int cols = value ? head_dim : group;
    if ((size_t(cols) * bits) % 8 != 0) {
        fprintf(stderr, "kvarn: store_tile: cols*bits must be byte aligned (cols=%d bits=%d)\n", cols, bits);
        return;
    }
    const size_t payload_off   = value ? layout->v_payload_off   : layout->k_payload_off;
    const size_t payload_bytes = value ? layout->v_payload_bytes : layout->k_payload_bytes;
    const size_t scale_off     = value ? layout->v_scale_off     : layout->k_scale_off;
    const size_t zp_off        = value ? layout->v_zp_off        : layout->k_zp_off;
    const size_t other_off     = value ? layout->v_other_off     : layout->k_other_off;

    const size_t smem_bytes = (size_t(rows) * cols + 3 * size_t(rows + cols) + 258) * sizeof(float);

    if (bits == 2) {
        cudaFuncSetAttribute((const void *) kvarn_store_tile_kernel<2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes);
        kvarn_store_tile_kernel<2><<<1, 128, smem_bytes>>>((const half *) tile_f16, (uint8_t *) record,
                rows, cols, iters, alpha, payload_off, payload_bytes, scale_off, zp_off, other_off);
        kvarn_launch_ok(__func__);
    } else if (bits == 4) {
        cudaFuncSetAttribute((const void *) kvarn_store_tile_kernel<4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes);
        kvarn_store_tile_kernel<4><<<1, 128, smem_bytes>>>((const half *) tile_f16, (uint8_t *) record,
                rows, cols, iters, alpha, payload_off, payload_bytes, scale_off, zp_off, other_off);
        kvarn_launch_ok(__func__);
    } else {
        fprintf(stderr, "kvarn: store_tile: bits %d not instantiated (2 and 4 available)\n", bits);
    }
}

// ---------------------------------------------------------------------------
// integration helpers: stage write / gather (semantics from ggml-kvarn.cpp
// kvarn_stage_write / kvarn_gather_stage_tile)
//   current : F32 [D, H, T]   (offset d + h*D + t*D*H)
//   indices : I32 [T]         absolute positions (ascending, contiguous)
//   stage   : F16 [D, stage_rows, H]; sink block 0 (pos<128), group g >= 1 in
//             ring block 1 + (g-1) % stage_slots
// ---------------------------------------------------------------------------

__global__ void kvarn_stage_write_kernel(
        const float * __restrict__ current, const int32_t * __restrict__ indices,
        half * __restrict__ stage, const int head_dim, const int head_dim_s, const int n_heads, const int n_slices,
        const int stage_rows, const int stage_slots) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim_s) {
        return;
    }
    const int pos   = indices[t];
    const int block = (pos < 128) ? 0 : 1 + ((pos / 128) - 1) % stage_slots;
    const int slot  = pos & 127;
    for (int s = 0; s < n_slices; ++s) {
        const int hs = h * n_slices + s;
        stage[size_t(block * 128 + slot) * head_dim_s + size_t(hs) * head_dim_s * stage_rows + d] =
                __float2half_rn(current[s * head_dim_s + d + size_t(h) * head_dim + size_t(t) * head_dim * n_heads]);
    }
}

// tile: F16 [rows, cols] row-major; K: rows=head_dim, cols=128 (chan=r, slot=c);
// V: rows=128, cols=head_dim (chan=c, slot=r). block: stage block to read.
__global__ void kvarn_gather_kernel(
        const half * __restrict__ stage, half * __restrict__ tile,
        const int head_dim, const int stage_rows, const int rows, const int cols,
        const int block, const int value, const int h) {
    const int r = blockIdx.x;
    const int c = threadIdx.x;
    if (c >= cols) {
        return;
    }
    const int chan = value ? c : r;
    const int slot = value ? r : c;
    tile[size_t(r) * cols + c] =
            stage[size_t(block * 128 + slot) * head_dim + size_t(h) * head_dim * stage_rows + chan];
}

void ggml_cuda_kvarn_stage_write(
        const float * current, const int32_t * indices, void * stage,
        int head_dim, int n_heads, int n_tokens, int stage_rows, int stage_slots, int n_slices,
        cudaStream_t stream) {
    if (n_tokens <= 0) {
        return;
    }
    const int head_dim_s = head_dim / n_slices;
    dim3 grid(n_tokens, n_heads);
    kvarn_stage_write_kernel<<<grid, head_dim_s, 0, stream>>>(
            current, indices, (half *) stage, head_dim, head_dim_s, n_heads, n_slices, stage_rows, stage_slots);
    kvarn_launch_ok(__func__);
}

void ggml_cuda_kvarn_gather(
        const void * stage, void * tile, int head_dim, int stage_rows, int n_heads,
        int rows, int cols, int block, bool value, int h) {
    (void) n_heads;
    kvarn_gather_kernel<<<rows, cols>>>(
            (const half *) stage, (half *) tile, head_dim, stage_rows, rows, cols, block, value ? 1 : 0, h);
    kvarn_launch_ok(__func__);
}

// ---------------------------------------------------------------------------
// combined K+V variants: one launch serves both sides (blockIdx.z selects the
// side). The record decode uses runtime bit widths to avoid a 6x6 template
// explosion; the per-side kernels above stay templated and bit-exact.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint8_t kvarn_unpack_rt(const uint8_t * __restrict__ src, int64_t index, int bits) {
    uint8_t value = 0;
    const size_t bit_offset = size_t(index) * size_t(bits);
    for (int b = 0; b < bits; ++b) {
        const size_t src_bit = bit_offset + size_t(b);
        value |= uint8_t(((src[src_bit / 8] >> (src_bit % 8)) & 1u) << b);
    }
    return value;
}

__global__ void kvarn_stage_write2_kernel(
        const float * __restrict__ k_cur, const float * __restrict__ v_cur,
        const int32_t * __restrict__ indices,
        half * __restrict__ k_stage, half * __restrict__ v_stage,
        const int k_dim, const int v_dim, const int k_dim_s, const int v_dim_s,
        const int n_heads, const int n_slices,
        const int k_rows, const int v_rows, const int k_slots, const int v_slots) {
    const int t    = blockIdx.x;
    const int h    = blockIdx.y;
    const int side = blockIdx.z;
    const int dim   = side == 0 ? k_dim   : v_dim;   // full head dim
    const int dim_s = side == 0 ? k_dim_s : v_dim_s; // 128-wide slice
    const int d    = threadIdx.x;
    if (d >= dim_s) {
        return;
    }
    const float * cur = side == 0 ? k_cur : v_cur;
    half * stage      = side == 0 ? k_stage : v_stage;
    const int rows    = side == 0 ? k_rows : v_rows;
    const int slots   = side == 0 ? k_slots : v_slots;

    const int pos   = indices[t];
    const int block = (pos < 128) ? 0 : 1 + ((pos / 128) - 1) % slots;
    const int slot  = pos & 127;
    for (int s = 0; s < n_slices; ++s) {
        const int hs = h * n_slices + s;
        stage[size_t(block * 128 + slot) * dim_s + size_t(hs) * dim_s * rows + d] =
                __float2half_rn(cur[s * dim_s + d + size_t(h) * dim + size_t(t) * dim * n_heads]);
    }
}

__global__ void kvarn_materialize_stage2_kernel(
        const int32_t * __restrict__ indices,
        const half * __restrict__ k_stage, const half * __restrict__ v_stage,
        half * __restrict__ k_out, half * __restrict__ v_out,
        const int32_t * __restrict__ state,
        const int k_dim, const int v_dim, const int k_dim_s, const int v_dim_s,
        const int n_heads, const int n_slices,
        const int k_rows, const int v_rows, const int k_slots, const int v_slots) {
    const int i    = blockIdx.x;
    const int hh   = blockIdx.y; // head-slice index (h*n_slices + s)
    const int side = blockIdx.z;
    const int dim   = side == 0 ? k_dim   : v_dim;   // full head dim
    const int dim_s = side == 0 ? k_dim_s : v_dim_s; // 128-wide slice
    const int h = hh / n_slices;
    const int s = hh % n_slices;
    const int d    = threadIdx.x;
    if (d >= dim_s) {
        return;
    }
    const half * stage = side == 0 ? k_stage : v_stage;
    half * out         = side == 0 ? k_out : v_out;
    const int rows     = side == 0 ? k_rows : v_rows;
    const int slots    = side == 0 ? k_slots : v_slots;

    const int sealed = state[2]; // groups sealed by this call
    const int pos = indices[i];
    int block;
    if (pos < 128) {
        block = 0; // permanent sink
    } else {
        const int g = pos / 128;
        if (g <= sealed) {
            return; // sealed groups are written by the records kernel
        }
        block = 1 + (g - 1) % slots;
    }
    const int slot = pos & 127;
    out[size_t(h) * dim + s * dim_s + d + size_t(i) * dim * n_heads] =
            stage[size_t(block * 128 + slot) * dim_s + size_t(hh) * dim_s * rows + d];
}

__global__ void kvarn_materialize_records2_kernel(
        const int32_t * __restrict__ indices,
        const uint8_t * __restrict__ k_records, const uint8_t * __restrict__ v_records,
        half * __restrict__ k_out, half * __restrict__ v_out,
        const int32_t * __restrict__ state,
        const int n_kv, const int k_dim, const int v_dim, const int n_heads, const int n_slices,
        const int k_bits, const int v_bits,
        const size_t k_record_bytes, const size_t v_record_bytes,
        const size_t k_payload_off, const size_t k_scale_off, const size_t k_zp_off, const size_t k_other_off,
        const size_t v_payload_off, const size_t v_scale_off, const size_t v_zp_off, const size_t v_other_off) {
    __shared__ int s_lo;
    __shared__ int s_hi;
    const int g    = 1 + blockIdx.x;
    const int hh   = blockIdx.y; // head-slice index (h*n_slices + s)
    const int side = blockIdx.z;
    const int tid  = threadIdx.x;
    if (g > state[2]) {
        return; // this group is not sealed yet (grid covers the whole record set)
    }

    if (tid == 0) {
        const int pos0 = g * 128;
        const int pos1 = pos0 + 128;
        int lo = 0, hi = n_kv;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (indices[mid] < pos0) { lo = mid + 1; } else { hi = mid; }
        }
        int up = lo, hi2 = n_kv;
        while (up < hi2) {
            const int mid = (up + hi2) >> 1;
            if (indices[mid] < pos1) { up = mid + 1; } else { hi2 = mid; }
        }
        s_lo = lo;
        s_hi = up;
    }
    __syncthreads();

    const int lo = s_lo;
    const int n = s_hi - lo;
    if (n <= 0) {
        return;
    }

    extern __shared__ half tile[]; // max(k_dim,v_dim)*128 halves (slices)
    const int dim  = side == 0 ? k_dim : v_dim;
    const int bits = side == 0 ? k_bits : v_bits;
    const int rows = side == 0 ? dim : 128;
    const int cols = side == 0 ? 128 : dim;
    const uint8_t * records = side == 0 ? k_records : v_records;
    const size_t record_bytes = side == 0 ? k_record_bytes : v_record_bytes;
    const size_t payload_off  = side == 0 ? k_payload_off : v_payload_off;
    const size_t scale_off    = side == 0 ? k_scale_off   : v_scale_off;
    const size_t zp_off       = side == 0 ? k_zp_off      : v_zp_off;
    const size_t other_off    = side == 0 ? k_other_off   : v_other_off;
    half * out = side == 0 ? k_out : v_out;

    const int h = hh / n_slices;
    const int s = hh % n_slices;
    const int dim_full = dim * n_slices;
    const int n_heads_s = n_heads * n_slices;

    const uint8_t * record = records + (size_t(g) * n_heads_s + hh) * record_bytes;
    const half * scale_s = (const half *) (record + scale_off);
    const half * zp_s    = (const half *) (record + zp_off);
    const half * other_s = (const half *) (record + other_off);

    const int nn = rows * cols;
    for (int e = tid; e < nn; e += blockDim.x) {
        const int r = e / cols;
        const int c = e % cols;
        const uint8_t code = kvarn_unpack_rt(record + payload_off, size_t(r) * cols + c, bits);
        tile[e] = __float2half_rn((float(code) * __half2float(scale_s[r]) + __half2float(zp_s[r]))
                * __half2float(other_s[c]));
    }
    __syncthreads();

    for (int e = tid; e < n * dim; e += blockDim.x) {
        const int j = e / dim;
        const int d = e % dim;
        const int slot = indices[lo + j] & 127;
        const half v = side == 0 ? tile[d * 128 + slot] : tile[slot * dim + d];
        out[size_t(h) * dim_full + s * dim + d + size_t(lo + j) * dim_full * n_heads] = v;
    }
}

// Entry points for the combined variants.
void ggml_cuda_kvarn_stage_write2(
        const float * k_cur, const float * v_cur, const int32_t * indices,
        void * k_stage, void * v_stage,
        int k_dim, int v_dim, int n_heads, int n_tokens, int n_slices,
        int k_rows, int v_rows, int k_slots, int v_slots, cudaStream_t stream) {
    if (n_tokens <= 0) {
        return;
    }
    const int k_dim_s = k_dim / n_slices;
    const int v_dim_s = v_dim / n_slices;
    const int dim_max = k_dim_s > v_dim_s ? k_dim_s : v_dim_s;
    dim3 grid(n_tokens, n_heads, 2);
    kvarn_stage_write2_kernel<<<grid, dim_max, 0, stream>>>(
            k_cur, v_cur, indices, (half *) k_stage, (half *) v_stage,
            k_dim, v_dim, k_dim_s, v_dim_s, n_heads, n_slices, k_rows, v_rows, k_slots, v_slots);
    kvarn_launch_ok("kvarn stage write2");
}

void ggml_cuda_kvarn_materialize2(
        const struct ggml_kvarn_layout * k_layout, const struct ggml_kvarn_layout * v_layout,
        const void * k_records, const void * v_records,
        const void * k_stage, const void * v_stage,
        const int32_t * indices, const int32_t * state, void * k_out, void * v_out,
        int n_kv, int k_dim, int v_dim, int n_heads, int k_rows, int v_rows,
        int k_slots, int v_slots, int n_record_groups, int n_slices, int k_bits, int v_bits, cudaStream_t stream) {
    if (n_kv <= 0) {
        return;
    }
    const int k_dim_s = k_dim / n_slices;
    const int v_dim_s = v_dim / n_slices;
    const int n_heads_s = n_heads * n_slices;
    {
        const int dim_max = k_dim_s > v_dim_s ? k_dim_s : v_dim_s;
        dim3 grid(n_kv, n_heads_s, 2);
        kvarn_materialize_stage2_kernel<<<grid, dim_max, 0, stream>>>(
                indices, (const half *) k_stage, (const half *) v_stage,
                (half *) k_out, (half *) v_out, state,
                k_dim, v_dim, k_dim_s, v_dim_s, n_heads, n_slices, k_rows, v_rows, k_slots, v_slots);
        kvarn_launch_ok("kvarn materialize stage2");
    }
    {
        const int dim_max = k_dim_s > v_dim_s ? k_dim_s : v_dim_s;
        const size_t smem_bytes = size_t(dim_max) * 128 * sizeof(half);
        // fixed geometry: one block per record slot, blocks skip per device state
        dim3 grid(n_record_groups, n_heads_s, 2);
        kvarn_materialize_records2_kernel<<<grid, 256, smem_bytes, stream>>>(
                indices, (const uint8_t *) k_records, (const uint8_t *) v_records,
                (half *) k_out, (half *) v_out, state,
                n_kv, k_dim_s, v_dim_s, n_heads, n_slices, k_bits, v_bits,
                k_layout->record_bytes, v_layout->record_bytes,
                k_layout->k_payload_off, k_layout->k_scale_off, k_layout->k_zp_off, k_layout->k_other_off,
                v_layout->v_payload_off, v_layout->v_scale_off, v_layout->v_zp_off, v_layout->v_other_off);
        kvarn_launch_ok("kvarn materialize records2");
    }
}

// ---------------------------------------------------------------------------
// seal: fused stage -> record producer for all groups completed in a call
// ---------------------------------------------------------------------------

void ggml_cuda_kvarn_seal(
        const struct ggml_kvarn_layout * layout, const void * stage, void * records,
        const int32_t * state,
        int head_dim, int n_heads, int stage_rows, int stage_slots,
        int n_record_groups, int n_slices, bool value, int bits, int iters, float alpha,
        cudaStream_t stream) {
    if (getenv("IK_KVARN_SKIP_SEAL") != nullptr) {
        return; // timing diagnostic only
    }
    const int head_dim_s = head_dim / n_slices;
    const int n_heads_s  = n_heads * n_slices;
    const int rows = value ? 128 : head_dim_s;
    const int cols = value ? head_dim_s : 128;
    const size_t payload_off   = value ? layout->v_payload_off   : layout->k_payload_off;
    const size_t payload_bytes = value ? layout->v_payload_bytes : layout->k_payload_bytes;
    const size_t scale_off     = value ? layout->v_scale_off     : layout->k_scale_off;
    const size_t zp_off        = value ? layout->v_zp_off        : layout->k_zp_off;
    const size_t other_off     = value ? layout->v_other_off     : layout->k_other_off;
    const size_t smem_bytes = (size_t(rows) * cols + 3 * size_t(rows + cols) + 258) * sizeof(float);

    // fixed geometry: one block per record group; blocks skip themselves
    // unless their group is in (sealed, g_first), read from `state` on device
    dim3 grid(n_record_groups, n_heads_s);
#define KVARN_LAUNCH_SEAL(B) do { \
        static bool kvarn_attr_done_##B = false; \
        if (!kvarn_attr_done_##B) { \
            cudaFuncSetAttribute((const void *) kvarn_seal_kernel<B>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes); \
            kvarn_attr_done_##B = true; \
        } \
        kvarn_seal_kernel<B><<<grid, 128, smem_bytes, stream>>>( \
                (const half *) stage, (uint8_t *) records, state, \
                head_dim_s, n_heads_s, stage_rows, stage_slots, value ? 1 : 0, iters, alpha, \
                layout->record_bytes, payload_off, payload_bytes, scale_off, zp_off, other_off); \
        kvarn_launch_ok("kvarn seal"); \
    } while (0)
    switch (bits) {
        case 2: KVARN_LAUNCH_SEAL(2); break;
        case 3: KVARN_LAUNCH_SEAL(3); break;
        case 4: KVARN_LAUNCH_SEAL(4); break;
        case 5: KVARN_LAUNCH_SEAL(5); break;
        case 6: KVARN_LAUNCH_SEAL(6); break;
        case 8: KVARN_LAUNCH_SEAL(8); break;
        default:
            fprintf(stderr, "kvarn: seal: bits %d not instantiated\n", bits);
            GGML_ABORT("kvarn: seal: unsupported bits");
    }
#undef KVARN_LAUNCH_SEAL
}

// ---------------------------------------------------------------------------
// materialize: records + stage -> native cell-major F16 span
// ---------------------------------------------------------------------------

void ggml_cuda_kvarn_materialize(
        const struct ggml_kvarn_layout * layout,
        const void * records, const void * stage, const int32_t * indices,
        const int32_t * state,
        void * out, int n_kv, int head_dim, int n_heads, int stage_rows, int stage_slots,
        int n_record_groups, int n_slices, bool value, int bits, cudaStream_t stream) {
    if (n_kv <= 0) {
        return;
    }
    const int head_dim_s = head_dim / n_slices;
    const int n_heads_s  = n_heads * n_slices;
    {
        dim3 grid(n_kv, n_heads_s);
        kvarn_materialize_stage_kernel<<<grid, head_dim_s, 0, stream>>>(
                indices, (const half *) stage, (half *) out, state,
                head_dim, head_dim_s, n_heads, n_slices, stage_rows, stage_slots);
        kvarn_launch_ok("kvarn materialize stage");
    }
    {
        const size_t payload_off = value ? layout->v_payload_off : layout->k_payload_off;
        const size_t scale_off   = value ? layout->v_scale_off   : layout->k_scale_off;
        const size_t zp_off      = value ? layout->v_zp_off      : layout->k_zp_off;
        const size_t other_off   = value ? layout->v_other_off   : layout->k_other_off;
        const size_t smem_bytes  = size_t(128) * size_t(head_dim_s) * sizeof(half);
        // fixed geometry: one block per record slot, blocks skip per device state
        dim3 grid(n_record_groups, n_heads_s);
#define KVARN_LAUNCH_MAT(B) do { \
            kvarn_materialize_records_kernel<B><<<grid, 256, smem_bytes, stream>>>( \
                    indices, (const uint8_t *) records, state, (half *) out, \
                    n_kv, head_dim_s, n_heads_s, n_slices, \
                    value ? 1 : 0, layout->record_bytes, payload_off, scale_off, zp_off, other_off); \
            kvarn_launch_ok("kvarn materialize records"); \
        } while (0)
        switch (bits) {
            case 2: KVARN_LAUNCH_MAT(2); break;
            case 3: KVARN_LAUNCH_MAT(3); break;
            case 4: KVARN_LAUNCH_MAT(4); break;
            case 5: KVARN_LAUNCH_MAT(5); break;
            case 6: KVARN_LAUNCH_MAT(6); break;
            case 8: KVARN_LAUNCH_MAT(8); break;
            default:
                fprintf(stderr, "kvarn: materialize: bits %d not instantiated\n", bits);
                GGML_ABORT("kvarn: materialize: unsupported bits");
        }
#undef KVARN_LAUNCH_MAT
    }
}

// ---------------------------------------------------------------------------
// KVarN decode attention: one block per query head; scores straight from the
// structured cache (records + F16 ring/sink), in-shared softmax with the
// mask, then a per-channel PV phase. v1: one query token, no sinks/SWA/
// softcap; the caller gates those conditions and n_kv <= 8192.
// ---------------------------------------------------------------------------
__global__ void kvarn_attn_kernel(
        const float * __restrict__ q, const int q_head_stride,
        const uint8_t * __restrict__ k_records, const half * __restrict__ k_stage,
        const uint8_t * __restrict__ v_records, const half * __restrict__ v_stage,
        const int32_t * __restrict__ indices, const half * __restrict__ mask,
        float * __restrict__ out,
        const int n_kv, const int Dk, const int Dk_s, const int Dv, const int Dv_s, const int n_slices,
        const int Hq, const int Hkv,
        const int k_rows, const int v_rows, const int slots,
        const int32_t * __restrict__ state,
        const int k_bits, const int v_bits, const float kq_scale,
        const size_t k_record_bytes, const size_t v_record_bytes,
        const size_t k_payload_off, const size_t k_scale_off, const size_t k_zp_off, const size_t k_other_off,
        const size_t v_payload_off, const size_t v_scale_off, const size_t v_zp_off, const size_t v_other_off) {
    extern __shared__ float smem[];
    float * S   = smem;         // [n_kv]
    float * red = smem + n_kv;  // [blockDim.x]

    const int sealed = state[2]; // groups sealed by this call
    const int hq = blockIdx.x;
    const int h = hq / (Hq / Hkv);
    const int tid = threadIdx.x;
    const float * qh = q + size_t(hq) * q_head_stride;

    // phase 1: scores (accumulated over the 128-wide channel slices)
    for (int i = tid; i < n_kv; i += blockDim.x) {
        const int pos = indices[i];
        float s = 0.0f;
        for (int sl = 0; sl < n_slices; ++sl) {
            const int hs = h * n_slices + sl;
            const float * qs = qh + sl * Dk_s;
            if (pos < 128) {
                const half * row = k_stage + size_t(pos) * Dk_s + size_t(hs) * Dk_s * k_rows;
                for (int d = 0; d < Dk_s; ++d) {
                    s += qs[d] * __half2float(row[d]);
                }
            } else {
                const int g = pos / 128;
                const int slot = pos & 127;
                if (g > sealed) {
                    const int block = 1 + (g - 1) % slots;
                    const half * row = k_stage + size_t(block * 128 + slot) * Dk_s + size_t(hs) * Dk_s * k_rows;
                    for (int d = 0; d < Dk_s; ++d) {
                        s += qs[d] * __half2float(row[d]);
                    }
                } else {
                    const uint8_t * rec = k_records + ((size_t(g) * Hkv + h) * n_slices + sl) * k_record_bytes;
                    const half * ks = (const half *) (rec + k_scale_off);
                    const half * kz = (const half *) (rec + k_zp_off);
                    const half * ko = (const half *) (rec + k_other_off);
                    float acc = 0.0f;
                    for (int d = 0; d < Dk_s; ++d) {
                        const uint8_t code = kvarn_unpack_rt(rec + k_payload_off, size_t(d) * 128 + slot, k_bits);
                        acc += qs[d] * (float(code) * __half2float(ks[d]) + __half2float(kz[d]));
                    }
                    s += acc * __half2float(ko[slot]);
                }
            }
        }
        S[i] = s * kq_scale + __half2float(mask[i]);
    }
    __syncthreads();

    // phase 2a: max
    float lmax = -INFINITY;
    for (int i = tid; i < n_kv; i += blockDim.x) {
        lmax = fmaxf(lmax, S[i]);
    }
    red[tid] = lmax;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red[tid] = fmaxf(red[tid], red[tid + step]);
        }
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();

    // phase 2b: exp + sum, probabilities in place
    float lsum = 0.0f;
    for (int i = tid; i < n_kv; i += blockDim.x) {
        const float v = S[i];
        const float e = (v == -INFINITY) ? 0.0f : expf(v - mx);
        S[i] = e;
        lsum += e;
    }
    red[tid] = lsum;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red[tid] += red[tid + step];
        }
        __syncthreads();
    }
    const float inv_sum = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;

    // phase 3: PV, one thread per output channel (full head width; loops for Dv > blockDim)
    for (int dow = tid; dow < Dv; dow += blockDim.x) {
        const int sthr = dow / Dv_s;
        const int dthr = dow % Dv_s;
        const int hs = h * n_slices + sthr;
        float a = 0.0f;
        for (int i = 0; i < n_kv; ++i) {
            const float p = S[i] * inv_sum;
            if (p == 0.0f) {
                continue;
            }
            const int pos = indices[i];
            float v;
            if (pos < 128) {
                v = __half2float(v_stage[size_t(pos) * Dv_s + size_t(hs) * Dv_s * v_rows + dthr]);
            } else {
                const int g = pos / 128;
                const int slot = pos & 127;
                if (g > sealed) {
                    const int block = 1 + (g - 1) % slots;
                    v = __half2float(v_stage[size_t(block * 128 + slot) * Dv_s + size_t(hs) * Dv_s * v_rows + dthr]);
                } else {
                    const uint8_t * rec = v_records + ((size_t(g) * Hkv + h) * n_slices + sthr) * v_record_bytes;
                    const half * vs = (const half *) (rec + v_scale_off);
                    const half * vz = (const half *) (rec + v_zp_off);
                    const half * vo = (const half *) (rec + v_other_off);
                    const uint8_t code = kvarn_unpack_rt(rec + v_payload_off, size_t(slot) * Dv_s + dthr, v_bits);
                    v = (float(code) * __half2float(vs[slot]) + __half2float(vz[slot])) * __half2float(vo[dthr]);
                }
            }
            a += p * v;
        }
        out[size_t(hq) * Dv + dow] = a;
    }
}

// ---------------------------------------------------------------------------
// op wrappers (dispatched from ggml-cuda.cu)
// ---------------------------------------------------------------------------

// Debug aid for determinism investigations: dump a device buffer to
// <dir>/<name>.bin. Enabled by IK_KVARN_DUMP_MAT=<dir>; the call range is
// selected with IK_KVARN_DUMP_STORE_FROM/_TO and IK_KVARN_DUMP_MAT_FROM/_TO
// (K side only, 0-based per-op counters across the whole process).
static void kvarn_dump_dev(const void * dev, size_t bytes, const char * name, cudaStream_t stream) {
    const char * dir = getenv("IK_KVARN_DUMP_MAT");
    if (dir == nullptr) {
        return;
    }
    // a D2H copy + host sync is illegal while the stream is being captured
    // (KVarN kernels participate in CUDA graph captures)
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone) {
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    std::vector<char> host(bytes);
    cudaMemcpyAsync(host.data(), dev, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    FILE * f = fopen(path, "wb");
    if (f != nullptr) {
        fwrite(host.data(), 1, bytes, f);
        fclose(f);
    }
}

static bool kvarn_dump_range(const char * from_env, const char * to_env, int counter) {
    const char * from = getenv(from_env);
    if (from == nullptr) {
        return false;
    }
    const char * to = getenv(to_env);
    return counter >= atoi(from) && counter <= (to != nullptr ? atoi(to) : atoi(from));
}

static int32_t kvarn_op_int(const struct ggml_tensor * dst, int index) {
    int32_t value;
    memcpy(&value, (const char *) dst->op_params + index * sizeof(int32_t), sizeof(value));
    return value;
}

static float kvarn_op_float(const struct ggml_tensor * dst, int index) {
    float value;
    memcpy(&value, (const char *) dst->op_params + index * sizeof(int32_t), sizeof(value));
    return value;
}

// The two ops mutate the cache tensors, so every tensor must be on this
// device; a mixed CPU/GPU assignment would silently write the wrong copy.
static bool kvarn_all_device(const struct ggml_tensor * const * tensors, int n) {
    for (int i = 0; i < n; ++i) {
        const struct ggml_tensor * t = tensors[i];
        if (t == nullptr || t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
            return false;
        }
    }
    return true;
}

// Combined K+V store (one launch for both stage writes, one seal launch per
// side when groups complete).
static void kvarn_store2_op(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_cur   = dst->src[0];
    const ggml_tensor * v_cur   = dst->src[1];
    const ggml_tensor * indices = dst->src[2];
    const ggml_tensor * k_stage = dst->src[3];
    const ggml_tensor * v_stage = dst->src[4];
    const ggml_tensor * k_rec   = dst->src[5];
    const ggml_tensor * v_rec   = dst->src[6];

    const ggml_tensor * srcs[7] = { k_cur, v_cur, indices, k_stage, v_stage, k_rec, v_rec };
    if (!kvarn_all_device(srcs, 7) || dst->buffer == nullptr || ggml_backend_buffer_is_host(dst->buffer)) {
        GGML_ABORT("KVarN store2: mixed CPU/GPU KV-cache offload is unsupported "
                   "(all tensors must be CUDA-resident)");
    }
    if (k_cur == nullptr || v_cur == nullptr) {
        GGML_ABORT("KVarN store2: both sides must be present");
    }

    const int   bits  = kvarn_op_int(dst, GGML_KVARN_OP_BITS);
    const int   other = kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS);
    const int   iters = kvarn_op_int(dst, GGML_KVARN_OP_ITERS);
    const float alpha = kvarn_op_float(dst, GGML_KVARN_OP_ALPHA);
    const int   S       = kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP);
    const int   n_slices = kvarn_op_int(dst, GGML_KVARN_OP_SLICES);

    const int kD = (int) k_cur->ne[0];
    const int vD = (int) v_cur->ne[0];
    const int H  = (int) k_cur->ne[1];
    const int T  = (int) k_cur->ne[2];
    const int k_rows = (int) k_stage->ne[1];
    const int v_rows = (int) v_stage->ne[1];
    const int slots  = S - 1;
    GGML_ASSERT(n_slices >= 1 && n_slices <= 4);
    GGML_ASSERT(kD % n_slices == 0 && vD % n_slices == 0);
    const int kD_s = kD / n_slices;
    const int vD_s = vD / n_slices;
    const int H_s  = H * n_slices;
    GGML_ASSERT(k_stage->ne[0] == kD_s && k_stage->ne[2] == H_s);
    GGML_ASSERT(v_stage->ne[0] == vD_s && v_stage->ne[2] == H_s);
    GGML_ASSERT(v_cur->ne[1] == H && v_cur->ne[2] == T);
    GGML_ASSERT(S >= 2 && k_rows >= S * 128 && v_rows >= S * 128);

    const struct ggml_kvarn_layout k_layout = ggml_kvarn_layout_make(kD_s, 128, bits, other);
    const struct ggml_kvarn_layout v_layout = ggml_kvarn_layout_make(vD_s, 128, bits, other);
    GGML_ASSERT((size_t) k_rec->ne[0] == k_layout.record_bytes);
    GGML_ASSERT((size_t) v_rec->ne[0] == v_layout.record_bytes);

    cudaStream_t stream = ctx.stream();

    // per-call group state (device memory; fixed launch geometry below)
    const ggml_tensor * state = dst->src[7];
    GGML_ASSERT(state != nullptr && state->type == GGML_TYPE_I32);
    // ONE shared state tensor; this op's 3 ints live at state + 3*layer
    const int32_t * st = (const int32_t *) state->data + 3*kvarn_op_int(dst, GGML_KVARN_OP_LAYER);

    if (getenv("IK_KVARN_SKIP_STORE") != nullptr) {
        return; // timing diagnostic only
    }

    // 1. seal completed groups for both sides (MUST run before the stage
    // writes). Unconditional fixed-geometry launches; every block skips
    // itself unless its group is in (sealed, g_first), read from `state`
    // on device so captured CUDA graphs replay with current cutoffs.
    const int n_record_groups = (int) (k_rec->ne[1] / H_s);
    ggml_cuda_kvarn_seal(&k_layout, k_stage->data, k_rec->data, st,
            kD, H, k_rows, slots, n_record_groups, n_slices, false, bits, iters, alpha, stream);
    ggml_cuda_kvarn_seal(&v_layout, v_stage->data, v_rec->data, st,
            vD, H, v_rows, slots, n_record_groups, n_slices, true, other, iters, alpha, stream);

    // 2. one launch writes both stages
    ggml_cuda_kvarn_stage_write2((const float *) k_cur->data, (const float *) v_cur->data,
            (const int32_t *) indices->data, k_stage->data, v_stage->data,
            kD, vD, H, T, n_slices, k_rows, v_rows, slots, slots, stream);

    // note: the CPU reference's 1.0f marker output is not written (nothing
    // consumes it), as in the single-side store.
}

// Combined K+V materialize (one launch for the sink/live copy, one for the
// sealed records, both serving K and V).
static void kvarn_materialize2_op(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_rec   = dst->src[0];
    const ggml_tensor * k_stage = dst->src[1];
    const ggml_tensor * indices = dst->src[2];
    const ggml_tensor * v_rec   = dst->src[3];
    const ggml_tensor * v_stage = dst->src[4];
    const ggml_tensor * v_flat  = dst->src[5];
    const ggml_tensor * state   = dst->src[6];

    const ggml_tensor * srcs[7] = { k_rec, k_stage, indices, v_rec, v_stage, v_flat, state };
    if (!kvarn_all_device(srcs, 7) || dst->buffer == nullptr || ggml_backend_buffer_is_host(dst->buffer)) {
        GGML_ABORT("KVarN materialize2: mixed CPU/GPU KV-cache offload is unsupported "
                   "(all tensors must be CUDA-resident)");
    }
    GGML_ASSERT(state != nullptr && state->type == GGML_TYPE_I32);
    const int32_t * st = (const int32_t *) state->data + 3*kvarn_op_int(dst, GGML_KVARN_OP_LAYER);

    const int k_bits = kvarn_op_int(dst, GGML_KVARN_OP_BITS);
    const int v_bits = kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS);
    const int S      = kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP);
    const int n_slices = kvarn_op_int(dst, GGML_KVARN_OP_SLICES);

    const int kD_s = (int) k_stage->ne[0];
    const int vD_s = (int) v_stage->ne[0];
    const int H_s  = (int) k_stage->ne[2];
    GGML_ASSERT(n_slices >= 1 && n_slices <= 4);
    GGML_ASSERT(H_s % n_slices == 0);
    const int H    = H_s / n_slices;
    const int kD   = kD_s * n_slices;   // full head dims
    const int vD   = vD_s * n_slices;
    const int n_kv = (int) dst->ne[1];
    const int k_rows = (int) k_stage->ne[1];
    const int v_rows = (int) v_stage->ne[1];
    const int slots  = S - 1;
    GGML_ASSERT(dst->ne[0] == kD_s * H_s);
    GGML_ASSERT(v_flat->ne[0] == vD_s * H_s && v_flat->ne[1] == n_kv);
    GGML_ASSERT(v_stage->ne[2] == H_s);
    GGML_ASSERT(S >= 2 && k_rows >= S * 128 && v_rows >= S * 128);

    const struct ggml_kvarn_layout k_layout = ggml_kvarn_layout_make(kD_s, 128, k_bits, v_bits);
    const struct ggml_kvarn_layout v_layout = ggml_kvarn_layout_make(vD_s, 128, k_bits, v_bits);
    GGML_ASSERT((size_t) k_rec->ne[0] == k_layout.record_bytes);
    GGML_ASSERT((size_t) v_rec->ne[0] == v_layout.record_bytes);

    const int n_record_groups = (int) (k_rec->ne[1] / H_s);

    ggml_cuda_kvarn_materialize2(&k_layout, &v_layout,
            k_rec->data, v_rec->data, k_stage->data, v_stage->data,
            (const int32_t *) indices->data, st, dst->data, v_flat->data,
            n_kv, kD, vD, H, k_rows, v_rows, slots, slots, n_record_groups, n_slices, k_bits, v_bits, ctx.stream());
}

void ggml_cuda_kvarn_attn_op(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k_rec   = dst->src[1];
    const ggml_tensor * k_stage = dst->src[2];
    const ggml_tensor * v_rec   = dst->src[3];
    const ggml_tensor * v_stage = dst->src[4];
    const ggml_tensor * indices = dst->src[5];
    const ggml_tensor * mask    = dst->src[6];
    const ggml_tensor * state   = dst->src[7];

    const ggml_tensor * srcs[8] = { q, k_rec, k_stage, v_rec, v_stage, indices, mask, state };
    if (!kvarn_all_device(srcs, 8) || dst->buffer == nullptr || ggml_backend_buffer_is_host(dst->buffer)) {
        GGML_ABORT("KVarN attn: mixed CPU/GPU KV-cache offload is unsupported "
                   "(all tensors must be CUDA-resident)");
    }
    GGML_ASSERT(state != nullptr && state->type == GGML_TYPE_I32);
    // attn keeps the layer index in its own slot 7
    const int32_t * st = (const int32_t *) state->data + 3*kvarn_op_int(dst, 7);

    const int   n_kv    = kvarn_op_int(dst, 0);
    const int   S       = kvarn_op_int(dst, 2);
    const int   k_bits  = kvarn_op_int(dst, 3);
    const int   v_bits  = kvarn_op_int(dst, 4);
    const float kq_scale = kvarn_op_float(dst, 5);
    const int   n_slices = kvarn_op_int(dst, 6);

    const int Dk   = (int) q->ne[0];            // full K head dim
    const int Dk_s = (int) k_stage->ne[0];
    const int Dv_s = (int) v_stage->ne[0];
    const int Dv   = Dv_s * n_slices;           // full V head dim
    const int Hq   = (int) q->ne[2];
    GGML_ASSERT(n_slices >= 1 && n_slices <= 4);
    GGML_ASSERT(Dk == Dk_s * n_slices);
    GGML_ASSERT(k_stage->ne[2] % n_slices == 0);
    const int Hkv  = (int) k_stage->ne[2] / n_slices;
    const int k_rows = (int) k_stage->ne[1];
    const int v_rows = (int) v_stage->ne[1];
    const int slots  = S - 1;
    GGML_ASSERT(Hkv > 0 && Hq % Hkv == 0);
    GGML_ASSERT(q->ne[1] == 1);
    GGML_ASSERT(n_kv > 0 && n_kv <= 8192);
    GGML_ASSERT((Dk_s == 64 || Dk_s == 128) && (Dv_s == 64 || Dv_s == 128));

    const struct ggml_kvarn_layout k_layout = ggml_kvarn_layout_make(Dk_s, 128, k_bits, v_bits);
    const struct ggml_kvarn_layout v_layout = ggml_kvarn_layout_make(Dv_s, 128, k_bits, v_bits);
    GGML_ASSERT((size_t) k_rec->ne[0] == k_layout.record_bytes);
    GGML_ASSERT((size_t) v_rec->ne[0] == v_layout.record_bytes);

    const size_t smem_bytes = (size_t(n_kv) + 256) * sizeof(float);
    kvarn_attn_kernel<<<Hq, 256, smem_bytes, ctx.stream()>>>(
            (const float *) q->data, (int) (q->nb[2] / sizeof(float)),
            (const uint8_t *) k_rec->data, (const half *) k_stage->data,
            (const uint8_t *) v_rec->data, (const half *) v_stage->data,
            (const int32_t *) indices->data, (const half *) mask->data,
            (float *) dst->data,
            n_kv, Dk, Dk_s, Dv, Dv_s, n_slices, Hq, Hkv, k_rows, v_rows, slots,
            st, k_bits, v_bits, kq_scale,
            k_layout.record_bytes, v_layout.record_bytes,
            k_layout.k_payload_off, k_layout.k_scale_off, k_layout.k_zp_off, k_layout.k_other_off,
            v_layout.v_payload_off, v_layout.v_scale_off, v_layout.v_zp_off, v_layout.v_other_off);
    kvarn_launch_ok("kvarn attn");
}

void ggml_cuda_kvarn_store_op(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (kvarn_op_int(dst, GGML_KVARN_OP_COMBINED) != 0) {
        kvarn_store2_op(ctx, dst);
        return;
    }
    const ggml_tensor * current = dst->src[0];
    const ggml_tensor * indices = dst->src[1];
    const ggml_tensor * stage   = dst->src[2];
    const ggml_tensor * records = dst->src[3];
    const ggml_tensor * state   = dst->src[4];

    const ggml_tensor * srcs[5] = { current, indices, stage, records, state };
    if (!kvarn_all_device(srcs, 5) ||
            dst->buffer == nullptr || ggml_backend_buffer_is_host(dst->buffer)) {
        GGML_ABORT("KVarN store: mixed CPU/GPU KV-cache offload is unsupported "
                   "(all tensors must be CUDA-resident)");
    }
    GGML_ASSERT(state != nullptr && state->type == GGML_TYPE_I32);
    const int32_t * st = (const int32_t *) state->data + 3*kvarn_op_int(dst, GGML_KVARN_OP_LAYER);

    const bool  value   = kvarn_op_int(dst, GGML_KVARN_OP_VALUE) != 0;
    const int   bits    = kvarn_op_int(dst, GGML_KVARN_OP_BITS);
    const int   other   = kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS);
    const int   iters   = kvarn_op_int(dst, GGML_KVARN_OP_ITERS);
    const float alpha   = kvarn_op_float(dst, GGML_KVARN_OP_ALPHA);
    const int   S       = kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP);
    const int   n_slices = kvarn_op_int(dst, GGML_KVARN_OP_SLICES);

    const int D   = (int) current->ne[0];   // full head dim
    const int D_s = (int) stage->ne[0];     // 128-wide slice
    const int H   = (int) current->ne[1];
    const int T   = (int) current->ne[2];
    const int H_s = (int) stage->ne[2];
    const int stage_rows  = (int) stage->ne[1];
    const int stage_slots = S - 1;
    GGML_ASSERT(n_slices >= 1 && n_slices <= 4);
    GGML_ASSERT(D == D_s * n_slices && H_s == H * n_slices);
    GGML_ASSERT(S >= 2 && stage_rows >= S * 128);
    const struct ggml_kvarn_layout layout = ggml_kvarn_layout_make(
            D_s, 128, value ? other : bits, value ? bits : other);
    GGML_ASSERT((size_t) records->ne[0] == layout.record_bytes);

    cudaStream_t stream = ctx.stream();

    if (getenv("IK_KVARN_SKIP_STORE") != nullptr) {
        return; // timing diagnostic only
    }

    // 1. seal every group completed in a previous call (one-call deferral).
    // This MUST run before the stage writes: a group's ring block may be
    // reused by a group of the current call (e.g. at ub=512, group 7 wraps to
    // block 1 while group 1 is being sealed from it), and the CPU reference
    // seals before writing for the same reason. Unconditional fixed-geometry
    // launch; every block skips itself unless its group is in (sealed,
    // g_first), read from `state` on device.
    {
        static int dbg_pre_k = 0;
        static int dbg_pre_v = 0;
        int & pctr = value ? dbg_pre_v : dbg_pre_k;
        if (kvarn_dump_range("IK_KVARN_DUMP_PRE_FROM", "IK_KVARN_DUMP_PRE_TO", pctr)) {
            char name[64];
            snprintf(name, sizeof(name), "pre_%c_%03d", value ? 'v' : 'k', pctr);
            kvarn_dump_dev(stage->data, (size_t) stage_rows * D * H * sizeof(uint16_t), name, stream);
        }
        ++pctr;
    }
    ggml_cuda_kvarn_seal(&layout, stage->data, records->data, st,
            D, H, stage_rows, stage_slots, (int) (records->ne[1] / H_s), n_slices, value, bits, iters, alpha, stream);

    // 2. stage write (ring addressing)
    ggml_cuda_kvarn_stage_write((const float *) current->data, (const int32_t *) indices->data,
            stage->data, D, H, T, stage_rows, stage_slots, n_slices, stream);

    // Note: the CPU reference writes a 1.0f marker into the op output; nothing
    // consumes it, so the CUDA path skips that launch (one less kernel per
    // store per layer).

    {
        static int dbg_store_k = 0;
        static int dbg_store_v = 0;
        int & ctr = value ? dbg_store_v : dbg_store_k;
        if (kvarn_dump_range("IK_KVARN_DUMP_STORE_FROM", "IK_KVARN_DUMP_STORE_TO", ctr)) {
            char name[64];
            snprintf(name, sizeof(name), "stage_%c_%03d", value ? 'v' : 'k', ctr);
            kvarn_dump_dev(stage->data, (size_t) stage_rows * D * H * sizeof(uint16_t), name, stream);
            snprintf(name, sizeof(name), "rec_%c_%03d", value ? 'v' : 'k', ctr);
            kvarn_dump_dev(records->data, ggml_nbytes(records), name, stream);
        }
        ++ctr;
    }
}

void ggml_cuda_kvarn_materialize_op(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (getenv("IK_KVARN_SKIP_MAT") != nullptr) {
        // timing diagnostic only: skip the materialize launches
        return;
    }
    if (kvarn_op_int(dst, GGML_KVARN_OP_COMBINED) != 0) {
        kvarn_materialize2_op(ctx, dst);
        return;
    }
    const ggml_tensor * records = dst->src[0];
    const ggml_tensor * stage   = dst->src[1];
    const ggml_tensor * indices = dst->src[2];
    const ggml_tensor * state   = dst->src[3];

    const ggml_tensor * srcs[4] = { records, stage, indices, state };
    if (!kvarn_all_device(srcs, 4) ||
            dst->buffer == nullptr || ggml_backend_buffer_is_host(dst->buffer)) {
        GGML_ABORT("KVarN materialize: mixed CPU/GPU KV-cache offload is unsupported "
                   "(all tensors must be CUDA-resident)");
    }
    GGML_ASSERT(state != nullptr && state->type == GGML_TYPE_I32);
    const int32_t * st = (const int32_t *) state->data + 3*kvarn_op_int(dst, GGML_KVARN_OP_LAYER);

    const bool value  = kvarn_op_int(dst, GGML_KVARN_OP_VALUE) != 0;
    const int  bits   = kvarn_op_int(dst, GGML_KVARN_OP_BITS);
    const int  other  = kvarn_op_int(dst, GGML_KVARN_OP_OTHER_BITS);
    const int  S      = kvarn_op_int(dst, GGML_KVARN_OP_STAGE_GRP);
    const int  n_slices = kvarn_op_int(dst, GGML_KVARN_OP_SLICES);

    const int D_s = (int) stage->ne[0];
    const int H_s = (int) stage->ne[2];
    GGML_ASSERT(n_slices >= 1 && n_slices <= 4);
    GGML_ASSERT(H_s % n_slices == 0);
    const int H = H_s / n_slices;
    const int D = D_s * n_slices;   // full head dim
    const int n_kv = (int) dst->ne[1];
    const int stage_rows  = (int) stage->ne[1];
    const int stage_slots = S - 1;
    GGML_ASSERT(dst->ne[0] == D_s * H_s);
    GGML_ASSERT(S >= 2 && stage_rows >= S * 128);

    const struct ggml_kvarn_layout layout = ggml_kvarn_layout_make(
            D_s, 128, value ? other : bits, value ? bits : other);
    GGML_ASSERT((size_t) records->ne[0] == layout.record_bytes);

    const int n_record_groups = (int) (records->ne[1] / H_s);

    ggml_cuda_kvarn_materialize(&layout, records->data, stage->data,
            (const int32_t *) indices->data, st, dst->data, n_kv, D, H,
            stage_rows, stage_slots, n_record_groups, n_slices, value, bits, ctx.stream());

    {
        static int dbg_mat_k = 0;
        static int dbg_mat_v = 0;
        int & ctr = value ? dbg_mat_v : dbg_mat_k;
        if (kvarn_dump_range("IK_KVARN_DUMP_MAT_FROM", "IK_KVARN_DUMP_MAT_TO", ctr)) {
            char name[64];
            snprintf(name, sizeof(name), "mat_%c_%03d", value ? 'v' : 'k', ctr);
            kvarn_dump_dev(dst->data, (size_t) n_kv * D * H * sizeof(uint16_t), name, ctx.stream());
        }
        ++ctr;
    }
}
