#pragma once

// KVarN MMA (tensor-core) batched kernels.
//
// Design (settled; see the KVarN CUDA design notes):
//   score(q,t) = k_other[t] * ( sum_d q[d]*k_scale[d]*code(d,t) + C_q )
//   C_q        = sum_d q[d]*k_zp[d]                     (per-query constant)
//
// Q' = q*k_scale (record-specific!) is quantized to int8 per query with a
// per-(query, record) scale qs; the 4-bit codes (0..15) are consumed directly
// as signed int8; the channel dimension is reduced with
// mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 (M=16 queries, N=8 keys,
// K=16 channels); epilogue: (acc*qs + C_q) * k_other[t].
//
// PV (planned): out(q,c) = v_other[c] * ( sum_t (P*v_scale)[t]*code(t,c) + C'_q )
//   C'_q = sum_t P[q,t]*v_zp[t]; fold v_scale into P, int8 mma vs int2 codes.
//
// STATUS: written 2026-10-01, not yet compiled or validated. Build on go.

#include <cstddef>
#include <cstdint>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Batched QK with tensor cores: M = 16 queries per record.
//   q        : [16 * head_dim] f32, a 16-query tile (rotated, one head slice)
//   records  : [record_bytes, n_records]
//   scores   : [16 * n_records * 128] f32, query-major (q * n_records*128)
void ggml_cuda_kvarn_mma_kq_records(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const float * q,
        int head_dim,
        int n_records,
        size_t record_bytes,
        float * scores);

// M=64 variant (4 m-tiles per warp; B fragments amortized 4x) — prefill path.
// q: [64 * head_dim]; scores: [64 * n_records * 128], query-major.
void ggml_cuda_kvarn_mma_kq_records64(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const float * q,
        int head_dim,
        int n_records,
        size_t record_bytes,
        float * scores);

// M=128 variant (2 warps x 4 m-tiles; record staging shared across warps).
// q: [128 * head_dim]; scores: [128 * n_records * 128], query-major.
void ggml_cuda_kvarn_mma_kq_records128(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const float * q,
        int head_dim,
        int n_records,
        size_t record_bytes,
        float * scores);

// Batched PV with tensor cores: M = 16 queries per record; accumulates over
// records into out (caller zeroes it). weights: [16 * 128] P for one record.
//   out[16 * head_dim] += v_other[c]*(sum_t (P*v_scale)[t]*code + C'_q)
void ggml_cuda_kvarn_mma_pv_records(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const float * weights,
        int head_dim,
        int n_records,
        size_t record_bytes,
        float * out);

// M=64 PV variant (prefill path); weights: [n_records][64][128] P values.
void ggml_cuda_kvarn_mma_pv_records64(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const float * weights,
        int head_dim,
        int n_records,
        size_t record_bytes,
        float * out);

// M=128 PV variant (2 warps x 4 m-tiles); weights: [n_records][128][128].
void ggml_cuda_kvarn_mma_pv_records128(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const float * weights,
        int head_dim,
        int n_records,
        size_t record_bytes,
        float * out);

#ifdef __cplusplus
}
#endif
