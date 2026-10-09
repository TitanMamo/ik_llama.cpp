#pragma once

// KVarN CUDA kernels.
//
// The record layout and the quantize/dequantize math are defined by the CPU
// reference in ggml/src/ggml-kvarn.cpp; the code below must stay bit-faithful
// to it (4-bit little-endian bit packing inside each payload byte, LSB first).
//
// This first entry point set is intentionally narrow: it covers the consumer
// side of ONE record (128 tokens, one 128-wide head slice) so the kernels can
// be validated against the CPU reference before attention integration.
//   K record: K[d,t]  = (code(d,t)*k_scale[d] + k_zp[d]) * k_other[t]
//   V record: V[t,c]  = (code(t,c)*v_scale[t] + v_zp[t]) * v_other[c]
// (K payload index d*128+t; V payload index t*head_dim+c; both row-major.)

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// scores[t] = k_other[t] * sum_d q[d] * (code(d,t)*k_scale[d] + k_zp[d])
// q: [head_dim] f32 (rotated query slice, one head)
// scores: [128] f32
void ggml_cuda_kvarn_kq_record(
        const struct ggml_kvarn_layout * layout,
        const void * record,
        const float * q,
        int head_dim,
        float * scores);

// tile[t*head_dim + c] = (code(t,c)*v_scale[t] + v_zp[t]) * v_other[c]
// tile: [128 * head_dim] f32 row-major, token-major
void ggml_cuda_kvarn_v_record(
        const struct ggml_kvarn_layout * layout,
        const void * record,
        int head_dim,
        float * tile);

// Batched record variants (MMVQ decode shape):
//   n_records records of one head slice; scores[rec*128 + t].
void ggml_cuda_kvarn_kq_records(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const float * q,
        int head_dim,
        int n_records,
        size_t record_bytes,
        float * scores);

//   out[c] += sum_t weights[rec*128 + t] * V_rec[t,c]  (accumulates over records)
void ggml_cuda_kvarn_pv_records(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const float * weights,
        int head_dim,
        int n_records,
        size_t record_bytes,
        float * out);

// K dequant: one record -> tile [head_dim, 128] f32 (channel-major).
void ggml_cuda_kvarn_k_record(
        const struct ggml_kvarn_layout * layout,
        const void * record,
        int head_dim,
        float * tile);

// Store (producer): quantize one tile [rows, cols] f16 row-major into a record.
// rows/cols are implied by `value`: K [head_dim, group], V [group, head_dim].
// bits 2 and 4 are instantiated; alpha=1 is the reference KVarN.
void ggml_cuda_kvarn_store_tile(
        const struct ggml_kvarn_layout * layout,
        const void * tile_f16,
        int head_dim,
        int group,
        bool value,
        int bits,
        int iters,
        float alpha,
        void * record);

// Stage write: current F32 [head_dim, n_heads, n_tokens] -> stage F16
// [head_dim_s, stage_rows, n_heads*n_slices] (head_dim_s = head_dim/n_slices).
// Ring addressing: the sink (pos < 128) lives in block 0, group g >= 1 in
// block 1 + (g-1) % stage_slots.
void ggml_cuda_kvarn_stage_write(
        const float * current,
        const int32_t * indices,
        void * stage,
        int head_dim,
        int n_heads,
        int n_tokens,
        int stage_rows,
        int stage_slots,
        int n_slices,
        cudaStream_t stream);

// Seal (producer): quantize the completed ring groups in (sealed, g_first) of
// `stage` into their records. `state` is a device I32[3]: [0]=g_first,
// [1]=sealed before this call, [2]=sealed after (materialize/attn cutoff).
// Fixed geometry: one block per (record group, head-slice); each block skips
// itself unless its group is in range, so captured CUDA graphs replays with
// current cutoffs. `bits` selects the payload width of this side, `layout`
// carries both sides' offsets.
void ggml_cuda_kvarn_seal(
        const struct ggml_kvarn_layout * layout,
        const void * stage,
        void * records,
        const int32_t * state,
        int head_dim,
        int n_heads,
        int stage_rows,
        int stage_slots,
        int n_record_groups,
        int n_slices,
        bool value,
        int bits,
        int iters,
        float alpha,
        cudaStream_t stream);

// Materialize: reconstruct the native cell-major F16 span [head_dim*n_heads,
// n_kv] (element (d,i,h) at h*head_dim + s*head_dim_s + d + i*head_dim*n_heads).
// Sink (pos < 128) and live ring groups are copied from `stage`; groups
// 1..state[2] are decoded from `records`. `indices` must be ascending (v1: one
// contiguous token stream); duplicate positions are handled by the
// lower/upper bounds.
void ggml_cuda_kvarn_materialize(
        const struct ggml_kvarn_layout * layout,
        const void * records,
        const void * stage,
        const int32_t * indices,
        const int32_t * state,
        void * out,
        int n_kv,
        int head_dim,
        int n_heads,
        int stage_rows,
        int stage_slots,
        int n_record_groups,
        int n_slices,
        bool value,
        int bits,
        cudaStream_t stream);

// Gather one stage block into a row-major F16 tile [rows, cols] for head `h`
// (K: rows=head_dim/cols=128; V: rows=128/cols=head_dim).
void ggml_cuda_kvarn_gather(
        const void * stage,
        void * tile,
        int head_dim,
        int stage_rows,
        int n_heads,
        int rows,
        int cols,
        int block,
        bool value,
        int h);

#ifdef __cplusplus
}
#endif
