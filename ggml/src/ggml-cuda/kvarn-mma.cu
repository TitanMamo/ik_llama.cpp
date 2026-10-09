#include "kvarn-mma.cuh"

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (int8 tensor core; 4 bytes/reg)
static __device__ __forceinline__ void kvarn_mma_m16n8k32_s8(
        int32_t c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ uint32_t kvarn_pack_s8(const int b0, const int b1, const int b2, const int b3) {
    return (uint32_t) (uint8_t) b0 | ((uint32_t) (uint8_t) b1 << 8) |
           ((uint32_t) (uint8_t) b2 << 16) | ((uint32_t) (uint8_t) b3 << 24);
}

// expand a nibble-pair uint16 [c0 | c1<<4 | (c2 | c3<<4)<<8] into pack_s8 byte order
static __device__ __forceinline__ uint32_t kvarn_expand_nibbles(const uint16_t u) {
    return (uint32_t) (u & 0x000F) | ((uint32_t) (u & 0x00F0) << 4) |
           ((uint32_t) (u & 0x0F00) << 8) | ((uint32_t) (u & 0xF000) << 12);
}

// 2-bit code of (token t, channel c) in the V payload (row-major [128, head_dim])
static __device__ __forceinline__ int kvarn_code2(const uint8_t * __restrict__ payload,
        const int head_dim, const int t, const int c) {
    const int idx = t * head_dim + c;
    return (payload[idx >> 2] >> (2 * (idx & 3))) & 0x3;
}

// ---------------------------------------------------------------------------
// QK: M = MT*16*W queries per record; W warps, each owning MT 16-query tiles.
// score(q,t) = k_other[t] * ( sum_d q[d]*k_scale[d]*code(d,t) + C_q )
//
// The K payload is transposed once per block into a nibble-pair byte table
//   spt2[t][p] = code(2p, t) | (code(2p+1, t) << 4)      p = channel pair
// (row pitch 68 -> 4-byte-row padding spreads shared banks). Each B fragment
// word is then ONE uint16 load + a 4-nibble expand, instead of 8 per-element
// nibble extractions + packing.
// ---------------------------------------------------------------------------

template <int MT, int W>
__global__ void kvarn_mma_kq_records_kernel(
        const uint8_t * __restrict__ records, const float * __restrict__ q,
        const int head_dim, const size_t record_bytes, float * __restrict__ scores,
        const size_t k_payload_off, const size_t k_scale_off,
        const size_t k_zp_off, const size_t k_other_off, const int score_stride) {
    constexpr int NQ     = MT * 16;
    constexpr int QPITCH = 132; // qi8 row pitch: padding spreads shared banks

    __shared__ __align__(16) int8_t qi8[W * NQ * QPITCH];
    __shared__ float  qscale[W * NQ];
    __shared__ float  cq[W * NQ];
    __shared__ float  ko[128];
    __shared__ __align__(16) float ksf[128];
    __shared__ __align__(16) float kzf[128];
    __shared__ __align__(4) uint8_t spt2[128 * 68];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int rec  = blockIdx.x;
    const uint8_t * __restrict__ record = records + size_t(rec) * record_bytes;
    const half * __restrict__ k_scale = (const half *) (record + k_scale_off);
    const half * __restrict__ k_zp    = (const half *) (record + k_zp_off);
    const half * __restrict__ k_other = (const half *) (record + k_other_off);

    for (int i = tid; i < 128; i += 32 * W) {
        ko[i]  = __half2float(k_other[i]);
        ksf[i] = __half2float(k_scale[i]);
        kzf[i] = __half2float(k_zp[i]);
    }

    // Q' = round(q*k_scale / qs) per query; C_q = sum_d q[d]*k_zp[d]
    // (float4 chunks; |q'| <= 127 by construction so no clamp is needed).
    // q == nullptr is a timing/debug switch that skips the prep.
    if (q != nullptr) {
        for (int gg = warp * NQ + lane; gg < (warp + 1) * NQ; gg += 32) {
            const float * qr = q + size_t(gg) * head_dim;
            float amax = 0.0f;
            for (int d = 0; d < head_dim; d += 4) {
                const float4 q4 = *(const float4 *) (qr + d);
                const float4 s4 = *(const float4 *) (ksf + d);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(q4.x * s4.x), fabsf(q4.y * s4.y)),
                                         fmaxf(fabsf(q4.z * s4.z), fabsf(q4.w * s4.w))));
            }
            const float qs    = fmaxf(amax / 127.0f, 1e-10f);
            const float invqs = 1.0f / qs;
            float cqv = 0.0f;
            int8_t * row = qi8 + size_t(gg) * QPITCH;
            for (int d = 0; d < head_dim; d += 4) {
                const float4 q4 = *(const float4 *) (qr + d);
                const float4 s4 = *(const float4 *) (ksf + d);
                const float4 z4 = *(const float4 *) (kzf + d);
                const uint32_t qw =
                        (uint32_t) (uint8_t) __float2int_rn(q4.x * s4.x * invqs)        |
                       ((uint32_t) (uint8_t) __float2int_rn(q4.y * s4.y * invqs) << 8)  |
                       ((uint32_t) (uint8_t) __float2int_rn(q4.z * s4.z * invqs) << 16) |
                       ((uint32_t) (uint8_t) __float2int_rn(q4.w * s4.w * invqs) << 24);
                *(uint32_t *) (row + d) = qw;
                cqv += q4.x * z4.x + q4.y * z4.y + q4.z * z4.z + q4.w * z4.w;
            }
            qscale[gg] = qs;
            cq[gg]     = cqv;
        }
    }

    // payload -> nibble-pair transposed table (group = 128 tokens, k_bits = 4;
    // source byte (d*64 + t/2) holds codes (d, 2u) in the low nibble and
    // (d, 2u+1) in the high nibble)
    const uint8_t * __restrict__ payload = record + k_payload_off;
    for (int i = tid; i < (head_dim / 2) * 128; i += 32 * W) {
        const int p  = i >> 7; // channel pair
        const int t  = i & 127;
        const int sh = 4 * (t & 1);
        const uint8_t lo = payload[128 * p + (t >> 1)];
        const uint8_t hi = payload[128 * p + 64 + (t >> 1)];
        spt2[t * 68 + p] = uint8_t(((lo >> sh) & 0xF) | (((hi >> sh) & 0xF) << 4));
    }
    __syncthreads();

    const int g  = lane >> 2;
    const int t4 = lane & 3;

    // A fragments: MT m-tiles x (head_dim/32) k-steps x 4 regs (uint32 loads)
    uint32_t a_frag[MT][4][4];
    for (int mt = 0; mt < MT; ++mt) {
        int ks = 0;
        for (int kk = 0; kk + 31 < head_dim; kk += 32, ++ks) {
            const int k0 = kk + 4 * t4;
            const int gg = warp * NQ + mt * 16;
            const int8_t * row0 = qi8 + size_t(gg + g) * QPITCH;
            const int8_t * row8 = qi8 + size_t(gg + g + 8) * QPITCH;
            a_frag[mt][ks][0] = *(const uint32_t *) (row0 + k0);
            a_frag[mt][ks][1] = *(const uint32_t *) (row8 + k0);
            a_frag[mt][ks][2] = *(const uint32_t *) (row0 + k0 + 16);
            a_frag[mt][ks][3] = *(const uint32_t *) (row8 + k0 + 16);
        }
    }

    for (int n0 = 0; n0 < 128; n0 += 8) {
        int32_t c[MT][4];
        for (int mt = 0; mt < MT; ++mt) { c[mt][0] = 0; c[mt][1] = 0; c[mt][2] = 0; c[mt][3] = 0; }
        int ks = 0;
        for (int kk = 0; kk + 31 < head_dim; kk += 32, ++ks) {
            const int p0 = (kk + 4 * t4) >> 1; // channel pair of k0 = kk + 4*t4
            const int t  = n0 + g;
            const uint8_t * row = spt2 + size_t(t) * 68;
            const uint32_t b0 = kvarn_expand_nibbles(*(const uint16_t *) (row + p0));
            const uint32_t b1 = kvarn_expand_nibbles(*(const uint16_t *) (row + p0 + 8));
            const uint32_t b[2] = { b0, b1 };
            for (int mt = 0; mt < MT; ++mt) {
                kvarn_mma_m16n8k32_s8(c[mt], a_frag[mt][ks], b);
            }
        }
        const int t0 = n0 + 2 * t4;
        const float o0 = ko[t0];
        const float o1 = ko[t0 + 1];
        for (int mt = 0; mt < MT; ++mt) {
            const int gq = warp * NQ + mt * 16 + g;
            float * s0 = scores + size_t(gq)     * score_stride + size_t(rec) * 128 + t0;
            float * s1 = scores + size_t(gq + 8) * score_stride + size_t(rec) * 128 + t0;
            s0[0] = (float(c[mt][0]) * qscale[gq]     + cq[gq])     * o0;
            s0[1] = (float(c[mt][1]) * qscale[gq]     + cq[gq])     * o1;
            s1[0] = (float(c[mt][2]) * qscale[gq + 8] + cq[gq + 8]) * o0;
            s1[1] = (float(c[mt][3]) * qscale[gq + 8] + cq[gq + 8]) * o1;
        }
    }
}

// ---------------------------------------------------------------------------
// PV: out(q,c) = v_other[c] * ( sum_t (P*v_scale)[t]*code(t,c) + C'_q )
//   C'_q = sum_t P[q,t]*v_zp[t]; P' = P*v_scale quantized to int8 per query.
//   M = MT*16*W queries per record; W warps (tokens are the k-dim, 128 -> 4
//   steps of 32, channels are n). out is accumulated with atomics.
// ---------------------------------------------------------------------------

// M = 16 (single warp), kept for the decode-shaped entry point.
__global__ void kvarn_mma_pv_records_kernel(
        const uint8_t * __restrict__ records, const float * __restrict__ weights,
        const int head_dim, const size_t record_bytes, float * __restrict__ out,
        const size_t v_payload_off, const size_t v_scale_off,
        const size_t v_zp_off, const size_t v_other_off) {
    __shared__ int8_t pq8[16 * 128];
    __shared__ float  ps[16];
    __shared__ float  cp[16];
    __shared__ float  vo[128];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int rec  = blockIdx.x;
    const uint8_t * __restrict__ record = records + size_t(rec) * record_bytes;
    const half * __restrict__ v_scale = (const half *) (record + v_scale_off);
    const half * __restrict__ v_zp    = (const half *) (record + v_zp_off);
    const half * __restrict__ v_other = (const half *) (record + v_other_off);

    if (warp == 0 && lane < 16) {
        const float * w = weights + size_t(rec) * 16 * 128 + size_t(lane) * 128;
        float amax = 0.0f;
        for (int t = 0; t < 128; ++t) {
            amax = fmaxf(amax, fabsf(w[t] * __half2float(v_scale[t])));
        }
        const float psc = fmaxf(amax / 127.0f, 1e-10f);
        float cpv = 0.0f;
        for (int t = 0; t < 128; ++t) {
            float qi = rintf(w[t] * __half2float(v_scale[t]) / psc);
            qi = fminf(fmaxf(qi, -127.0f), 127.0f);
            pq8[lane * 128 + t] = (int8_t) qi;
            cpv += w[t] * __half2float(v_zp[t]);
        }
        ps[lane] = psc;
        cp[lane] = cpv;
    }
    for (int i = lane; i < head_dim; i += 32) {
        vo[i] = __half2float(v_other[i]);
    }
    __shared__ uint8_t sp2[4096];
    for (int i = lane; i < 4096 / 16; i += 32) {
        ((uint4 *) sp2)[i] = ((const uint4 *) (record + v_payload_off))[i];
    }
    __syncthreads();

    const int g    = lane >> 2;
    const int t4   = lane & 3;
    const uint8_t * __restrict__ payload = sp2;

    for (int c0 = 0; c0 + 7 < head_dim; c0 += 8) {
        int32_t c[4] = { 0, 0, 0, 0 };
        for (int kk = 0; kk < 128; kk += 32) {
            const int k0 = kk + 4 * t4;
            uint32_t a[4];
            a[0] = kvarn_pack_s8(pq8[g * 128 + k0], pq8[g * 128 + k0 + 1],
                                 pq8[g * 128 + k0 + 2], pq8[g * 128 + k0 + 3]);
            a[1] = kvarn_pack_s8(pq8[(g + 8) * 128 + k0], pq8[(g + 8) * 128 + k0 + 1],
                                 pq8[(g + 8) * 128 + k0 + 2], pq8[(g + 8) * 128 + k0 + 3]);
            a[2] = kvarn_pack_s8(pq8[g * 128 + k0 + 16], pq8[g * 128 + k0 + 17],
                                 pq8[g * 128 + k0 + 18], pq8[g * 128 + k0 + 19]);
            a[3] = kvarn_pack_s8(pq8[(g + 8) * 128 + k0 + 16], pq8[(g + 8) * 128 + k0 + 17],
                                 pq8[(g + 8) * 128 + k0 + 18], pq8[(g + 8) * 128 + k0 + 19]);
            const int cn = c0 + g;
            uint32_t b[2];
            b[0] = kvarn_pack_s8(kvarn_code2(payload, head_dim, k0, cn),     kvarn_code2(payload, head_dim, k0 + 1, cn),
                                 kvarn_code2(payload, head_dim, k0 + 2, cn), kvarn_code2(payload, head_dim, k0 + 3, cn));
            b[1] = kvarn_pack_s8(kvarn_code2(payload, head_dim, k0 + 16, cn), kvarn_code2(payload, head_dim, k0 + 17, cn),
                                 kvarn_code2(payload, head_dim, k0 + 18, cn), kvarn_code2(payload, head_dim, k0 + 19, cn));
            kvarn_mma_m16n8k32_s8(c, a, b);
        }
        const int cc = c0 + 2 * t4;
        const float o0 = vo[cc];
        const float o1 = vo[cc + 1];
        atomicAdd(&out[size_t(g)     * head_dim + cc],     (float(c[0]) * ps[g]     + cp[g])     * o0);
        atomicAdd(&out[size_t(g)     * head_dim + cc + 1], (float(c[1]) * ps[g]     + cp[g])     * o1);
        atomicAdd(&out[size_t(g + 8) * head_dim + cc],     (float(c[2]) * ps[g + 8] + cp[g + 8]) * o0);
        atomicAdd(&out[size_t(g + 8) * head_dim + cc + 1], (float(c[3]) * ps[g + 8] + cp[g + 8]) * o1);
    }
}

// M = MT*16*W (multi-warp); staging and the per-record scale vectors are
// built once per block; A fragments are hoisted out of the channel loop.
template <int MT, int W>
__global__ void kvarn_mma_pv_records_w_kernel(
        const uint8_t * __restrict__ records, const float * __restrict__ weights,
        const int head_dim, const size_t record_bytes, float * __restrict__ out,
        const size_t v_payload_off, const size_t v_scale_off,
        const size_t v_zp_off, const size_t v_other_off) {
    constexpr int NQ = MT * 16;

    __shared__ __align__(16) int8_t pq8[W * NQ * 128];
    __shared__ float  ps[W * NQ];
    __shared__ float  cp[W * NQ];
    __shared__ float  vo[128];
    __shared__ __align__(16) float vsf[128];
    __shared__ __align__(16) float vzf[128];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int rec  = blockIdx.x;
    const uint8_t * __restrict__ record = records + size_t(rec) * record_bytes;
    const half * __restrict__ v_scale = (const half *) (record + v_scale_off);
    const half * __restrict__ v_zp    = (const half *) (record + v_zp_off);
    const half * __restrict__ v_other = (const half *) (record + v_other_off);

    for (int i = tid; i < 128; i += 32 * W) {
        vsf[i] = __half2float(v_scale[i]);
        vzf[i] = __half2float(v_zp[i]);
    }
    for (int i = tid; i < head_dim; i += 32 * W) {
        vo[i] = __half2float(v_other[i]);
    }

    // P' = round(P*v_scale / ps) per query; C'_q = sum_t P*v_zp (float4 chunks,
    // no clamp needed). weights == nullptr is a timing/debug switch.
    if (weights != nullptr) {
        for (int gg = warp * NQ + lane; gg < (warp + 1) * NQ; gg += 32) {
            const float * w = weights + size_t(rec) * (W * NQ) * 128 + size_t(gg) * 128;
            float amax = 0.0f;
            for (int t = 0; t < 128; t += 4) {
                const float4 w4 = *(const float4 *) (w + t);
                const float4 s4 = *(const float4 *) (vsf + t);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(w4.x * s4.x), fabsf(w4.y * s4.y)),
                                         fmaxf(fabsf(w4.z * s4.z), fabsf(w4.w * s4.w))));
            }
            const float psc   = fmaxf(amax / 127.0f, 1e-10f);
            const float invps = 1.0f / psc;
            float cpv = 0.0f;
            int8_t * row = pq8 + size_t(gg) * 128;
            for (int t = 0; t < 128; t += 4) {
                const float4 w4 = *(const float4 *) (w + t);
                const float4 s4 = *(const float4 *) (vsf + t);
                const float4 z4 = *(const float4 *) (vzf + t);
                const uint32_t qw =
                        (uint32_t) (uint8_t) __float2int_rn(w4.x * s4.x * invps)        |
                       ((uint32_t) (uint8_t) __float2int_rn(w4.y * s4.y * invps) << 8)  |
                       ((uint32_t) (uint8_t) __float2int_rn(w4.z * s4.z * invps) << 16) |
                       ((uint32_t) (uint8_t) __float2int_rn(w4.w * s4.w * invps) << 24);
                *(uint32_t *) (row + t) = qw;
                cpv += w4.x * z4.x + w4.y * z4.y + w4.z * z4.z + w4.w * z4.w;
            }
            ps[gg] = psc;
            cp[gg] = cpv;
        }
    }

    __shared__ uint8_t sp2[4096];
    const uint8_t * __restrict__ vpayload = record + v_payload_off;
    for (int i = tid; i < head_dim * 32; i += 32 * W) {
        sp2[i] = vpayload[i];
    }
    __syncthreads();

    const int g   = lane >> 2;
    const int t4  = lane & 3;
    const uint8_t * __restrict__ payload = sp2;

    uint32_t a_frag[MT][4][4];
    for (int mt = 0; mt < MT; ++mt) {
        int ks = 0;
        for (int kk = 0; kk < 128; kk += 32, ++ks) {
            const int k0 = kk + 4 * t4;
            const int gg = warp * NQ + mt * 16;
            a_frag[mt][ks][0] = kvarn_pack_s8(pq8[(gg + g) * 128 + k0], pq8[(gg + g) * 128 + k0 + 1],
                                              pq8[(gg + g) * 128 + k0 + 2], pq8[(gg + g) * 128 + k0 + 3]);
            a_frag[mt][ks][1] = kvarn_pack_s8(pq8[(gg + g + 8) * 128 + k0], pq8[(gg + g + 8) * 128 + k0 + 1],
                                              pq8[(gg + g + 8) * 128 + k0 + 2], pq8[(gg + g + 8) * 128 + k0 + 3]);
            a_frag[mt][ks][2] = kvarn_pack_s8(pq8[(gg + g) * 128 + k0 + 16], pq8[(gg + g) * 128 + k0 + 17],
                                              pq8[(gg + g) * 128 + k0 + 18], pq8[(gg + g) * 128 + k0 + 19]);
            a_frag[mt][ks][3] = kvarn_pack_s8(pq8[(gg + g + 8) * 128 + k0 + 16], pq8[(gg + g + 8) * 128 + k0 + 17],
                                              pq8[(gg + g + 8) * 128 + k0 + 18], pq8[(gg + g + 8) * 128 + k0 + 19]);
        }
    }

    for (int c0 = 0; c0 + 7 < head_dim; c0 += 8) {
        int32_t c[MT][4];
        for (int mt = 0; mt < MT; ++mt) { c[mt][0] = 0; c[mt][1] = 0; c[mt][2] = 0; c[mt][3] = 0; }
        int ks = 0;
        for (int kk = 0; kk < 128; kk += 32, ++ks) {
            const int k0 = kk + 4 * t4;
            const int cn = c0 + g;
            uint32_t b[2];
            b[0] = kvarn_pack_s8(kvarn_code2(payload, head_dim, k0, cn),     kvarn_code2(payload, head_dim, k0 + 1, cn),
                                 kvarn_code2(payload, head_dim, k0 + 2, cn), kvarn_code2(payload, head_dim, k0 + 3, cn));
            b[1] = kvarn_pack_s8(kvarn_code2(payload, head_dim, k0 + 16, cn), kvarn_code2(payload, head_dim, k0 + 17, cn),
                                 kvarn_code2(payload, head_dim, k0 + 18, cn), kvarn_code2(payload, head_dim, k0 + 19, cn));
            for (int mt = 0; mt < MT; ++mt) {
                kvarn_mma_m16n8k32_s8(c[mt], a_frag[mt][ks], b);
            }
        }
        const int cc = c0 + 2 * t4;
        const float o0 = vo[cc];
        const float o1 = vo[cc + 1];
        for (int mt = 0; mt < MT; ++mt) {
            const int gq = warp * NQ + mt * 16 + g;
            atomicAdd(&out[size_t(gq)     * head_dim + cc],     (float(c[mt][0]) * ps[gq]     + cp[gq])     * o0);
            atomicAdd(&out[size_t(gq)     * head_dim + cc + 1], (float(c[mt][1]) * ps[gq]     + cp[gq])     * o1);
            atomicAdd(&out[size_t(gq + 8) * head_dim + cc],     (float(c[mt][2]) * ps[gq + 8] + cp[gq + 8]) * o0);
            atomicAdd(&out[size_t(gq + 8) * head_dim + cc + 1], (float(c[mt][3]) * ps[gq + 8] + cp[gq + 8]) * o1);
        }
    }
}

// ---------------------------------------------------------------------------
// host entry points
// ---------------------------------------------------------------------------

static bool kvarn_mma_launch_ok(const char * what) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kvarn: %s failed: %s\n", what, cudaGetErrorString(err));
        return false;
    }
    return true;
}

void ggml_cuda_kvarn_mma_kq_records(
        const struct ggml_kvarn_layout * layout, const void * records, const float * q,
        int head_dim, int n_records, size_t record_bytes, float * scores) {
    if (head_dim <= 0 || head_dim > 128 || head_dim % 8 != 0) {
        fprintf(stderr, "kvarn: mma_kq: head_dim %d out of range (8..128, /8)\n", head_dim);
        return;
    }
    kvarn_mma_kq_records_kernel<1, 1><<<n_records, 32>>>(
            (const uint8_t *) records, q, head_dim, record_bytes, scores,
            layout->k_payload_off, layout->k_scale_off, layout->k_zp_off, layout->k_other_off,
            n_records * 128);
    kvarn_mma_launch_ok(__func__);
}

void ggml_cuda_kvarn_mma_kq_records64(
        const struct ggml_kvarn_layout * layout, const void * records, const float * q,
        int head_dim, int n_records, size_t record_bytes, float * scores) {
    if (head_dim <= 0 || head_dim > 128 || head_dim % 8 != 0) {
        fprintf(stderr, "kvarn: mma_kq64: head_dim %d out of range (8..128, /8)\n", head_dim);
        return;
    }
    kvarn_mma_kq_records_kernel<4, 1><<<n_records, 32>>>(
            (const uint8_t *) records, q, head_dim, record_bytes, scores,
            layout->k_payload_off, layout->k_scale_off, layout->k_zp_off, layout->k_other_off,
            n_records * 128);
    kvarn_mma_launch_ok(__func__);
}

void ggml_cuda_kvarn_mma_kq_records128(
        const struct ggml_kvarn_layout * layout, const void * records, const float * q,
        int head_dim, int n_records, size_t record_bytes, float * scores) {
    if (head_dim <= 0 || head_dim > 128 || head_dim % 8 != 0) {
        fprintf(stderr, "kvarn: mma_kq128: head_dim %d out of range (8..128, /8)\n", head_dim);
        return;
    }
    kvarn_mma_kq_records_kernel<4, 2><<<n_records, 64>>>(
            (const uint8_t *) records, q, head_dim, record_bytes, scores,
            layout->k_payload_off, layout->k_scale_off, layout->k_zp_off, layout->k_other_off,
            n_records * 128);
    kvarn_mma_launch_ok(__func__);
}

void ggml_cuda_kvarn_mma_pv_records(
        const struct ggml_kvarn_layout * layout, const void * records, const float * weights,
        int head_dim, int n_records, size_t record_bytes, float * out) {
    if (head_dim <= 0 || head_dim > 128 || head_dim % 8 != 0) {
        fprintf(stderr, "kvarn: mma_pv: head_dim %d unsupported (8..128, /8)\n", head_dim);
        return;
    }
    kvarn_mma_pv_records_kernel<<<n_records, 32>>>(
            (const uint8_t *) records, weights, head_dim, record_bytes, out,
            layout->v_payload_off, layout->v_scale_off, layout->v_zp_off, layout->v_other_off);
    kvarn_mma_launch_ok(__func__);
}

void ggml_cuda_kvarn_mma_pv_records64(
        const struct ggml_kvarn_layout * layout, const void * records, const float * weights,
        int head_dim, int n_records, size_t record_bytes, float * out) {
    if (head_dim <= 0 || head_dim > 128 || head_dim % 8 != 0) {
        fprintf(stderr, "kvarn: mma_pv64: head_dim %d unsupported (8..128, /8)\n", head_dim);
        return;
    }
    kvarn_mma_pv_records_w_kernel<4, 1><<<n_records, 32>>>(
            (const uint8_t *) records, weights, head_dim, record_bytes, out,
            layout->v_payload_off, layout->v_scale_off, layout->v_zp_off, layout->v_other_off);
    kvarn_mma_launch_ok(__func__);
}

void ggml_cuda_kvarn_mma_pv_records128(
        const struct ggml_kvarn_layout * layout, const void * records, const float * weights,
        int head_dim, int n_records, size_t record_bytes, float * out) {
    if (head_dim <= 0 || head_dim > 128 || head_dim % 8 != 0) {
        fprintf(stderr, "kvarn: mma_pv128: head_dim %d unsupported (8..128, /8)\n", head_dim);
        return;
    }
    kvarn_mma_pv_records_w_kernel<4, 2><<<n_records, 64>>>(
            (const uint8_t *) records, weights, head_dim, record_bytes, out,
            layout->v_payload_off, layout->v_scale_off, layout->v_zp_off, layout->v_other_off);
    kvarn_mma_launch_ok(__func__);
}
