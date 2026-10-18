#pragma once

// Internal KVarN helpers. The public layout struct and op constructors live in
// ggml.h; the CPU compute entry points below are dispatched from ggml.c.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// The KVarN tile is a fixed 128-token group. v1 supports 64- and 128-wide head
// dimensions; larger heads are conceptually split into 128-wide slices (not
// implemented yet).
#define GGML_KVARN_GROUP   128
#define GGML_KVARN_MAX_DIM 128

// op_params layout shared by GGML_OP_KVARN_STORE and GGML_OP_KVARN_MATERIALIZE
// (slot 4 is only used by the store; the materialize writes 0 there).
enum {
    GGML_KVARN_OP_BITS       = 0, // payload bits for this side
    GGML_KVARN_OP_ITERS      = 1, // sinkhorn iterations (store only)
    GGML_KVARN_OP_VALUE      = 2, // 1 when the op handles the V side
    GGML_KVARN_OP_STAGE_GRP  = 3, // number of 128-row stage blocks (S)
    GGML_KVARN_OP_OWNER      = 4, // newest group being staged, or -1
    GGML_KVARN_OP_OTHER_BITS = 5, // payload bits of the other side (record layout)
    GGML_KVARN_OP_ALPHA      = 6, // VarN damping exponent (float bits; store only)
    GGML_KVARN_OP_SEALED     = 7, // highest group written to a record
    GGML_KVARN_OP_G_FIRST    = 8, // first group touched by this store call, or -1
    GGML_KVARN_OP_COMBINED   = 9, // 1 for the K+V combined variants (store2/materialize2)
    GGML_KVARN_OP_SLICES     = 10, // 128-wide channel slices per head (1/2/4); attn keeps it in slot 6
    GGML_KVARN_OP_LAYER      = 11, // layer index into the device state tensor (also attn slot 7)
    GGML_KVARN_OP_SOFTCAP    = 12, // logit softcap (float bits; attn only, 0 = off)
};

// CPU compute entry points (GGML_OP_KVARN_STORE / GGML_OP_KVARN_MATERIALIZE /
// GGML_OP_KVARN_ATTN). The combined variants (KVARN_OP_COMBINED == 1) handle
// K and V in one node.
void ggml_kvarn_compute_store    (struct ggml_tensor * dst, int ith, int nth);
void ggml_kvarn_compute_materialize(struct ggml_tensor * dst, int ith, int nth);
void ggml_kvarn_compute_attn     (struct ggml_tensor * dst, int ith, int nth);

#ifdef __cplusplus
}
#endif
