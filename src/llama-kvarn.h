#pragma once

// KVarN structured KV-cache quantization — llama-side parameters and helpers.
//
// v1 scope: dense full-attention layers, unified KV, one contiguous token
// stream per cache, head_dim in {64, 128}, group 128, sink 128, stage 2.
// The heavy math and the record format live in ggml (ggml-kvarn.cpp); this
// module owns naming, validation and sizing so the cache/graph integration can
// stay thin.

#include "llama.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

#define LLAMA_KVARN_GROUP       128
#define LLAMA_KVARN_SINK_TOKENS 128
// minimum ring stage size (sink block + one live group); the actual size is
// derived from n_ubatch so that lazy sealing is exact for any batch shape
#define LLAMA_KVARN_STAGE_GROUPS 2

// struct llama_kvarn_params + the public parse/validate entry points live in
// include/llama.h; the helpers below are internal sizing/layout utilities.

// "kvarnN" with N in {2, 3, 4, 5, 6, 8} (matches the beellama presets).
bool llama_kvarn_bits_from_name(const char * name, int32_t & bits);

bool llama_kvarn_head_dim_supported(int64_t head_dim);

// Record layout for one (group, head) pair of the given cache side pair.
struct ggml_kvarn_layout llama_kvarn_layout(const struct llama_kvarn_params & params, int64_t head_dim);

// Bytes needed for the packed records / F16 staging of one layer.
size_t llama_kvarn_records_bytes(
        const struct llama_kvarn_params & params, int64_t head_dim,
        int64_t n_groups, int64_t n_head_kv);
size_t llama_kvarn_stage_bytes(int64_t head_dim, int64_t n_head_kv, int32_t stage_groups);
