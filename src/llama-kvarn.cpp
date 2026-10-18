#include "llama-kvarn.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

static bool kvarn_valid_bits(int32_t bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

bool llama_kvarn_bits_from_name(const char * name, int32_t & bits) {
    if (name == nullptr) {
        return false;
    }
    if (std::strncmp(name, "kvarn", 5) != 0) {
        return false;
    }

    char * end = nullptr;
    const long value = std::strtol(name + 5, &end, 10);
    if (end == name + 5 || *end != '\0') {
        return false;
    }
    if (value < 2 || value > 8 || !kvarn_valid_bits(int32_t(value))) {
        return false;
    }

    bits = int32_t(value);
    return true;
}

bool llama_kvarn_params_from_cache_types(
        struct llama_kvarn_params * params,
        const char * cache_type_k,
        const char * cache_type_v,
        const char ** error) {
    if (error != nullptr) {
        *error = nullptr;
    }
    if (params == nullptr) {
        return false;
    }

    *params = {};

    int32_t k_bits = 0;
    int32_t v_bits = 0;
    const bool k_is_kvarn = llama_kvarn_bits_from_name(cache_type_k, k_bits);
    const bool v_is_kvarn = llama_kvarn_bits_from_name(cache_type_v, v_bits);

    // reject malformed "kvarn" prefixes instead of silently falling through
    if (!k_is_kvarn && cache_type_k != nullptr && std::strncmp(cache_type_k, "kvarn", 5) == 0) {
        if (error != nullptr) { *error = "invalid KVarN cache type"; }
        return false;
    }
    if (!v_is_kvarn && cache_type_v != nullptr && std::strncmp(cache_type_v, "kvarn", 5) == 0) {
        if (error != nullptr) { *error = "invalid KVarN cache type"; }
        return false;
    }

    if (!k_is_kvarn && !v_is_kvarn) {
        return true; // KVarN not selected
    }

    if (k_is_kvarn && !v_is_kvarn) {
        v_bits = k_bits; // promote the other side (beellama behavior)
    } else if (!k_is_kvarn && v_is_kvarn) {
        k_bits = v_bits;
    }

    params->enabled        = true;
    params->key_bits       = k_bits;
    params->value_bits     = v_bits;
    params->group          = LLAMA_KVARN_GROUP;
    params->sinkhorn_iters = 16;
    params->sink_tokens    = LLAMA_KVARN_SINK_TOKENS;
    params->alpha          = 1.0f;
    // experimental knob: alpha=0 drops the variance normalization entirely
    // (stores s**0 = 1, i.e. Hadamard-only RTN), used to decompose the KVarN
    // benefit into the F16 window and the normalization parts
    if (const char * a = getenv("IK_KVARN_ALPHA")) {
        params->alpha = (float) atof(a);
    }

    return true;
}

bool llama_kvarn_head_dim_supported(int64_t head_dim) {
    // >128-wide heads are stored as independent 128-wide channel slices
    return head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512;
}

const char * llama_kvarn_validate(
        const struct llama_kvarn_params * params,
        int64_t head_dim_k,
        int64_t head_dim_v) {
    if (params == nullptr || !params->enabled) {
        return nullptr;
    }
    if (!kvarn_valid_bits(params->key_bits) || !kvarn_valid_bits(params->value_bits)) {
        return "KVarN supports 2-, 3-, 4-, 5-, 6-, and 8-bit cache payloads";
    }
    if (params->group != LLAMA_KVARN_GROUP) {
        return "KVarN currently requires a group size of 128 tokens";
    }
    if (params->sinkhorn_iters <= 0) {
        return "KVarN requires at least one Sinkhorn iteration";
    }
    if (params->sink_tokens != LLAMA_KVARN_SINK_TOKENS) {
        return "KVarN currently requires exactly 128 unquantized sink tokens";
    }
    if (!(params->alpha >= 0.0f && params->alpha <= 1.0f)) {
        return "KVarN alpha must be within [0, 1]";
    }
    if (!llama_kvarn_head_dim_supported(head_dim_k) || !llama_kvarn_head_dim_supported(head_dim_v)) {
        return "KVarN currently supports 64- and 128-dimensional key/value heads";
    }
    if (head_dim_k != head_dim_v) {
        return "KVarN requires identical K and V head dimensions (one record holds both sides)";
    }
    return nullptr;
}

struct ggml_kvarn_layout llama_kvarn_layout(const struct llama_kvarn_params & params, int64_t head_dim) {
    return ggml_kvarn_layout_make(head_dim, params.group, params.key_bits, params.value_bits);
}

size_t llama_kvarn_records_bytes(
        const struct llama_kvarn_params & params, int64_t head_dim,
        int64_t n_groups, int64_t n_head_kv) {
    const int64_t hd_s     = head_dim > 128 ? 128 : head_dim;
    const int64_t n_slices = head_dim > 128 ? head_dim / 128 : 1;
    const struct ggml_kvarn_layout layout = llama_kvarn_layout(params, hd_s);
    return layout.record_bytes * size_t(n_groups) * size_t(n_head_kv) * size_t(n_slices);
}

size_t llama_kvarn_stage_bytes(int64_t head_dim, int64_t n_head_kv, int32_t stage_groups) {
    const int64_t hd_s     = head_dim > 128 ? 128 : head_dim;
    const int64_t n_slices = head_dim > 128 ? head_dim / 128 : 1;
    return size_t(std::max<int32_t>(stage_groups, LLAMA_KVARN_STAGE_GROUPS)) * LLAMA_KVARN_GROUP * size_t(hd_s) *
            size_t(n_head_kv) * size_t(n_slices) * sizeof(uint16_t);
}
