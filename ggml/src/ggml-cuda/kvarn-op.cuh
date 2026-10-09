#pragma once

#include "common.cuh"

// Op-level KVarN entry points dispatched from ggml-cuda.cu for
// GGML_OP_KVARN_STORE / GGML_OP_KVARN_MATERIALIZE. The two ops carry the
// side-effect state (stage, records), so they must run on the backend where
// those tensors live; the implementations assert that every tensor is
// device-resident and abort otherwise (no silent mixed-offload fallback).
//
// The kernels themselves live in kvarn.cu; the record format and the VarN math
// are defined bit-faithfully by the CPU reference in ggml/src/ggml-kvarn.cpp.
void ggml_cuda_kvarn_store_op      (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_kvarn_materialize_op(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_kvarn_attn_op       (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
