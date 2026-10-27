#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Spark2_5: pre-norm decoder with
//   - a fused q|k|v projection (attention.key_length = head_dim, not n_embd / n_head)
//   - a head-wise sigmoid gate on the attention output (attn_gate, one scalar per head),
//     applied before the output projection
//   - GELU feed-forward (LLM_FFN_GELU, parallel gate/up)
//   - hybrid layer types: sliding_attention layers use the SWA mask, a full-width rope
//     (rope.dimension_count_swa) and rope.freq_base_swa; full_attention layers use the
//     partial rotary width (rope.dimension_count) and rope.freq_base
//   - tied output embedding
//
// build_std_attention already implements the qkv split, the per-layer rope base selection
// (n_swa > 0 -> rope_freq_base_train_swa) and the head-wise sigmoid gate via wqkv_gate, so
// the graph is assembled from it like build_mellum/build_qwen3.
ggml_cgraph * llm_build_context::build_spark2_5() {
    ggml_cgraph * gf = new_graph_custom();

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    struct ggml_tensor * inp_pos     = build_inp_pos();
    struct ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    for (int il = 0; il < n_layer; ++il) {
        const bool    is_swa     = hparams.swa_layers[il] != 0;
        const int64_t n_embd_head = hparams.n_embd_head_v(il);

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(il));

        auto KQ_mask_l = is_swa ? KQ_mask_swa : KQ_mask;

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr, KQ_mask_l,
                nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f,
                is_swa ? (int) hparams.n_swa : 0, il, true, false, true);

        // feed-forward network (gelu, parallel gate/up)
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf, true);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
