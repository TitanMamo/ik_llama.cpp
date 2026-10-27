# Spark2.5 (`spark2_5`) port checklist

Executable companion: `qat_tools/tools/verify_spark2_5.py` — one command, PASS/FAIL per gate, works on
any `spark2_5` GGUF (ours, Bl4ckyou's, a future one). Everything below is what that file checks, with
the evidence an implementer can read in the three references.

Line numbers are pinned to:
* **HF** `modeling_spark.py` / `configuration_spark.py` from `XHToken/Spark-X2.6-4B` (config
  `transformers_version: 4.57.1`),
* **exllamav3** `exllamav3/architecture/spark2_5.py` (the version that carried the port),
* **ours** `TitanMamo/ik_llama.cpp`, branch `spark2_5`, commit `51299eea`, diff `k2-t16...spark2_5`
  (16 files, +282/−1). Paths are repo-relative.

```
python verify_spark2_5.py spark.gguf                       # G1-G4, CPU only, ~10 s
python verify_spark2_5.py spark.gguf --llama-tokenize ./build/bin/llama-tokenize   # + C++ tokenizer
python verify_spark2_5.py spark.gguf --text --llama-cli ./build/bin/llama-cli --ref-dir ./Spark-X2.6-4B
```

`SKIP` never means pass: it means the gate could not be evaluated (no reference GGUF, no safetensors,
no binary). Exit status is 1 if anything FAILed.

---

## 1. Fused QKV is `[Q;K;V]` along the output dim — and prove it numerically

The projection is one matmul whose output rows are exactly Q, then K, then V:

* **HF** `modeling_spark.py:149-153` — `q_dim = n_head*head_dim`, `kv_dim = n_kv*head_dim`,
  `qkv_out_dim = q_dim + 2*kv_dim`, `q_k_v_proj = nn.Linear(hidden, qkv_out_dim)`.
* **HF** `modeling_spark.py:170-173` — the split: `q = qkv[..., :q_dim]`,
  `k = qkv[..., q_dim:q_dim+kv_dim]`, `v = qkv[..., q_dim+kv_dim:]`. Verbatim slicing, no reshape, no
  interleave.
* **exllamav3** `spark2_5.py:181` — `key_fused_qkv = "q_k_v_proj"`, fed to the shared
  `Attention` module, which splits by the same `[q | k | v]` convention.
* **ours** `src/llama-load-tensors.cpp:1805-1806` — `layer.wqkv` is created with
  `{n_embd, n_embd_q + n_embd_k_i + n_embd_v_i}`; `src/graphs/build_spark2_5.cpp:33-42` runs
  `build_std_attention`, whose `llm_build_mul_mat_qkv` splits that tensor with the same three offsets.

**Why an interleaved write is the classic failure.** Several fused-QKV layouts exist in the wild
(`[Q;K;V]`, per-head `[q_h0 k_h0 v_h0 q_h1 ...]`, and Qwen-style `[q | gate]`); a converter that
copies `q_k_v_proj` into a per-head-interleaved `attn_qkv` (or into separate `q/k/v` with a head-wise
gather) produces a file that **loads, runs, and emits plausible text** — the projections are dense, so
every output row is a valid linear combination, just of the wrong weights. Nothing crashes and PPL is
only somewhat worse. It is also invisible to shape checks: `[6144, 2560]` is `[6144, 2560]` either
way.

**The numeric proof (G4).** Compare GGUF rows against the HF tensor at the *same* indices:

| row | what it is | catches |
|---|---|---|
| 0, 1 | first two rows of Q head 0 | a global transpose/offset error |
| 4096, 4097 | first two rows of K head 0 (q_dim = 4096) | a Q/K boundary shift |
| 5120, 5121 | first two rows of V head 0 (q_dim+kv_dim = 5120) | a K/V boundary shift |

Rows 0-1 stay identical under a per-head interleave *by construction* (the permutation fixes the
first head), so the boundary rows are what actually move — that is why all six are probed.

Measured on our F16 GGUF: `max|Δ| = 0, 0, 0, 0, 0, 0` (bf16→f16 is exact, so a verbatim copy must be
bit-identical). Negative control — the same file rewritten with a 4q/1k/1v per-group interleave:
`max|Δ| = 0, 0, 0.379, 0.536, 0.218, 0.177`, and the best-matching HF row for GGUF row 4096 is
**HF row 4608** (`k` head 0 landed where `q` head 16 belongs). The kit prints both numbers, so a
failure tells you *where* the layout went wrong, not just that it did.

On a quantised file `max|Δ|` cannot be 0; G4 then uses a tolerance (`--tol-q`, default 0.02; measured
0.003 on the published Q8_0) **and** requires every probe row's best cosine match to be its own index.

The same measurement was made independently on the encoder side before the port existed: rows
0/1/4096/4097/5120/5121 bit-identical, while the interleaved hypothesis differs by **1.11** against a
weight RMS of 0.043 (`qat_tools/K2_SESSION_2026-10-25_paper.md:1666-1676`) — i.e. the wrong layout is
~26σ, not a rounding question.

## 2. Head-wise attention gate: one scalar per *query head*, sigmoid, before `out_proj`

* **HF** `modeling_spark.py:154` — `g_proj = nn.Linear(hidden_size, num_heads)` — output width **16**,
  not `n_head*head_dim` (4096) and not `hidden` (2560). `modeling_spark.py:180` reshapes it to
  `[b, s, n_head, 1]`, i.e. one scalar per head, broadcast over the head's 256 channels.
* **HF** `modeling_spark.py:198-206` — `gate = sigmoid(gate_score.float())`, then
  `attn_output = attn_output * gate`, and only afterwards `modeling_spark.py:208-209` reshapes to
  `[b, s, n_head*head_dim]` and applies `out_proj`. Applying the gate *after* `out_proj` is a silent
  bug: same shapes, wrong result.
* **exllamav3** `spark2_5.py:182` — `key_g = "g_proj"`, and `spark2_5.py:88-92` refuses any activation
  other than `sigmoid` (the shared kernel path only implements `x *= sigmoid(y)`).
* **ours** `src/llama-load-tensors.cpp:1809` — `attn_gate` created as `{n_embd, n_head_i}`;
  `src/llama-build-context.cpp:3775-3781` — reshape to `[1, n_head, n_tokens]`, `sigmoid`, `mul`
  (kept as two ops; `GGML_OP_FUSED_MUL_UNARY` has no CPU implementation), applied to the attention
  output inside `build_std_attention`, before `wo`.
* Converter guard: `convert_hf_to_gguf.py:7410-7418` rejects a `g_proj` whose output width is not
  exactly `num_attention_heads`.

Every checkpoint in this family sets `gate_attn_act_mode: "sigmoid"`,
`headwise_attn_output_gate: true` (`config.json`); a port that supports only "large" gates
(`n_head*head_dim`) must not silently reuse that path.

## 3. Hybrid attention: 27 sliding / 9 full — and the off-by-one that bit us

`layer_types` has 36 entries; `full_attention` sits at **3, 7, 11, …, 35** (every 4th layer, 9 total)
and the other 27 are `sliding_attention` (`config.json`, `configuration_spark.py:93-99`).

* **HF** `modeling_spark.py:358-359` builds two masks, `create_causal_mask` and
  `create_sliding_window_causal_mask`; `modeling_spark.py:376-379` picks per layer. Sliding layers also
  get `self_attn.sliding_window = config.sliding_window` (`modeling_spark.py:224-229`).
* **ours** `src/graphs/build_spark2_5.cpp:31-42` selects `KQ_mask_swa` per layer from
  `hparams.swa_layers[il]`, and `src/llama-hparams.cpp:719-738` requires
  `attention.sliding_window_pattern` whenever `sliding_window > 0`.

**The off-by-one.** HF's mask is `masking_utils.py:97`: `kv_idx > q_idx - sliding_window`, i.e. the
allowed distance `d = q - kv` is `[0, sliding_window-1]` = **512 keys** for `sliding_window: 512`. When
HF hands the same number to a flash-attention backend it passes **`sliding_window - 1`**
(`integrations/flash_paged.py:50`: `(module.sliding_window - 1, 0)`), because FA's window argument is
an *inclusive* left bound. An implementation that applies 512 as an inclusive bound keeps 513 keys —
one key older than anything HF ever attends to. It loads, it runs, it is wrong at every position past
512, and it shows up as a slow quality decay on long prompts rather than as a crash.

* **The bug we had:** the K2/exllamav3 encoder path applied the config value as an inclusive bound and
  kept 513 keys; its Hessians for the affected layers were collected on 513-key windows before the fix
  landed (`qat_tools/K2_SESSION_2026-10-25_paper.md:1671-1675`).
* **exllamav3's fix:** `spark2_5.py:107-118` — `layer_sliding_window()` returns
  `sliding_window - 1` for sliding layers, with the comment naming `flash_paged.py:50`.
* **llama.cpp convention:** masks are exclusive, `src/llama.cpp:6519` drops a cell when
  `pos - cell_pos >= n_swa_eff`, so `n_swa = 512` straight from the GGUF gives exactly 512 keys.
  Our graph passes the raw value (`src/graphs/build_spark2_5.cpp:42`); subtracting one there would give
  511. **Both conventions are correct — mixing them is not.** Gate 3 pins `sliding_window = 512` in the
  file; the 513-vs-512 question is a property of *your mask code*, and G5 is what catches it in text.

## 4. Per-type RoPE: 64 @ 5e6 on full layers, 256 @ 1e4 on sliding layers, NEOX pairing

`config.json` → `rope_parameters.full_attention = {partial_rotary_factor: 0.25, rope_theta: 5e6}`,
`rope_parameters.sliding_attention = {partial_rotary_factor: 1.0, rope_theta: 1e4}`, with
`head_dim: 256`. So full layers rotate the first **64** of 256 channels at base 5e6, sliding layers
rotate **all 256** at base 1e4.

* **HF** `modeling_spark.py:31-40` — `inv_freq` over `rope_head_dim = head_dim * partial_rotary_factor`,
  `freqs = outer(t, inv_freq)`, then **`cat([freqs, freqs])`** — duplicated frequencies, i.e. NEOX
  pairing, not GPT-J half-split. `modeling_spark.py:43-57` rotates `x[..., :rope_head_dim]` with
  `rotate_half` and concatenates the untouched remainder; `modeling_spark.py:369-374` builds one
  cos/sin table **per layer type**.
* **exllamav3** `spark2_5.py:74,81` — `rope_style = RopeStyle.NEOX` on both `RopeSettings`, with the
  per-type theta and factor.
* **ours** `src/llama.cpp:10357` returns `LLAMA_ROPE_TYPE_NEOX` for the arch; GGUF carries
  `rope.dimension_count 64` / `rope.dimension_count_swa 256` / `rope.freq_base 5e6` /
  `rope.freq_base_swa 1e4` (`convert_hf_to_gguf.py:7378-7390`), and the graph selects the SWA base by
  passing `n_swa` into `build_std_attention` (`src/graphs/build_spark2_5.cpp:42` →
  `src/llama-build-context.cpp:3359`); the rotary *width* comes from `rope_n_rot(il)`
  (`src/llama-hparams.h:551`, `swa_layers[il] ? n_rot_swa : n_rot`).

Two naming traps: GGUF's `rope.dimension_count` is a count of **scalar dimensions** (64 → 32 frequency
entries), not a pair count; and `partial_rotary_factor` multiplies **`head_dim` (256)**, never
`hidden_size/num_heads` (160 — see §6).

## 5. Tied head — the checkpoint has no `lm_head.weight`

* **HF** `modeling_spark.py:408` (`_tied_weights_keys = ["lm_head.weight"]`) and
  `modeling_spark.py:464-468`: when `tie_word_embeddings`, logits are `F.linear(hidden, embedding.weight)`
  — the `lm_head` module exists but is never used. The published safetensors index contains no
  `lm_head.weight` key.
* **exllamav3** `spark2_5.py:209-211,224` — `head_alt_key = "model.embed_tokens"` when the tensor is
  absent.
* **ours** `src/llama-load-tensors.cpp:1779-1786` — `output` is `TENSOR_NOT_REQUIRED`; if absent it is
  created as `TENSOR_DUPLICATED` from `token_embd`. The converter emits no `output.weight`
  (G1 fails the file if one appears: a stray untied head is a different model).

Related checkpoint naming: the embedding is `model.embedding.weight` (not `model.embed_tokens.weight`
like every other Llama-shaped repo) — exllamav3 patches it at `spark2_5.py:94-100`, our converter maps
it through `tensor_mapping.py:162`-style rules.

## 6. The things that are not "gates" but silently move logits

* **`head_dim` (256) ≠ `hidden_size / n_head` (2560/16 = 160).** Q is 4096 wide, not 2560. Any code
  that derives the head width from `hidden_size` builds a model that loads and is wrong. In GGUF this
  is why `attention.key_length = attention.value_length = 256` exist; in HF it is
  `config.head_dim` (`configuration_spark.py:87`). Our loader reads it explicitly
  (`src/llama-load-tensors.cpp:1794-1801`).
* **Attention scale is `1/sqrt(head_dim)` = 1/16**, not `1/sqrt(rope_dim)` = 1/8
  (`modeling_spark.py:146`). With 64 rotated dims a "natural" `1/sqrt(64)` is off by 2× in the softmax
  temperature.
* **Exact GELU** (`hidden_act: "gelu"`, `modeling_spark.py:126-132`), gate×up in parallel
  (`LLM_FFN_PAR`, `src/graphs/build_spark2_5.cpp:50`) — not SiLU, not tanh-approx.
* **RMSNorm computed in f32** with the weight applied in f32 then cast back
  (`modeling_spark.py:96-107`); softmax in f32 (`modeling_spark.py:90`).
* **No biases anywhere** (`attention_bias: false`, `mlp_bias: false`).
* **Tokenizer**: GPT-2 BPE, 131072 tokens, 130716 merges, `add_bos_token: false`,
  `add_eos_token: false`, bos 0 / eos 1 / pad 2 / unk 5. The pre-tokenizer is a `Sequence`
  (`\p{N}{1,3}` split + digit splitting + the CJK/kana run + a main regex); our port hard-codes those
  three patterns in `src/llama-vocab.cpp:392-403` and keys them off `tokenizer.ggml.pre == "spark2_5"`
  (`src/llama-vocab.cpp:1999`). The GGUF `pre` string is validated against the HF tokenizer by hashing
  the ids of `get_vocab_base_pre`'s check text (`convert_hf_to_gguf.py:609`,
  `0a766d03…7941ed`).
* **`rope_parameters` is per layer type in `config.json`**, and transformers ≥5 rewrites that key into
  its own `RopeParameters` object; read it as plain JSON if your loader does not understand it.

## 7. Converter validations worth copying

Each of these turns a silent numerical bug into a hard failure at conversion time
(`convert_hf_to_gguf.py:7362-7418`):

1. `layer_types` length == `num_hidden_layers`, every entry in `{sliding_attention, full_attention}`.
2. `gate_attn_act_mode == "sigmoid"` **and** `headwise_attn_output_gate is True`, else refuse.
3. `hidden_act == "gelu"`, else refuse.
4. `q_k_v_proj.weight.shape[0] == (n_head + 2*n_kv) * head_dim` — a fused width that is not exactly
   `[q; k; v]` is a layout bug, not a variant.
5. `g_proj.weight.shape[0] == num_attention_heads` — one scalar per head.
6. Write `vocab_size`, `sliding_window`, `sliding_window_pattern` (36 bools),
   `rope.dimension_count(_swa)` and `rope.freq_base(_swa)`; refuse if the pattern disagrees with the
   layer count.
7. Emit no `output.weight` for a tied checkpoint.

## 8. What the kit cannot check

* **Anything about attention internals at run time** — the 512-vs-513 window, the gate's *position*
  relative to `out_proj`, the scale factor and the rope pairing are pinned only indirectly, by G5's
  24-token text parity against the torch reference. A port can pass G1-G4 and still attend over 513
  keys; G5 catches it only if the divergence reaches the sampled text (it did for the encoder bug).
* **G5 needs a GPU and ~9 GB of VRAM/RAM** for the bf16 reference, and it is one prompt. It is a smoke
  test, not proof: two implementations that share a mistake agree with each other. The numeric
  authorities are G4 (bit-level, CPU) and PPL against the HF model.
* **Quantised files**: G4's exactness applies to F16/F32/BF16. For Q8_0 the kit uses a tolerance plus a
  best-match check; other quant types are refused rather than guessed (convert to F16 to run the
  exact check).
* **Not checked at all**: the chat template, the MTP/draft head (Spark has none), KV-cache layout,
  multi-sequence/batching behaviour, and the 1M-token `max_position_embeddings` path (untested beyond
  what G5 exercises).
* The kit reads the GGUF itself, so it cannot tell you *why* your converter wrote something — only
  whether the bytes agree with the published model.

## 9. Measured on this box (RTX 3060 12 GB, engine/fork as of commit 51299eea)

| check | result |
|---|---|
| F16 GGUF (our converter) vs published Q8_0 GGUF | 290 tensors, all shapes and 16 `spark2_5.*` keys identical |
| `blk.0.attn_qkv` rows 0/1/4096/4097/5120/5121 vs HF | `max|Δ| = 0` on all six (F16) |
| published Q8_0, same rows | `max|Δ| ≤ 0.0033` (quantisation only), best match = same row |
| tokenizer vs HF | 8/8 probe strings id-identical; 131072 tokens / 130716 merges identical |
| tokenizer vs `llama-tokenize` (C++ pre-tokenizer) | 8/8 probes id-identical |
| PPL, `wiki.test.raw -c 512 --chunks 20` | F16 23.8006 ± 1.175 (Q8_0 24.1464 ± 1.195) |
| Q8_0 quantised from our F16 | per-chunk PPL and text **bit-identical** to the published Q8_0 |
| G5 text parity, 24 greedy tokens | 24/24 common with the standalone torch reference |
| negative control: interleaved `attn_qkv` | G4 FAIL with `max|Δ|` 0.38-0.54 and `4096 -> 4608` |
| negative control: `sliding_window 511`, all-sliding pattern | G3 FAIL on both |

The known non-port caveat: `llama-perplexity -c 600 --chunks 2` with the default `-ub 512` gives
`[2]nan` on Spark **and** on gemma-4-E4B (`[1]387.5,[2]10079.3`), and both are clean with
`-b/-ub = n_ctx`. It is a pre-existing SWA/ubatch interaction in this fork, not something the port
introduced; unrelated to any gate above.
