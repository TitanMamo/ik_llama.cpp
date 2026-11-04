# LiLiCorr drafter (experimental)

> **Status: correct, but slower than no drafter on this hardware.** Published so the remaining
> defect can be found in the open. Numbers and the hypotheses already ruled out are in the commit
> message of `lilicorr: LiLiCorr drafter for Nemotron 3.5 Bolt (experimental)` and in
> `STATUS.md` of the original branch.

NVIDIA's LiLiCorr is a DFlash-family speculative drafter for the Nemotron 3.5 Bolt model. It loads
as a separate draft model and runs through the existing DFlash machinery - `llama_batch` is
untouched, the anchor goes through `build_inp_mtp_states`, and the per-tensor ModelOpt scales go
through `w_s` in `llm_build_lora_mm`.

## Requirement: NVFP4

The drafter GGUF (`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf`) is **NVFP4-quantised**
(25 of its 153 tensors), so this branch needs the NVFP4 support that the same branch adds. Without
it the drafter does not load. The target file can be any quant of Bolt, e.g. the Unsloth
`UD-IQ1_S` (no NVFP4).

## Running

```bash
llama-cli -m nemotron-bolt-74b.gguf -md nemotron-bolt-lilicorr.gguf \
  --spec-type dflash:n_max=7 --n-cpu-moe 28 --moe-cache-mib 1024 -ngl 99 -ngld 99
```

- The draft context is bounded to the drafter's sliding window (1024) plus the block size unless
  `-cd` is given, which keeps the injected feature window small at any target context length.
- `p_min` (per-stage parameter, default 0.75 as a cap) controls how confidently a draft token must
  be ranked to be kept; the cutoff starts at zero (no truncation) and only becomes strict after a
  fully rejected block.
- Draft proposals are never skipped and there is no backoff: every decode is followed by a draft
  call. Measured on the documented prompt, three guard configurations gave 93/135 accepted drafts
  without skipping, 0/28 with a skip after a rejected block, and 1/35 with skip plus an empty-walk
  signal, so a skip leaves the draft state behind the target window.

## Measured configurations

RTX 3060 12 GB, Bolt 74B-A9B, 6 CPU threads (`--n-cpu-moe 28`), greedy, 119-token prompt, n=128:

| configuration | decode | draft acceptance | correct? |
|---|---|---|---|
| no drafter | 23.51 t/s | - | yes |
| drafter, speculative checkpoints disabled | 34.09 t/s | 100/156 (64 %) | **no** - the target's recurrent state is never rewound |
| drafter, correct rewind (as published) | 11.3-14.8 t/s | 10-16 % | yes |

The 34.09 t/s arm is the drafter's real capability, but it is not a valid speedup: it accepts
drafts against a state that the target never rewinds. With a correct rewind the acceptance collapses
to ~10 %, the block count quadruples, and the drafter costs more than it saves.

The open problem is therefore not the rewind itself (that is correct, and a dense `ngram_simple`
drafter is byte-identical to its non-spec run through the same checkpoint path) but the drop in
acceptance once the target state is rewound: the drafter's feature window has to stay identical to
the target's committed history across rewinds.
