#!/usr/bin/env python3
"""Port-verification kit for the spark2_5 architecture (XHToken/Spark-X2.6-4B).

One command, PASS/FAIL per gate, no GPU except for G5 (--text):

  G1 arch + tensor set      general.architecture, the 290-tensor set, shapes derived from the
                            file's own hparams, and every spark2_5.* key against the published
                            XHToken/Spark-X2.6-4B Q8_0 GGUF when a reference copy is available.
  G2 tokenizer              model/type, pre-tokenizer hash, 131072 tokens, 130716 merges, and
                            id-for-id equality with the HF tokenizer on 8 probe strings.
  G3 semantics keys         sliding window 512, 27/9 layer pattern (full at 3,7,...,35), per-type
                            rope (64 @ 5e6 full, 256 @ 1e4 sliding), head dim 256, tied head.
  G4 fused-QKV row order    rows 0,1,4096,4097,5120,5121 of blk.0.attn_qkv vs HF
                            model.layers.0.self_attn.q_k_v_proj -- CPU only, needs the safetensors.
  G5 text parity (--text)   24 greedy tokens from a llama.cpp binary vs a torch reference.

Everything is read from the GGUF itself (own reader, no gguf-py), so the kit does not depend on
the state of the tree under test.  Exit status: 0 = every executed gate PASSed, 1 = any FAIL,
2 = bad usage (argparse).  SKIP means "could not be checked here", never "passed".

  ./verify_spark2_5.py /path/to/spark.gguf
  ./verify_spark2_5.py /path/to/spark.gguf --llama-tokenize ./build/bin/llama-tokenize
  ./verify_spark2_5.py /path/to/spark.gguf --download-ref /tmp/ref
  ./verify_spark2_5.py /path/to/spark.gguf --text --llama-cli ./build/bin/llama-cli --ref-dir ./Spark-X2.6-4B
"""

import argparse
import json
import mmap
import os
import struct
import subprocess
import sys
import urllib.request

import numpy as np

# ---------------------------------------------------------------- published reference constants

REF_REPO = "XHToken/Spark-X2.6-4B"
REF_FILE = "Spark-X2.6-4B-Q8_0.gguf"
REF_CANDIDATES = [
    "/mnt/Data/hf/spark-gguf/" + REF_FILE,
    "/mnt/Data/qat_k2/bl4ckyou/" + REF_FILE,
]
REF_URL = f"https://huggingface.co/{REF_REPO}/resolve/main/{REF_FILE}?download=true"

# sha256 of the token id list of convert_hf_to_gguf.py's check text (get_vocab_base_pre).
PRE_HASH = "0a766d034107bc736a3f2dc4968fd62e54a3570f1454443e0c5a4cc6bd7941ed"
CHKTXT = ("\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ "
          "(multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 "
          "3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на "
          "Български \'\'\'\'\'\'```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, "
          "'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL")

# the published hparams; G3 compares against these, G1 derives shapes from the file's own values.
EXPECT = {
    "spark2_5.block_count": 36,
    "spark2_5.embedding_length": 2560,
    "spark2_5.feed_forward_length": 10240,
    "spark2_5.attention.head_count": 16,
    "spark2_5.attention.head_count_kv": 4,
    "spark2_5.attention.key_length": 256,
    "spark2_5.attention.value_length": 256,
    "spark2_5.attention.sliding_window": 512,
    "spark2_5.rope.dimension_count": 64,
    "spark2_5.rope.dimension_count_swa": 256,
    "spark2_5.rope.freq_base": 5e6,
    "spark2_5.rope.freq_base_swa": 1e4,
    "spark2_5.vocab_size": 131072,
    "spark2_5.attention.layer_norm_rms_epsilon": 1e-6,
}
N_TOKENS, N_MERGES = 131072, 130716
N_FULL = {3, 7, 11, 15, 19, 23, 27, 31, 35}          # full_attention layer indices
PROBE_ROWS = [0, 1, 4096, 4097, 5120, 5121]          # q rows 0-1, first k rows, first v rows
QKV_HF = "model.layers.0.self_attn.q_k_v_proj.weight"

PROBES = [
    "The three most important properties of a good algorithm are",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "1234567890 3.14159 0x1F + 42 = 43; 2026-10-27T20:31:00Z",
    "我想在apple工作1314151天，薪资是多少？",
    "Die Straße ist schön, aber ich gehe über die Brücke.",
    "  \n\t\n   spaced   out\n\nlines\t and\ttabs  ",
    "emoji test: 🚀😶‍🌫️✅🦙 and a zero​width space",
    "إذا كان الجو جميلاً، فسنذهب إلى الحديقة مع الأصدقاء.",
]

# GGML tensor types we can decode for the row check.
F32, F16, Q8_0, BF16 = 0, 1, 8, 30
DTYPE_NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", BF16: "BF16"}
DTYPE_SIZE = {F32: 4, F16: 2, BF16: 2}

rows = []          # (gate, status, detail)


def report(gate, status, detail):
    rows.append((gate, status, detail))


# ------------------------------------------------------------------------------------ GGUF reader

class GGUF:
    """Minimal GGUF v2/v3 reader: metadata, tensor table, and row-level dequantisation."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file (magic {magic!r})")
        self.version, n_tensors, n_kv = struct.unpack("<IQQ", self.f.read(20))
        self.kv = {}
        for _ in range(n_kv):
            key = self._str()
            self.kv[key] = self._value(struct.unpack("<I", self.f.read(4))[0])
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            n_dims = struct.unpack("<I", self.f.read(4))[0]
            ne = struct.unpack("<" + "Q" * n_dims, self.f.read(8 * n_dims))
            dtype = struct.unpack("<I", self.f.read(4))[0]
            offset = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (ne, dtype, offset)
        align = self.kv.get("general.alignment", 32)
        self.data_off = (self.f.tell() + align - 1) // align * align
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def _str(self):
        (n,) = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8", "replace")

    def _value(self, t):
        if t in (0, 1):   return struct.unpack("<B" if t == 0 else "<b", self.f.read(1))[0]
        if t in (2, 3):   return struct.unpack("<H" if t == 2 else "<h", self.f.read(2))[0]
        if t in (4, 5):   return struct.unpack("<I" if t == 4 else "<i", self.f.read(4))[0]
        if t == 6:        return struct.unpack("<f", self.f.read(4))[0]
        if t == 7:        return bool(struct.unpack("<B", self.f.read(1))[0])
        if t == 8:        return self._str()
        if t == 10:       return struct.unpack("<Q", self.f.read(8))[0]
        if t == 11:       return struct.unpack("<q", self.f.read(8))[0]
        if t == 12:       return struct.unpack("<d", self.f.read(8))[0]
        if t == 9:
            et, n = struct.unpack("<IQ", self.f.read(12))
            return [self._value(et) for _ in range(n)]
        raise SystemExit(f"GGUF: unsupported metadata type {t}")

    def row(self, name, index, nrows=1):
        """Rows `index .. index+nrows-1` of a 2-D tensor, dequantised to float32."""
        ne, dtype, offset = self.tensors[name]
        ne0 = ne[0]
        base = self.data_off + offset
        if dtype in DTYPE_SIZE:
            start = base + index * ne0 * DTYPE_SIZE[dtype]
            nbytes = ne0 * nrows * DTYPE_SIZE[dtype]
            raw = self.mm[start:start + nbytes]
            if dtype == F32:
                return np.frombuffer(raw, dtype="<f4").reshape(nrows, ne0).copy()
            if dtype == F16:
                return np.frombuffer(raw, dtype="<f2").reshape(nrows, ne0).astype(np.float32)
            u = np.frombuffer(raw, dtype="<u2").reshape(nrows, ne0).astype(np.uint32) << 16
            return u.view(np.float32)
        if dtype == Q8_0:
            nb = ne0 // 32
            start = base + index * nb * 34
            nbytes = nb * 34 * nrows
            raw = np.frombuffer(self.mm[start:start + nbytes], dtype=np.uint8)
            raw = raw.reshape(nrows, nb, 34)
            d = raw[:, :, :2].copy().view(np.float16).reshape(nrows, nb).astype(np.float32)
            q = raw[:, :, 2:].copy().view(np.int8).astype(np.float32)
            return (d[:, :, None] * q).reshape(nrows, ne0)
        raise SystemExit(f"GGUF: {name} has dtype {DTYPE_NAMES.get(dtype, dtype)}; "
                         "G4 supports F32/F16/BF16/Q8_0 (convert to F16 to run the exact check)")


# ------------------------------------------------------------------------------- safetensors

class SafeTensors:
    """Row access into a HF safetensors shard, numpy only (no torch, no safetensors pkg)."""

    def __init__(self, path):
        self.f = open(path, "rb")
        (n,) = struct.unpack("<Q", self.f.read(8))
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def tensor(self, key):
        if key not in self.header:
            raise KeyError(key)
        info = self.header[key]
        s, e = info["data_offsets"]
        dt = info["dtype"]
        shape = info["shape"]
        raw = self.mm[self.base + s:self.base + e]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        else:
            raise SystemExit(f"safetensors: unsupported dtype {dt} for {key}")
        return arr.reshape(shape)


def find_safetensors(model_dir, key):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            m = json.load(f)["weight_map"]
        if key in m:
            return os.path.join(model_dir, m[key])
    cand = os.path.join(model_dir, "model.safetensors")
    return cand if os.path.exists(cand) else None


# ------------------------------------------------------------------------------------- helpers

def hf_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def download_ref(dest_dir):
    """Fetch the published Q8_0 GGUF (~4.4 GB) so G1 can compare against it."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, REF_FILE)
    if os.path.exists(dest):
        return dest
    print(f"downloading {REF_URL}\n     -> {dest}")
    with urllib.request.urlopen(REF_URL, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    return dest


# --------------------------------------------------------------------------------------- gates

def gate1(gg: GGUF, ref_path=None):
    arch = gg.kv.get("general.architecture")
    report("G1 arch", "PASS" if arch == "spark2_5" else "FAIL",
           f"general.architecture = {arch!r}")

    n_layer = gg.kv.get("spark2_5.block_count")
    n_embd = gg.kv.get("spark2_5.embedding_length")
    n_ff = gg.kv.get("spark2_5.feed_forward_length")
    n_head = gg.kv.get("spark2_5.attention.head_count")
    n_kv = gg.kv.get("spark2_5.attention.head_count_kv")
    hd = gg.kv.get("spark2_5.attention.key_length")
    n_vocab = gg.kv.get("spark2_5.vocab_size")
    missing = [k for k, v in (("block_count", n_layer), ("embedding_length", n_embd),
                              ("feed_forward_length", n_ff), ("head_count", n_head),
                              ("head_count_kv", n_kv), ("key_length", hd),
                              ("vocab_size", n_vocab)) if v is None]
    if missing or arch != "spark2_5":
        report("G1 tensor set", "FAIL", f"cannot derive shapes, missing keys: {missing}")
        return

    want = {"token_embd.weight": (n_embd, n_vocab), "output_norm.weight": (n_embd,)}
    per_layer = {
        "attn_norm.weight":  (n_embd,),
        "attn_qkv.weight":   (n_embd, (n_head + 2 * n_kv) * hd),
        "attn_gate.weight":  (n_embd, n_head),
        "attn_output.weight": (n_head * hd, n_embd),
        "ffn_norm.weight":   (n_embd,),
        "ffn_gate.weight":   (n_embd, n_ff),
        "ffn_up.weight":     (n_embd, n_ff),
        "ffn_down.weight":   (n_ff, n_embd),
    }
    for i in range(n_layer):
        for suffix, shape in per_layer.items():
            want[f"blk.{i}.{suffix}"] = shape

    bad = []
    for name, shape in want.items():
        if name not in gg.tensors:
            bad.append(f"missing {name}")
        elif tuple(gg.tensors[name][0]) != shape:
            bad.append(f"{name} shape {tuple(gg.tensors[name][0])} != {shape}")
    extra = sorted(set(gg.tensors) - set(want))
    if extra:
        bad.append(f"{len(extra)} unexpected tensor(s): {extra[:3]}")
    if "output.weight" in gg.tensors:
        bad.append("output.weight present: the head must be tied (Spark has no lm_head.weight)")
    report("G1 tensor set", "PASS" if not bad else "FAIL",
           f"{len(gg.tensors)} tensors, {len(want)} expected, 36x8 + embd + norm"
           + ("" if not bad else " | " + "; ".join(bad[:4])))

    ref_path = ref_path or next((p for p in REF_CANDIDATES if os.path.exists(p)), None)
    if ref_path is None:
        report("G1 vs published", "SKIP",
               f"no reference copy; download {REF_URL} (4.4 GB) or pass --ref-gguf")
        return
    ref = GGUF(ref_path)
    diffs = []
    if set(ref.tensors) != set(gg.tensors):
        only_ref = sorted(set(ref.tensors) - set(gg.tensors))
        only_me = sorted(set(gg.tensors) - set(ref.tensors))
        diffs.append(f"name set differs (ref-only {only_ref[:2]}, file-only {only_me[:2]})")
    else:
        shp = [n for n in ref.tensors if tuple(ref.tensors[n][0]) != tuple(gg.tensors[n][0])]
        if shp:
            diffs.append(f"{len(shp)} shape mismatch(es): {shp[:2]}")
    ref_keys = {k: v for k, v in ref.kv.items() if k.startswith("spark2_5.")}
    my_keys = {k: v for k, v in gg.kv.items() if k.startswith("spark2_5.")}
    for k in sorted(set(ref_keys) | set(my_keys)):
        a, b = ref_keys.get(k, "<absent>"), my_keys.get(k, "<absent>")
        if isinstance(a, float) and isinstance(b, float):
            same = abs(a - b) <= 1e-6 * max(1.0, abs(a))
        else:
            same = a == b
        if not same:
            diffs.append(f"{k}: ref={a!r} file={b!r}")
    report("G1 vs published", "PASS" if not diffs else "FAIL",
           f"{os.path.basename(ref_path)}: {len(ref_keys)} spark2_5 keys, {len(ref.tensors)} tensors"
           + ("" if not diffs else " | " + "; ".join(diffs[:5])))


def gate2(gg: GGUF, model_dir, tok_bin):
    import hashlib
    model = gg.kv.get("tokenizer.ggml.model")
    pre = gg.kv.get("tokenizer.ggml.pre")
    tokens = gg.kv.get("tokenizer.ggml.tokens") or []
    merges = gg.kv.get("tokenizer.ggml.merges") or []
    report("G2 tok type", "PASS" if (model == "gpt2" and pre == "spark2_5") else "FAIL",
           f"model={model!r} pre={pre!r}")
    report("G2 tok size", "PASS" if (len(tokens) == N_TOKENS and len(merges) == N_MERGES) else "FAIL",
           f"{len(tokens)} tokens (want {N_TOKENS}), {len(merges)} merges (want {N_MERGES})")

    if model_dir is None or not os.path.isdir(model_dir):
        report("G2 vs HF", "SKIP", "no HF model directory (--ref-dir)")
        return
    tok = hf_tokenizer(model_dir)
    chkhsh = hashlib.sha256(str(tok.encode(CHKTXT)).encode()).hexdigest()
    # the hash identifies the HF pre-tokenizer; it only says something about this file if the
    # file claims that pre-tokenizer (otherwise G2 tok type has already failed)
    ok_hash = chkhsh == PRE_HASH and pre == "spark2_5"
    report("G2 pre hash", "PASS" if ok_hash else "FAIL",
           f"sha256(ids(check_text)) = {chkhsh[:16]}..., file pre = {pre!r} "
           f"(want {PRE_HASH[:16]}... + 'spark2_5')")

    hf_vocab = tok.get_vocab()
    inv = tok.convert_ids_to_tokens(list(range(N_TOKENS)))
    same_vocab = all(inv[i] == tokens[i] for i in range(N_TOKENS))
    hf_merges = getattr(tok, "backend_tokenizer", None)
    # merges.txt order is what the GGUF stores; compare against the HF file when present
    mpath = os.path.join(model_dir, "merges.txt")
    same_merges = None
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#version")]
        same_merges = lines == merges
    report("G2 vocab/merges", "PASS" if same_vocab and same_merges is not False else "FAIL",
           f"token strings {'identical' if same_vocab else 'DIFFER'}, "
           f"merges {'identical' if same_merges else ('DIFFER' if same_merges is False else 'not compared (no merges.txt)')}")

    if tok_bin and os.path.exists(tok_bin):
        bad = []
        for k, p in enumerate(PROBES):
            out = subprocess.run([tok_bin, "-m", gg.path, "--ids", "--log-disable", "--stdin"],
                                 input=p, capture_output=True, text=True, timeout=600, errors="replace")
            got = [int(x) for x in out.stdout.strip().strip("[]").split(",") if x.strip()]
            if out.returncode != 0 or not got:
                tail = (out.stderr.strip().splitlines() or ["no stderr"])[-1]
                bad.append(f"tokenizer binary failed: {tail[:100]}")
                break
            want = tok.encode(p)
            if got != want:
                bad.append(f"probe {k}: llama.cpp {got[:12]} != HF {want[:12]} "
                           f"({len(got)} vs {len(want)} ids)")
        report("G2 vs llama.cpp", "PASS" if not bad else "FAIL",
               f"{len(PROBES) - len(bad)}/{len(PROBES)} probes id-identical"
               + ("" if not bad else " | " + bad[0]))
    else:
        report("G2 vs llama.cpp", "SKIP",
               "pass --llama-tokenize ./build/bin/llama-tokenize to test the C++ pre-tokenizer")


def gate3(gg: GGUF):
    bad = []
    for key, want in EXPECT.items():
        got = gg.kv.get(key)
        if got is None:
            bad.append(f"{key} absent")
        elif isinstance(want, float):
            if abs(got - want) > 1e-6 * max(1.0, abs(want)):
                bad.append(f"{key}={got} want {want}")
        elif got != want:
            bad.append(f"{key}={got} want {want}")
    report("G3 hparams", "PASS" if not bad else "FAIL",
           f"{len(EXPECT)} keys" + ("" if not bad else " | " + "; ".join(bad[:4])))

    pat = gg.kv.get("spark2_5.attention.sliding_window_pattern")
    full = set()
    if isinstance(pat, list) and len(pat) == 36 and all(isinstance(v, (bool, np.bool_)) for v in pat):
        full = {i for i, v in enumerate(pat) if not v}          # bool form: True = sliding
    elif isinstance(pat, list) and len(pat) == 1 and pat[0] > 0:
        # int form: pat[0] = N means N sliding layers then one full layer (i % (N+1) == N)
        full = {i for i in range(36) if i % (pat[0] + 1) == pat[0]}
    if full:
        n_slide = 36 - len(full)
        ok = full == N_FULL and n_slide == 27
        report("G3 layer pattern", "PASS" if ok else "FAIL",
               f"{n_slide} sliding / {len(full)} full, full at {sorted(full)}"
               + ("" if ok else f" (want full at {sorted(N_FULL)})"))
    else:
        report("G3 layer pattern", "FAIL", f"sliding_window_pattern = {pat!r}")
    report("G3 tied head", "PASS" if "output.weight" not in gg.tensors else "FAIL",
           "no output.weight (tied to token_embd)" if "output.weight" not in gg.tensors
           else "output.weight present")


def gate4(gg: GGUF, model_dir, tol_q):
    if model_dir is None or not os.path.isdir(model_dir):
        report("G4 qkv order", "SKIP", "no HF model directory (--ref-dir)")
        return
    st_path = find_safetensors(model_dir, QKV_HF)
    if st_path is None:
        report("G4 qkv order", "SKIP", f"{QKV_HF} not in the safetensors index")
        return
    hf = SafeTensors(st_path).tensor(QKV_HF)                 # [6144, 2560]
    gg0 = gg.row("blk.0.attn_qkv.weight", 0, gg.tensors["blk.0.attn_qkv.weight"][0][1])
    dtype = gg.tensors["blk.0.attn_qkv.weight"][1]
    if hf.shape != gg0.shape:
        report("G4 qkv order", "FAIL", f"shape {gg0.shape} vs HF {hf.shape}")
        return
    deltas = [float(np.max(np.abs(gg0[r] - hf[r]))) for r in PROBE_ROWS]
    worst = max(deltas)
    exact = dtype in (F32, F16, BF16)
    tol = 1e-6 if exact else tol_q
    ok = worst <= tol
    detail = (f"[{DTYPE_NAMES.get(dtype, dtype)}] rows {PROBE_ROWS}: max|d| = "
              + ", ".join(f"{d:.3g}" for d in deltas) + f" (tol {tol:g})")
    if not ok:
        # explain a permuted/interleaved write: where does each row actually live?
        hfn = hf / (np.linalg.norm(hf, axis=1, keepdims=True) + 1e-9)
        guesses = []
        for r in PROBE_ROWS:
            sims = hfn @ (gg0[r] / (np.linalg.norm(gg0[r]) + 1e-9))
            guesses.append(f"{r}->{int(np.argmax(sims))}")
        detail += f" | best HF row match: {', '.join(guesses)} (want r->r)"
    elif not exact:
        hfn = hf / (np.linalg.norm(hf, axis=1, keepdims=True) + 1e-9)
        moved = [r for r in PROBE_ROWS
                 if int(np.argmax(hfn @ (gg0[r] / (np.linalg.norm(gg0[r]) + 1e-9)))) != r]
        if moved:
            ok = False
            detail += f" | rows {moved} best-match a different HF row"
    report("G4 qkv order", "PASS" if ok else "FAIL", detail)


class HFWeights:
    """Lazy shard loader: HF key -> torch tensor, numpy/safetensors only (no transformers)."""

    def __init__(self, model_dir, torch):
        self.dir, self.torch, self.shards = model_dir, torch, {}
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        self.map = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {}

    def get(self, key):
        shard = self.map.get(key, "model.safetensors")
        if shard not in self.shards:
            self.shards[shard] = SafeTensors(os.path.join(self.dir, shard))
        return self.torch.from_numpy(np.array(self.shards[shard].tensor(key), copy=True))


def torch_reference(model_dir, prompt, ntok, device="cuda"):
    """Literal transcription of modeling_spark.py, independent of the installed transformers.

    No remote code, no version coupling: embedding -> 36 x (fused q|k|v, per-type NEOX rope,
    GQA attention with the HF sliding window, per-head sigmoid gate before out_proj, GELU MLP),
    final norm, tied head.  One full forward per generated token (no cache) -- fine for 24.
    """
    import torch
    import torch.nn.functional as F

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    W = HFWeights(model_dir, torch)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    dt = torch.bfloat16
    n_layer, n_head, n_kv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, hidden, eps = cfg["head_dim"], cfg["hidden_size"], cfg["rms_norm_eps"]
    q_dim, kv_dim = n_head * hd, n_kv * hd
    swa, types = cfg["sliding_window"], cfg["layer_types"]
    rope_cfg = cfg["rope_parameters"]

    def t(key):
        return W.get(key).to(dev, dt)

    w = {"embed": t("model.embedding.weight"), "norm": t("model.norm.weight")}
    for i in range(n_layer):
        p = f"model.layers.{i}."
        w[i] = {"in": t(p + "input_layernorm.weight"), "post": t(p + "post_attention_layernorm.weight"),
                "qkv": t(p + "self_attn.q_k_v_proj.weight"), "g": t(p + "self_attn.g_proj.weight"),
                "o": t(p + "self_attn.out_proj.weight"), "gate": t(p + "mlp.gate_proj.weight"),
                "up": t(p + "mlp.up_proj.weight"), "down": t(p + "mlp.down_proj.weight")}

    def rms(x, weight):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
        return (weight.float() * h).to(x.dtype)

    def rope(seq_len, base, factor):
        rd = int(hd * factor)
        inv = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=dev) / rd))
        fr = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=dev), inv)
        fr = torch.cat([fr, fr], dim=-1)
        return fr.cos(), fr.sin(), rd

    def apply_rope(x, cos, sin, rd):
        orig = x.dtype
        x = x.float()
        rot, pas = x[..., :rd], x[..., rd:]
        half = rd // 2
        rot = torch.cat([-rot[..., half:], rot[..., :half]], -1) * sin + rot * cos
        return torch.cat([rot, pas], -1).to(orig)

    ids = prompt_ids(model_dir, prompt)

    def forward(seq):
        T = len(seq)
        x = w["embed"][torch.tensor(seq, device=dev)].unsqueeze(0)
        idx = torch.arange(T, device=dev)
        causal = idx[None, :] <= idx[:, None]                                   # [T, T] bool
        cos_full, sin_full, rd_full = rope(T, rope_cfg["full_attention"]["rope_theta"],
                                           rope_cfg["full_attention"]["partial_rotary_factor"])
        cos_swa, sin_swa, rd_swa = rope(T, rope_cfg["sliding_attention"]["rope_theta"],
                                        rope_cfg["sliding_attention"]["partial_rotary_factor"])
        # HF: sliding_window_overlay -> kv_idx > q_idx - sliding_window, i.e. d = q-kv in [0, 511]
        swa_mask = causal & (idx[None, :] > idx[:, None] - swa)
        for i in range(n_layer):
            L = w[i]
            sliding = types[i] == "sliding_attention"
            cos, sin, rd = (cos_swa, sin_swa, rd_swa) if sliding else (cos_full, sin_full, rd_full)

            h = rms(x, L["in"])
            qkv = h @ L["qkv"].T
            q, k, v = qkv[..., :q_dim], qkv[..., q_dim:q_dim + kv_dim], qkv[..., q_dim + kv_dim:]
            gate = (h @ L["g"].T).view(1, T, n_head, 1)
            q = q.view(1, T, n_head, hd).transpose(1, 2)
            k = k.view(1, T, n_kv, hd).transpose(1, 2)
            v = v.view(1, T, n_kv, hd).transpose(1, 2)
            q = apply_rope(q, cos[None, None], sin[None, None], rd)
            k = apply_rope(k, cos[None, None], sin[None, None], rd)
            k = k.repeat_interleave(n_head // n_kv, dim=1)
            v = v.repeat_interleave(n_head // n_kv, dim=1)
            scores = (q @ k.transpose(-1, -2)) * (1.0 / hd ** 0.5)
            scores = scores.masked_fill(~(swa_mask if sliding else causal)[None, None], float("-inf"))
            probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
            attn = (probs @ v) * torch.sigmoid(gate.float()).to(q.dtype).transpose(1, 2)
            x = x + (attn.transpose(1, 2).reshape(1, T, -1) @ L["o"].T)

            h = rms(x, L["post"])
            x = x + (F.gelu(h @ L["gate"].T) * (h @ L["up"].T)) @ L["down"].T
        return (rms(x, w["norm"]) @ w["embed"].T)[0]

    out = list(ids)
    with torch.no_grad():
        for _ in range(ntok):
            logits = forward(out)
            out.append(int(torch.argmax(logits[-1])))
    return out[len(ids):]


def prompt_ids(model_dir, text):
    """Prompt ids via the HF tokenizer (the only place G5 needs transformers)."""
    return hf_tokenizer(model_dir).encode(text)


def gate5(gg: GGUF, model_dir, cli_bin, ngl, ntok):
    if not cli_bin or not os.path.exists(cli_bin):
        report("G5 text parity", "SKIP", "--text needs --llama-cli <path>")
        return
    if model_dir is None or not os.path.isdir(model_dir):
        report("G5 text parity", "SKIP", "--text needs --ref-dir with the HF checkpoint")
        return
    prompt = PROBES[0]
    out = subprocess.run([cli_bin, "-m", gg.path, "-p", prompt, "-n", str(ntok), "--temp", "0",
                          "-ngl", str(ngl), "-c", "4096", "--no-display-prompt", "--no-warmup"],
                         capture_output=True, text=True, timeout=1800, errors="replace")
    txt = out.stdout
    for marker in ("\nllama_perf", "\nllama_print_timings", "\nllama :"):
        if marker in txt:
            txt = txt.split(marker)[0]
    txt = txt.strip()

    ref_ids = torch_reference(model_dir, prompt, ntok)
    tok = hf_tokenizer(model_dir)
    ref = tok.decode(ref_ids, skip_special_tokens=True).strip()

    n = 0
    for a, b in zip(txt, ref):
        if a != b:
            break
        n += 1
    n_tok = len(tok.encode(txt[:n])) if n else 0
    ok = n_tok >= min(20, ntok)
    report("G5 text parity", "PASS" if ok else "FAIL",
           f"common prefix {n} chars (~{n_tok}/{ntok} tokens) | llama.cpp: {txt[:60]!r} | "
           f"torch: {ref[:60]!r}")


# ---------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Port-verification kit for spark2_5 GGUFs")
    ap.add_argument("gguf", help="GGUF to verify")
    ap.add_argument("--ref-gguf", help="published Spark-X2.6-4B-Q8_0.gguf to compare against")
    ap.add_argument("--download-ref", metavar="DIR",
                    help="download the published Q8_0 GGUF (4.4 GB) into DIR if it is absent")
    ap.add_argument("--ref-dir", default=os.environ.get("SPARK_HF_DIR"),
                    help="HF checkpoint directory (tokenizer + safetensors); "
                         "default: ./Spark-X2.6-4B, then /mnt/Data/hf/Spark-X2.6-4B")
    ap.add_argument("--llama-tokenize", default=os.environ.get("LLAMA_TOKENIZE"),
                    help="llama-tokenize binary for the tokenizer id check")
    ap.add_argument("--tol-q", type=float, default=0.02,
                    help="max|d| tolerance for the QKV row check on quantised files (default 0.02)")
    ap.add_argument("--text", action="store_true", help="run G5 text parity (needs torch + GPU)")
    ap.add_argument("--llama-cli", default=os.environ.get("LLAMA_CLI"))
    ap.add_argument("--ngl", type=int, default=99, help="GPU layers for G5 (0 = CPU)")
    ap.add_argument("--ntok", type=int, default=24, help="greedy tokens for G5")
    args = ap.parse_args()
    if args.download_ref:
        args.ref_gguf = download_ref(args.download_ref)
    if not args.ref_dir:
        for cand in ("Spark-X2.6-4B", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "..", "hf", "Spark-X2.6-4B"),
                     "/mnt/Data/hf/Spark-X2.6-4B"):
            if os.path.isdir(cand):
                args.ref_dir = cand
                break

    gg = GGUF(args.gguf)
    print(f"file : {args.gguf}")
    print(f"arch : {gg.kv.get('general.architecture')}  tensors: {len(gg.tensors)}  "
          f"kv: {len(gg.kv)}\n")

    def run(gate, fn, *a):
        """A broken gate must not swallow the table -- report it as FAIL and carry on."""
        try:
            fn(*a)
        except Exception as e:                                   # noqa: BLE001
            report(gate, "FAIL", f"gate error: {type(e).__name__}: {str(e)[:160]}")

    run("G1", gate1, gg, args.ref_gguf)
    run("G2", gate2, gg, args.ref_dir, args.llama_tokenize)
    run("G3", gate3, gg)
    run("G4", gate4, gg, args.ref_dir, args.tol_q)
    if args.text:
        run("G5", gate5, gg, args.ref_dir, args.llama_cli, args.ngl, args.ntok)
    else:
        report("G5 text parity", "SKIP", "not requested (--text)")

    w = max(len(g) for g, _, _ in rows)
    for gate, status, detail in rows:
        print(f"{status:4s}  {gate:<{w}}  {detail}")
    fails = [r for r in rows if r[1] == "FAIL"]
    skips = [r for r in rows if r[1] == "SKIP"]
    print(f"\n{len(rows) - len(fails) - len(skips)} pass, {len(fails)} fail, {len(skips)} skip")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
