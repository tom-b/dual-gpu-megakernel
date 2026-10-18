# Qwen3.8-27B megakernel on two GPUs

Modified to run on two RTX 2000 Ada cards. Build with make ARCH=sm_89. On my test, doubled token generation speed to 50 tokens/sec
on agentic test that had to read about 20K of context from source and markdown files.

Everything below is the same test as the original fork.

A two-GPU fork of the Qwen3.8-27B CUDA megakernel by **L-Forster**, from [open-jet](https://github.com/L-Forster/open-jet) (`megakernel/`). The megakernel itself is their work: one persistent kernel launch per speculative-decoding cycle (main model, verification, MTP drafting), the batched prefill kernels, the OpenAI-compatible server, the benchmarks and the quality tooling. On one RTX 3090 it already runs 1.4-1.9x faster than llama.cpp.

**This fork splits it tensor-parallel across two GPUs.** On two RTX 3090s it decodes at 164 tok/s, 1.64x llama.cpp's best two-GPU setup on the same cards, with the same output quality, over plain PCIe (no NVLink). It also loads Q4_K / Q5_K / Q6_K / Q8_0 weights in any per-tensor mix.

| greedy decode, 4 chat prompts x 512 tokens (mean tok/s) | one RTX 3090 (upstream's measurement) | two RTX 3090s (this fork) |
|---|---|---|
| llama.cpp, best setup measured | 63.1 (MTP 2 drafts) | 99.7 (tensor split, MTP 4) |
| megakernel | 106.7 | **164.0** |
| speed-up | 1.69x | **1.64x** |

## Contents

- [Quick start](#quick-start)
- What this fork adds: [Two GPUs](#two-gpus-this-fork), [Results](#results), [Quality](#quality), [FAQ](#faq)
- The engine (upstream's design): [How it works](#how-it-works), [Prefill](#prefill), [Server](#server), [The GGUF](#the-gguf)
- [Limitations](#limitations), [Files](#files), [License and credits](#license-and-credits)

## Quick start

Requirements: CUDA 12+ (12.8+ for RTX 50 series), one or two sm_86 / sm_89 / sm_120 GPUs with 24 GB, the Q4_K_M GGUF of Qwen3.8-27B, and a llama.cpp build next to this directory (only `libllama`, used vocab-only for tokenization; a CPU-only build is enough). For two GPUs, see the P2P requirement in [Two GPUs](#two-gpus-this-fork).

```sh
# llama.cpp (tag b10246) as ../llama.cpp, CPU-only shared libllama
git clone --depth 1 --branch b10246 https://github.com/ggml-org/llama.cpp ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF
cmake --build ../llama.cpp/build -j

make   # single-GPU and TP2 CLI + server libraries, tools; ~15 min (mostly the TP2 compile)

# one GPU
./qwen38_mk -m ../models/Qwen3.8-27B-Q4_K_M.gguf -p "Write quicksort in C" -n 512 \
            -d 4 --draft-vocab draft_vocab_32k.bin
# two GPUs (devices 0,1 by default)
MK_DEVICES=0,1 ./qwen38_mk_tp2 -m ../models/Qwen3.8-27B-Q4_K_M.gguf -p "Write quicksort in C" -n 512 \
            -d 4 --draft-vocab draft_vocab_32k.bin
# OpenAI-compatible server on two GPUs
python3 server/mk_server.py -m ../models/Qwen3.8-27B-Q4_K_M.gguf --lib ./libqwen38mk_tp2.so -c 100000
```

`make ARCH=sm_89` overrides the detected GPU architecture. `docker/build.Dockerfile` plus `scripts/box-sync.sh <ssh-host> <dir>` and `scripts/box-build.sh <ssh-host> <dir>` build everything inside a CUDA container on a remote GPU host.

CLI flags: `-d` drafts per cycle (0-4; with 4 the depth drops to 3 for a cycle after a rejected draft), `-c` KV capacity, `--kv f16|q8|auto` (KV cache format), `--think [xhigh|medium|low]` (reasoning mode; the chat template's default effort is xhigh), `--temp` / `--top-p` / `--top-k` / `--min-p` / `--seed` (sampling; default greedy), `--raw` (no chat template), `--ids`, `--prompt-ids f` (prompt as token ids), `--turn2 text` / `--turn2-file f` (a second user turn appended to the conversation), `--score ids.txt` (teacher-forced agreement), `--kld ref` (KL divergence, perplexity and agreement against a reference file, see [Quality](#quality)). Environment: `MK_DEVICES=a,b` (TP2 cards), `MK_PROFILE=1` (per-phase timing, with a "min-composed" estimate that filters out time-slicing by other GPU users), `MK_PF_PROFILE=1` (per-kernel prefill timing), `MK_PREFILL=mk` (prefill through the decode megakernel instead).

## Two GPUs (this fork)

`qwen38_mk_tp2` / `libqwen38mk_tp2.so` (`-DMK_TP=2`) split the model over two identical GPUs. Each card runs its own copy of the decode megakernel on its half of the weights; the cards exchange partial sums directly over PCIe peer-to-peer with per-item flags (no cross-card grid barrier) and add them in rank order, so the replicated residual stream stays bit-identical on both cards. The batched prefill runs one host thread per card with the same rank-order all-reduce.

What each card holds: its attention q / KV heads, its Gated-DeltaNet value heads (interleaved by key head: card c owns heads h with `(h % 16) / 8 == c`), its half of the FFN rows (8,704, zero-padded to 9,216 for the GEMV tiling), its half of the lm_head vocabulary (top-k / argmax merged across cards), its K half of the MTP `eh_proj`, and its half of the MTP draft vocabulary. Embeddings and norms are replicated. Weights take 8.7 GB per card instead of 16.7 GB on one, which leaves room for long contexts with an fp16 KV cache.

**P2P is required** (the engine refuses to start without it). GeForce cards have PCIe P2P disabled in NVIDIA's stock driver; the measurements here use the 595.84 open kernel module with the community P2P patch (tinygrad's, as packaged in aikitoria/open-gpu-kernel-modules), Resizable BAR on and `iommu=pt`. Check with `nvidia-smi topo -p2p r` (both directions `OK`). Workstation and data-center cards support P2P with the stock driver.

Two-GPU specific work beyond the split itself, each measured with `MK_PROFILE=1`:

- **Overlapped all-reduce**: the residual projections (attention output, DeltaNet output, FFN down) compute all their items, park the partials in shared memory and send them, then wait and combine, instead of waiting on the peer item by item.
- **Warp-pair FFN gate/up**: two warps split each 16-row item and share one activation load (double-buffered), cutting the FFN gate/up phase from 102.5 to 84.5 µs.
- **Balanced GEMV tails**: when a phase's row pairs do not divide evenly over the 1,312 warps, the last round is spread one row per warp.
- **Cross-phase L2 prefetch**: before each grid barrier, blocks prefetch the first weight rows of the next phase (and of the next layer's projections).
- **Split MTP draft vocabulary**: each card holds and scores half of the draft rows (gathered at load from both cards' vocabulary halves over P2P), cutting the MTP lm_head from 220 to 132 µs per cycle.

## Results

**Setup**: two RTX 3090s (24 GB), each power-limited to 280 W (stock: 350 W), each on PCIe 4.0 x8 through the CPU (no NVLink); Ryzen 7 5800X; driver 595.84 (open, P2P patch); CUDA 12.9; unsloth's Qwen3.8-27B Q4_K_M; greedy decoding; best of 2 runs, back-to-back on otherwise idle GPUs. llama.cpp is tag b10246 built with CUDA, `llama-server --flash-attn on -ctk q8_0 -ctv q8_0 -ngl 99 --spec-type draft-mtp --spec-draft-n-max N -sm <layer|tensor>` on the same two cards; the megakernel uses `-d 4` and its automatic KV format (fp16 at these sizes). Reproduce with `bench/tp2_bench.py`.

### Two GPUs: decode on 4 chat prompts (512 tokens, tok/s)

| prompt | llama.cpp layer split, MTP 2 | llama.cpp layer split, MTP 4 | llama.cpp tensor split, MTP 4 | megakernel TP2 |
|---|---|---|---|---|
| Python function (code) | 84.5 | 91.9 | 128.0 | **213.9** (1.67x) |
| hash map explanation | 75.5 | 74.7 | 102.8 | **168.2** (1.64x) |
| short story | 61.6 | 49.0 | 65.0 | **109.2** (1.68x) |
| Flask to FastAPI plan | 74.6 | 75.6 | 102.9 | **164.8** (1.60x) |
| **mean** | 74.1 | 72.8 | 99.7 | **164.0** (1.64x) |

Speed-ups are against llama.cpp's fastest configuration, tensor split with 4 drafts. Layer split (llama.cpp's default) runs the cards one after the other; tensor split is 35% faster.

### Two GPUs: a source file in the prompt, then a coding task (256 tokens)

| context (tokens) | decode, llama.cpp tensor MTP 4 | decode, megakernel TP2 | prompt processing, llama.cpp (best) | prompt processing, megakernel TP2 |
|---|---|---|---|---|
| 3,599 | 83.4 | **133.2** (1.60x) | 1,470 | **1,872** (1.27x) |
| 15,751 | 80.2 | **136.3** (1.70x) | 1,623 | **1,787** (1.10x) |
| 29,586 | 76.9 | **128.8** (1.67x) | 1,549 | **1,662** (1.07x) |
| 56,017 | 66.6 | **105.4** (1.58x) | 1,385 | **1,457** (1.05x) |

(tok/s; llama.cpp's prompt processing is its best of the three configurations, layer split with 2 drafts at every size.) The decode lead holds at long context; the prompt-processing lead shrinks as attention over the growing prompt dominates.

### Two GPUs: draft acceptance

| prompt | llama.cpp tensor MTP 4: drafted tokens accepted | megakernel TP2: tokens per cycle (max 5) |
|---|---|---|
| Python function | 93.9% | 4.46 |
| hash map | 69.7% | 3.43 |
| short story | 34.0% | 2.16 |
| Flask plan | 69.8% | 3.36 |

Both engines draft with the model's own MTP head, so acceptance depends mostly on how predictable the text is. The megakernel's lead does not come from accepting more: verifying 5 tokens in one pass costs it almost the same as verifying 1, because each pass reads the weights once. Without drafting, both engines are close to the memory-bandwidth limit (one GPU: llama.cpp 40.4 tok/s, megakernel 42.9).

### One GPU (upstream's measurements)

Measured by upstream on one RTX 3090 against `llama-server` with open-jet's flags (MTP 2 drafts). This fork's single-GPU build runs at the same speed as upstream (105.7 vs 106.7 tok/s mean on the 4 prompts, measured back-to-back on the setup above).

| workload | llama.cpp | megakernel | |
|---|---|---|---|
| greedy, 4 chat prompts x 512 tokens (mean) | 63.1 | 106.7 | 1.69x |
| greedy, Python function (code) | 72.5 | 139.7 | 1.93x |
| reasoning mode through the HTTP servers (thinking on, temp 1.0, top_p 0.95, top_k 20; 5 coding prompts x 2 seeds, up to 1,024 tokens) | 50.9 | 78.0 | 1.53x |
| decode with 3.6K tokens of code in context | 56.3 | 95.3 | 1.69x |
| decode with 29.5K tokens of code in context | 52.3 | 72.7 | 1.39x |
| prompt processing, 1.8K to 16K tokens | 1,050 to 1,140 | 1,466 to 1,595 | ~1.4x |
| pasting 2.7K / 12K new tokens into a running conversation | 2.41 / 10.45 s | 1.68 / 8.10 s | |

The context rows predate the sequence-token draft vocabulary and double-buffered attention, which add 1-8% to the megakernel's decode there. With the q8_0 KV cache (chosen automatically when fp16 does not fit, above ~80K tokens on one card) decode is ~1-2% and prefill 2-7% slower than with fp16. Single-GPU benchmarks: `bench/code_bench.py`, `bench/think_bench.py`, `bench/bench.py`.

## Quality

The output is not byte-identical to llama.cpp's, and the two-GPU output is not byte-identical to the one-GPU output: the arithmetic is ordered and rounded differently (two cards add their partial sums separately), so where two tokens are almost tied the other one can win, and the generation then continues differently. Whether that costs quality is measured against llama.cpp's next-token distributions: top-token agreement, mean KL divergence and perplexity.

**This fork against upstream** (same reference for all three: llama.cpp b10246 batched on the CPU, q8_0 KV, 4,092 scored positions of this repository's code and README, each with at least 1,024 tokens of context; `--score-path mixed --kv q8`):

| engine | top token agrees | KL divergence (mean, nats) | perplexity (reference 4.1372) |
|---|---|---|---|
| upstream megakernel, one GPU | 96.75% | 0.00406 | 4.1356 |
| this fork, one GPU | 96.75% | 0.00406 | 4.1356 |
| this fork, two GPUs | 96.75% | 0.00413 | 4.1313 |

The one-GPU build matches upstream to the last digit. The two-GPU build disagrees with the reference at the same 133 positions and has the same perplexity within noise: splitting across cards costs no quality. (Absolute agreement is lower than in the table below because this reference is llama.cpp batched on the CPU, which rounds differently from llama.cpp on a GPU; the comparison that matters is between the three rows.)

**Upstream's study against llama.cpp generating token by token** (their measurement and their tooling, `tools/llama_ref` and `--kld`, which the check above reuses; one GPU; 13,299 positions of code and prose, each after a 1,024-token prompt; reference: llama.cpp on the GPU, q8_0 KV, as open-jet runs it):

| against llama.cpp generating token by token | top token agrees | KL divergence (mean, nats) | perplexity (reference 2.4263) |
|---|---|---|---|
| llama.cpp, batched instead | 98.68% | 0.0019 | 2.4283 |
| llama.cpp, fp16 KV cache instead | 99.17% | 0.0007 | 2.4284 |
| megakernel (prefill kernels, then the decode kernel), q8_0 KV | 99.08% | 0.0009 | 2.4274 |
| megakernel, fp16 KV | 99.11% | 0.0008 | 2.4273 |
| megakernel, prefill kernels only, q8_0 KV | 99.01% | 0.0009 | 2.4275 |

The megakernel is as close to llama.cpp as llama.cpp's own configurations are to each other; its log-loss differs by +0.0005 +- 0.0004 (not significant). Most disagreements are near-ties (median lead of llama.cpp's token 0.03-0.04 nats); in 2-3 of ~120 llama.cpp's token led by more than 0.5 nats, as in llama.cpp against itself (4 of 176). At 30K context (1,999 positions after a 28,000-token prompt) it agrees on 99.65% with KL 0.0007 (fp16 KV: 99.55%, 0.0007; llama.cpp batched: 99.35%, 0.0009), log-loss +0.0018 +- 0.001 (llama.cpp batched +0.0017); at 96K (1,999 positions after 94,000 tokens, q8_0 KV) on 99.60% with KL 0.0008, log-loss +0.0012 +- 0.001 (llama.cpp batched: 99.45%, 0.0014, +0.0015 +- 0.002). Each long-context figure is from a single sequence of code.

Speculative verification is exact: greedy output is the main model's greedy choice, and with sampling the token distribution is the same as without drafts (2,000-sample test per position). Verifying 1 vs 4 tokens per pass rounds slightly differently, so on near-ties the output can differ between `-d` settings.

To reproduce:

```sh
make tools/llama_ref
tools/llama_ref -m model.gguf -f code.txt -f prose.txt --score-from 1024 -b 1 --ref-out llama.ref
tools/llama_ref -m model.gguf --kld llama.ref                                 # llama.cpp batched: the noise floor
./qwen38_mk     -m model.gguf --kld llama.ref --score-path mixed --kv q8      # one GPU
./qwen38_mk_tp2 -m model.gguf --kld llama.ref --score-path mixed --kv q8      # two GPUs
```

`llama_ref` teacher-forces 2,048-token chunks of the text (`--chunk`, `-b 1` for token by token, `--max-chunks` to limit) and stores, per scored position, the 256 most likely tokens (`logit_stats.h`). `--score-path` selects the megakernel's batched prefill kernels (`prefill`, the default), the decode kernel (`decode`), or the prefill kernels for the unscored prompt and the decode kernel for the scored positions (`mixed`, what a generation runs).

## FAQ

**Does it run on two 3090s? Is one instance on two cards better than two instances?** Yes (`qwen38_mk_tp2`, `--lib libqwen38mk_tp2.so`). One instance on two cards gives one user 164 tok/s and twice the KV memory. Two single-GPU instances give two users ~107 tok/s each, which is more total throughput if you have two independent streams.

**More than two GPUs?** No. The split is written for exactly two cards (`MK_TP=2`).

**Is the output byte-identical to llama.cpp?** No, and neither is llama.cpp to itself across its own settings. See [Quality](#quality): agreement, KL divergence and perplexity are within llama.cpp's own spread, and the two-GPU build matches the one-GPU build.

**Was the llama.cpp baseline fair?** On two GPUs it was run in all three relevant configurations, including tensor split (both cards working on every layer, like this engine) with the same 4 drafts; the speed-ups above are against the fastest. llama.cpp tensor split with 2 drafts was not measured; with layer split, 2 drafts beat 4 by 2% on average.

**What is the draft acceptance?** See [the acceptance table](#two-gpus-draft-acceptance): 4.46 tokens per cycle on code, 2.16 on a short story.

**How long can the context be?** The model's native context is 262,144 tokens. One GPU: fp16 KV up to ~80K tokens, q8_0 above that (147K per upstream). Two GPUs: each card holds half the KV cache and half the weights; by memory size the full 262,144 tokens fit with an fp16 KV cache. The longest context benchmarked here is 56K.

**What power?** Both cards are capped at 280 W (`nvidia-smi -pl 280`), below the 350 W stock limit.

**How fast is prompt processing?** 1,457 to 1,872 tok/s on two GPUs (1.05-1.27x llama.cpp), ~1,500 to 1,600 tok/s on one; see the tables in [Results](#results).

**Parallel requests / continuous batching?** No. One sequence at a time; the speed comes from verifying 5 tokens per weight read, which batching would compete with.

**Vision (mmproj)?** No, text only.

**Which quants?** unsloth's Q4_K_M is the tuned and benchmarked file. Q4_K, Q5_K, Q6_K and Q8_0 tensors load in any per-tensor mix (Q4_K_M, UD-Q4_K_M, Q5_K_M, Q6_K, Q8_0 files), but only Q4_K_M has been validated against real output; the others are untuned and unvalidated. IQ*, Q2/Q3 and other types are refused at load. A Q8_0 file (28.7 GB) only fits on two cards.

**RTX 4090 / 5090?** It builds for sm_89 and sm_120 and runs on 24 GB+ cards, but it is tuned and measured on 3090s only. Two-GPU mode needs P2P, which GeForce cards only have with the patched driver.

**NVLink?** Not used and not needed; everything above is over PCIe 4.0 x8 per card.

**Pass rate on coding tasks?** Not measured yet.

**Compared with other single-model engines (HyperQwen, ninfer, ...)?** Not measured here.

## The engine (upstream's design)

Everything in this part is L-Forster's design from open-jet, described as upstream documents it. This fork extends it in two places: the per-tensor quant types (upstream accepts unsloth's Q4_K_M layout) and the two-GPU split described in [Two GPUs](#two-gpus-this-fork).

### How it works

Decoding runs an entire speculative cycle in one persistent cooperative kernel launch:

```
main model over T = 1 + n_draft tokens  ->  lm_head + verification on device (greedy, or exact
speculative sampling)  ->  MTP catch-up rows for the accepted tokens  ->  n_draft MTP draft steps
```

The ~340 phases of a cycle (projections, DeltaNet core, attention, FFN, lm_head, MTP) are separated by a software grid barrier; the host only reads back the accepted tokens. Prompts go through batched tensor-core kernels instead (see [Prefill](#prefill)). Where the speed comes from:

- **Weights stay in their GGUF k-quant encoding** (Q4_K / Q5_K / Q6_K / Q8_0), read once per cycle regardless of T. Q6_K and Q8_0 are re-laid-out at load time (values unchanged) so every lane does aligned 16/8-byte loads; Q6_K blocks carry their scales and `d` in one 8-byte word.
- **dp4a GEMV with shared activations**: activations are int8 (per-16/32 scales, as llama.cpp's q8_1); each warp streams 2-4 weight rows at once and reuses every shared-memory activation load across them, so T = 5 costs almost the same memory time as T = 1. Weight loads bypass L1 and are marked L2 evict-first.
- **No launches, few round trips**: activations are re-quantised by the producing phase (including the next RMS norm's weight), so consumers only async-copy (`cp.async`) them.
- **Gated-DeltaNet rollback by lazy replay**: the recurrent state is committed only through the surely-accepted token; accepted drafts are replayed from a small (k, v, g, beta) ring at the start of the next cycle, so rejected drafts cost no state snapshots or extra state traffic.
- **Attention**: split-K flash decoding with tensor-core MMAs (Q.K^T and P.V in fp16 with fp32 accumulation, online softmax), double-buffered 32-position K/V tiles in shared memory (~5% faster cycles at 30K context).
- **MTP drafting on device**, with the draft lm_head restricted to the 32,768 most frequent tokens (`draft_vocab_32k.bin`, built by `tools/vocab_freq` from code and docs; covers ~99.4% of tokens), plus the tokens of the current sequence missing from it, added on the fly (identifiers of pasted code: +3-8% decode speed with code in context). Verification always uses the full vocabulary, so this affects only the acceptance rate, never the output.
- **Sampling** keeps the 32 best logits of each verified row; a draft is accepted with probability p(draft) and a rejected one is resampled from the rest of the distribution, which keeps the sampled distribution exact.
- **L2 warm-up**: before each barrier, blocks prefetch (evict-last) what their next latency-bound phase reads first (DeltaNet state, conv weights, norm weights, RoPE rows, first K/V tile).

### Prefill

A prompt (all but its last token) is processed in chunks of up to 2,048 tokens by ordinary kernel launches (`prefill.cuh`), which leave exactly the state the megakernel expects (trunk and MTP K/V caches, committed DeltaNet state, conv ring, hidden-state history); the last prompt token then goes through the megakernel, which produces the first token and the first MTP drafts. The same path appends to a running conversation (`MKEngine::prefill` on a non-empty sequence, e.g. the next user turn): it starts at the committed DeltaNet position and also re-feeds the last emitted token and any accepted drafts not yet folded into the recurrent state; the output is token-identical to processing the whole conversation as one fresh prompt.

- **GEMMs on fp16 tensor cores straight from the k-quants** (`prefill_gemm.cuh`, ~85% of prefill time): 128 x 128 tiles; weights stay in their GGUF blocks and are dequantised to fp16 in registers (exact small integers via the `0x6400` trick, then one fp16 scale and min per sub-block); activations are fp16 with the elements of each 16-group permuted so that a 4-byte weight word is directly an MMA A fragment. Products accumulate in fp16 over k = 128, then in fp32. ~100-108 TFLOPS on all four quant types (`bench/mmq_bench.cu`). FFN gate and up share one GEMM whose epilogue writes `silu(gate) * up` as the down projection's fp16 input.
- **DeltaNet in chunked (WY) form on tensor cores**: per 32-token chunk, `A = tril(beta K K^T)` with the gate decays, `T = (I + A)^-1` (one warp, forward substitution), then `U = T beta V - T beta gamma (K S)`, `O = gamma Q S + P U` and `S <- gamma_last S + K^T (decay U)`. Value columns are independent, so a block owns (head, 64 columns) and each warp keeps its 128 x 8 slice of the state in fp32 MMA accumulators, turning it (and K S, U) into fp16 MMA operands in registers with `movmatrix`; gates are carried as log g, so tiny or zero gates are exact. ~2% of prefill time; 3.5e-4 relative error against a float64 reference of the recurrence.
- **Attention**: causal flash attention over the KV cache on tensor cores.

### Server

`server/mk_server.py` serves the engine (through `libqwen38mk.so` or, with `--lib`, `libqwen38mk_tp2.so`; C API in `capi.cpp`) with an OpenAI-compatible `/v1/chat/completions`, streaming or not, plus `/health` and `/v1/models`. Flags: `-m`, `--host`, `--port` (18080), `-c` (32768), `-d` (4), `--draft-vocab`, `--lib`, `--kv auto|f16|q8`.

- Prompts are rendered with the chat template stored in the GGUF (jinja2): `chat_template_kwargs` `enable_thinking` / `preserve_thinking` / `reasoning_effort`, OpenAI `tools`. Reasoning is streamed as `reasoning_content`; Qwen's XML tool calls are returned as structured `tool_calls`.
- Successive requests of a conversation only prefill what is new: the prompt is matched against the engine's sequence (prompts + emitted tokens), and when the re-rendered history diverges after the end of the previous prompt (e.g. a reformatted tool call) the engine rewinds to a checkpoint saved there. Anything else starts over (batched prefill).
- Sampling: temperature, top_p, top_k (<= 32), min_p, seed; presence / repetition penalties are ignored (with a warning). One request at a time.
- KV cache: fp16, or q8_0 when fp16 does not fit the requested context.

Upstream's [open-jet](https://github.com/L-Forster/open-jet) uses this server in place of `llama-server` (`openjet setup` builds the single-GPU engine).

### The GGUF

Per-tensor quant types are this fork's extension. Qwen3.8-27B, llama.cpp arch `qwen35`: 48 Gated-DeltaNet + 16 gated-attention layers and one MTP block. Every quantized weight tensor may be Q4_K, Q5_K, Q6_K or Q8_0, chosen per tensor:

| tensors | accepted types |
|---|---|
| `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_qkv`, `attn_gate`, `ssm_out`, `ffn_gate`, `ffn_up`, `ffn_down`, `token_embd`, `output`, `nextn.eh_proj` | Q4_K, Q5_K, Q6_K or Q8_0, per tensor |
| `ffn_gate` and `ffn_up` of a layer | must share a type |
| `ssm_alpha`, `ssm_beta` | Q4_K, Q5_K, Q6_K or Q8_0; converted to F32 at load (a few seconds) |

Any other type is refused at load, naming the tensor. A plain `llama-quantize` from the BF16 GGUF produces a supported file (Q4_K_M stores `ssm_out`, `ssm_alpha`, `ssm_beta` and `nextn.eh_proj` as Q4_K, which all loads):

```sh
llama-quantize Qwen3.8-27B-BF16-00001-of-00002.gguf Qwen3.8-27B-Q4_K_M.gguf Q4_K_M
```

## Limitations

- Specialised to this model's shapes. Only Q4_K_M is tuned and validated (see the FAQ on quants).
- Tuned and measured on RTX 3090s (sm_86); builds for sm_89 and sm_120.
- Two GPUs at most, both identical, with PCIe P2P.
- Sampling supports temperature, top-k (<= 32), top-p and min-p (llama.cpp's order); no presence / repetition penalties.
- Text only; one sequence and one checkpoint: a conversation is reused while it grows (or diverges only after the end of the previous prompt); editing an earlier part, e.g. context compaction, re-prefills it. No slot save / restore.

## Files

- `qwen38_mk.cu` / `qwen38_mk.h`: loader, repacking, the megakernel (single-GPU and TP2) and the `MKEngine` API
- `prefill.cuh` / `prefill_gemm.cuh`: batched prefill kernels and their quantised fp16 GEMM
- `capi.cpp`: C API (engine + libllama tokenizer) for `libqwen38mk.so` / `libqwen38mk_tp2.so`
- `server/mk_server.py`: OpenAI-compatible HTTP server
- `main.cu`: CLI (tokenization via libllama)
- `gguf.h`: minimal mmap GGUF reader
- `draft_vocab_32k.bin`: MTP draft-head token subset (int32 ids)
- `tools/vocab_freq.cpp`: builds a draft vocabulary from a list of text files
- `tools/llama_ref.cpp` / `logit_stats.h`: llama.cpp's next-token distributions and the comparison with them (KL divergence, perplexity, top-token agreement)
- `bench/tp2_bench.py`: two-GPU benchmark against llama-server; `bench/code_bench.py`, `bench/think_bench.py`, `bench/bench.py`: single-GPU benchmarks; `bench/*_bench.cu`: GEMV / GEMM microbenchmarks
- `docker/build.Dockerfile`, `scripts/box-*.sh`: containerised build on a remote GPU host

## License and credits

The megakernel is the work of **L-Forster** in [open-jet](https://github.com/L-Forster/open-jet): the decode megakernel, the batched prefill kernels, the loader, the server and CLI, the single-GPU benchmarks (`bench/code_bench.py`, `bench/think_bench.py`, `bench/bench.py`, `bench/*_bench.cu`), the quality tooling (`tools/llama_ref`, `logit_stats.h`, `--kld`) and the draft vocabulary. This repository is a modified version of `megakernel/` at commit `2993f47`, imported with its git history, licensed under the GNU Affero General Public License v3.0 and distributed under the same license (see `LICENSE`).

Modifications, October 2026: sources moved to the repository root; per-tensor Q4_K / Q5_K / Q6_K / Q8_0 weights in decode and prefill; tensor parallelism over two GPUs via PCIe P2P (decode and prefill) with the two-GPU optimisations; `bench/tp2_bench.py`; the two-GPU benchmarks and quality checks above.
