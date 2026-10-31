# Nanbeige4.3-33B-INT4 on one M5 Ultra (TensorFold native, Nanbeige's own INT4 release)

Serves [Nanbeige4.3-33B-INT4](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-INT4), Nanbeige's own int4 release of
its 33B dense Llama-architecture model (512K-token context), as published, with TensorFold's native Metal engine, as
an OpenAI-compatible server with reasoning, tool calls, structured output, the prompt cache and two streams, with
[Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) drafting for it from its bf16 release (replies
unchanged). Nothing is converted: the checkpoint's packed int4 weights (21.0 GiB) are read in place.

The INT4 release is the fast one: one stream decodes at 41-43 tokens/s against 16.3 for the bf16 release; with the
drafter, 79-81 on the harness's ruler and about the same as without it on ordinary chats
([drafter](#drafter-nanbeige41-3b)). Its quantization is Nanbeige's, and it costs something against the bf16
release (mean KL 0.087, top-1 90.8%, [below](#quality-what-the-int4-release-costs)); for the exact model, serve the
bf16 release with [nanbeige43-33b-bf16-m5-ultra](https://github.com/BobClawblaw/nanbeige43-33b-bf16-m5-ultra).

The engine's `llama` Metal family is under review upstream as
[ashhart/TensorFold#1415](https://github.com/ashhart/TensorFold/pull/1415) (the family) and
[#1417](https://github.com/ashhart/TensorFold/pull/1417) (the drafter); reading the INT4 release is the next change on
top of them, branch `llama-int4` on [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold/tree/llama-int4),
pinned in `recipe.toml`.

## Quick start

Needs an M5 Mac with the weights' 21 GiB plus a stream's caches free (the M5's tensor units; M1-M4 are refused),
Python 3.11+, git, `hf` and [Zig 0.17.0](https://ziglang.org/download/).

```bash
git clone https://github.com/BobClawblaw/TensorFold ~/github/TensorFold
hf download Nanbeige/Nanbeige4.3-33B-INT4 --revision d834b414cf1c47e42640de33c147b7cf8a9321ef --local-dir ~/models/Nanbeige4.3-33B-INT4
hf download Nanbeige/Nanbeige4.1-3B --revision 9c4555c37921c0982af8fffeee1e0a6e6b953c4f --local-dir ~/models/Nanbeige4.1-3B
cp local.example.toml local.toml      # then set zig (and paths, if yours differ)
python3 -m mac build                  # the pinned engine and libtfgrammar, under ~/.cache/nanbeige43-33b-int4
python3 -m mac up                     # http://127.0.0.1:8896/v1, model "nanbeige"
python3 -m mac chat "Write a Python function that merges overlapping intervals."
```

`python3 -m mac down|status|logs|command|validate` stop, inspect and check it. To serve the LAN on login:
`python3 -m mac launchd --port 8888` writes `~/Library/LaunchAgents/com.nanbeige.nanbeige43-33b-int4.plist` (not
loaded), then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nanbeige.nanbeige43-33b-int4.plist`.

## Settings

`recipe.toml` holds the defaults; `local.toml` overrides them for one Mac, and an environment variable of the same
name in upper case overrides both (`PORT=8888 HOST=0.0.0.0 python3 -m mac up`).

| setting | default | |
|---|---|---|
| `context` | 131072 | a stream's window; the release allows 524,288 (`context = 524288`, `parallel = 1`; [memory](#settings)) |
| `parallel` | 2 | streams decoded together, each reply equal to its run alone |
| `max_tokens` | 65536 | a reply's default ceiling; thinking is on by default and counts |
| `temperature`, `top_p`, `top_k` | 1.0, 0.95, off | the model card's agentic settings for requests that set none (it recommends 0.6 for reasoning tasks) |
| `prompt_cache_gib` | 8 | kept prompt states between turns; they give way to streams |
| `load_limit_gb` | 150 | weights, scratch and every stream's caches together (`LLAMA_LOAD_LIMIT_GB`; 0: 75% of RAM) |
| `drafts` | true | drafted windows (replies are identical with them off) |
| `drafter` | true | Nanbeige4.1-3B proposing the next tokens, verified by the 33B ([drafter](#drafter-nanbeige41-3b)); false: the lane engine's copy drafts alone |

**Memory.** The weights take 21.0 GiB, plus 301 MiB of scratch where prompt chunks make each projection's bf16
weights. Every one of the 52 layers keeps bf16 keys and values, 208 KiB a token: 26 GiB a stream at 131,072, 104 GiB
at 524,288. The drafter adds its 7.33 GiB and 64 KiB a token for each stream's cache. Streams and kept prompts share
what `load_limit_gb` leaves; a stream that needs more room evicts kept prompts first. 150 GB holds two 131,072-token
streams with the drafter (about 97 GiB in all). One 524,288-token stream (`parallel = 1`) comes to about 126 GiB without
the drafter and 165 GiB with it (`load_limit_gb = 180`); the full window was not run on this checkpoint.

**Thinking** is on by default, as in the chat template: replies reason in `reasoning_content` before the answer in
`content`. A request sends `"chat_template_kwargs": {"enable_thinking": false}` to turn it off. The server forwards
only the thinking switch to the template: earlier turns' reasoning is left out of the prompt (the template's
`preserve_thinking` default) and tool calls take the template's XML form (its `tool_call_format` default), which the
server parses into `tool_calls`.

## How the engine reads the INT4 release

The checkpoint is compressed-tensors `pack-quantized`: each projection's weights are asymmetric int4 codes in groups
of 32 along the row, with a bf16 scale and a 4-bit zero point a group; `lm_head` and the embedding stay bf16.
compressed-tensors decompresses a weight as `bf16((q - z) * scale)`, and the engine computes exactly that value:

- Decode windows (up to 32 rows) read the packed codes directly (`ll_q4_matvec`: a simdgroup an output row, fp32 sums
  rounded once to bf16).
- Prompt chunks make the projection's bf16 weights in a scratch (`ll_q4_dequant`), then run the bf16 release's
  tensor-unit GEMM over them.

So the model the engine runs is the checkpoint's own bf16 model, with the bf16 release's rounding points everywhere
else. The engine admits exactly this scheme and refuses other quantizations by name. Every new kernel is registered
with TensorFold's kernel contract.

## What works, and how it was checked

All on one M5 Ultra (256 GB, macOS 27), checkpoint revision d834b414, 31 Oct 2026, with an idle 4 GB server of
another model the only other one loaded (`evidence/int4-*` without the drafter, `evidence/int4draft-*` with it). The
plans were run on engine e269c888. The pinned commit 19c80efa is the same work rebased onto newer TensorFold main, with
one change: the serving line names the drafter among the draft sources. On the pinned code the 1,500-position logits
of both releases are byte-identical to the measured builds, drafted replies equal plain ones, the drafted quick plan
passes in full on the same commit one main earlier (`evidence/int4draft-quick-2c819649`), and tool calls and resumed
turns pass on the pinned commit itself (`evidence/int4draft-tools-turns-19c80efa`).

- **Text, greedy and sampled.** Against a float64 pass over the same checkpoint (TensorFold's
  `llama_f64_reference.py`: the weights decompressed as above, float64 throughout, rounded once at the end), 1,500
  teacher-forced positions: mean KL 0.0044, top-1 97.1%. Hugging Face's own code on this checkpoint (transformers
  4.57.1, bf16 on the CPU) is further from it: KL 0.0346, top-1 96.6%. One-row steps and 16-row windows give the same
  logits bit for bit after a 1,500-token prompt. Seeded sampling repeats (9/9).
- **Long context.** A passphrase at 10%, 50% and 90% of a 128,579-token prompt comes back exactly (3/3, greedy,
  thinking off) on this recipe's default server (131,072-token window, drafter on); the cold prompt takes 254 s to the
  first token (506 tokens/s). Cold prompts to 61,501 tokens run at 846-1,257 tokens/s (`evidence/int4-prefill`).
- **Tools and structured output.** The harness's structured check (json_object, json_schema, regex, choice,
  grammar) 18/18 and drafted == plain 12/12; its tool cases (auto, required, named, thinking, streamed) 5/5.
- **Prompt cache.** A turn resumed from a kept state (5,493 of 5,649 tokens) equals the same request after a
  restart, and starts in 0.39 s against 4.6 s cold.
- **Concurrency.** A 60-turn soak (3 chats of 20 turns beside a decoding stream) and 12 concurrent replies each equal
  the request alone; the suite's streaming 4/4, multi-turn recall 8/8, stop 3/3, thinking 10/10 and max_tokens 3/3.
- **Drafts.** The checkpoint has no draft head; Nanbeige4.1-3B drafts for it ([below](#drafter-nanbeige41-3b)).
  Drafted greedy tokens equal plain greedy tokens on ten prompts in the engine's own tool, and the quick plan passes in
  full with the drafter (`evidence/int4draft-quick`: structured drafted == plain 12/12, sampling 9/9 equal to
  undrafted, soak 60/60, concurrent 12/12, the suite's drafts check 12/12).
- **Vision.** Not applicable: Nanbeige4.3-33B is text-only.

## Quality: what the INT4 release costs

Nanbeige trained the INT4 checkpoint with quantization-aware training (on-policy distillation from the bf16 model, then
reinforcement learning on the INT4 model itself, as its model card says); the engine reproduces it closely (above), so
the difference from the bf16 release is the release's own. On the same 1,500 positions:

| | mean KL against HF's bf16 release | top-1 agreement |
|---|---:|---:|
| HF transformers on the INT4 release | 0.082 | 90.8% |
| TensorFold on the INT4 release (this recipe) | 0.087 | 90.8% |
| TensorFold on the bf16 release ([nanbeige43-33b-bf16-m5-ultra](https://github.com/BobClawblaw/nanbeige43-33b-bf16-m5-ultra)) | 0.0015 | 99.1% |

No quality suite (benchmarks of answers) was run on this recipe, so how much the KL above changes answers is not
measured here.

## Measured

Through [recipe-harness](https://github.com/BobClawblaw/recipe-harness) (targets `nanbeige43-33b-int4-m5` and, with
the drafter, `nanbeige43-33b-int4draft-m5`), greedy, 256-token replies, engine e269c888; the raw results are in
`evidence/`.

| | streams | prose | code | structured | list | first token |
|---|---:|---:|---:|---:|---:|---:|
| undrafted | 1 | 42.8 | 42.0 | 42.9 | 41.4 | 0.27-0.30 s |
| undrafted | 2 | 74.6 | 53.4 | 66.6 | 74.7 | 0.41-0.45 s |
| drafted (default) | 1 | 79.4 | 79.6 | 81.0 | 80.5 | 0.27-0.31 s |
| drafted (default) | 2 | 74.4 | 74.6 | 73.7 | 74.8 | 0.41-0.46 s |

Tokens a second, all streams together (`evidence/int4-speed`, `evidence/int4draft-speed`). Undrafted, one stream reads
the 21 GiB of weights each step (23.8 ms), 2.5-2.6 times the bf16 release's 16.3 tokens/s. The ruler's texts repeat
themselves, which the drafter continues (92% of drafts kept); ordinary chats gain little or nothing
([drafter](#drafter-nanbeige41-3b)). With two streams, shared rounds mostly run undrafted.

| cold prompt | 2,109 | 8,254 | 32,829 | 61,501 tokens |
|---|---:|---:|---:|---:|
| tokens a second | 1,185 | 1,257 | 1,041 | 846 |
| time to first token | 1.78 s | 6.57 s | 31.6 s | 72.7 s |

A fresh server, each prompt starting with words no earlier one did, undrafted (`evidence/int4-prefill`; with thinking
on, within 4%). With the drafter loaded the same prompts ran 6-8% slower in this run (thinking on, 4-15%;
`evidence/int4draft-prefill`), though the drafter reads a prompt only after the 33B's first token. Prompt chunks make
each projection's bf16 weights before the bf16 release's GEMM, so prompts run 4-6% slower than the bf16 release's at
2k-8k tokens and 2-3% at 33k-61k. A multi-turn chat's third turn, resumed from 5,493 of its 5,649 tokens, starts in
0.39 s against 4.6 s cold (`evidence/int4-speed`).

In the undrafted quick plan (`evidence/int4-quick`) the ruler after the serving checks ran at 9.9-43.0 tokens/s
while another job's GPU tests ran on the Mac; the speed plan's ruler after the same checks, on an otherwise idle Mac,
is 42.1-43.0 (`evidence/int4-speed`).

## Drafter: Nanbeige4.1-3B

Decode is bound by reading the weights each step. Speculative decoding has a smaller model propose the next few
tokens and the 33B verify them all in one window, which costs about as much as one step; every token the 33B keeps is
one it would have drawn itself, so replies are the same. [Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)
is a plain Llama on the same tokenizer, so the engine serves it as the drafter beside the 33B, from its bf16 release
as published (Nanbeige/Nanbeige4.1-3B @ 9c4555c3, 7.33 GiB; quantized drafters are refused by name). Each stream gets
a drafter cache (64 KiB a token) in the same memory budget as the 33B's, and the depth follows measured acceptance and
round times.

With the INT4 33B the drafter has less to win than with the bf16 release: a 33B step is 24 ms against the drafter's
11 ms, so a draft must land often to pay. Checked in the engine's own tool (`tf-llama-run ... draft`), which decodes
greedily through the drafter's windows, on eight chat prompts (128 tokens, thinking on and off), every drafted reply
equal to the plain one token for token:

| prompt | plain | drafted | tokens a round | drafts kept |
|---|---:|---:|---:|---:|
| Fibonacci function, thinking off | 41.3 tok/s | 44.5 | 1.86 | 59 of 84 |
| Fibonacci function, thinking on | 41.8 | 39.0 | 1.60 | 48 of 94 |
| why the sky is blue, thinking off | 41.4 | 43.4 | 1.78 | 56 of 84 |
| why the sky is blue, thinking on | 41.4 | 38.9 | 1.59 | 48 of 92 |
| Hangzhou in two sentences (Chinese), thinking off | 41.4 | 35.6 | 1.38 | 35 of 99 |
| Hangzhou in two sentences (Chinese), thinking on | 41.1 | 38.6 | 1.62 | 49 of 92 |
| 9.11 or 9.8, thinking off | 39.5 | 41.0 | 1.88 | 60 of 84 |
| 9.11 or 9.8, thinking on | 37.9 | 42.9 | 1.78 | 56 of 85 |

On these chats it about breaks even (40.5 tokens/s drafted against 40.7 plain on average, -14% to +13% a prompt), and
the harness suite's drafted check gives 42 against 39. Text the reply continues predictably, such as the ruler's
repetitive texts, is where it pays: 79-81 tokens/s against 41-43 (replies that copy or edit given text should gain the
same way, but were not measured). The drafter is on by default; `drafter = false` saves its 7.33 GiB and caches and
gives the plain speed everywhere.

After a long prompt the drafter reads the prompt itself before it drafts, which it does only after the 33B's first
token: the first token keeps its time, but the second waits for the drafter. On the 128,579-token passphrase prompts,
9-10-token replies took 33-47 s after their first token; at 110,347 tokens, 30 s. For long documents, `drafter = false`
answers sooner.

## Notes

- No quality suite was run on this recipe; the checks above are exactness against a float64 pass and Hugging Face's
  own code, the harness's serving checks and its speed rulers.
- `logprobs` requests are refused by name: the release's tokenizer decodes through a `Strip` step, which the engine's
  tokenizer does not count as token-local, so the server cannot give a token's bytes on its own.
- The family refuses what it has not been checked on: tied heads, head sizes other than 128, Llama 3's scaled RoPE,
  biased projections, and quantizations other than this checkpoint's.

## License

The recipe is under the [MIT License](LICENSE). Nanbeige4.3-33B-INT4 and Nanbeige4.1-3B are published under Apache 2.0
on Hugging Face ([33B INT4](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-INT4),
[3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)).
