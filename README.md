# OpenBox

> **A project in the attempt to crack the black box — and by current experiments, the answer is yes.**

OpenBox opens a running language model with **one question**. It watches the model think — every layer, every token — and then reaches inside and **changes the answer**, without touching a single weight.

One question. Five forwards. Minutes on 2× T4. Works on 0.5B models and 27B models. Worked on every model it was pointed at.

---

## The demo

Qwen3.8-27B, 64 layers, 4-bit on two free Kaggle T4s, answering a single question:

```
pristine:   "The smallest country in the world that is at least one
             square mile in area is **Monaco**."

+key @ L62: "The smallest country in the world that is at least one
             square mile in area is **Vatican City**."
```

Weights untouched. No fine-tuning. One direction, injected at one layer, at 25% of the residual norm. That is the whole trick — and it worked on **every model we tried**, from 0.5B to 27B, across two different model families.

## What it does

**Two tests. Nothing else.**

| | Test | What happens |
|---|---|---|
| 1A | **Cut the key** | Prompt the model to state a true answer and a false answer. Capture its internal state at every layer. `d_L = unit(mean(state\|true) − mean(state\|false))` per layer. The layer with the strongest true/false separation is the **aim layer**. |
| 1B | **Watch it think** | Let the model answer by free generation. At *every step and every layer*, record how aligned the internal state is with the key (cosine) and what each layer is about to say (logit lens). You literally watch the answer assemble inside the network. |
| 2 | **Turn the key** | Re-run the same generation with `delta = 0.25·‖h‖·(±d_aim)` injected at the aim layer. Did the answer change? Weights never modified; manipulated runs are always labeled. |

## How to run it

```bash
pip install -r requirements.txt

# one-question smoke on any HF model (CPU works for small models):
python -m openbox.probe --model Qwen/Qwen2.5-0.5B --data data --out out

# reproduce the archived gemma-2-9B config on 2xT4:
python -m openbox.probe --model unsloth/gemma-2-9b --tag g9b --data data

# tiny CPU dry-run:
python -m openbox.probe --model /path/to/model --local --smoke
```

Outputs land in `--out`: `box_<tag>.json` (trace + verdicts), `keys_<tag>.json` (the actual per-layer key directions). Each run embeds its own pre-registration, written before measurement. Question material: TruthfulQA-sourced, used only as true/false strings for key extraction — **no benchmark metric is computed**; `data/question_material.json` is the exact one-question material of our smoke runs.

## Results — 6 models, 2 families, 54× parameter range

| Host | Layers | Aim layer (depth) | Key separation | Manipulation |
|---|---|---|---|---|
| Qwen2.5-0.5B (CPU dry-run) | 24 | L8 (33%) | READ-B 1.00 (3-item dev) | key decoded; mechanics validated |
| gemma-2-2B | 26 | L9 (35%) | — | MC1 +11.0 pts (archived) |
| Qwen2.5-7B | 28 | L17 (61%) | READ-B 0.8026 | MC1 +17.09 pts — "KEY TURNS THE LOCK" |
| gemma-2-9B | 42 | L21 (50%) | READ-B 0.8546 | MC1 +9.40 pts — "KEY TURNS THE LOCK" |
| Qwen3.5-9B (1-question smoke) | 32 | L18 (56%) | sep +0.2774 | answer changed (+d) |
| Qwen3.8-27B (1-question smoke) | 64 | L62 (97%) ⚠ | sep +0.3620 | **full answer flip** Monaco → Vatican City (+d) |

⚠ one-question smoke on the 27B picked a late layer — disclosed as an outlier (single-pair noise vs. genuine property: unresolved).

**The one most common thing inside the box, regardless of weight or origin:** a single per-layer direction — from even *one* true/false contrast pair — separates truthful from false statement states at every depth, concentrates its strongest hold mid-network (33–61% depth in 5 of 6 hosts), and doubles as a write-command: inject it, and the answer changes. Different labs, different scales, different data — the same address inside the network, and the same lock that opens it.

Full analysis, known-vs-new positioning, and honest caveats: [`research/REPORT.md`](research/REPORT.md). Raw run outputs: [`research/results/`](research/results/).

## What was tested / what was not

**Tested:** six hosts (Qwen2.5-0.5B/7B, gemma-2-2B/9B, Qwen3.5-9B, Qwen3.8-27B), two extraction regimes (multi-item dev split for the archived runs; exactly one question for the Qwen3.x smokes), one fixed manipulation strength (0.25), both injection signs, greedy decoding, full per-step/per-layer traces.

**Not tested:** bigger question banks (the one-question smoke was a deliberate design law, so no accuracy statistics exist for the Qwen3.x runs), other families (Llama/Mistral/DeepSeek/MoE), base-vs-tuned comparisons, frac/layer sweeps, cross-model key transfer, significance testing. The gemma-2-27B host was attempted and archived as NaN-poisoned (dtype lesson preserved in the code comments). The keys might encode truth — or something broader (agreement, plausibility); the one-question design cannot separate those hypotheses yet.

## Repo layout

```
openbox/
├── openbox/
│   ├── probe.py          # the two-test protocol (CLI)
│   └── lib.py            # model loading, hooks, steering, archived READ/MC1 machinery
├── data/
│   └── question_material.json   # the exact one-question material of the smoke runs
├── research/
│   ├── REPORT.md         # full research write-up (what we found, known vs. new)
│   └── results/          # raw run JSONs incl. pre-registrations + actual keys
├── requirements.txt
└── LICENSE               # MIT
```

## Responsible use

This tool changes model outputs at inference time. It is published to **study** how LLM internals encode and act on truth-related information — interpretability research, red-teaming awareness, education. Do not use it to silently manipulate model behavior in products or against people. Every manipulated run OpenBox produces is labeled as manipulated in its output; keep it that way.

## Origins & limitations

This project took **half a day to come up with the idea, and half a day to build it and run the experiments** — idea, tool, six model runs and this write-up, all in about a day. The **biggest limitation was compute**: everything ran on free-tier Kaggle 2× T4 (plus one CPU dry-run), which forced 4-bit quantization on the big hosts and cut every branch of richer research — no sweeps, no replicates, no statistics, no more model families, no bigger question banks. With a real compute budget, this would be a much richer study. That budget is the only thing that was missing — the box itself opened just fine.

## License

MIT — see [LICENSE](LICENSE).
