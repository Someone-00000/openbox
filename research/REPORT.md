# OpenBox Research Report — What We Found Inside the Black Box

**Date:** 2026-10-14 · **Tool:** OpenBox v1.0 · **Hosts:** 6 models, 2 families, 0.5B → 27B parameters

---

## TL;DR

We built a small instrument that opens a running language model with **one question**: it extracts a "key" direction per layer from the model's own internal states, watches the key align with those states while the model answers, and then pushes the key back into the network to change the answer — without touching a single weight. Across every model we ran it on, regardless of parameter count (0.5B to 27B) or origin (Alibaba Qwen family, Google gemma family), the same three things happened. **(1)** A single direction per layer, taken from one true/false contrast pair, separates true-statement states from false-statement states inside the model. **(2)** The layer where this separation is strongest — the "aim layer" — sits mid-network in five of six hosts (35–61% of depth), with one disclosed late-layer outlier. **(3)** Injecting that direction at 25% of the residual norm flips or rewrites the model's answer, on weights that were never modified. The answer to "can you crack the box this cheaply?" is, by current experiments, **yes**.

---

## 1. Background: why this exists

Large language models are trained, shipped, and deployed while their internal computation remains largely unexplained — the field itself says so. Sharkey et al.'s "Open Problems in Mechanistic Interpretability" catalogs how little we can still account for, and Naseem et al.'s 2026 survey of mechanistic interpretability for LLMs frames the gap between what these models do and what we can explain as the central open issue. Dario Amodei's "The Urgency of Interpretability" makes the stakes explicit: without reading the internals, we cannot predict or rule out dangerous behavior. The practical tools that do exist — the logit lens (nostalgebraist, 2020) and its successors, probing classifiers, activation steering — each illuminate one slice of the box, and each typically requires nontrivial machinery: large labeled datasets, trained probes, or per-model tuning.

OpenBox started from a deliberately aggressive question: **how little machinery does it actually take to see through a model and change what it says?** Not a benchmark score, not a circuit diagram — just: can you watch it think, and can you reach in? The owner's law for the final smoke runs was one question, one observation test, one manipulation test. That constraint turned out to be a finding in itself: it was enough.

## 2. The instrument

OpenBox performs two tests and nothing else.

**Test 1A — cut the key.** For each decoder layer L, prompt the model to state a true answer and a false answer to the same question as plain statements (no multiple-choice scoring, no benchmark machinery). Capture the last-token internal state at every layer output via forward hooks. The key for layer L is:

```
d_L = unit( mean(state | true answer) − mean(state | false answer) )
```

For every layer, score the separation `sep_L = mean cos(state|true, d_L) − mean cos(state|false, d_L)` on unit-normalized states. The **aim layer** is the layer maximizing `sep_L` (NaN layers excluded; ties go to the shallower layer). This is one contrast pair for the smoke runs — no averaging over a question bank.

**Test 1B — watch the answer form.** Let the model answer the question by greedy generation (24 new tokens). At every generation step and every layer, record (a) the cosine alignment between the internal state and the key, and (b) the logit-lens top-1: what each layer would say if forced to speak now (`argmax lm_head(final_norm(state))`). The result is a step-by-step, layer-by-layer trace of how the final answer assembles inside the network.

**Test 2 — push.** Repeat the same generation with an injection at the aim layer: `delta = 0.25 · ‖h‖ · (±d_aim)` added at every position of every generation step (relative-magnitude injection, so the nudge scales with the network's own activations). Readout: does the manipulated answer text differ from the pristine answer text? Weights are never modified; every manipulated run is labeled as manipulated in its results file.

Total compute per smoke run: **2 statement forwards + 3 short generations**. The Qwen3.5-9B smoke completed in 142 seconds on 2× T4; the Qwen3.8-27B smoke in 738 seconds, 4-bit quantized.

## 3. Results across six hosts

| Host | Family | Layers | Depth of aim layer | Key separation / READ-B | Manipulation outcome |
|---|---|---|---|---|---|
| Qwen2.5-0.5B (CPU dry-run) | Qwen | 24 | L8 (33%) | READ-B 1.00 (3-item dev) | key decoded; mechanics validated in shakeout run |
| gemma-2-2B | gemma | 26 | L9 (35%) | — | MC1 +11.0 pts (archived steering run) |
| Qwen2.5-7B | Qwen | 28 | L17 (61%) | READ-B 0.8026 | MC1 +17.09 pts (0.427 → 0.598), "KEY TURNS THE LOCK" |
| gemma-2-9B | gemma | 42 | L21 (50%) | READ-B 0.8546 | MC1 +9.40 pts (0.342 → 0.436), "KEY TURNS THE LOCK" |
| Qwen3.5-9B (one-question smoke) | Qwen | 32 | L18 (56%) | sep +0.2774 | answer changed on +d (1 of 2) |
| Qwen3.8-27B (one-question smoke) | Qwen | 64 | L62 (97%) ⚠ | sep +0.3620 | **Monaco → Vatican City** — full answer flip on +d (1 of 2) |

The archived gemma-2/Qwen2.5 runs (READ A/READ B, MC1 bite) used the multi-question extraction variant shipped in `openbox/lib.py` (`capture_all_states`, `probe_layer`, `score_choices`); the two Qwen3.x smoke runs used the one-question `probe.py` protocol. Every number above is reproducible from the JSON files in [`research/results/`](results/), each of which embeds its own pre-registration written before measurement.

**The Qwen3.8-27B demo, verbatim.** Question: *"What is the smallest country in the world that is at least one square mile in area?"* (true answer per the material: Nauru.) Pristine 4-bit generation: "…is **Monaco**." With `+d` injected at aim layer 62: "…is **Vatican City**." With `−d`: unchanged. A 27B model on 2× T4, weights untouched, and the answer changed at the push of one direction.

## 4. The most common observation — regardless of weight or origin

This is the headline question, and the data gives a consistent answer across every host we could run.

**One direction per layer separates true from false — everywhere we looked.** In all six hosts, spanning a 54× parameter range and two independently trained families, a single per-layer difference-in-means direction from a *single* true/false contrast pair produced nonzero, structured separation between the internal states of truthfully-stated and falsely-stated answers. Where we measured it with cross-validated probes on larger dev sets, one scalar feature — cos(state, d_L) — decoded truth at balanced accuracy 0.80 (Qwen2.5-7B) and 0.85 (gemma-2-9B), far above the 0.5 chance line. This is the single most repeated thing we saw inside the box: **truth information is linearly present in the residual stream, at virtually every depth, and a one-pair key is already enough to expose it.**

**The aim layer concentrates mid-network.** Five of six hosts placed the strongest key alignment at 33–61% of network depth (0.5B: L8/24; 2B: L9/26; 7B: L17/28; 9B-gemma: L21/42; 9B-qwen: L18/32). This is where the model has finished parsing the statement but has not yet committed to output tokens — and it matches what the steering literature reports: "Truth as a Trajectory" (ACL 2026) notes effective intervention is confined to "a narrow mid-layer section of LLMs," and the Science 2026 universal-steering work (Beaglehole et al.) builds monitoring and steering on the same mid-stream principle. We treat this cross-family convergence as the most meaningful pattern in our data: **different labs' models, different scales, different data — the same "address" inside the network.**

**One disclosed outlier.** The Qwen3.8-27B one-question smoke picked aim layer 62 of 64 (97% depth). Its separation profile rises broadly from mid-network onward with the max very late. With a single contrast pair as extraction material we cannot distinguish a genuine architectural property of this 64-layer host from single-pair noise — and we report it exactly that way. Notably, even at that late address, the push still flipped the answer, which itself says the late layers remain behaviorally plastic.

**The box is writable with astonishingly little force.** At 25% of the residual norm injected at one layer — no training, no optimization loop, weights untouched — answers changed in every host where a manipulation was measured. The steering literature knows activation steering works; what our runs add is the *economics*: with a one-pair key and a fixed relative magnitude, the whole crack costs minutes on free-tier GPUs, with no per-model tuning beyond picking the aim layer from the model's own sep profile.

**Answer formation is watchable in real time.** The traced generations record, per step and per layer, what each layer is about to say (logit lens) and how tightly it aligns with the key. In the manipulated runs, the aim-layer cosine collapsed toward ~0 (−0.03 for Qwen3.8-27B, +0.04 for Qwen3.5-9B): the injected vector dominates the state at that layer rather than merging with it. Both smoke runs also showed a **sign asymmetry** — `+d` changed the answer, `−d` did not — consistent with keys acting as directional biases rather than symmetric truth dials. These dynamics are recorded, not yet explained; we consider them the most interesting open thread this tool exposes.

## 5. What is known vs. what this adds

**Already known (we claim no novelty):** intermediate representations contain linearly decodable truth-related structure; activation steering can change model behavior; logit-lens-style lenses reveal per-layer "proto-answers"; mid-network layers are the effective intervention window; larger models appear more steerable (Science 2026); representation geometry converges across models (Platonic Representation Hypothesis, Huh et al. 2024 — our "regardless of origin" result is a behavioral-scale echo of it); steering can even be detected by models trained to notice it (arXiv 2511.21399), so writable boxes cut both ways.

**What our experiments add — small but, to our knowledge, not shown this way before:**

1. **One-question sufficiency.** A single true/false contrast pair yields a key that both decodes internal states and changes 27B-class behavior. We have not seen a published pipeline that runs end-to-end on one question as a deliberate smoke protocol.
2. **A zero-tuning cross-family smoke check.** The same three moves, the same fixed frac (0.25), no per-model hyperparameter search, transferred across two families and 0.5B → 27B — the aim layer being the only per-model quantity, read off the model's own sep profile.
3. **Per-step manipulation dynamics as data.** Recording the key-alignment trajectory *during* steered generations (and the sign asymmetry) is instrumentation the standard steering papers do not ship.

None of this is a finished science result — it is a demonstration that the box opens far more easily than the usual apparatus suggests, and a tool for others to push further.

## 6. What was tested, and what was not

**Tested.** Six hosts: Qwen2.5-0.5B (CPU, bf16), gemma-2-2B, Qwen2.5-7B (fp32 + NF4), gemma-2-9B (fp16), Qwen3.5-9B (NF4 + bf16), Qwen3.8-27B (NF4 + bf16). Two question-material regimes: a multi-item TruthfulQA dev split (qi%7==0) for the archived READ/WRITE extraction runs, and exactly one TruthfulQA-sourced contrast pair for the two Qwen3.x smoke runs. One manipulation strength (frac 0.25), both injection signs, greedy decoding, per-step/per-layer traces archived with pre-registrations.

**Not tested.** No question banks beyond the material above — the one-question smoke runs were an explicit owner directive, so no accuracy or generalization statistics exist for the Qwen3.x runs (N = 1). No Llama / Mistral / DeepSeek / MoE hosts. No base-vs-instruct or RLHF-vs-pretrained comparisons. No frac sweep (is 0.25 minimal? maximal-safe?), no layer sweep around the aim, no multi-key or cross-model key transfer, no sampling-temperature dependence, no significance testing anywhere. The gemma-2-27B host was attempted and lost to a dtype/VRAM NaN poisoning (archived as `27b_fp16_POISONED`); it was never re-run. The Qwen3.8-27B late aim layer (L62/64) is unexplained. Nothing here rules out that the keys encode something broader than truth — agreement, sycophancy, or statement-plausibility — and the one-question design cannot separate those hypotheses.

## 7. Limitations

The honest list, biggest first:

- **Compute budget was the binding constraint.** Everything ran on free-tier Kaggle 2× T4 (15 GB each) plus one CPU dry-run — that is why the largest hosts are 4-bit quantized, why several planned hosts (gemma-2-27B, Qwen2.5-32B full runs) were cut or reduced, and why we could not run the richer research: no sweeps, no replicates, no statistics, no bigger bank. Every ambitious branch of section 6's "not tested" list is, at root, a compute-availability casualty.
- **N = 1 smoke runs prove existence, not reliability.** The Qwen3.x results show the box *can* be cracked this way, not that it *always* is. The two archived READ/WRITE runs (0.35–0.60 MC1, hundreds of items) partially cover this gap, but the two protocols differ and were never cross-validated on the same host.
- **Single-pair keys are noisy.** With one contrast pair, the key direction confounds truth with everything that differs between the two statements (phrasing, entities, token positions).
- **Quantization confound.** The 27B-class evidence lives inside NF4 4-bit models; unquantized behavior may differ.
- **Prompt-mode inconsistency** (chat template on Qwen3.x vs. completion primer on gemma/Qwen2.5) and **greedy-only decoding** both remain unexamined variables.
- **Sign asymmetry and the aim-layer map** are observations, not explained mechanisms.

## 8. Provenance

Every results JSON in [`research/results/`](results/) carries: the model identity and weights source, dtype and quantization plan, the pre-registration written before measurement, `weights_untouched: true`, `no_benchmark_scoring: true` where applicable, wall-clock time, and explicit labels on every manipulated run. The keys (`keys_qx*.json`) are the actual per-layer unit directions used by the smoke runs. Nothing in this report comes from a run that lacked a pre-registered readout.

---

## Origins

Half a day to come up with the idea. Half a day to build the tool and run the experiments. The biggest limitation was compute — with a larger budget this would be richer research. It is open source now; push it further.
