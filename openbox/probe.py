#!/usr/bin/env python3
"""
OpenBox — crack the black box with ONE question. Two tests, nothing else.

  TEST 1 OBSERVATION — see through the model while it answers:
       A) the key: one direction per layer, extracted from the internal
          states of the model stating a TRUE vs a FALSE answer (plain
          prompted statements; no multiple-choice machinery, no benchmark
          scoring).
       B) the model answers a question by free generation; during
          generation we record, at EVERY step and EVERY layer: the
          alignment of the internal state with the key (cos) and the
          logit-lens top-1 (what each layer is about to say) — how the
          answer forms inside the box.

  TEST 2 MANIPULATION — reach in and change the answer:
       the same generation, but the key is injected at the aim layer while
       the model is answering (both signs +d / -d) -> did the answer change?

Weights are never modified; every manipulated run is labeled in the output.

usage:
  python -m openbox.probe --model <hf-id-or-path> [--tag <size-tag>]
                          [--data <dir>] [--out <dir>] [options]

examples:
  # the exact one-question smoke (any HF model):
  python -m openbox.probe --model Qwen/Qwen2.5-0.5B --data data --out out

  # reproduce the archived gemma-2-9B extraction config (2xT4):
  python -m openbox.probe --model unsloth/gemma-2-9b --tag g9b --data data

  # CPU dry-run with a tiny local model:
  python -m openbox.probe --model /path/to/model --local --n-questions 1 --max-new 8
"""
import argparse, datetime, os, sys, time, traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    p = argparse.ArgumentParser(
        prog="openbox.probe",
        description="OpenBox: observe and steer an LLM's internals with one question.")
    p.add_argument("--model", required=True,
                   help="HF repo id or local dir with config.json + safetensors; "
                        "'kaggle' = auto-discover a mounted model under /kaggle/input")
    p.add_argument("--tag", default=None,
                   help="size tag for the proven dtype plans "
                        "(g2b|g9b|27b|q1p5b|q7b|q32b|qx35|qx38); default: generic bf16 full")
    p.add_argument("--data", default=None,
                   help="dir with truthfulqa_mc.json (question MATERIAL) or "
                        "question_material.json ({q,true,false} list); "
                        "default: ./data, then /kaggle/input auto-discovery")
    p.add_argument("--out", default="openbox_out", help="output dir (default ./openbox_out)")
    p.add_argument("--n-questions", type=int, default=1,
                   help="question MATERIAL items to use (default 1 — the original "
                        "owner law: one question is enough for a smoke check)")
    p.add_argument("--max-new", type=int, default=24, help="max new tokens per generation")
    p.add_argument("--frac", type=float, default=0.25,
                   help="manipulation strength: delta = frac * ||h|| * (±d_aim)")
    p.add_argument("--deadline-h", type=float, default=1.5, help="wall-clock deadline (hours)")
    p.add_argument("--local", action="store_true",
                   help="CPU dry-run: full-precision bf16, no quantization")
    p.add_argument("--smoke", action="store_true", help="alias of --n-questions 1 --max-new 8")
    return p.parse_args()


ARGS = parse_args()
SIZE = ARGS.tag or "model"
LOCAL = ARGS.local
SMOKE = ARGS.smoke
N_QUESTIONS = 1 if SMOKE else max(1, ARGS.n_questions)
MAX_NEW = 8 if SMOKE else max(4, ARGS.max_new)
FRAC = ARGS.frac
DEADLINE_H = ARGS.deadline_h
OUT = ARGS.out

if LOCAL:
    MODEL_PATH = ARGS.model
    DATA = ARGS.data or "data"
else:
    MODEL_PATH = None if ARGS.model == "kaggle" else ARGS.model

    def _find_file(name, root):
        for dp, _dn, fn in os.walk(root):
            if name in fn:
                return os.path.join(dp, name)
        return None

    DATA = ARGS.data or _find_file("truthfulqa_mc.json", ".") and \
        os.path.dirname(_find_file("truthfulqa_mc.json", ".")) or "data"
    if os.path.isdir("/kaggle/input"):
        hit = _find_file("truthfulqa_mc.json", "/kaggle/input")
        if hit:
            DATA = os.path.dirname(hit)
        if MODEL_PATH is None:
            for dp, _dn, fn in os.walk("/kaggle/input"):
                if "config.json" in fn and any(f.endswith(".safetensors") for f in fn):
                    MODEL_PATH = dp
                    break

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib as L  # noqa: E402

import torch  # noqa: E402

R = {"meta": {}, "complete": False}
T00 = time.time()
WDIR = OUT


def dump():
    L.save_json(os.path.join(WDIR, f"box_{SIZE}.json"), R)


def fail(err):
    R["fatal"] = err
    R["wall_seconds"] = round(time.time() - T00, 1)
    dump()
    L.log("[FATAL]", err.splitlines()[-1] if err else "")
    sys.exit(1)


def main():
    mdir = MODEL_PATH or L.download_model(SIZE)
    L.ensure_deps(mdir)
    if not LOCAL and SIZE in L.QUANT_SIZES:
        L.ensure_quant_deps()
    model, tok = L.load_model(SIZE, mdir, local=LOCAL)
    layers, final_norm = L.get_text_stack(model)
    n_layers = len(layers)
    hidden = model.config.get_text_config().hidden_size \
        if hasattr(model.config, "get_text_config") else model.config.hidden_size
    lm_head_w = model.get_output_embeddings().weight
    prompt_mode = "chat" if getattr(tok, "chat_template", None) else "completion"
    L.log(f"[boot] OPEN BOX size={SIZE} layers={n_layers} hidden={hidden} "
          f"mode={prompt_mode} dtype_plan={L.DTYPE_PLAN.get(SIZE, 'bfloat16')}")

    # ------------ question bank (material, not a benchmark) -----------------
    tq = os.path.join(DATA, "truthfulqa_mc.json")
    qm = os.path.join(DATA, "question_material.json")
    dev = []
    if os.path.exists(tq):
        rows = L.load_json(tq)["rows"]
        for qi in range(0, len(rows), 7):
            mc = rows[qi]["mc1_targets"]
            ti = mc["labels"].index(1)
            fi = next(i for i, l in enumerate(mc["labels"]) if l == 0)
            dev.append({"q": rows[qi]["question"],
                        "true": str(mc["choices"][ti]),
                        "false": str(mc["choices"][fi])})
    elif os.path.exists(qm):
        for r in L.load_json(qm)["questions"]:
            dev.append({"q": r["q"], "true": r["true"], "false": r["false"]})
    else:
        fail(f"question material not found under {DATA} "
             f"(need truthfulqa_mc.json or question_material.json)")
    dev = dev[:N_QUESTIONS]
    N = len(dev)

    # ------------ pre-registration (before any measurement) -----------------
    R["meta"] = {
        "protocol": "OPEN BOX v2 (two tests: observation + manipulation)",
        "tool": "OpenBox",
        "size": SIZE,
        "model": ARGS.model if ARGS.model != "kaggle" else "kaggle-mounted",
        "weights_source": ("local dry-run model" if LOCAL else
                           ("kaggle-mounted model" if ARGS.model == "kaggle"
                            else ARGS.model)),
        "dtype_storage": L.DTYPE_PLAN.get(SIZE, "bfloat16"),
        "quant": SIZE in L.QUANT_SIZES and not LOCAL,
        "quant_compute": L.QUANT_COMPUTE.get(SIZE),
        "layers": n_layers, "hidden": hidden,
        "prompt_mode": prompt_mode,
        "weights_untouched": True,
        "no_benchmark_scoring": True,
        "utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    R["prereg"] = {
        "test1_observation": {
            "key_extraction": "per layer L: d_L = unit(mean(state|true answer) "
                              "- mean(state|false answer)); states = last token "
                              "of the model stating each answer; one contrast "
                              "pair per question; NO multiple-choice scoring",
            "aim_rule": "layer maximizing sep_L = mean cos(state|true, d_L) - "
                        "mean cos(state|false, d_L); NaN layers excluded; "
                        "ties -> shallower",
            "generation": f"greedy, max_new_tokens={MAX_NEW}, one question at "
                          "a time; at every step and layer: state captured via "
                          "forward hooks; recorded = cos(state, d_L) and "
                          "logit-lens top-1 (final norm -> lm_head)",
            "lens": "per step, all layers: top-1 token id via lm_head(final_norm"
                    "(state)); stored as id matrix [steps x layers]",
        },
        "test2_manipulation": {
            "injection": f"delta = {FRAC} * ||h|| * (+/-)d_aim at every "
                         "position of every generation step (relative "
                         "magnitude injection)",
            "signs": ["+d", "-d"],
            "readout": "manipulated answer text vs pristine answer text; "
                       "changed = normalized strings differ",
        },
        "question_bank": "question MATERIAL only (true/false answer strings "
                         "for key extraction); no benchmark metric is computed",
        "n_questions": N, "smoke": bool(SMOKE),
        "deadline_h": DEADLINE_H,
    }
    dump()
    L.log(f"[prereg] N={N} max_new={MAX_NEW} frac={FRAC} mode={prompt_mode}")

    # ================= TEST 1A: key extraction ==============================
    L.log(f"[test1a] extracting key from {N} true/false contrast pairs")
    st_t = {Li: [] for Li in range(n_layers)}
    st_f = {Li: [] for Li in range(n_layers)}

    def capture_last(prompts):
        """forward 1-2 prompts (padded), return per-layer last-token states"""
        enc = [tok(p).input_ids for p in prompts]
        B = len(enc)
        Lm = max(len(e) for e in enc)
        ids = torch.full((B, Lm), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((B, Lm), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e)
            am[i, :len(e)] = 1
        ids, am = ids.to(model.device), am.to(model.device)
        li = torch.tensor([len(e) - 1 for e in enc], device=ids.device)
        seen = {}

        def mk(Li):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                sel = torch.arange(hs.shape[0], device=hs.device)
                lloc = li[:hs.shape[0]].to(hs.device)
                seen[Li] = hs[sel, lloc].float().cpu()
            return hook
        hs_ = [layers[Li].register_forward_hook(mk(Li)) for Li in range(n_layers)]
        try:
            model(input_ids=ids, attention_mask=am)
        finally:
            for h in hs_:
                h.remove()
        return [seen[Li] for Li in range(n_layers)]

    def build_prompt(q, ans=None):
        if prompt_mode == "chat":
            msgs = [{"role": "user", "content": q}]
            if ans is not None:
                msgs.append({"role": "assistant", "content": ans})
            try:
                return tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=(ans is None),
                                               enable_thinking=False)
            except TypeError:
                return tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=(ans is None))
        return L.mc_prompt(q) if ans is None else f"{L.QA_PRIMER}\n\nQ: {q}\nA: {ans}"

    for di, item in enumerate(dev):
        for store, ans in ((st_t, item["true"]), (st_f, item["false"])):
            per_layer = capture_last([build_prompt(item["q"], ans)])
            for Li in range(n_layers):
                store[Li].append(per_layer[Li][0])
        if (di + 1) % 10 == 0 or di == N - 1:
            L.log(f"  key extract {di+1}/{N} ({time.time()-T00:.0f}s)")

    key = {}
    prof = {}
    for Li in range(n_layers):
        T_ = torch.stack(st_t[Li])
        F_ = torch.stack(st_f[Li])
        d = T_.mean(0) - F_.mean(0)
        dn = d.norm().item()
        u = d / d.norm() if dn == dn and dn > 0 else torch.zeros(hidden)
        # unit-normalize states before cos (else values are raw projections,
        # unbounded and cross-layer incomparable)
        Tn = T_ / T_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Fn = F_ / F_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        ct = (Tn @ u).mean().item()
        cf = (Fn @ u).mean().item()
        ms = ((T_.norm(dim=1).mean() + F_.norm(dim=1).mean()) / 2).item()
        prof[str(Li)] = {"dir_norm_pre": round(dn, 4),
                         "mean_state_norm": round(ms, 2),
                         "cos_true": round(ct, 4), "cos_false": round(cf, 4),
                         "sep": round(ct - cf, 4)}
        key[str(Li)] = [round(x, 6) for x in u.tolist()]
    L.save_json(os.path.join(WDIR, f"keys_{SIZE}.json"),
                {"meta": R["meta"], "aim_rule": R["prereg"]["test1_observation"]["aim_rule"],
                 "keys": key})

    nan_layers = [k for k, v in prof.items() if v["sep"] != v["sep"]]
    cand = {k: v["sep"] for k, v in prof.items() if v["sep"] == v["sep"]}
    if not cand:
        fail("all layers NaN during key extraction - aborting (dtype issue)")
    aim = sorted(cand, key=lambda k: (-cand[k], int(k)))[0]
    AIM = int(aim)
    L.log(f"  aim = L{AIM} (sep {cand[aim]:+.4f}); NaN layers: "
          f"{len(nan_layers)}{'' if not nan_layers else ' ' + str(nan_layers[:8])}")

    R["test1_observation"] = {
        "key_extraction": {"per_layer": prof, "aim": AIM,
                           "aim_sep": round(cand[aim], 4),
                           "nan_layers_excluded": len(nan_layers),
                           "n_pairs": N}}
    dump()

    # ------------ generation machinery (Test 1B + Test 2) -------------------
    lens_head = lm_head_w

    def trace_generate(prompt, steer=None):
        """greedy generation with full internal tracing. steer=None or
        (unit_tensor, sign). Returns dict with text, cos matrix, lens ids."""
        layers_, _ = L.get_text_stack(model)
        step = [-1]
        seen = {}
        sh = None

        def mk(Li, first=False):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                if first:
                    step[0] += 1
                s = step[0]
                seen.setdefault(s, {})[Li] = hs[:, -1, :].float().cpu()[0]
            return hook

        hooks = [layers_[0].register_forward_hook(mk(0, first=True))]
        hooks += [layers_[Li].register_forward_hook(mk(Li))
                  for Li in range(1, n_layers)]
        if steer is not None:
            u, sign = steer
            sh = L.relative_steer_hook(model, AIM, u * sign, FRAC)
        try:
            enc = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=MAX_NEW,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
        finally:
            for h in hooks:
                h.remove()
            if sh is not None:
                sh.remove()
        steps = sorted(seen.keys())
        cos = [[None] * n_layers for _ in steps]
        nan = False
        for si, s in enumerate(steps):
            for Li in range(n_layers):
                v = seen[s][Li]
                u = key_t[Li]
                vn = v.norm()
                c = float((v / vn.clamp_min(1e-8)) @ u) if vn > 0 else 0.0
                if c != c:
                    nan = True
                    c = None
                cos[si][Li] = None if c is None else round(c, 4)
        # logit lens at every step, all layers: batched GEMM per step
        lens = [[None] * n_layers for _ in steps]
        for si, s in enumerate(steps):
            stack = torch.stack([seen[s][Li] for Li in range(n_layers)]).to(
                lens_head.device, lens_head.dtype)
            hn = final_norm(stack)
            logits = torch.nn.functional.linear(hn, lens_head)
            top = logits.argmax(dim=-1).tolist()
            lens[si] = top
        first_tok = tok.decode(out[0, enc["input_ids"].shape[1]:
                                    enc["input_ids"].shape[1] + 1]).strip()
        return {"text": text.strip(), "first_tok": first_tok,
                "n_steps": len(steps), "cos": cos, "lens_top1": lens,
                "nan": nan}

    key_t = [torch.tensor(key[str(Li)]) for Li in range(n_layers)]

    # ================= TEST 1B: observation =================================
    L.log(f"[test1b] observing {N} generations (pristine)")
    gens = []
    for di, item in enumerate(dev):
        p = build_prompt(item["q"])
        tr = trace_generate(p)
        gens.append({"qi": di, "question": item["q"],
                     "true_answer": item["true"], "false_answer": item["false"],
                     "pristine": tr})
        L.log(f"  obs {di+1}/{N}: \"{tr['text'][:60]}\" "
              f"({tr['n_steps']} steps, nan={tr['nan']})")
        if (di + 1) % 5 == 0 or di == N - 1:
            R["test1_observation"]["generations"] = gens
            dump()
        if time.time() - T00 > DEADLINE_H * 3600:
            R["test1_observation"]["generations"] = gens
            R["test1_observation"]["note"] = "deadline hit mid-observation"
            dump()
            L.log("[halt] deadline during observation")
            return
    R["test1_observation"]["generations"] = gens
    dump()

    # ================= TEST 2: manipulation =================================
    L.log(f"[test2] manipulating {N} generations (+d and -d at L{AIM}, f{FRAC})")
    mruns = []
    for sign, sname in ((+1.0, "+d"), (-1.0, "-d")):
        for di, item in enumerate(dev):
            p = build_prompt(item["q"])
            tr = trace_generate(p, steer=(key_t[AIM], sign))
            pristine_text = gens[di]["pristine"]["text"]
            norm = lambda s: " ".join(s.lower().split())
            mruns.append({"qi": di, "sign": sname, "text": tr["text"],
                          "changed": norm(tr["text"]) != norm(pristine_text),
                          "cos_aim_mean": (None if tr["nan"] else
                                           round(sum(r[AIM] for r in tr["cos"]
                                                     if r[AIM] is not None)
                                                 / max(len(tr["cos"]), 1), 4)),
                          "nan": tr["nan"]})
            if (di + 1) % 10 == 0 or di == N - 1:
                L.log(f"  {sname} {di+1}/{N}: "
                      f"\"{tr['text'][:50]}\" changed={mruns[-1]['changed']}")
        if time.time() - T00 > DEADLINE_H * 3600:
            break
    R["test2_manipulation"] = {"aim": AIM, "frac": FRAC, "runs": mruns,
                               "n_changed_plus": sum(1 for m in mruns
                                                     if m["sign"] == "+d" and m["changed"]),
                               "n_changed_minus": sum(1 for m in mruns
                                                      if m["sign"] == "-d" and m["changed"])}
    n_ch = R["test2_manipulation"]["n_changed_plus"] + \
        R["test2_manipulation"]["n_changed_minus"]
    R["test2_manipulation"]["verdict"] = (
        f"BOX OPEN, KEY TURNS THE LOCK: answers changed on {n_ch} of {len(mruns)} "
        f"manipulations (+d: {R['test2_manipulation']['n_changed_plus']}, "
        f"-d: {R['test2_manipulation']['n_changed_minus']})"
        if n_ch else "no behavioral change at this strength (honest null)")
    dump()

    R["verdicts"] = {
        "test1": (f"BOX OPEN: key aligns with internal states at aim L{AIM} "
                  f"(sep {cand[aim]:+.4f}); answer formation traced across "
                  f"{n_layers} layers x {MAX_NEW} steps with per-step lens"),
        "test2": R["test2_manipulation"]["verdict"],
    }
    R["complete"] = True
    R["wall_seconds"] = round(time.time() - T00, 1)
    dump()
    L.log(f"[done] box_{SIZE}.json wall={R['wall_seconds']}s")
    L.log("[verdict T1]", R["verdicts"]["test1"])
    L.log("[verdict T2]", R["verdicts"]["test2"])


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        fail(traceback.format_exc())
