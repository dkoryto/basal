# Weight interpolation (`basal-interpolate`)

`basal-interpolate` mixes two checkpoints that share a base model, tensor by tensor
([#42](https://github.com/rkinas/basal/issues/42); WiSE-FT, model soups):

    θ(α) = (1 − α) · θ_old + α · θ_new

α is the weight of the new checkpoint: α = 0 gives the old weights, α = 1 the new ones.
- Each tensor is mixed in fp32 and stored in the new checkpoint's dtype.
- Integer tensors are taken from the new checkpoint.
- The output keeps the new checkpoint's shard layout and holds one output shard in memory at a time. Peak memory is
  the largest shard plus one tensor in fp32.
- The merge is written to a temporary folder and renamed only when every shard is written, so a failure leaves no
  partial checkpoint. `--out` must not exist or must be empty.
- The result is deterministic: the same inputs give byte-identical safetensors.

    basal-interpolate Remek/basal-1.5-mini Remek/basal-2.0-mini --alpha 0.25 0.5 0.75 --out merged/mini \
        --meta-from old
    basal-calibrate panel.jsonl --model merged/mini-a0.5 --save-rows rows/mini-a0.5.jsonl --out calib/mini-a0.5.json

The tool refuses a pair before writing anything unless the two checkpoints have:
- the same tensor names, shapes and dtypes;
- unquantized weights, with no `quantization_config`. FP8 and NVFP4 exports store encoded weights and scales, which
  do not interpolate, so merge the unquantized checkpoints and quantize the result.
- the same architecture (`config.json`);
- the same vocabulary (`tokenizer.json`: vocab, BPE merges, added tokens).

Configs and tokenizers are compared by content, so a file written by an older transformers release passes.

The tokenizers must also split text the same way, which the tool checks on probe texts. The `Bielik-*-v3.0-Instruct`
base models have basal-2.0's vocabulary but an older tokenizer layout, which adds a word boundary after special tokens
and at a leading space. They are accepted only with `--allow-tokenizer-difference`.

The merged model is served with the `--meta-from` tokenizer only. With basal-2.0 on the new side, that is the tokenizer
basal-2.0 was fine-tuned with.

**Prompt format.** The non-weight files (`basal.json`, tokenizer, chat template, `CALIBRATION*.json`) come from
`--meta-from`: `new` (default), `old`, or a path / repository id. The merged model is served in that checkpoint's
prompt format. For a basal-1.5 / basal-2.0 pair:
- `--meta-from old` gives a [format 1](FORMATS.md) model: letters, two option orders averaged.
- The default gives a [format 2](FORMAT2.md) model: codes, one order.

The format moves accuracy as much as the weights do (results below), so evaluate a merge in the format it will ship
with.

**Calibration.** The copied `CALIBRATION.json` belongs to the source checkpoint. Fit a new one for every merged model
with `basal-calibrate`.

## Results on public panels

These are the measurements behind #42 (October 2026). The panels are public datasets, not the issue author's private
evaluation sets.

**Engine.** basal 2.4.1, `--mode fast-nocompile`, CUDA on a GB10 (torch 2.11, cu130).

**Panels.**
- **EN:** 10 tasks × 400 = 4000 items: arc_easy, arc_challenge, openbookqa, commonsense_qa, boolq, hellaswag,
  winogrande, sciq, piqa, mmlu.
- **PL:** 6 tasks × 400 = 2400 items: belebele `pol_Latn`, and from KLEJ polemo2-in, dyk, cdsc-e, cbd, psc.

**Calibration and ECE.** Every candidate gets its own temperature fit (FIT 70 %). The calibrated ECE is read on the
held-out CERT 30 %.

**Statistics.**
- Δ is the macro accuracy difference against a reference, with a 95 % CI from 2000 paired bootstrap resamples of items
  within each task. Each resample keeps the panel's task mix.
- There is no multiple-comparison correction, so read a CI that barely clears zero as a lead, not a result.

**Checks.**
- Every merge was built twice, and the safetensors SHA256 matched.
- Repeated runs agreed on 3999–4000 of 4000 argmaxes.
- A merge at α = 0 with `--meta-from old` reproduced basal-1.5-4.5B exactly (4000 / 4000 in EN, Δp = 0).

### 4.5B: basal-1.5-4.5B → basal-2.0-4.5B

| candidate | EN acc | Δ vs 2.0 | PL acc | Δ vs 2.0 |
|---|---:|---|---:|---|
| 2.0 | 82.8 | ref | 89.4 | ref |
| 1.5 | 83.0 | +0.20 [−0.75, +1.15] | 88.8 | −0.58 [−1.50, +0.42] |
| α = 0.25 | 82.5 | −0.30 | 89.5 | +0.12 |
| α = 0.5 | 83.0 | +0.18 [−0.50, +0.87] | 89.6 | +0.21 [−0.42, +0.87] |
| α = 0.75 | 82.9 | +0.10 | 89.5 | +0.04 |
| α = 0.9 | 82.8 | +0.05 | 89.4 | −0.04 |
| α = 0 (1.5 weights, format 2) | 82.0 | −0.80 | 88.4 | −1.00 [−1.92, −0.04] |

All rows use format 2. The grid also covered α 0.6, 0.85 and 0.95, which showed the same picture. No merge beats 2.0.

α = 0.5 in format 1 reaches 84.1 in EN, +1.30 [+0.53, +2.07] over 2.0. That gain comes from the two option orders,
not from the weights:
- basal-2.0-4.5B with `--orders 2` scores 84.0 in EN, +1.20 [+0.60, +1.82] over one order.
- Against that baseline the merge adds +0.10 [−0.60, +0.78].

In PL a second order does not help (89.2 against 89.4).

### mini: basal-1.5-mini → basal-2.0-mini

| candidate | EN acc | PL acc | PL Δ vs 2.0 `--orders 2` |
|---|---:|---:|---|
| 1.5 | 73.7 | 83.6 | +1.75 |
| 2.0 | 72.7 | 82.4 | +0.54 |
| 2.0, `--orders 2` | 74.1 | 81.8 | ref |
| α = 0.25, format 1 | 74.3 | 84.0 | +2.17 [+1.12, +3.21] |
| **α = 0.5, format 1** | **74.5** | **84.6** | **+2.75 [+1.87, +3.67]** |
| α = 0.75, format 2, `--orders 2` | 74.4 | 82.8 | +0.92 [+0.29, +1.58] |

basal-2.0-mini loses Polish against basal-1.5-mini (−1.21 [−2.38, −0.12]). α = 0.5 in format 1 wins it back and is
better than both endpoints:
- PL: +1.00 [+0.13, +1.79] against 1.5.
- EN: +0.80 [+0.12, +1.45] against 1.5, and +0.38 [−0.32, +1.08] against 2.0 with two orders.
- Calibrated ECE (CERT) stays at 0.021, the same as the endpoints.

### max: basal-1.5-max → basal-2.0-max

| candidate | EN acc | Δ vs 1.5 | PL acc | Δ vs 1.5 |
|---|---:|---|---:|---|
| 1.5 | 87.1 | ref | 89.2 | ref |
| 2.0 | 88.1 | +1.05 [+0.25, +1.93] | 89.5 | +0.29 |
| α = 0.5 | 88.1 | +1.05 | 89.7 | +0.46 [−0.21, +1.08] |
| α = 0.75 | 88.2 | +1.10 | 89.4 | +0.17 |
| α = 0.9 | 88.1 | +1.02 | 89.5 | +0.29 |

All rows use format 2. The merges match 2.0 but do not add to it.

### omni: basal-omni-1.0-max → basal-omni-1.1-max

| candidate | EN acc | Δ vs 1.0 | PL acc | Δ vs 1.0 |
|---|---:|---|---:|---|
| 1.0 | 89.7 | ref | 88.0 | ref |
| 1.1 | 89.6 | −0.08 [−0.52, +0.38] | 86.6 | −1.42 [−2.08, −0.79] |
| α = 0.25 | 89.8 | +0.10 | 87.8 | −0.17 [−0.46, +0.12] |
| α = 0.5 | 89.7 | +0.05 | 87.2 | −0.79 [−1.29, −0.29] |
| α = 0.75 | 89.6 | −0.08 | 86.7 | −1.33 [−1.96, −0.71] |

On these panels 1.1 loses 1.4 points of Polish against 1.0. Lowering α wins it back step by step.

No merge beats 1.0 here. A merge is worth it only if 1.1 is better on other data; α 0.25–0.5 is the range to check
there.

### WiSE-FT: Bielik-*-v3.0-Instruct → basal-2.0

These merges use `--allow-tokenizer-difference`, and they are served with basal-2.0's tokenizer.

| candidate | EN acc | Δ vs 2.0 | PL acc | Δ vs 2.0 |
|---|---:|---|---:|---|
| 4.5B: Bielik-4.5B-v3.0-Instruct (α = 0) | 79.8 | −2.95 [−4.10, −1.77] | 81.6 | −7.83 [−9.42, −6.42] |
| 4.5B: α = 0.5 | 83.1 | +0.30 [−0.50, +1.10] | 89.5 | +0.04 |
| 4.5B: α = 0.75 | 83.0 | +0.18 | 90.0 | +0.62 [+0.04, +1.21] |
| 4.5B: α = 0.9 | 82.8 | +0.02 | 89.7 | +0.29 [−0.04, +0.63] |
| mini: Bielik-1.5B-v3.0-Instruct (α = 0) | 69.0 | −3.72 | 59.5 | −22.87 |
| mini: α = 0.5 | 73.0 | +0.23 | 82.8 | +0.38 |
| mini: α = 0.75 | 72.9 | +0.13 | 83.5 | +1.08 [+0.25, +1.92] |
| mini: α = 0.9 | 73.1 | +0.40 [−0.05, +0.83] | 82.6 | +0.21 |

All rows use format 2. Mixing a quarter of the instruct base back in (α = 0.75) gives a small Polish gain on both
sizes, and English does not change. Both CIs only just clear zero.

### Summary

- The accuracy and calibrated ECE of every pair move smoothly with α. There is no collapse in the middle.
- mini: α = 0.5 in format 1 is better than both endpoints in Polish.
- 4.5B and max: no merge improves on 2.0.
- 4.5B in English: the gain comes from two option orders (`--orders 2`), not from merging.
- omni: α sets the trade between 1.0's Polish and whatever 1.1 adds elsewhere.

## Reproducing

`tools/interpolation/` holds the scripts:
- **Panels:** `panel_en.py` and `panel_pl.py` build the panels from Hugging Face datasets, with a fixed seed.
- **Analysis:** `analyze.py` turns `basal-calibrate --save-rows` files into the tables above: accuracy per task,
  macro accuracy, raw and calibrated ECE, and the paired bootstrap.
- **Repeatability:** `compare_runs.py` compares a run with its repeat.

    uv run --with datasets python tools/interpolation/panel_en.py --out panel_en.jsonl
    uv run --with datasets python tools/interpolation/panel_pl.py --out panel_pl.jsonl
    basal-interpolate Remek/basal-1.5-4.5B Remek/basal-2.0-4.5B --alpha 0.5 --out merged/4.5B
    for m in Remek/basal-2.0-4.5B merged/4.5B-a0.5; do n=$(basename $m)
      basal-calibrate panel_en.jsonl --model $m --mode fast-nocompile --save-rows rows/$n.jsonl --out calib/$n.json
    done
    uv run --with numpy python tools/interpolation/analyze.py ref=rows/basal-2.0-4.5B.jsonl a0.5=rows/4.5B-a0.5.jsonl \
        --panel panel_en.jsonl --ref ref --out report.md

The two-order runs add `--orders 2` to `basal-calibrate`. On the GB10, one 4.5B candidate takes about 20 minutes on
the EN panel with two runs sharing the GPU. A max or omni candidate takes about an hour.
