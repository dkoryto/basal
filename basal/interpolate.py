"""Linear weight interpolation of two checkpoints that share a base model (WiSE-FT / model soups, issue #42).

    theta(alpha) = (1 - alpha) * theta_old + alpha * theta_new

alpha is the weight of the new checkpoint: 0 gives the old weights, 1 the new ones. Each tensor is mixed in fp32 and
stored in the dtype of the new checkpoint; tensors that are not floating point (position ids) are taken from the new
checkpoint. The output keeps the new checkpoint's shard layout and holds one output shard in memory at a time (peak:
the largest shard of the new checkpoint plus one tensor in fp32). The result is written to a temporary folder next to
--out and renamed only once every shard is written, so a failed merge leaves no partial checkpoint.

The checkpoints must have the same tensor names, shapes and dtypes, unquantized weights (no quantization_config: FP8 /
NVFP4 exports store encoded weights and scales, which do not interpolate), the same architecture (config.json,
compared without serialization-only keys) and the same vocabulary (tokenizer.json: vocab, BPE merges, added tokens).
The tokenizers must also split text the same way (checked on probe texts); a pair whose vocabulary matches but whose
splitting rules differ, such as the Bielik-*-v3.0-Instruct base models (an older tokenizers layout that adds a word
boundary after special tokens) against basal-2.0, is refused unless --allow-tokenizer-difference is given: the merged
model is served with the --meta-from tokenizer only.

The non-weight files (basal.json, tokenizer, chat template, config, CALIBRATION*.json) come from --meta-from (default:
the new checkpoint), so the merged model is served in that checkpoint's prompt format: merging basal-1.5 and basal-2.0
weights with --meta-from old gives a format 1 model, otherwise format 2. The prompt format moves accuracy as much as
the weights do (docs/INTERPOLATION.md), so evaluate a merge together with the format it ships with. The shipped
calibration is only a starting point: fit one per merged model with basal-calibrate.

    basal-interpolate Remek/basal-1.5-mini Remek/basal-2.0-mini --alpha 0.25 0.5 0.75 --out merged/mini --meta-from old
"""
import argparse
import json
import shutil
from pathlib import Path

# not copied from --meta-from: the model card and checksums describe the source checkpoint, the evidence head and
# the thinking adapter were trained on its weights
SKIP = {".gitattributes", "README.md", "SHA256SUMS", "evidence_head.pt"}
# weight files of any format, and their indexes: the merged weights are the only ones in the output
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack", ".onnx"}
WEIGHT_PREFIXES = ("model.safetensors.index", "pytorch_model", "tf_model", "flax_model", "model.onnx")
# config.json keys that do not describe the architecture (serialization, runtime defaults)
CONFIG_IGNORE = {"transformers_version", "torch_dtype", "dtype", "use_cache"}
# safetensors dtypes of encoded (quantized) weights
QUANTIZED_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E8M0", "F4", "U8", "I8", "U4", "I4"}
PROBES = ["Zażółć gęślą jaźń. Ala ma kota, 123 zł!", " leading space\n\nnew lines\ttab", "x  y   ", "",
          "<s>[INST] Pytanie? [/INST]", "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n",
          "emoji 🙂 and ünïcödé", "A) tak\nB) nie\nOdpowiedź:"]


def shards(model_dir):
    """tensor name -> safetensors file holding it."""
    from safetensors import safe_open

    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        out = {}
        for k, v in json.loads(index.read_text())["weight_map"].items():
            p = Path(v)
            # the shard paths are reused under the output folder: nothing absolute, nothing that leaves the model
            if p.is_absolute() or ".." in p.parts or not (d / p).resolve().is_relative_to(d.resolve()):
                raise SystemExit(f"{index}: shard path {v!r} is outside the model folder")
            out[k] = d / p
        return out
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            out.update(dict.fromkeys(s.keys(), f))
    return out


def config_key(path):
    """config.json as compared: rope_theta folded into rope_parameters (the transformers 5 layout), keys that do not
    describe the architecture dropped."""
    c = json.loads(Path(path).read_text())
    if "rope_theta" in c:
        c.setdefault("rope_parameters", {"rope_theta": c["rope_theta"], "rope_type": "default"})
        if c.get("rope_scaling") is None:
            c.pop("rope_scaling", None)
        c.pop("rope_theta")
    return {k: v for k, v in c.items() if k not in CONFIG_IGNORE}


def tokenizer_key(path):
    """What fixes the meaning of a token id: vocabulary, BPE merges (string or pair form) and added tokens with their
    flags."""
    t = json.loads(Path(path).read_text())
    m = t.get("model", {})
    merges = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in m.get("merges") or []]
    return m.get("type"), m.get("vocab"), merges, t.get("added_tokens", [])


def tokenization_difference(po, pn):
    """The first probe text the two tokenizers split differently, as a message, or None."""
    from tokenizers import Tokenizer

    a, b = Tokenizer.from_file(str(po)), Tokenizer.from_file(str(pn))
    for text in PROBES:
        for special in (True, False):
            ea, eb = a.encode(text, add_special_tokens=special), b.encode(text, add_special_tokens=special)
            if ea.ids != eb.ids:
                return f"{text!r}: {ea.tokens[:12]} vs {eb.tokens[:12]}"
    return None


def check_pair(old, new, allow_tokenizer_difference=False):
    """Refuses (SystemExit) a pair that cannot be interpolated; returns the tensor -> shard maps."""
    from safetensors import safe_open

    for name in ("config.json", "tokenizer.json"):
        missing = [d for d in (old, new) if not (Path(d) / name).is_file()]
        if missing:
            raise SystemExit(f"{name} missing in {missing[0]}: cannot check that the checkpoints share an "
                             "architecture / tokenizer")
    for d in (old, new):
        if "quantization_config" in json.loads((Path(d) / "config.json").read_text()):
            raise SystemExit(f"{d} is quantized (quantization_config): interpolate the unquantized checkpoints and "
                             "quantize the result")
    po, pn = Path(old) / "config.json", Path(new) / "config.json"
    if po.read_bytes() != pn.read_bytes() and config_key(po) != config_key(pn):
        raise SystemExit("config.json differs: the checkpoints do not share an architecture")
    po, pn = Path(old) / "tokenizer.json", Path(new) / "tokenizer.json"
    if po.read_bytes() != pn.read_bytes():
        if tokenizer_key(po) != tokenizer_key(pn):
            raise SystemExit("tokenizer.json differs: the checkpoints do not share a vocabulary")
        diff = tokenization_difference(po, pn)
        if diff and not allow_tokenizer_difference:
            raise SystemExit(f"the tokenizers share the vocabulary but split text differently ({diff}); the merged "
                             "model is served with the --meta-from tokenizer only. Pass --allow-tokenizer-difference "
                             "if that is intended (a base model written by an older tokenizers release)")

    a, b = shards(old), shards(new)
    if not b:
        raise SystemExit(f"{new}: no safetensors weights")
    if a.keys() != b.keys():
        raise SystemExit(f"different tensors: only old {sorted(a.keys() - b.keys())[:5]}, "
                         f"only new {sorted(b.keys() - a.keys())[:5]}")
    for k in b:  # shapes and dtypes from the headers, before anything is written
        with safe_open(a[k], "pt") as so, safe_open(b[k], "pt") as sn:
            so_, sn_ = so.get_slice(k), sn.get_slice(k)
            if so_.get_shape() != sn_.get_shape():
                raise SystemExit(f"{k}: shape {tuple(so_.get_shape())} vs {tuple(sn_.get_shape())}")
            dt = {so_.get_dtype(), sn_.get_dtype()}
            if dt & QUANTIZED_DTYPES and k.endswith("weight"):
                raise SystemExit(f"{k}: quantized weights ({'/'.join(sorted(dt))}) do not interpolate")
    return a, b


def interpolate(old, new, alpha, out, meta_from=None, allow_tokenizer_difference=False):
    """Writes the merged checkpoint to out (a new or empty directory) and returns its path."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"alpha {alpha} outside [0, 1]")
    old, new, out = Path(old), Path(new), Path(out)
    meta_from = Path(meta_from or new)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"{out} exists and is not empty; remove it or choose another --out")
    a, b = check_pair(old, new, allow_tokenizer_difference)
    tmp = out.parent / f".{out.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        by_file = {}  # tensors grouped by the new checkpoint's shard (path relative to the model, as in the index)
        for k, f in b.items():
            by_file.setdefault(f.relative_to(new), []).append(k)
        for rel, keys in by_file.items():
            tensors = {}
            for k in keys:
                with safe_open(a[k], "pt") as so, safe_open(b[k], "pt") as sn:  # no handle outlives its tensor
                    t_old, t_new = so.get_tensor(k), sn.get_tensor(k)
                if t_new.is_floating_point():
                    tensors[k] = torch.lerp(t_old.float(), t_new.float(), alpha).to(t_new.dtype).contiguous()
                else:
                    tensors[k] = t_new.clone()
                del t_old, t_new
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, tmp / rel, metadata={"format": "pt"})
            del tensors
        if (new / "model.safetensors.index.json").exists():
            shutil.copy(new / "model.safetensors.index.json", tmp)

        for f in meta_from.rglob("*"):
            rel = f.relative_to(meta_from)
            if not f.is_file() or rel.name in SKIP or rel.parts[0] == "think" or rel.parts[0].startswith("."):
                continue
            if f.suffix in WEIGHT_SUFFIXES or rel.name.startswith(WEIGHT_PREFIXES):
                continue
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, tmp / rel)
        info = json.loads((tmp / "basal.json").read_text()) if (tmp / "basal.json").exists() else {}
        info.update({"name": f"{info.get('name', new.name)}-interp-a{alpha:g}",
                     "interpolation": {"old": str(old), "new": str(new), "alpha": alpha,
                                       "rule": "(1 - alpha) * old + alpha * new, fp32 accumulate"}})
        (tmp / "basal.json").write_text(json.dumps(info, indent=1) + "\n")
        if out.exists():
            out.rmdir()  # empty, checked above
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])  # no docstring under python -OO
    ap.add_argument("old", help="local directory or Hugging Face repo id")
    ap.add_argument("new", help="local directory or Hugging Face repo id")
    ap.add_argument("--alpha", type=float, nargs="+", required=True, help="weight of the new checkpoint, in [0, 1]")
    ap.add_argument("--out", required=True, help="output prefix: one new directory <out>-a<alpha> per alpha")
    ap.add_argument("--meta-from", default=None, help="checkpoint whose non-weight files (prompt format, tokenizer, "
                                                      "calibration) are copied: old, new (default) or a path / repo id")
    ap.add_argument("--allow-tokenizer-difference", action="store_true",
                    help="accept tokenizers with the same vocabulary that split text differently (see above)")
    a = ap.parse_args(argv)
    if any(not 0.0 <= x <= 1.0 for x in a.alpha):
        ap.error("--alpha values must be in [0, 1]")
    names = [f"{a.out}-a{x:g}" for x in a.alpha]
    if len(set(names)) != len(names):
        ap.error(f"--alpha values give the same output directory: {sorted({n for n in names if names.count(n) > 1})}")
    from huggingface_hub import snapshot_download

    def local(m):
        return Path(m) if Path(m).exists() else Path(snapshot_download(m))

    old, new = local(a.old), local(a.new)
    meta = {"old": old, "new": new, None: new}.get(a.meta_from) or local(a.meta_from)
    for alpha, out in zip(a.alpha, names):
        print(interpolate(old, new, alpha, out, meta, a.allow_tokenizer_difference))


if __name__ == "__main__":
    main()
