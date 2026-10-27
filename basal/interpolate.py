"""Linear weight interpolation of two checkpoints that share a base model (WiSE-FT / model soups, issue #42).

    theta(alpha) = (1 - alpha) * theta_old + alpha * theta_new

alpha is the weight of the new checkpoint: 0 gives the old weights, 1 the new ones. Each tensor is mixed in fp32 and
stored in the dtype of the new checkpoint, one tensor at a time, so memory stays near the size of one model; tensors
that are not floating point (position ids) are taken from the new checkpoint. The checkpoints must have the same
tensor names and shapes, the same architecture (config.json, compared without serialization-only keys) and the same
token ids (tokenizer.json: vocabulary, BPE merges, added tokens).

The non-weight files (basal.json, tokenizer, chat template, config, CALIBRATION*.json) come from --meta-from (default:
the new checkpoint), so the merged model is served in that checkpoint's prompt format: merging basal-1.5 and basal-2.0
weights with --meta-from on the basal-1.5 side gives a format 1 model, otherwise format 2. The prompt format moves
accuracy as much as the weights do (docs/INTERPOLATION.md), so evaluate a merge together with the format it ships with.
The shipped calibration is only a starting point: fit one per merged model with basal-calibrate.

    basal-interpolate Remek/basal-1.5-mini Remek/basal-2.0-mini --alpha 0.25 0.5 0.75 --out merged/mini \\
        --meta-from Remek/basal-1.5-mini
"""
import argparse
import json
import shutil
from pathlib import Path

# not copied from --meta-from: the model card and checksums describe the source checkpoint, the evidence head and
# the thinking adapter were trained on its weights
SKIP = {".gitattributes", "README.md", "SHA256SUMS", "evidence_head.pt"}
# config.json keys that do not describe the architecture (serialization, runtime defaults)
CONFIG_IGNORE = {"transformers_version", "torch_dtype", "dtype", "use_cache"}


def shards(model_dir):
    """tensor name -> safetensors file holding it."""
    from safetensors import safe_open

    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        return {k: d / v for k, v in json.loads(index.read_text())["weight_map"].items()}
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            out.update({k: f for k in s.keys()})
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
    """What fixes the token ids: vocabulary, BPE merges (string or pair form) and added tokens. Older and newer
    tokenizers releases write the same tokenizer with different normalizer / pre_tokenizer layouts (the
    Bielik-*-v3.0-Instruct base models against basal-2.0)."""
    t = json.loads(Path(path).read_text())
    m = t.get("model", {})
    merges = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in m.get("merges") or []]
    return m.get("type"), m.get("vocab"), merges, [(x["id"], x["content"]) for x in t.get("added_tokens", [])]


def check_pair(old, new):
    a, b = shards(old), shards(new)
    if not b:
        raise SystemExit(f"{new}: no safetensors weights")
    if a.keys() != b.keys():
        raise SystemExit(f"different tensors: only old {sorted(a.keys() - b.keys())[:5]}, "
                         f"only new {sorted(b.keys() - a.keys())[:5]}")
    for name, key in (("config.json", config_key), ("tokenizer.json", tokenizer_key)):
        po, pn = Path(old) / name, Path(new) / name
        if po.read_bytes() != pn.read_bytes() and key(po) != key(pn):
            raise SystemExit(f"{name} differs: the checkpoints do not share an architecture / tokenizer")
    return a, b


def interpolate(old, new, alpha, out, meta_from=None):
    """Writes the merged checkpoint to out (a directory) and returns its path."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"alpha {alpha} outside [0, 1]")
    meta_from = Path(meta_from or new)
    a, b = check_pair(old, new)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    by_file = {}  # tensors grouped by the new checkpoint's shard, written to shards of the same name
    for k, f in b.items():
        by_file.setdefault(f.name, []).append(k)
    handles = {}

    def get(path, key):
        if path not in handles:
            handles[path] = safe_open(path, "pt")
        return handles[path].get_tensor(key)

    for fname, keys in by_file.items():
        tensors = {}
        for k in keys:
            t_old, t_new = get(a[k], k), get(b[k], k)
            if t_old.shape != t_new.shape:
                raise SystemExit(f"{k}: shape {tuple(t_old.shape)} vs {tuple(t_new.shape)}")
            if t_new.is_floating_point():
                tensors[k] = torch.lerp(t_old.float(), t_new.float(), alpha).to(t_new.dtype).contiguous()
            else:
                tensors[k] = t_new.clone()
        save_file(tensors, out / fname, metadata={"format": "pt"})
    if (Path(new) / "model.safetensors.index.json").exists():
        shutil.copy(Path(new) / "model.safetensors.index.json", out)

    for f in meta_from.rglob("*"):
        rel = f.relative_to(meta_from)
        if f.is_file() and f.suffix != ".safetensors" and rel.name not in SKIP and rel.parts[0] != "think" \
                and not rel.name.startswith("model.safetensors.index") and not rel.parts[0].startswith("."):
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, out / rel)
    info = json.loads((out / "basal.json").read_text()) if (out / "basal.json").exists() else {}
    info.update({"name": f"{info.get('name', Path(new).name)}-interp-a{alpha:g}",
                 "interpolation": {"old": str(old), "new": str(new), "alpha": alpha,
                                   "rule": "(1 - alpha) * old + alpha * new, fp32 accumulate"}})
    (out / "basal.json").write_text(json.dumps(info, indent=1) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])  # no docstring under python -OO
    ap.add_argument("old", help="local directory or Hugging Face repo id")
    ap.add_argument("new", help="local directory or Hugging Face repo id")
    ap.add_argument("--alpha", type=float, nargs="+", required=True, help="weight of the new checkpoint, in [0, 1]")
    ap.add_argument("--out", required=True, help="output prefix: one directory <out>-a<alpha> per alpha")
    ap.add_argument("--meta-from", default=None, help="checkpoint whose non-weight files (prompt format, tokenizer, "
                                                      "calibration) are copied: old, new (default) or a path / repo id")
    a = ap.parse_args(argv)
    if any(not 0.0 <= x <= 1.0 for x in a.alpha):
        ap.error("--alpha values must be in [0, 1]")
    from huggingface_hub import snapshot_download

    def local(m):
        return Path(m) if Path(m).exists() else Path(snapshot_download(m))

    old, new = local(a.old), local(a.new)
    meta = {"old": old, "new": new, None: new}.get(a.meta_from) or local(a.meta_from)
    for alpha in a.alpha:
        print(interpolate(old, new, alpha, f"{a.out}-a{alpha:g}", meta))


if __name__ == "__main__":
    main()
