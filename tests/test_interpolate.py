"""basal-interpolate: alpha 0 / 1 reproduce the checkpoints, 0.5 is the fp32 mean stored in bf16, integer tensors come
from the new checkpoint, shards are kept, mismatched checkpoints are refused (but not a config or tokenizer written
by another transformers / tokenizers release), and the non-weight files come from --meta-from (on tiny checkpoints)."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from basal.interpolate import interpolate, main

TOKENIZER = {"model": {"type": "BPE", "vocab": {"<s>": 0, "a": 1, "b": 2, "ab": 3}, "merges": [["a", "b"]]},
             "added_tokens": [{"id": 0, "content": "<s>"}], "normalizer": None,
             "pre_tokenizer": {"type": "Metaspace", "replacement": "▁"}}


def make_ckpt(d, seed, sharded=False, extra=None, shape=(4, 8), config=None, tokenizer=TOKENIZER, prompt=2):
    d = Path(d)
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    t = {"a.weight": torch.randn(shape, generator=g).to(torch.bfloat16),
         "b.weight": torch.randn(3, generator=g).to(torch.bfloat16),
         "pos.ids": torch.arange(5, dtype=torch.int64) + seed}
    t.update(extra or {})
    if sharded:
        s1 = {"a.weight": t["a.weight"]}
        s2 = {k: v for k, v in t.items() if k != "a.weight"}
        save_file(s1, d / "model-00001-of-00002.safetensors")
        save_file(s2, d / "model-00002-of-00002.safetensors")
        wm = {k: "model-00001-of-00002.safetensors" for k in s1}
        wm.update({k: "model-00002-of-00002.safetensors" for k in s2})
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    else:
        save_file(t, d / "model.safetensors")
    config = config or {"architectures": ["LlamaForCausalLM"], "hidden_size": 8,
                        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
                        "dtype": "bfloat16", "transformers_version": "5.17.0"}
    (d / "config.json").write_text(json.dumps(config))
    (d / "tokenizer.json").write_text(json.dumps(tokenizer))
    (d / "basal.json").write_text(json.dumps({"name": f"m{seed}", "layout": "qfirst", "prompt": prompt}))
    for f in ("README.md", "SHA256SUMS", "evidence_head.pt"):
        (d / f).write_text("x")
    (d / "think").mkdir()
    (d / "think" / "x.json").write_text("{}")
    return d, t


def load_dir(d):
    out = {}
    for f in Path(d).glob("*.safetensors"):
        out.update(load_file(f))
    return out


@pytest.fixture
def pair(tmp_path):
    old, to = make_ckpt(tmp_path / "old", 1, prompt=1)
    new, tn = make_ckpt(tmp_path / "new", 2)
    return old, new, to, tn


def test_alpha0_is_old(pair, tmp_path):
    old, new, to, _ = pair
    r = load_dir(interpolate(old, new, 0.0, tmp_path / "o"))
    for k in ("a.weight", "b.weight"):
        assert torch.equal(r[k], to[k]) and r[k].dtype == torch.bfloat16


def test_alpha1_is_new(pair, tmp_path):
    old, new, _, tn = pair
    r = load_dir(interpolate(old, new, 1.0, tmp_path / "o"))
    for k in tn:
        assert torch.equal(r[k], tn[k])


def test_alpha_half(pair, tmp_path):
    old, new, to, tn = pair
    r = load_dir(interpolate(old, new, 0.5, tmp_path / "o"))
    for k in ("a.weight", "b.weight"):
        assert torch.equal(r[k], ((to[k].float() + tn[k].float()) / 2).to(torch.bfloat16))
    assert torch.equal(r["pos.ids"], tn["pos.ids"])  # not floating point: from the new checkpoint


def test_sharded(tmp_path):
    old, to = make_ckpt(tmp_path / "old", 1, sharded=True)
    new, tn = make_ckpt(tmp_path / "new", 2, sharded=True)
    out = interpolate(old, new, 0.5, tmp_path / "o")
    assert (out / "model.safetensors.index.json").exists()
    assert sorted(p.name for p in out.glob("*.safetensors")) == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    r = load_dir(out)
    assert torch.equal(r["a.weight"], ((to["a.weight"].float() + tn["a.weight"].float()) / 2).to(torch.bfloat16))
    assert set(r) == set(tn)


@pytest.mark.parametrize("kw", [{"config": {"architectures": ["Gemma4"], "hidden_size": 8}},
                                {"extra": {"c.weight": torch.zeros(2)}},
                                {"shape": (4, 9)},
                                {"tokenizer": {**TOKENIZER, "model": {**TOKENIZER["model"], "merges": []}}}])
def test_mismatch_refused(tmp_path, kw):
    old, _ = make_ckpt(tmp_path / "old", 1)
    new, _ = make_ckpt(tmp_path / "new", 2, **kw)
    with pytest.raises(SystemExit):
        interpolate(old, new, 0.5, tmp_path / "o")


def test_other_serialization_accepted(tmp_path):
    """The Bielik-*-v3.0-Instruct base models were written by transformers 4.x and an older tokenizers: rope_theta at
    the top level, torch_dtype, merges as strings, a Prepend / Replace normalizer instead of the Metaspace
    pre-tokenizer. Same architecture and token ids, so the pair is accepted."""
    config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 8, "rope_theta": 1000000, "rope_scaling": None,
              "torch_dtype": "bfloat16", "transformers_version": "4.44.2", "use_cache": True}
    tokenizer = {"model": {"type": "BPE", "vocab": TOKENIZER["model"]["vocab"], "merges": ["a b"]},
                 "added_tokens": TOKENIZER["added_tokens"], "pre_tokenizer": None,
                 "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}]}}
    old, _ = make_ckpt(tmp_path / "old", 1, config=config, tokenizer=tokenizer)
    new, _ = make_ckpt(tmp_path / "new", 2)
    out = interpolate(old, new, 0.5, tmp_path / "o")
    assert json.loads((out / "tokenizer.json").read_text()) == TOKENIZER  # from the new checkpoint


def test_meta_files_and_basal_json(pair, tmp_path):
    old, new, _, _ = pair
    out = interpolate(old, new, 0.25, tmp_path / "o")
    for f in ("README.md", "SHA256SUMS", "evidence_head.pt", "think/x.json"):
        assert not (out / f).exists(), f
    assert (out / "config.json").exists() and (out / "tokenizer.json").exists()
    info = json.loads((out / "basal.json").read_text())
    assert (info["layout"], info["prompt"], info["name"]) == ("qfirst", 2, "m2-interp-a0.25")
    assert info["interpolation"]["alpha"] == 0.25
    assert info["interpolation"]["old"] == str(old) and info["interpolation"]["new"] == str(new)


def test_cli_meta_from_old(pair, tmp_path):
    old, new, _, _ = pair
    main([str(old), str(new), "--alpha", "0.5", "1", "--out", str(tmp_path / "m"), "--meta-from", "old"])
    info = json.loads((tmp_path / "m-a0.5" / "basal.json").read_text())
    assert (info["prompt"], info["name"]) == (1, "m1-interp-a0.5")  # format of the old checkpoint
    assert (tmp_path / "m-a1" / "model.safetensors").exists()


def test_cli_alpha_range(pair, tmp_path):
    old, new, _, _ = pair
    with pytest.raises(SystemExit):
        main([str(old), str(new), "--alpha", "1.5", "--out", str(tmp_path / "m")])
