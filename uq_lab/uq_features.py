"""uq_features.py -- per-layer feature extraction with resumable caching.

Cache: <root>/features/<dataset>_<split>_<encoder>.npy  float16 [N, L, d] (L = relative-depth taps)
       plus a sidecar .json with meta (encoder id, library version, taps, preprocessing, image size, split, dataset).
Pooling (decided once for all results): mean over patch tokens (prefix/CLS tokens excluded) for ViTs, spatial mean
for ConvNets, applied to the raw block output (no final norm) at every tap, so depths are treated identically.
"""
import json
import time

import numpy as np

from uq_config import BLINDSPOT_ENCODERS, ENCODERS, get_device


def build_encoder(name: str, device):
    """Return (model.eval(), preprocess_fn, tap_modules, info). preprocess_fn maps a uint8 [B,32,32,3] array to a
    normalised float tensor [B,3,S,S] on `device`. For open_clip only the IMAGE tower is used."""
    import torch
    import torch.nn.functional as F
    lib, ident, tag = ENCODERS[name] if name in ENCODERS else BLINDSPOT_ENCODERS[name]
    if lib == "timm":
        import timm
        kw = {"img_size": 224} if "dinov2" in ident else {}
        if tag == "random":
            torch.manual_seed(0)
        model = timm.create_model(ident, pretrained=(tag != "random"), num_classes=0, **kw)
        cfg = model.pretrained_cfg
        mean, std = cfg["mean"], cfg["std"]
        if hasattr(model, "blocks"):
            # MlpMixer has no prefix tokens and no num_prefix_tokens attribute.
            default_prefix = 0 if type(model).__name__ == "MlpMixer" else 1
            blocks, kind = list(model.blocks), "vit"
            n_prefix = int(getattr(model, "num_prefix_tokens", default_prefix))
        elif hasattr(model, "stages"):
            blocks, kind, n_prefix = [b for s in model.stages for b in s.blocks], "conv", 0
        elif hasattr(model, "layers"):        # Swin: stage outputs are NHWC
            blocks, kind, n_prefix = [b for s in model.layers for b in s.blocks], "nhwc", 0
        else:                                 # ResNet bottlenecks across layer1..layer4
            blocks, kind, n_prefix = [b for s in (model.layer1, model.layer2, model.layer3, model.layer4)
                                      for b in s], "conv", 0
        version = timm.__version__
    else:
        import open_clip
        clip, _, _ = open_clip.create_model_and_transforms(ident, pretrained=tag, force_quick_gelu=(tag == "openai"))
        model = clip.visual
        if hasattr(model, "trunk"):           # timm-backed towers (SigLIP): no CLS token, MAP pooling
            mean, std = model.trunk.pretrained_cfg["mean"], model.trunk.pretrained_cfg["std"]
            blocks, kind = list(model.trunk.blocks), "vit"
            n_prefix = int(getattr(model.trunk, "num_prefix_tokens", 0))
        else:
            mean, std = open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD
            blocks, kind, n_prefix = list(model.transformer.resblocks), "vit", 1
        version = open_clip.__version__
    model = model.eval().to(device)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def preprocess(x_uint8):
        x = torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        return (x - m) / s

    batch_first = bool(getattr(getattr(model, "transformer", None), "batch_first", True))
    info = dict(library=lib, identifier=ident, tag=tag, version=version, kind=kind, n_prefix=n_prefix,
                n_blocks=len(blocks), batch_first=batch_first, mean=list(mean), std=list(std),
                force_quick_gelu=(lib == "open_clip" and tag == "openai"))
    return model, preprocess, blocks, info


def relative_depth_taps(blocks, rel_depths) -> list:
    """Map relative depths to block indices: idx = round(depth * n_blocks) - 1 over the flattened block list
    (ViT blocks; ConvNeXt blocks across stages), so architectures are compared by RELATIVE depth."""
    n = len(blocks)
    return [max(0, int(round(r * n)) - 1) for r in rel_depths]


def pool(x, kind: str = "vit", n_prefix: int = 1, batch_first: bool = True):
    """tokens [B,T,d] -> mean over patch tokens (prefix excluded); conv [B,C,H,W] -> spatial mean."""
    if kind == "conv":
        return x.float().mean(dim=(2, 3))
    if kind == "nhwc":
        return x.float().mean(dim=(1, 2))
    if not batch_first:
        x = x.transpose(0, 1)
    return x[:, n_prefix:, :].float().mean(1)


def extract_split(cfg, encoder: str, dataset: str, split: str, resume: bool = True, images=None) -> None:
    """Extract and cache features for one (encoder, dataset, split). Resumable: progress is written to the sidecar
    every `checkpoint_every` batches; a crash loses at most that much work. No multiprocessing."""
    import torch
    import uq_data as D
    out = cfg.feature_path(dataset, split, encoder).with_suffix(".npy")
    meta_path = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if images is None:
        assert dataset == "cifar10"
        images, _ = D.load_cifar10(train=(split == "train"))
    N = len(images)
    device = get_device()
    model, prep, blocks, info = build_encoder(encoder, device)
    taps = relative_depth_taps(blocks, cfg.rel_depths)
    store = {}
    hooks = [blocks[t].register_forward_hook(lambda mod, i, o, t=t: store.__setitem__(t, o)) for t in taps]
    with torch.no_grad():
        prep(images[:2])
        probe = model(prep(images[:2]))
        tap_dims = [pool(store[t], info["kind"], info["n_prefix"], info["batch_first"]).shape[1] for t in taps]
        d = max(tap_dims)                      # taps of different width (ConvNeXt stages) are zero-padded to d
    done = 0
    if resume and out.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        done = meta.get("done", 0)
        arr = np.lib.format.open_memmap(out, mode="r+")
        if done >= N:
            print(f"[skip] {out.name} complete")
            return
    else:
        arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(N, len(taps), d))
    meta = dict(info, encoder=encoder, dataset=dataset, split=split, taps=taps, rel_depths=list(cfg.rel_depths),
                image_size=224, upsample="bicubic 32->224", pooling="mean patch tokens / spatial mean, raw block out",
                N=N, d=d, tap_dims=tap_dims, done=done)
    bs, t0 = cfg.batch_size, time.time()
    with torch.no_grad():
        for b, start in enumerate(range(done, N, bs)):
            x = prep(images[start:start + bs])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type != "cpu"):
                model(x)
            feats = [pool(store[t], info["kind"], info["n_prefix"], info["batch_first"]) for t in taps]
            feats = [torch.nn.functional.pad(f, (0, d - f.shape[1])) for f in feats]
            arr[start:start + len(x)] = torch.stack(feats, 1).cpu().numpy().astype(np.float16)
            if b % 50 == 0 or start + bs >= N:
                arr.flush()
                meta["done"] = min(N, start + bs)
                meta_path.write_text(json.dumps(meta, indent=1))
                rate = (meta["done"] - done) / (time.time() - t0 + 1e-9)
                print(f"{encoder} {split} {meta['done']}/{N} {rate:.0f} img/s", flush=True)
    for h in hooks:
        h.remove()
    del probe


def load_features(cfg, encoder: str, dataset: str, split: str, layer: int | None = None) -> np.ndarray:
    """Return float32 features [N, L, d] or [N, d] for one tap index into cfg.rel_depths."""
    path = cfg.feature_path(dataset, split, encoder).with_suffix(".npy")
    arr = np.load(path, mmap_mode="r")
    if layer is None:
        return np.asarray(arr, dtype=np.float32)
    dims = json.loads(path.with_suffix(".json").read_text()).get("tap_dims")
    d = dims[layer] if dims else arr.shape[2]
    return np.asarray(arr[:, layer, :d], dtype=np.float32)


def benchmark_throughput(cfg, encoder: str, n: int = 512) -> float:
    """Images/second on this machine (fp16 autocast)."""
    import torch
    import uq_data as D
    imgs, _ = D.load_cifar10(train=False)
    device = get_device()
    model, prep, _, _ = build_encoder(encoder, device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type != "cpu"):
        model(prep(imgs[:cfg.batch_size]))
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        for s in range(0, n, cfg.batch_size):
            model(prep(imgs[s:s + cfg.batch_size]))
        if device.type == "mps":
            torch.mps.synchronize()
    return n / (time.time() - t0)


if __name__ == "__main__":
    import sys
    from uq_config import Config
    cfg = Config()
    encs = sys.argv[1:] or list(ENCODERS)
    for e in encs:
        for split in ("test", "train"):
            extract_split(cfg, e, "cifar10", split)
