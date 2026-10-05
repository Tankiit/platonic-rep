"""uq_config.py -- central config and pre-registered predictions (plumbing only; implemented).

Conventions: flat directory, plain imports, Mac-first (MPS), no multiprocessing.
Heavy imports (torch, timm) are lazy so the theory tests run without them.
"""
from dataclasses import dataclass, field
from pathlib import Path

# name -> (library, identifier, pretrained_tag).
# TODO: verify every identifier with timm.list_models() / open_clip.list_pretrained() before extraction,
#       and check licences. Choose 4-5 that differ in objective AND architecture, or alignment range is too narrow.
ENCODERS = {
    "dinov2_b":    ("timm",      "vit_base_patch14_dinov2.lvd142m",             None),
    "clip_b16":    ("open_clip", "ViT-B-16",                                    "openai"),
    "vit_sup_b16": ("timm",      "vit_base_patch16_224.augreg_in21k_ft_in1k",   None),
    "convnext_s":  ("timm",      "convnext_small.fb_in22k_ft_in1k",             None),
    "mae_b16":     ("timm",      "vit_base_patch16_224.mae",                    None),
}

# Blind-spot follow-up (BLINDSPOT_PLAN.md, B3 needs >= 8 encoders). Not part of the frozen PREREG.
# tag 'random' = same architecture, random init (trivial baseline; excluded from natural pairs).
BLINDSPOT_ENCODERS = dict(ENCODERS, **{
    "deit_b16":    ("timm",      "deit_base_patch16_224.fb_in1k",                  None),
    "swin_b":      ("timm",      "swin_base_patch4_window7_224.ms_in22k_ft_in1k",  None),
    "resnet50":    ("timm",      "resnet50.a1_in1k",                               None),
    "mixer_b16":   ("timm",      "mixer_b16_224.goog_in21k_ft_in1k",               None),
    "siglip_b16":  ("open_clip", "ViT-B-16-SigLIP",                                "webli"),
    "vit_rand_b16": ("timm",     "vit_base_patch16_224",                           "random"),
})
FAMILIES = {"dinov2_b": "ssl", "mae_b16": "ssl", "clip_b16": "lang", "siglip_b16": "lang",
            "vit_sup_b16": "sup_vit", "deit_b16": "sup_vit", "swin_b": "sup_vit",
            "convnext_s": "sup_conv", "resnet50": "sup_conv", "mixer_b16": "mlp", "vit_rand_b16": "random"}

# Pre-registered predictions and falsifiers. FROZEN 2026-10-02 before any real-data result (see PREREG_FROZEN.txt).
PREREG = {
    "E0_gate": dict(
        claim="Pipeline recovers known AU on Synthetic.",
        rule="Spearman(model AU, true AU) >= 0.5, else stop and debug.",
        note="Frozen natural-image encoders may represent synthetic shapes poorly; that is an encoder failure, not a pipeline one."),
    "E2": dict(
        claim="With identical features (alignment = 1), EU agreement drops with head-data mismatch; AU agreement stays near its ceiling-normalised level.",
        falsifier_a="Partial Spearman of EU between heads trained on 10% vs 100% stays > 0.9 -> data term negligible in practice.",
        falsifier_b="AU agreement degrades as much as EU agreement -> thesis becomes 'alignment identifies neither'."),
    "E2b": dict(
        claim="Rotation + isotropic rescaling keeps CKA/mKNN fixed but changes EU at fixed weight decay (Prop A); tail reshaping keeps CKA ~ 1 but moves d_eff and EU (Prop B).",
        control="With weight decay re-tuned per transform, the scale effect should mostly vanish."),
    "E4": dict(
        claim="At high calibrated alignment, head and data terms dominate the variance of |W_A - W_B|.",
        falsifier="Representation term > 80% Shapley share -> representation dominates; thesis restricted to low-alignment regime."),
    "H8": dict(
        claim="Class-level alignment rises with collapse (NC1 falls); residual alignment does not; EU agreement tracks residual alignment.",
        gate="share_W (share of item-to-item EU variance from the residual) must be high, else drop H8.",
        falsifier="Residual alignment also high AND tracks EU agreement -> no gap."),
    "S_rho": dict(
        claim="S_rho at the head's own rho predicts EU agreement across encoder pairs better than CKA, CCA, mKNN.",
        falsifier="S_rho no better than CKA (partial correlation controlling for confidence and human AU)."),
}

@dataclass
class Config:
    root: Path = Path("uq_runs")
    datasets: tuple = ("cifar10h", "synthetic", "treeversity6")
    encoders: tuple = tuple(ENCODERS.keys())
    rel_depths: tuple = (0.5, 0.75, 1.0)
    head_fracs: tuple = (0.1, 0.25, 1.0)
    n_boot: int = 50                 # TODO: fix via M-stability curve (uq_heads.choose_M)
    seeds: tuple = (0, 1, 2, 3, 4)
    k_nn: int = 10                   # TODO: Groger et al. suggest K >= 200 for calibrated global metrics; k here is for per-item mKNN
    n_perm: int = 100
    rho_grid: tuple = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    wd_grid: tuple = (1e-5, 1e-4, 1e-3, 1e-2)
    batch_size: int = 64
    image_size: int = 224            # CIFAR 32px is upsampled; state the blur as a shared limitation
    cache_dtype: str = "float16"
    eu_interval: tuple = (0.05, 0.95)  # quantile width, NOT max-min (grows with M)

    def feature_path(self, dataset: str, split: str, encoder: str) -> Path:
        return self.root / "features" / f"{dataset}_{split}_{encoder}.npz"

    def result_path(self, name: str) -> Path:
        return self.root / "results" / f"{name}.csv"

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

LINALG_NOTE = "Run eigh/solve/inv in float64 on CPU (MPS linalg is unstable); keep GPU/MPS for matmuls and extraction."
