"""Score-prior network and VP-SDE helpers.

Extracted VERBATIM from the SMC notebook so the CT experiment loads the same
architecture the checkpoint was trained with.  Do not edit: if the notebook's
model changes, re-extract rather than hand-patching.

Forward SDE (as used throughout):   x_t = (1-t) x_0 + sqrt(t(2-t)) eps
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TimeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class SmallUNetScore(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.enc1 = TimeBlock(in_ch, base_ch, emb_dim)
        self.enc2 = TimeBlock(base_ch, base_ch * 2, emb_dim)
        self.enc3 = TimeBlock(base_ch * 2, base_ch * 4, emb_dim)
        self.pool = nn.AvgPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = TimeBlock(base_ch * 4, base_ch * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = TimeBlock(base_ch * 2, base_ch, emb_dim)
        self.out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        # t should be in [0,1], shape (B,)
        emb = self.time_mlp(t)
        e1 = self.enc1(x, emb)
        e2 = self.enc2(self.pool(e1), emb)
        e3 = self.enc3(self.pool(e2), emb)
        u2 = self.up2(e3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), emb)
        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), emb)
        return self.out(d1)


def dsm_loss(model: nn.Module, x0: torch.Tensor,
             dsm_t_min: float = 1e-3, dsm_t_max: float = 0.98,
             dsm_weight_by_sigma2: bool = True) -> torch.Tensor:
    """Denoising score-matching loss.  Only needed if you retrain the prior;
    the CT experiment loads a checkpoint and never calls this."""
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device) * (dsm_t_max - dsm_t_min) + dsm_t_min
    sigma = torch.sqrt(torch.clamp(t * (2.0 - t), min=1e-8)).view(B, 1, 1, 1)
    mean = (1.0 - t).view(B, 1, 1, 1) * x0
    eps = torch.randn_like(x0)
    xt = mean + sigma * eps
    target = -eps / sigma
    pred = model(xt, t)
    sq = (pred - target) ** 2
    if dsm_weight_by_sigma2:
        sq = sq * sigma ** 2
    return sq.mean()


# ---------------------------------------------------------------------------
# VP-SDE helpers (same parameterisation as the notebooks)
# ---------------------------------------------------------------------------

def sde_mean_std(t: torch.Tensor):
    """x_t = mean(t) * x_0 + std(t) * eps."""
    return (1.0 - t), torch.sqrt(torch.clamp(t * (2.0 - t), min=1e-8))


def build_score_model(base_ch: int, emb_dim: int, device) -> nn.Module:
    return SmallUNetScore(in_ch=1, base_ch=int(base_ch), emb_dim=int(emb_dim)).to(device)


def _extract_state_dict(ckpt):
    """Find the weights inside a checkpoint dict.

    The SMC notebook saves as ``{"model": state_dict, "optimizer": ..., ...}``
    and loads with ``ckpt["model"]``.  That key is checked first; the others are
    fallbacks for checkpoints saved by other scripts.
    """
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model", "model_state_dict", "state_dict", "net", "ema", "weights"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # A bare state_dict is a dict whose values are all tensors.
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise KeyError(
        "Could not find weights in the checkpoint. Top-level keys: "
        f"{sorted(ckpt.keys())}. Pass the right one via state_dict_key=."
    )


def infer_model_dims(state):
    """Recover (in_ch, base_ch, emb_dim) from a SmallUNetScore state_dict.

    Lets the notebook build a model that matches the checkpoint instead of
    guessing, which otherwise fails with a wall of missing-key errors.
    """
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    dims = {}
    if "time_mlp.1.weight" in state:
        dims["emb_dim"] = int(state["time_mlp.1.weight"].shape[0])
    if "enc1.conv1.weight" in state:
        w = state["enc1.conv1.weight"]
        dims["base_ch"] = int(w.shape[0])
        dims["in_ch"] = int(w.shape[1])
    if "out.weight" in state:
        dims.setdefault("in_ch", int(state["out.weight"].shape[0]))
    return dims


def load_score_checkpoint(model, path, device, state_dict_key=None, strict=True,
                          verbose=True):
    """Load a checkpoint into ``model``.  Returns True on success.

    Raises a readable error on a dimension mismatch rather than dumping every
    missing parameter name.
    """
    import pathlib
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        return False
    try:
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location=device)

    state = ckpt[state_dict_key] if state_dict_key else _extract_state_dict(ckpt)
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}

    want = infer_model_dims(state)
    have = {"emb_dim": int(model.time_mlp[1].weight.shape[0]),
            "base_ch": int(model.enc1.conv1.weight.shape[0]),
            "in_ch": int(model.enc1.conv1.weight.shape[1])}
    bad = {k: (have[k], want[k]) for k in want if k in have and have[k] != want[k]}
    if bad:
        raise ValueError(
            "Checkpoint architecture does not match the model you built:\n" +
            "\n".join(f"    {k}: model has {h}, checkpoint has {w}"
                       for k, (h, w) in bad.items()) +
            "\n  Fix the corresponding cfg fields (base_channels / time_emb_dim), "
            "or call build_score_model_for_checkpoint(path, device)."
        )

    model.load_state_dict(state, strict=strict)
    model.eval()
    if verbose and isinstance(ckpt, dict):
        meta = {k: ckpt[k] for k in ("global_step", "saved_at_unix_time")
                if k in ckpt and not torch.is_tensor(ckpt[k])}
        if "config" in ckpt and isinstance(ckpt["config"], dict):
            keep = {k: v for k, v in ckpt["config"].items()
                    if k in ("IMG_SIZE", "BASE_CHANNELS", "TIME_EMB_DIM",
                             "NUM_TRAIN_STEPS", "SEED", "DATA_MODE")}
            if keep:
                meta["config"] = keep
        if meta:
            print("  checkpoint metadata:", meta)
    return True


def build_score_model_for_checkpoint(path, device, in_ch=1):
    """Build a SmallUNetScore sized to match the checkpoint, and load it."""
    import pathlib
    path = pathlib.Path(path).expanduser()
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    dims = infer_model_dims(_extract_state_dict(ckpt))
    model = SmallUNetScore(in_ch=dims.get("in_ch", in_ch),
                           base_ch=dims["base_ch"],
                           emb_dim=dims["emb_dim"]).to(device)
    load_score_checkpoint(model, path, device)
    return model, dims

# ---------------------------------------------------------------------------
# Cylinder / ellipse phantom dataset (extracted verbatim from the SMC notebook,
# where it is named SyntheticMRIDataset).  This is the distribution the saved
# prior was trained on unless you pointed DATA_MODE at fastMRI.
# ---------------------------------------------------------------------------

class SyntheticMRIDataset(Dataset):
    """Small synthetic phantom dataset for code debugging only."""
    def __init__(self, n: int = 2048, img_size: int = 64, seed: int = 0):
        self.n = int(n)
        self.img_size = int(img_size)
        self.rng = np.random.default_rng(seed)
        self.seeds = self.rng.integers(0, 2**31 - 1, size=self.n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(int(self.seeds[idx]))
        H = W = self.img_size
        yy, xx = np.mgrid[-1:1:complex(H), -1:1:complex(W)]
        img = np.zeros((H, W), dtype=np.float32)
        n_ellipses = rng.integers(4, 10)
        for _ in range(n_ellipses):
            cx, cy = rng.uniform(-0.45, 0.45, size=2)
            ax, ay = rng.uniform(0.08, 0.35, size=2)
            theta = rng.uniform(0, np.pi)
            c, s = np.cos(theta), np.sin(theta)
            x0 = xx - cx
            y0 = yy - cy
            xr = c * x0 + s * y0
            yr = -s * x0 + c * y0
            mask = (xr / ax) ** 2 + (yr / ay) ** 2 <= 1.0
            img[mask] += rng.uniform(0.15, 1.0)
        # Mild smoothness in Fourier domain.
        k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm="ortho"))
        rr = np.sqrt(xx**2 + yy**2)
        filt = np.exp(-(rr / 0.75) ** 4)
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k * filt), norm="ortho")).real
        img = img - img.min()
        img = img / (img.max() + 1e-6)
        img = 2.0 * img - 1.0
        return torch.from_numpy(img.astype(np.float32))[None]


CylinderPhantomDataset = SyntheticMRIDataset   # clearer alias


def sample_cylinder_image(index: int = 0, img_size: int = 64, seed: int = 0, device=None):
    """One ground-truth image (1,1,H,W) in [-1,1] from the same distribution the
    prior was trained on.  Prefer this over an ad-hoc phantom: an out-of-
    distribution test image confounds sampler quality with prior mismatch."""
    ds = SyntheticMRIDataset(n=max(1, index + 1), img_size=img_size, seed=seed)
    x = ds[int(index)][None]
    return x if device is None else x.to(device)
