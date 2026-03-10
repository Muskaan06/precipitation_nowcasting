"""
Training loop for NowcastNet GAN on rainfall satellite data.

NowcastNet input structure (from lightning.py):
    input_shape_dict has TWO components concatenated along dim-1:
        1. raw rainfall context frames  (input_len = 4 frames)
        2. evolution-network prediction (output_len = 6 frames)
    x_before.shape = (B, input_len + output_len, H, W) = (B, 10, H, W)

Evolution prediction (evo signal fed to SPADE layers in the decoder):
    Computed with pysteps LK optical flow — same pipeline as
    save_intensities_and_motion_hdf.py. No data leakage.
        1. LK optical flow on context frames -> motion field (2, H, W)
        2. Warp last context frame with that flow
        3. Repeat warped frame output_len times -> (B, output_len, H, W)

Data format:
    Each .npz = one set, shape (24, 3, 112, 112)
        axis-0 : 24 frames, axis-1 : 3 channels (ch0=rain mm/hr), axis-2/3 : 112x112
Dataset split: 70 / 15 / 15
CSI at 0.1, 1.0, 5.0, 10.0 mm/hr accumulated over the full epoch.
"""

import os
import glob
import math

os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["KMP_WARNINGS"]        = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from pysteps import motion as pysteps_motion
from pysteps.utils import conversion, transformation

from model.nowcastnet.nowcastnet_model_code import NowcasnetGenerator, TemporalDiscriminator


# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
CFG = {
    "data_dir":         "data/rainfall_npz",
    "npz_key":          None,
    "input_len":        4,          # raw context frames
    "output_len":       6,          # target / evo-prediction frames
    "rain_channel":     0,          # ch0 = rain (mm/hr)
    "img_size":         112,
    "motion_method":    "lk",       # pysteps method: "lk" or "darts"
    "latent_dim":       32,
    "h_noise":          8,          # noise map spatial size
    "alpha":            6.0,        # adversarial loss weight
    "beta":             20.0,       # grid-cell loss weight
    "glr":              2e-4,
    "dlr":              2e-4,
    "b1":               0.5,
    "b2":               0.999,
    "num_epochs":       50,
    "batch_size":       8,
    "generation_steps": 1,
    "seed":             42,
    "num_workers":      4,
    "save_dir":         "checkpoints/nowcastnet",
    "csi_thresholds":   [0.1, 1.0, 5.0, 10.0],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ------------------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------------------

class RainfallNPZDataset(Dataset):
    """
    Each .npz = one set, shape (24, 3, H, W).
    Returns:
        X : (input_len,  3, H, W)  float32  — context frames
        Y : (output_len, 3, H, W)  float32  — target frames
    """
    def __init__(self, file_paths, input_len=4, output_len=6, npz_key=None):
        self.file_paths = file_paths
        self.input_len  = input_len
        self.output_len = output_len
        s = np.load(file_paths[0])
        self.npz_key = npz_key or list(s.files)[0]
        arr = s[self.npz_key]
        print(f"[data] npz shape per file: {arr.shape}")
        s.close()
        assert arr.shape[0] == 24, f"Expected time axis=24, got {arr.shape}"
        assert input_len + output_len <= 24

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        n = np.load(self.file_paths[idx])
        s = n[self.npz_key].astype(np.float32)
        n.close()
        return (torch.from_numpy(s[:self.input_len]),
                torch.from_numpy(s[self.input_len:self.input_len + self.output_len]))


def build_dataloaders(cfg):
    fps = sorted(glob.glob(os.path.join(cfg["data_dir"], "*.npz")))
    assert fps, f"No .npz files in '{cfg['data_dir']}'"
    print(f"[data] Found {len(fps)} files")

    ds  = RainfallNPZDataset(fps, cfg["input_len"], cfg["output_len"], cfg["npz_key"])
    N   = len(ds)
    ntr = int(0.70 * N)
    nv  = int(0.15 * N)
    nte = N - ntr - nv

    g = torch.Generator().manual_seed(cfg["seed"])
    tr_ds, v_ds, te_ds = random_split(ds, [ntr, nv, nte], generator=g)
    print(f"[data] train={len(tr_ds)}  val={len(v_ds)}  test={len(te_ds)}")

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=(cfg["device"] == "cuda"))
    return (DataLoader(tr_ds, shuffle=True, **kw),
            DataLoader(v_ds,  **kw),
            DataLoader(te_ds, **kw))


# ------------------------------------------------------------------------------
# EVOLUTION NETWORK  (pysteps LK — mirrors save_intensities_and_motion_hdf.py)
# ------------------------------------------------------------------------------

_ZR_A   = 223.0
_ZR_B   = 1.53
_ZERO   = -15.0
_THRESH = 0.1


def _to_dbz(rain_np: np.ndarray) -> np.ndarray:
    """(T, H, W) mm/hr -> dBZ, matching save_intensities_and_motion_hdf.py."""
    meta = {"transform": None, "zerovalue": _ZERO, "threshold": _THRESH, "unit": "mm/h"}
    rr, m2 = conversion.to_rainrate(rain_np, meta, zr_a=_ZR_A, zr_b=_ZR_B)
    dbz, _ = transformation.dB_transform(rr, m2)
    dbz[~np.isfinite(dbz)] = _ZERO
    return dbz


def _grid_cpu(H: int, W: int) -> torch.Tensor:
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    return torch.cat([xx.view(1, 1, H, W), yy.view(1, 1, H, W)], dim=1).float()


def compute_evo_prediction(
    rain_context: torch.Tensor,   # (B, input_len, H, W)  mm/hr
    output_len: int,
    method: str = "lk",
) -> torch.Tensor:
    """
    Linear-persistence evolution prediction via pysteps LK optical flow.

    Steps (per sample in the batch):
        1. Convert context frames to dBZ
        2. Run pysteps LK -> motion field (2, H, W)
        3. Warp last context frame with the motion field
        4. Repeat the warped frame output_len times

    Returns: (B, output_len, H, W)  mm/hr, same device as input.
    """
    B, T, H, W = rain_context.shape
    dev        = rain_context.device
    rain_np    = rain_context.detach().cpu().numpy()
    oflow      = pysteps_motion.get_method(method)
    grid       = _grid_cpu(H, W)

    evo_list = []
    for b in range(B):
        dbz    = _to_dbz(rain_np[b])                                            # (T,H,W)
        mf     = torch.from_numpy(oflow(dbz).astype(np.float32)).unsqueeze(0)  # (1,2,H,W)
        X0     = torch.from_numpy(rain_np[b, -1]).unsqueeze(0).unsqueeze(0)    # (1,1,H,W)
        vg     = grid.clone() - mf
        vg[:, 0] = 2.0 * vg[:, 0] / max(W - 1, 1) - 1.0
        vg[:, 1] = 2.0 * vg[:, 1] / max(H - 1, 1) - 1.0
        warped = F.grid_sample(X0, vg.permute(0, 2, 3, 1),
                               mode="bilinear", padding_mode="zeros",
                               align_corners=True).squeeze(0)                   # (1,H,W)
        evo_list.append(warped.expand(output_len, -1, -1))                      # (T,H,W)

    return torch.stack(evo_list, dim=0).to(dev)                                 # (B,T,H,W)


# ------------------------------------------------------------------------------
# CSI METRIC
# ------------------------------------------------------------------------------

class CSIMetric:
    """Critical Success Index accumulated over a full epoch."""

    def __init__(self, thresholds: list):
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        self._tp = {t: 0.0 for t in self.thresholds}
        self._fp = {t: 0.0 for t in self.thresholds}
        self._fn = {t: 0.0 for t in self.thresholds}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """pred, target: (B, T, H, W) or (B, H, W) in mm/hr."""
        if pred.dim() == 4:
            pred   = pred.mean(dim=1)
            target = target.mean(dim=1)
        for t in self.thresholds:
            p = pred   >= t
            q = target >= t
            self._tp[t] += ( p &  q).sum().item()
            self._fp[t] += ( p & ~q).sum().item()
            self._fn[t] += (~p &  q).sum().item()

    def compute(self) -> dict:
        out = {}
        for t in self.thresholds:
            denom = self._tp[t] + self._fp[t] + self._fn[t]
            out[f"csi@{t}"] = self._tp[t] / denom if denom > 0 else float("nan")
        return out


# ------------------------------------------------------------------------------
# LOSS FUNCTIONS
# ------------------------------------------------------------------------------

def adversarial_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    BCE with logits — numerically stable, no [0,1] requirement on y_hat.
    y_hat : raw logits from discriminator  (any range)
    y     : labels in {0, 1}
    """
    return F.binary_cross_entropy_with_logits(y_hat, y)


def grid_cell_regularizer(
    generated_samples: torch.Tensor,   # (gen_steps, B, T, H, W)
    batch_targets:     torch.Tensor,   # (B, T, H, W)
    generation_steps:  int,
) -> torch.Tensor:
    """
    Grid-cell regularizer from lightning.py.
    MaxPool2d applied directly to (B, T, H, W) treating T as channels.
    """
    mp     = nn.MaxPool2d(kernel_size=5, stride=2)
    pooled = [mp(generated_samples[i]) for i in range(generation_steps)]
    x_pred = torch.mean(torch.stack(pooled, dim=0), dim=0)
    return torch.mean(torch.abs(x_pred - mp(batch_targets)))


# ------------------------------------------------------------------------------
# TRAIN / EVAL EPOCH
# ------------------------------------------------------------------------------

def _sample_noise(batch_size: int, cfg: dict, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, cfg["latent_dim"],
                       cfg["h_noise"], cfg["h_noise"], device=device)


def _gen_forward(generator, x_before: torch.Tensor, cfg: dict) -> torch.Tensor:
    z = _sample_noise(x_before.shape[0], cfg, x_before.device).to(x_before.dtype)
    return generator(x_before, z)


def run_epoch(
    generator, discriminator,
    loader, csi_metric,
    opt_g, opt_d,
    device, cfg,
    training: bool = True,
) -> dict:
    generator.train(training)
    discriminator.train(training)
    csi_metric.reset()

    totals, n_batches = {}, 0
    rain_ch    = cfg["rain_channel"]
    gen_steps  = cfg["generation_steps"]
    output_len = cfg["output_len"]
    alpha      = cfg["alpha"]
    beta       = cfg["beta"]
    _first     = True

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)

            # Extract rain channel -> (B, T, H, W)
            rain_X = X[:, :, rain_ch, :, :] if X.dim() == 5 else X
            rain_Y = Y[:, :, rain_ch, :, :] if Y.dim() == 5 else Y

            if _first:
                print(f"[run_epoch] rain_X={rain_X.shape}  rain_Y={rain_Y.shape}")
                _first = False

            B = rain_X.shape[0]

            # Evolution prediction: pysteps LK warp, no data leakage
            with torch.no_grad():
                evo_pred = compute_evo_prediction(
                    rain_X, output_len, method=cfg["motion_method"]
                )   # (B, output_len, H, W)

            # Build x_before = cat(raw_context, evo_prediction)  (B, 10, H, W)
            # Matches lightning.py: x_before = cat([inputs[k] for k in inputs])
            # Generator extracts evo signal as x[:, -n_after:] internally.
            x_before = torch.cat([rain_X, evo_pred], dim=1)
            x_after  = rain_Y

            # Discriminator real sequence: cat(raw_context, real_target)
            real  = torch.cat([rain_X, x_after], dim=1)
            valid = torch.ones( B, 1, device=device, dtype=x_before.dtype)
            fake  = torch.zeros(B, 1, device=device, dtype=x_before.dtype)

            # ── Generator step ─────────────────────────────────────────────────
            if training:
                opt_g.zero_grad()
                opt_d.zero_grad()

            preds             = [_gen_forward(generator, x_before, cfg)
                                 for _ in range(gen_steps)]
            generated_samples = torch.stack(preds, dim=0)   # (gen_steps, B, T, H, W)

            disc_scores = torch.stack([
                adversarial_loss(
                    discriminator(torch.cat([rain_X, generated_samples[i]], dim=1)),
                    valid
                ) for i in range(gen_steps)
            ])
            g_adv = disc_scores.mean()
            cell  = grid_cell_regularizer(generated_samples, x_after, gen_steps)
            g_tot = alpha * g_adv + beta * cell

            if training:
                g_tot.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                opt_g.step()

            # ── Discriminator step ─────────────────────────────────────────────
            if training:
                opt_d.zero_grad()

            with torch.no_grad():
                fake_pred = _gen_forward(generator, x_before, cfg)
            fake_seq = torch.cat([rain_X, fake_pred], dim=1).detach()

            real_loss = adversarial_loss(discriminator(real),     valid)
            fake_loss = adversarial_loss(discriminator(fake_seq), fake)
            d_loss    = (real_loss + fake_loss) / 2

            if training:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                opt_d.step()

            # ── CSI ────────────────────────────────────────────────────────────
            with torch.no_grad():
                csi_metric.update(generated_samples.mean(0).detach(), x_after)

            for k, v in {
                "g_loss": g_tot.item(), "g_adv": g_adv.item(), "cell": cell.item(),
                "d_loss": d_loss.item(), "d_real": real_loss.item(),
                "d_fake": fake_loss.item(),
            }.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

    avg = {k: v / n_batches for k, v in totals.items()}
    avg.update(csi_metric.compute())
    return avg


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def _fmt(d: dict) -> str:
    return "  ".join(
        f"{k}={'nan' if isinstance(v, float) and math.isnan(v) else f'{v:.4f}'}"
        for k, v in d.items()
    )


def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    device = CFG["device"]
    print(f"[train] Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(CFG)

    # channel_in = input_len + output_len (raw context + evo prediction)
    channel_in = CFG["input_len"] + CFG["output_len"]

    generator = NowcasnetGenerator(
        channel_in=channel_in,
        latent_dim=CFG["latent_dim"],
        n_after=CFG["output_len"],
    ).to(device)

    discriminator = TemporalDiscriminator(
        in_channel=channel_in,
        img_size=CFG["img_size"],
    ).to(device)

    ng = sum(p.numel() for p in generator.parameters()     if p.requires_grad)
    nd = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"[model] Generator: {ng:,}  Discriminator: {nd:,}")

    opt_g = torch.optim.Adam(generator.parameters(),
                             lr=CFG["glr"], betas=(CFG["b1"], CFG["b2"]))
    opt_d = torch.optim.Adam(discriminator.parameters(),
                             lr=CFG["dlr"], betas=(CFG["b1"], CFG["b2"]))

    csi = CSIMetric(CFG["csi_thresholds"])
    os.makedirs(CFG["save_dir"], exist_ok=True)
    best_g = float("inf")
    best_p = os.path.join(CFG["save_dir"], "best_generator.pt")

    for epoch in range(1, CFG["num_epochs"] + 1):
        tr = run_epoch(generator, discriminator, train_loader,
                       csi, opt_g, opt_d, device, CFG, training=True)
        va = run_epoch(generator, discriminator, val_loader,
                       csi, opt_g, opt_d, device, CFG, training=False)

        print(f"\n[Epoch {epoch:03d}/{CFG['num_epochs']}]")
        print(f"  TRAIN  loss: {_fmt({k:v for k,v in tr.items() if not k.startswith('csi')})}")
        print(f"         CSI:  {_fmt({k:v for k,v in tr.items() if     k.startswith('csi')})}")
        print(f"  VAL    loss: {_fmt({k:v for k,v in va.items() if not k.startswith('csi')})}")
        print(f"         CSI:  {_fmt({k:v for k,v in va.items() if     k.startswith('csi')})}")

        torch.save({
            "epoch":         epoch,
            "generator":     generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g":         opt_g.state_dict(),
            "opt_d":         opt_d.state_dict(),
            "val_g_loss":    va["g_loss"],
            "cfg":           CFG,
        }, os.path.join(CFG["save_dir"], f"epoch_{epoch:03d}.pt"))

        if va["g_loss"] < best_g:
            best_g = va["g_loss"]
            torch.save(generator.state_dict(), best_p)
            print(f"  ✓ New best {best_g:.4f}  ->  {best_p}")

    print("\n[test] Loading best generator ...")
    generator.load_state_dict(torch.load(best_p, map_location=device))
    te = run_epoch(generator, discriminator, test_loader,
                   csi, opt_g, opt_d, device, CFG, training=False)
    print(f"[test] loss: {_fmt({k:v for k,v in te.items() if not k.startswith('csi')})}")
    print(f"[test] CSI:  {_fmt({k:v for k,v in te.items() if     k.startswith('csi')})}")


if __name__ == "__main__":
    main()
