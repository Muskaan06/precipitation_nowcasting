"""
finetune.py
===========
Fine-tuning script for TUPANN on 112×112 satellite rainfall data.

Aligns exactly with evaluate.py on:
  - RAIN_MAX_MMHR = 60.0
  - arr[4:, 0, :, :] frame slice
  - input_length=9 / target_length=18 (parametric, not hardcoded)
  - Dataset returns only (x, y) — no flow/intensity auxiliaries
  - Model instantiation with arch args that TUPANN actually accepts:
      (input_len, target_length, img_size, embed_dim, reduc_factor,
       channels, dropout, maxvit_dim, maxvit_depth, maxvit_heads, window_size)
    NOTE: maxvit_head_dim and pad_size are NOT constructor args in model_arch.py
          and are therefore absent from the TUPANN() call here.
  - Checkpoint loading: _load_stub_modules() + shape-mismatch filtering

Freeze strategy (default):
  - autoencoder frozen  (matches Stage-2 of train.py)
  - latent_model trained
  Pass --unfreeze_autoencoder to also update autoencoder weights.

Usage
-----
    # Fine-tune (latent_model only, autoencoder frozen):
    python finetune.py \\
        --data_dir /path/to/IMC_Combined \\
        --ckpt     /path/to/0--best-epoch=7.ckpt \\
        --save_dir ./ft_checkpoints \\
        --max_epochs 20

    # Also unfreeze the autoencoder:
    python finetune.py ... --unfreeze_autoencoder
"""

import os
import random
import sys
import types

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset

from model_arch import TUPANN, AutoencoderKL


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing  — identical to evaluate.py and train.py (with corrected R_max)
# ═══════════════════════════════════════════════════════════════════════════════

RAIN_MAX_MMHR: float = 60.0          # matches evaluate.py (train.py used 128.0)
_LOG_DENOM:    float = float(np.log1p(RAIN_MAX_MMHR))


def preprocess_rain(x: np.ndarray) -> np.ndarray:
    """Normalise raw IMERG mm/hr → [0, 1] log-rain space."""
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    x = np.clip(x, 0.0, RAIN_MAX_MMHR)
    return np.log1p(x) / _LOG_DENOM


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset  — identical to evaluate.py
# ═══════════════════════════════════════════════════════════════════════════════

class NPZDataset(Dataset):
    """
    Loads .npz files produced by the IMC_Combined pipeline.

    Frame slice  : arr[4:, 0, :, :]  — skips the first 4 channels (matches
                   evaluate.py; train.py used arr[:, 0, :, :]).
    Returns      : (x, y) tensors only — no optical-flow / intensity GT needed
                   for fine-tuning.
    """
    def __init__(self, files, input_length: int = 9, target_length: int = 18):
        self.files         = files
        self.input_length  = input_length
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=True) as f:
            arr = np.array(f['array'], dtype=np.float32).copy()

        rain   = arr[4:, 0, :, :]          # (T', H, W) — skip first 4 frames
        rain   = preprocess_rain(rain)

        needed = self.input_length + self.target_length
        if rain.shape[0] < needed:
            pad  = np.zeros((needed - rain.shape[0], *rain.shape[1:]), dtype=np.float32)
            rain = np.concatenate([rain, pad], axis=0)

        x = torch.from_numpy(rain[:self.input_length]).contiguous()
        y = torch.from_numpy(
                rain[self.input_length : self.input_length + self.target_length]
            ).contiguous()
        return x, y                         # (T_in, H, W), (T_out, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers  — identical to evaluate.py
# ═══════════════════════════════════════════════════════════════════════════════

def _load_stub_modules():
    """
    Register stub modules so torch.load can unpickle the Lightning checkpoint
    without the original training environment installed.
    """
    class LazyStub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('_'):
                raise AttributeError(name)
            cls = type(name, (object,), {'__init__': lambda s, *a, **k: None})
            setattr(self, name, cls)
            return cls

    for mod_name in [
        'src', 'src.models', 'src.models.metnet', 'src.models.metnet.metnet',
        'src.utils', 'src.utils.lightning_utils', 'src.utils.train_utils',
        'src.utils.metrics', 'src.utils.losses',
        'lightning', 'lightning.pytorch',
        'precipitation_nowcasting',
        'precipitation_nowcasting.model',
        'precipitation_nowcasting.model.tupann',
        'precipitation_nowcasting.model.tupann.autoenc_lightning',
        'precipitation_nowcasting.model.tupann.utils',
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = LazyStub(mod_name)


def load_pretrained_weights(
    model:            nn.Module,
    ckpt_path:        str,
    autoenc_ckpt_path: str = "",
    device:           torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Load pretrained weights into an already-instantiated TUPANN.

    Replicates the shape-safe loading logic from evaluate.py:
      1. Optionally load a separate autoencoder checkpoint first.
      2. Load the main checkpoint, skipping any shape-mismatched buffers.
         (The grid buffer stored in the checkpoint is (1,2,256,256) while
          our model registers (1,2,img_size,img_size) — this key is silently
          skipped so strict=False does not raise.)
    """
    _load_stub_modules()

    # ── Step 1: optional separate autoencoder weights ─────────────────────────
    if autoenc_ckpt_path and os.path.isfile(autoenc_ckpt_path):
        print(f"  [ckpt] Loading autoencoder from: {autoenc_ckpt_path}")
        ae_raw = torch.load(autoenc_ckpt_path, map_location=device, weights_only=False)
        ae_sd  = ae_raw["state_dict"] if "state_dict" in ae_raw else ae_raw

        # Re-prefix bare keys so they match TUPANN's autoencoder.* namespace
        ae_sd_reprefixed = {}
        for k, v in ae_sd.items():
            new_k = k if k.startswith("autoencoder.") else f"autoencoder.{k}"
            ae_sd_reprefixed[new_k] = v

        model_sd = model.state_dict()
        filtered = {k: v for k, v in ae_sd_reprefixed.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"         Autoencoder loaded : {len(filtered)} tensors "
              f"| missing={len(missing)}  unexpected={len(unexpected)}")

    # ── Step 2: main checkpoint ───────────────────────────────────────────────
    print(f"  [ckpt] Loading main model from: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        sd = dict(raw_ckpt["state_dict"])

        model_sd = model.state_dict()
        filtered, skipped = {}, []
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                skipped.append(
                    f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}"
                )
            else:
                filtered[k] = v

        if skipped:
            print(f"  [ckpt] Skipped {len(skipped)} shape-mismatched buffer(s):")
            for s in skipped:
                print(f"         {s}")

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        epoch = raw_ckpt.get("epoch", "?")
        print(f"  [ckpt] Loaded  <- {ckpt_path}  (epoch {epoch})")
        print(f"         Missing   : {len(missing)}")
        print(f"         Unexpected: {len(unexpected)}")
        if missing:
            print(f"         First 10 missing: {missing[:10]}")
    else:
        model.load_state_dict(raw_ckpt, strict=False)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Lightning Module
# ═══════════════════════════════════════════════════════════════════════════════

class TUPANNFinetune(pl.LightningModule):
    """
    Fine-tunes a pretrained TUPANN on your satellite dataset.

    Training objective:
        L = MSE(pred, y)   — regression loss on future normalised rain frames,
                             consistent with the Stage-2 l_target in train.py.

    Freeze strategy (default):
        - autoencoder frozen  — mirrors Stage-2 in train.py where the VED is
                                loaded and frozen before training latent_model.
        - latent_model trained

    Pass --unfreeze_autoencoder to also update autoencoder weights.

    TUPANN constructor args used here match model_arch.py exactly:
        input_len, target_length, img_size, embed_dim, reduc_factor,
        channels, dropout, maxvit_dim, maxvit_depth, maxvit_heads, window_size.

    NOTE: maxvit_head_dim is NOT a TUPANN constructor argument (it is an
    implicit consequence of maxvit_dim // maxvit_heads inside LatentMaxViT).
    pad_size is also not a model arg; it was used externally in evaluate.py
    to pad inputs before passing to the model — that logic lives in the caller,
    not in TUPANN itself. Both are removed from the model instantiation here.
    """
    def __init__(
        self,
        ckpt:               str,
        autoenc_ckpt:       str   = "",
        lr:                 float = 1e-4,
        weight_decay:       float = 1e-5,
        # ── Data dims ──────────────────────────────────────────────────────
        input_length:       int   = 9,
        target_length:      int   = 18,
        img_size:           int   = 112,
        # ── Model arch — must match pretrained checkpoint exactly ──────────
        embed_dim:          int   = 4,
        reduc_factor:       int   = 4,
        channels:           int   = 128,
        dropout:            float = 0.2,
        maxvit_dim:         int   = 64,
        maxvit_depth:       int   = 4,
        maxvit_heads:       int   = 32,   # passed as `heads` to LatentMaxViT
        window_size:        int   = 8,    # passed as `window_size` to LatentMaxViT
        # ── Fine-tune strategy ─────────────────────────────────────────────
        unfreeze_autoencoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Build model — only args accepted by TUPANN.__init__ are passed ──
        # See model_arch.py:
        #   TUPANN(input_len, target_length, img_size, embed_dim, reduc_factor,
        #          channels, dropout, maxvit_dim, maxvit_depth, maxvit_heads,
        #          window_size)
        # maxvit_head_dim is NOT a parameter; pad_size is NOT a parameter.
        self.model = TUPANN(
            input_len     = input_length,
            target_length = target_length,
            img_size      = img_size,
            embed_dim     = embed_dim,
            reduc_factor  = reduc_factor,
            channels      = channels,
            dropout       = dropout,
            maxvit_dim    = maxvit_dim,
            maxvit_depth  = maxvit_depth,
            maxvit_heads  = maxvit_heads,
            window_size   = window_size,
        )

        # ── Load pretrained weights (shape-safe, stub-module-aware) ─────────
        load_pretrained_weights(
            self.model, ckpt, autoenc_ckpt,
            device=torch.device("cpu"),  # Lightning moves tensors to correct device
        )

        # ── Freeze / unfreeze autoencoder (mirrors Stage-2 in train.py) ─────
        for p in self.model.autoencoder.parameters():
            p.requires_grad_(unfreeze_autoencoder)

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.model.parameters())
        print(f"\n  Trainable params : {n_trainable:,} / {n_total:,}")

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        # TUPANN.forward returns (B, target_length, H, W)
        return self.model(x)

    # ── Shared step ───────────────────────────────────────────────────────────
    def _step(self, batch, split: str):
        x, y = batch                            # (B, T_in, H, W), (B, T_out, H, W)
        pred = self(x)                          # (B, T_out, H, W)

        # Clamp to valid normalised range before computing loss — mirrors
        # evaluate.py's pred.clamp(0.0, 1.0) in collect_predictions().
        pred = pred.clamp(0.0, 1.0)

        # MSE on normalised [0,1] values — matches evaluate.py metric and
        # Stage-2 l_target in train.py.
        loss = nn.functional.mse_loss(pred, y)
        self.log(f"{split}/loss", loss, prog_bar=True, sync_dist=(split != "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    def configure_optimizers(self):
        # Only optimise parameters that require gradients (respects freeze flag).
        # Mirrors train.py Stage-2 which only passes latent_model.parameters().
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable,
            lr           = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
            betas        = (0.9, 0.999),
        )
        # Deferred T_max so scheduler is built after trainer attaches
        # (avoids AttributeError: 'NoneType' has no attribute 'max_epochs').
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max   = self.trainer.max_epochs,
            eta_min = self.hparams.lr * 1e-2,
        )
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader builder  — matches evaluate.py split logic (seed-stable shuffle)
# ═══════════════════════════════════════════════════════════════════════════════

def build_loaders(data_dir, train_split, batch_size, num_workers,
                  input_length, target_length, seed=42):
    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir) if f.endswith('.npz')
    ])
    random.seed(seed)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_split)

    train_ds = NPZDataset(all_files[:split_idx], input_length, target_length)
    val_ds   = NPZDataset(all_files[split_idx:], input_length, target_length)

    print(f"  Total  files  : {len(all_files)}")
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n── Fine-tune configuration ──────────────────────────────────────")
    print(f"  Checkpoint       : {args.ckpt}")
    print(f"  Autoenc ckpt     : {args.autoenc_ckpt or '(none)'}")
    print(f"  Data dir         : {args.data_dir}")
    print(f"  img_size         : {args.img_size}")
    print(f"  input_length     : {args.input_length}   target_length : {args.target_length}")
    print(f"  channels         : {args.channels}   embed_dim : {args.embed_dim}")
    print(f"  maxvit_dim       : {args.maxvit_dim}   maxvit_depth : {args.maxvit_depth}")
    print(f"  maxvit_heads     : {args.maxvit_heads}   window_size : {args.window_size}")
    print(f"  Unfreeze AE      : {args.unfreeze_autoencoder}")
    print(f"  LR               : {args.lr}   WD : {args.weight_decay}")
    print("─────────────────────────────────────────────────────────────────\n")

    train_loader, val_loader = build_loaders(
        data_dir      = args.data_dir,
        train_split   = args.train_split,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        input_length  = args.input_length,
        target_length = args.target_length,
        seed          = args.seed,
    )

    model = TUPANNFinetune(
        ckpt                 = args.ckpt,
        autoenc_ckpt         = args.autoenc_ckpt,
        lr                   = args.lr,
        weight_decay         = args.weight_decay,
        input_length         = args.input_length,
        target_length        = args.target_length,
        img_size             = args.img_size,
        embed_dim            = args.embed_dim,
        reduc_factor         = args.reduc_factor,
        channels             = args.channels,
        dropout              = args.dropout,
        maxvit_dim           = args.maxvit_dim,
        maxvit_depth         = args.maxvit_depth,
        maxvit_heads         = args.maxvit_heads,
        window_size          = args.window_size,
        unfreeze_autoencoder = args.unfreeze_autoencoder,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath    = args.save_dir,
            monitor    = "val/loss",
            filename   = "tupann_ft-{epoch:03d}-{val/loss:.4f}",
            mode       = "min",
            save_last  = True,
            save_top_k = 3,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs        = args.max_epochs,
        accelerator       = "gpu",
        devices           = args.devices,
        callbacks         = callbacks,
        gradient_clip_val = 1.0,       # stabilises fine-tuning
        log_every_n_steps = 10,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"\n[Done] Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fine-tune TUPANN on satellite data")

    # ── Paths ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",    "-d",  default="/home/muskaan06/Desktop/Research/nowcasting/datasets/selected_sets_2024_SI_112", type=str)
    p.add_argument("--ckpt",               default="/home/muskaan06/Desktop/Research/nowcasting/precipitation_nowcasting/model/tupann/model_weights/model_weights/tupann/0--best-epoch=7.ckpt", type=str,
                   help="Path to pretrained TUPANN checkpoint")
    p.add_argument("--autoenc_ckpt",       default="/home/muskaan06/Desktop/Research/nowcasting/precipitation_nowcasting/model/tupann/model_weights/model_weights/tupann_autoenc/0--best-epoch=85.ckpt", type=str,
                   help="Optional separate autoencoder checkpoint")
    p.add_argument("--save_dir",    "-sd", default="./ft_checkpoints", type=str)

    # ── Data ───────────────────────────────────────────────────────────────────
    p.add_argument("--input_length",       default=9,    type=int)
    p.add_argument("--target_length",      default=18,   type=int)
    p.add_argument("--img_size",           default=112,  type=int)
    p.add_argument("--train_split", "-ts", default=0.7,  type=float)
    p.add_argument("--batch_size",  "-bs", default=2,    type=int)
    p.add_argument("--num_workers", "-nw", default=0,    type=int)
    p.add_argument("--seed",               default=42,   type=int)

    # ── Model arch (must match pretrained checkpoint) ──────────────────────────
    # These map directly to TUPANN.__init__ parameters in model_arch.py.
    # maxvit_head_dim is excluded — it is not a TUPANN constructor parameter.
    # pad_size      is excluded — it is not a TUPANN constructor parameter.
    p.add_argument("--embed_dim",          default=4,    type=int)
    p.add_argument("--reduc_factor",       default=4,    type=int)
    p.add_argument("--channels",           default=128,  type=int)
    p.add_argument("--dropout",            default=0.2,  type=float)
    p.add_argument("--maxvit_dim",         default=64,   type=int)
    p.add_argument("--maxvit_depth",       default=4,    type=int)
    p.add_argument("--maxvit_heads",       default=32,   type=int)
    p.add_argument("--window_size",        default=8,    type=int)

    # ── Fine-tune strategy ─────────────────────────────────────────────────────
    p.add_argument("--unfreeze_autoencoder", action="store_true",
                   help="Also update autoencoder weights (default: frozen)")

    # ── Training ───────────────────────────────────────────────────────────────
    p.add_argument("--max_epochs", "-e",   default=5,    type=int)
    p.add_argument("--devices",    "-dv",  default=1,    type=int)
    p.add_argument("--lr",                 default=1e-4, type=float)
    p.add_argument("--weight_decay",       default=1e-5, type=float)

    main(p.parse_args())