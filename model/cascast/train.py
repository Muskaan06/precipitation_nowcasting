"""
train.py
========
Training script for CasCast on satellite rainfall data — structured the same
way as ../tupann/train.py and ../earthformer/train.py:

  - NPZDataset        : one sample per .npz file, explicit 'array' key
  - preprocess_rain() : clamp -> ln(1+R) / ln(1+128) -> [0, 1]
  - LightningModule    : self-contained, no repo base classes
  - main()            : argparse CLI + pl.Trainer + ModelCheckpoint

CasFormer (model_arch.py, architecture unchanged) is a DiT that predicts the
noise added to the target rain sequence, conditioned on a "coarse
prediction" of the same sequence. The coarse prediction is supplied by a
frozen Earthformer model (see diffusion_utils.py). Our dataset uses a 30-min
timestep (unlike SEVIR's 5-min), so instead of the SEVIR-pretrained
earthformer_sevir.pt we default to our own Earthformer checkpoint trained on
this dataset at 4 input / 6 output frames
(../earthformer/checkpoints/lightning_logs_4_6). Falls back to a persistence
baseline (repeat the last observed frame) if `--coarse_ckpt none` is passed,
so the pipeline stays runnable standalone.

Usage
-----
    python train.py --data_dir /path/to/data --save_dir ./checkpoint
    python train.py --data_dir /path/to/data --coarse_ckpt none   # persistence baseline
"""

import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from nowcast.precipitation_nowcasting.model.cascast.model_arch import CasFormer
from nowcast.precipitation_nowcasting.model.cascast.diffusion_utils import (
    GaussianDiffusion,
    load_coarse_forecaster,
    run_coarse_forecaster,
    persistence_coarse,
    DEFAULT_COARSE_CKPT,
    EARTHFORMER_INPUT_LENGTH,
    EARTHFORMER_TARGET_LENGTH,
)


# =============================================================================
# 1. Rain Preprocessing (same as tupann/train.py)
# =============================================================================

RAIN_MAX_MMHR: float = 128.0
_LOG_NORM_DENOM: float = float(np.log1p(RAIN_MAX_MMHR))


def preprocess_rain(x: np.ndarray) -> np.ndarray:
    """Normalises raw IMERG mm/hr to [0, 1] log-rain space."""
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    x = np.clip(x, 0.0, RAIN_MAX_MMHR)
    x = np.log1p(x) / _LOG_NORM_DENOM
    return x


# =============================================================================
# 2. Dataset
# =============================================================================

class NPZDataset(Dataset):
    def __init__(self, files, input_length=EARTHFORMER_INPUT_LENGTH, target_length=EARTHFORMER_TARGET_LENGTH):
        self.files = files
        self.input_length = input_length
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=True) as f:
            arr = np.array(f['array'], dtype=np.float32).copy()

        rain = arr[:, 0, :, :]
        rain = preprocess_rain(rain)

        x = torch.from_numpy(rain[:self.input_length]).contiguous()
        y = torch.from_numpy(rain[self.input_length:self.input_length + self.target_length]).contiguous()
        return x, y


# =============================================================================
# 3. CasCast Lightning module (diffusion training of CasFormer)
# =============================================================================

class CasCastLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, input_len=EARTHFORMER_INPUT_LENGTH, target_length=EARTHFORMER_TARGET_LENGTH,
                 img_size=112, patch_size=4, hidden_size=384, enc_hidden_size=128,
                 enc_depth=4, latent_depth=8, num_heads=6, mlp_ratio=4.0,
                 diffusion_timesteps=1000, coarse_ckpt=DEFAULT_COARSE_CKPT):
        super().__init__()
        self.save_hyperparameters()

        castformer_config = dict(
            input_size=img_size,
            patch_size=patch_size,
            in_channels=2,                  # noisy target frame + coarse cond frame, per frame
            out_channels=target_length,     # unpatchify emits (t c) stacked in the channel dim, c=1/frame
            hidden_size=hidden_size,
            enc_hidden_size=enc_hidden_size,
            split_num=target_length,
            enc_depth=enc_depth,
            latent_depth=latent_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.castformer = CasFormer(arch="DiT-custom", config=castformer_config)
        self.diffusion = GaussianDiffusion(timesteps=diffusion_timesteps)

        self.coarse_model = load_coarse_forecaster(coarse_ckpt, device="cpu")
        if self.coarse_model is not None:
            self.add_module("coarse_model", self.coarse_model)

    def _coarse_prediction(self, x):
        if self.coarse_model is not None:
            with torch.no_grad():
                return run_coarse_forecaster(self.coarse_model, x)
        return persistence_coarse(x, self.hparams.target_length)

    def _step(self, batch, split):
        x, y = batch
        self.diffusion.to(y.device)

        coarse = self._coarse_prediction(x)

        y_ = y.unsqueeze(2)             # (B, T, 1, H, W)
        coarse_ = coarse.unsqueeze(2)   # (B, T, 1, H, W)

        t = torch.randint(0, self.hparams.diffusion_timesteps, (y.shape[0],), device=y.device)
        x_noisy, noise = self.diffusion.q_sample(y_, t)

        noise_pred = self.castformer(x=x_noisy, timesteps=t, cond=coarse_)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=(split == "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.castformer.parameters(), lr=self.hparams.lr)


# =============================================================================
# 4. Execution block
# =============================================================================

def build_loaders(data_dir, train_split, batch_size, num_workers, input_length, target_length):
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_split)

    train_loader = DataLoader(
        NPZDataset(all_files[:split_idx], input_length, target_length),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        NPZDataset(all_files[split_idx:], input_length, target_length),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader = build_loaders(
        args.data_dir, args.train_split, args.batch_size, args.num_workers,
        args.input_length, args.target_length,
    )

    coarse_ckpt = None if args.coarse_ckpt.lower() == "none" else args.coarse_ckpt
    model = CasCastLightning(
        lr=args.lr, input_len=args.input_length, target_length=args.target_length,
        img_size=args.img_size, patch_size=args.patch_size, hidden_size=args.hidden_size,
        enc_hidden_size=args.enc_hidden_size, enc_depth=args.enc_depth,
        latent_depth=args.latent_depth, num_heads=args.num_heads,
        diffusion_timesteps=args.diffusion_timesteps, coarse_ckpt=coarse_ckpt,
    )

    ckpt_cb = ModelCheckpoint(dirpath=args.save_dir, monitor="val_loss", filename="cascast_best",
                              mode="min", save_last=True, auto_insert_metric_name=False)
    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu", devices=args.devices, callbacks=[ckpt_cb])
    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", default ="/media/kpdubey/8.0 TB Volume/Muskaan/nowcast/IMC_Combined")
    parser.add_argument("--save_dir", "-sd", default="./checkpoint")
    parser.add_argument("--coarse_ckpt", "-cckpt", default=DEFAULT_COARSE_CKPT, type=str,
                        help="Frozen Earthformer checkpoint used as the deterministic coarse "
                             "forecaster that CasFormer refines. Defaults to our own Earthformer "
                             "checkpoint trained on this dataset (4 in / 6 out, checkpoints/"
                             "lightning_logs_4_6). Pass 'none' to fall back to a persistence baseline.")
    parser.add_argument("--train_split", "-ts", default=0.7, type=float)
    parser.add_argument("--batch_size", "-bs", default=16, type=int)
    parser.add_argument("--num_workers", "-nw", default=0, type=int)
    parser.add_argument("--max_epochs", "-e", default=5, type=int)
    parser.add_argument("--devices", "-dv", default=1, type=int)
    parser.add_argument("--lr", "-lr", default=1e-4, type=float)
    parser.add_argument("--input_length", "-il", default=EARTHFORMER_INPUT_LENGTH, type=int)
    parser.add_argument("--target_length", "-tl", default=EARTHFORMER_TARGET_LENGTH, type=int)
    parser.add_argument("--img_size", "-is", default=112, type=int)
    parser.add_argument("--patch_size", "-ps", default=4, type=int)
    parser.add_argument("--hidden_size", default=384, type=int)
    parser.add_argument("--enc_hidden_size", default=128, type=int)
    parser.add_argument("--enc_depth", default=4, type=int)
    parser.add_argument("--latent_depth", default=8, type=int)
    parser.add_argument("--num_heads", default=6, type=int)
    parser.add_argument("--diffusion_timesteps", default=1000, type=int)
    args = parser.parse_args()
    main(args)
