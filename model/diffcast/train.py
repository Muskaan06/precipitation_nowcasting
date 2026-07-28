import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from model_arch import DiffCastBackbone
from patch_utils import mmhr_to_intensity_with_normalize

# ── Data utilities ────────────────────────────────────────────────────────────

class NPZDataset(Dataset):
    """
    Unlike the other backends' channels-last/channel-less conventions,
    DiffCast's backbones consume (T, C, H, W) — channel at axis 1 per sample.
    """
    def __init__(self, files, T_in=8, T_out=8):
        self.files = files
        self.T_in = T_in
        self.T_out = T_out

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:, 0, :, :], dtype=np.float32)  # (T_total, H, W)

        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.expand_dims(arr, axis=1)  # (T_total, 1, H, W)
        x_raw = torch.from_numpy(arr[:self.T_in]).float()
        y_raw = torch.from_numpy(arr[self.T_in:self.T_in + self.T_out]).float()

        return mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)

# ── Lightning module ──────────────────────────────────────────────────────────

class DiffCastLightning(pl.LightningModule):
    """
    Backbone-only training (PhyDNet or SimVP) — the official DiffCast repo
    does not ship a working diffusion-training loss, only inference/sampling
    against a pretrained checkpoint (see evaluate.py Mode B).

    - input : (B, T_in, 1, 112, 112)
    - output: (B, T_out, 1, 112, 112)
    """
    def __init__(self, lr=1e-3, backbone="phydnet", img_width=112, T_in=8, T_out=8):
        super().__init__()
        self.save_hyperparameters()

        self.model = DiffCastBackbone(
            backbone=backbone,
            C=1,
            H=img_width,
            W=img_width,
            T_in=T_in,
            T_out=T_out,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def forward(self, x):
        return self.model.predict(x, compute_loss=False)[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Each backbone's own canonical training loss (PhyDNet: per-step MSE +
        # moment-matrix physics constraint; SimVP: plain MSE) — used directly,
        # not reimplemented.
        pred, loss = self.model.predict(x, frames_gt=y, compute_loss=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred, loss = self.model.predict(x, frames_gt=y, compute_loss=True)
        # PhyDNet's raw `loss` includes the physics term and isn't directly
        # comparable to SimVP's plain MSE, so checkpoint selection uses a
        # plain MSE computed separately here, not the backbone's own loss.
        val_mse = torch.mean((pred - y) ** 2)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", val_mse, prog_bar=True)
        return val_mse

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    all_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir) if f.endswith('.npz')
    ])
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * args.train_split)

    train_loader = DataLoader(
        NPZDataset(all_files[:split_idx], T_in=args.T_in, T_out=args.T_out),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        NPZDataset(all_files[split_idx:], T_in=args.T_in, T_out=args.T_out),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_mse",
        filename=f"diffcast-{args.backbone}-best-{{epoch}}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(
        DiffCastLightning(
            lr=args.lr,
            backbone=args.backbone,
            img_width=args.img_width,
            T_in=args.T_in,
            T_out=args.T_out,
        ),
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      "-d",   required=True, type=str)
    parser.add_argument("--train_split",   "-ts",  default=0.7,   type=float)
    parser.add_argument("--batch_size",    "-bs",  default=8,     type=int)
    parser.add_argument("--num_workers",   "-nw",  default=1,     type=int)
    parser.add_argument("--max_epochs",    "-e",   default=5,     type=int)
    parser.add_argument("--devices",       "-dv",  default=1,     type=int)
    parser.add_argument("--lr",            "-lr",  default=1e-3,  type=float)
    parser.add_argument("--save_dir",      "-sd",  default="../../model_weights/diffcast", type=str)
    # DiffCast backbone args
    parser.add_argument("--backbone",      choices=["phydnet", "simvp"], default="phydnet",
                        help="phydnet matches the one public pretrained diffusion checkpoint; "
                             "simvp requires T_out %% T_in == 0.")
    parser.add_argument("--T_in",          default=8,     type=int)
    parser.add_argument("--T_out",         default=8,     type=int)
    parser.add_argument("--img_width",     default=112,   type=int)
    args = parser.parse_args()
    main(args)
