import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from earthformer import EarthformerModel
from csi_eval import soft_csi_loss, hard_csi


class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        arr = arr[arr.files[0]]           # (24, 3, 112, 112)
        arr = np.nan_to_num(arr, nan=0.0) # no-rain pixels → 0

        # permute to channels-last for Earthformer: (T, C, H, W) -> (T, H, W, C)
        x = torch.tensor(arr[:4],   dtype=torch.float32).permute(0, 2, 3, 1)  # (4, 112, 112, 3)
        y = torch.tensor(arr[4:10], dtype=torch.float32).permute(0, 2, 3, 1)  # (6, 112, 112, 3)

        return x, y


class EarthformerLightning(EarthformerModel):
    def __init__(self, lr=1e-4, weight_decay=0.0, threshold=1.0, **kwargs):
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        self.threshold = threshold

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = soft_csi_loss(pred, y, threshold=self.threshold)  # differentiable CSI for backprop
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = soft_csi_loss(pred, y, threshold=self.threshold)
        csi  = hard_csi(pred, y, threshold=self.threshold)       # hard CSI for monitoring
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_CSI",  csi,  prog_bar=True)


def main(args):
    all_files = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.npz')])
    random.shuffle(all_files)
    split_idx = int(len(all_files) * args.train_split)

    train_loader = DataLoader(NPZDataset(all_files[:split_idx]), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(NPZDataset(all_files[split_idx:]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # monitor hard CSI on validation (higher is better)
    checkpoint_cb = ModelCheckpoint(monitor="val_CSI", filename="earthformer-best-{epoch}", save_top_k=3, mode="max")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(EarthformerLightning(lr=args.lr, threshold=args.threshold), train_loader, val_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    "-d",  required=True, type=str)
    parser.add_argument("--train_split", "-ts", default=0.8,   type=float)
    parser.add_argument("--batch_size",  "-bs", default=2,     type=int)
    parser.add_argument("--num_workers", "-nw", default=4,     type=int)
    parser.add_argument("--max_epochs",  "-e",  default=50,    type=int)
    parser.add_argument("--devices",     "-dv", default=1,     type=int)
    parser.add_argument("--lr",          "-lr", default=1e-4,  type=float)
    parser.add_argument("--threshold",   "-t",  default=1.0,   type=float)
    args = parser.parse_args()
    main(args)