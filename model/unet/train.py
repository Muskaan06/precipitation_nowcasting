import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from unet import UNet


class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        arr = arr[arr.files[0]]           # (24, 3, 112, 112)
        arr = np.nan_to_num(arr, nan=0.0) # no rain pixels → 0

        x = torch.tensor(arr[:4],   dtype=torch.float32).reshape(12, 112, 112)
        y = torch.tensor(arr[4:10], dtype=torch.float32).reshape(18, 112, 112)

        return x, y


class UNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model   = UNet(channels_in=12, channels_out=18)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def main(args):
    all_files = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.npz')])
    random.shuffle(all_files)
    split_idx = int(len(all_files) * args.train_split)
    print(split_idx)

    train_loader = DataLoader(NPZDataset(all_files[:split_idx]), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(NPZDataset(all_files[split_idx:]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[ModelCheckpoint(monitor="val_loss", filename="unet-best-{epoch}", save_top_k=3, mode="min")],
    )
    
    trainer.fit(UNetLightning(lr=args.lr), train_loader, val_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    "-d",  required=True, type=str)
    parser.add_argument("--train_split", "-ts", default=0.7,   type=float)
    parser.add_argument("--batch_size",  "-bs", default=2,     type=int)
    parser.add_argument("--num_workers", "-nw", default=2,     type=int)
    parser.add_argument("--max_epochs",  "-e",  default=5,    type=int)
    parser.add_argument("--devices",     "-dv", default=1,     type=int)
    parser.add_argument("--lr",          "-lr", default=1e-4,  type=float)
    parser.add_argument("--weight_decay", "-wd", default=1e-5,  type=float)
    args = parser.parse_args()
    main(args)