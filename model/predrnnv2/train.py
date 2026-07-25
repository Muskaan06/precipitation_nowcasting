import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from model_arch import PredRNNv2Core
from patch_utils import mmhr_to_intensity_with_normalize
from scheduling import reserve_schedule_sampling_exp

# ── Data utilities ────────────────────────────────────────────────────────────

class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:, 0, :, :], dtype=np.float32)

        arr = np.expand_dims(arr, axis=-1)
        arr = np.nan_to_num(arr, nan=0.0)
        x_raw = torch.from_numpy(arr[:6]).float()    # T_in
        y_raw = torch.from_numpy(arr[6:16]).float()  # T_out

        return mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)

# ── Lightning module ──────────────────────────────────────────────────────────

class PredRNNv2Lightning(pl.LightningModule):
    """
    PredRNN-V2 expects channels-last input: (B, T, H, W, C)
    - input : (B, 6, 112, 112, 1)
    - output: (B, 10, 112, 112, 1)
    """
    def __init__(self, lr=1e-3, num_layers=4, num_hidden=(64, 64, 64, 64),
                 filter_size=5, stride=1, patch_size=4, layer_norm=False,
                 decouple_beta=0.1, img_width=112, input_length=6, total_length=16,
                 reverse_scheduled_sampling=False,
                 r_sampling_step_1=25000, r_sampling_step_2=50000, r_exp_alpha=5000):
        super().__init__()
        self.save_hyperparameters()

        self.model = PredRNNv2Core(
            num_layers=num_layers,
            num_hidden=num_hidden,
            img_channel=1,
            img_width=img_width,
            patch_size=patch_size,
            filter_size=filter_size,
            stride=stride,
            layer_norm=layer_norm,
            input_length=input_length,
            total_length=total_length,
            decouple_beta=decouple_beta,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams.reverse_scheduled_sampling:
            mask_true = reserve_schedule_sampling_exp(
                step=self.global_step,
                batch_size=x.shape[0],
                input_length=self.hparams.input_length,
                total_length=self.hparams.total_length,
                r_sampling_step_1=self.hparams.r_sampling_step_1,
                r_sampling_step_2=self.hparams.r_sampling_step_2,
                r_exp_alpha=self.hparams.r_exp_alpha,
                device=x.device,
            )
            pred, decouple_loss = self.model(x, y, mask_true)
        else:
            pred, decouple_loss = self(x)

        mae = torch.mean(torch.abs(pred - y))
        mse = torch.mean((pred - y) ** 2)
        loss = mae + mse + self.hparams.decouple_beta * decouple_loss

        self.log("train_mae", mae, prog_bar=True)
        self.log("train_mse", mse, prog_bar=True)
        self.log("train_decouple_loss", decouple_loss, prog_bar=False)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Always pure autoregression here (mask_true=None), matching the
        # exact conditions evaluate.py uses at real inference time.
        x, y = batch
        pred, decouple_loss = self(x)

        mae = torch.mean(torch.abs(pred - y))
        mse = torch.mean((pred - y) ** 2)
        loss = mae + mse + self.hparams.decouple_beta * decouple_loss

        self.log("val_mae", mae, prog_bar=True)
        self.log("val_mse", mse, prog_bar=True)
        self.log("val_decouple_loss", decouple_loss, prog_bar=False)
        self.log("val_loss", loss, prog_bar=True)
        return loss

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
        NPZDataset(all_files[:split_idx]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        NPZDataset(all_files[split_idx:]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    num_hidden = tuple(int(x) for x in args.num_hidden.split(","))

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_mse",
        filename="predrnnv2-best-{epoch}",
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
        PredRNNv2Lightning(
            lr=args.lr,
            num_layers=len(num_hidden),
            num_hidden=num_hidden,
            filter_size=args.filter_size,
            stride=args.stride,
            patch_size=args.patch_size,
            layer_norm=args.layer_norm,
            decouple_beta=args.decouple_beta,
            reverse_scheduled_sampling=args.reverse_scheduled_sampling,
            r_sampling_step_1=args.r_sampling_step_1,
            r_sampling_step_2=args.r_sampling_step_2,
            r_exp_alpha=args.r_exp_alpha,
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
    parser.add_argument("--save_dir",      "-sd",  default="../../model_weights/predrnnv2", type=str)
    # PredRNN-V2 architecture args
    parser.add_argument("--num_hidden",    default="64,64,64,64", type=str,
                        help="Comma-separated hidden channels, one per ST-LSTM layer")
    parser.add_argument("--filter_size",   default=5,     type=int)
    parser.add_argument("--stride",        default=1,     type=int)
    parser.add_argument("--patch_size",    default=4,     type=int)
    parser.add_argument("--layer_norm",    action="store_true", default=False)
    parser.add_argument("--decouple_beta", default=0.1,   type=float)
    # Reverse scheduled sampling (optional training curriculum)
    parser.add_argument("--reverse_scheduled_sampling", action="store_true", default=False,
                        help="Blend ground-truth/generated frames during training via a decaying mask "
                             "(predrnn-pytorch's RSS curriculum). Off by default: plain teacher-force-then-autoregress.")
    parser.add_argument("--r_sampling_step_1", default=25000, type=int)
    parser.add_argument("--r_sampling_step_2", default=50000, type=int)
    parser.add_argument("--r_exp_alpha",       default=5000,  type=int)
    args = parser.parse_args()
    main(args)
