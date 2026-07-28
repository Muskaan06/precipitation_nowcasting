import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_arch import DiffCastBackbone
from patch_utils import mmhr_to_intensity_with_normalize

# ── Data utilities (identical to train.py) ───────────────────────────────────

class NPZDataset(Dataset):
    def __init__(self, files, T_in=8, T_out=8):
        self.files = files
        self.T_in = T_in
        self.T_out = T_out

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:, 0, :, :], dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.expand_dims(arr, axis=1)
        x_raw = torch.from_numpy(arr[:self.T_in]).float()
        y_raw = torch.from_numpy(arr[self.T_in:self.T_in + self.T_out]).float()

        return mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)

# ── Pretrained checkpoint loading ────────────────────────────────────────────

def load_pretrained_weights(model, checkpoint_path, device):
    """
    Loads a DiffCastBackbone's weights from either a Lightning .ckpt
    (keys prefixed 'model.') or a raw state_dict, non-strictly (in case
    architecture flags don't exactly match the checkpoint's).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state, strict=False)
        print("Loaded PyTorch Lightning checkpoint (strict=False).")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded plain state-dict checkpoint (strict=False).")

    return model

# ── Freeze strategy ───────────────────────────────────────────────────────────

def apply_freeze(model, backbone="phydnet", freeze_mode="none"):
    """
    model: a DiffCastBackbone instance.

    PhyDNet (model.net.encoder): "first_n" freezes the physics branch
    (.phycell — presumed to capture generic short-range dynamics);
    "all_but_last" additionally freezes the ConvLSTM appearance branch
    (.convcell), leaving only the decoder path trainable.

    SimVP (model.net): "first_n" freezes the spatial encoder (.enc, generic
    low-level features); "all_but_last" additionally freezes the temporal
    translator (.hid), leaving only the decoder (.dec) trainable.
    """
    if freeze_mode == "none":
        return model

    if backbone == "phydnet":
        encoder = model.net.encoder
        if freeze_mode == "first_n":
            for p in encoder.phycell.parameters():
                p.requires_grad = False
        elif freeze_mode == "all_but_last":
            for p in encoder.phycell.parameters():
                p.requires_grad = False
            for p in encoder.convcell.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    elif backbone == "simvp":
        net = model.net
        if freeze_mode == "first_n":
            for p in net.enc.parameters():
                p.requires_grad = False
        elif freeze_mode == "all_but_last":
            for p in net.enc.parameters():
                p.requires_grad = False
            for p in net.hid.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model

# ── Lightning module ──────────────────────────────────────────────────────────

class DiffCastFinetune(pl.LightningModule):
    def __init__(self, lr=1e-5, backbone="phydnet", img_width=112, T_in=8, T_out=8,
                 freeze_mode="none", max_epochs=5):
        super().__init__()
        self.save_hyperparameters()

        self.model = DiffCastBackbone(
            backbone=backbone, C=1, H=img_width, W=img_width,
            T_in=T_in, T_out=T_out,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def forward(self, x):
        return self.model.predict(x, compute_loss=False)[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred, loss = self.model.predict(x, frames_gt=y, compute_loss=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred, loss = self.model.predict(x, frames_gt=y, compute_loss=True)
        val_mse = torch.mean((pred - y) ** 2)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", val_mse, prog_bar=True)
        return val_mse

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # NOTE: --backbone/--T_in/--T_out/--img_width must match the architecture
    # the checkpoint was originally trained with, or load_pretrained_weights'
    # strict=False load will silently drop mismatched tensors.
    module = DiffCastFinetune(
        lr=args.lr,
        backbone=args.backbone,
        img_width=args.img_width,
        T_in=args.T_in,
        T_out=args.T_out,
        freeze_mode=args.freeze_mode,
        max_epochs=args.max_epochs,
    )

    module.model = load_pretrained_weights(module.model, args.checkpoint, device)
    module.model = apply_freeze(module.model, backbone=args.backbone, freeze_mode=args.freeze_mode)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_mse",
        filename=f"diffcast-{args.backbone}-finetune-best-{{epoch}}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=1.0,
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      "-d",   required=True, type=str)
    parser.add_argument("--checkpoint",    "-ckpt", required=True, type=str)
    parser.add_argument("--train_split",   "-ts",  default=0.7,   type=float)
    parser.add_argument("--batch_size",    "-bs",  default=8,     type=int)
    parser.add_argument("--num_workers",   "-nw",  default=1,     type=int)
    parser.add_argument("--max_epochs",    "-e",   default=5,     type=int)
    parser.add_argument("--devices",       "-dv",  default=1,     type=int)
    parser.add_argument("--lr",            "-lr",  default=1e-5,  type=float)
    parser.add_argument("--save_dir",      "-sd",  default="../../model_weights/diffcast", type=str)
    # Architecture args — must match the checkpoint being loaded
    parser.add_argument("--backbone",      choices=["phydnet", "simvp"], default="phydnet")
    parser.add_argument("--T_in",          default=8,     type=int)
    parser.add_argument("--T_out",         default=8,     type=int)
    parser.add_argument("--img_width",     default=112,   type=int)
    # Freeze strategy
    parser.add_argument("--freeze_mode",   choices=["none", "first_n", "all_but_last"], default="none")
    args = parser.parse_args()
    main(args)
