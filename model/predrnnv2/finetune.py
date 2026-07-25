import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_arch import PredRNNv2Core
from patch_utils import mmhr_to_intensity_with_normalize

# ── Data utilities (identical to train.py) ───────────────────────────────────

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
        x_raw = torch.from_numpy(arr[:6]).float()
        y_raw = torch.from_numpy(arr[6:16]).float()

        return mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)

# ── Pretrained checkpoint loading ────────────────────────────────────────────

def load_pretrained_weights(model, checkpoint_path, device):
    """
    Loads a PredRNNv2Core's weights from either a Lightning .ckpt
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

def apply_freeze(model, freeze_mode="none", freeze_layers=2):
    """
    model: a PredRNNv2Core instance.
    - "none"        : everything trainable.
    - "first_n"     : freeze the first `freeze_layers` ST-LSTM cells (lower
                       layers, presumed to capture generic short-range
                       spatiotemporal dynamics); upper cells + conv_last +
                       adapter stay trainable.
    - "all_but_last": freeze every ST-LSTM cell + adapter, train only conv_last
                       (cheapest finetune).
    """
    if freeze_mode == "none":
        return model

    if freeze_mode == "first_n":
        for i in range(min(freeze_layers, len(model.cell_list))):
            for p in model.cell_list[i].parameters():
                p.requires_grad = False

    elif freeze_mode == "all_but_last":
        for cell in model.cell_list:
            for p in cell.parameters():
                p.requires_grad = False
        for p in model.adapter.parameters():
            p.requires_grad = False

    else:
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    return model

# ── Lightning module ──────────────────────────────────────────────────────────

class PredRNNv2Finetune(pl.LightningModule):
    def __init__(self, lr=1e-5, num_layers=4, num_hidden=(64, 64, 64, 64),
                 filter_size=5, stride=1, patch_size=4, layer_norm=False,
                 decouple_beta=0.1, img_width=112, input_length=6, total_length=16,
                 freeze_mode="none", freeze_layers=2, max_epochs=5):
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

    def _step(self, batch, stage):
        x, y = batch
        pred, decouple_loss = self(x)

        mae = torch.mean(torch.abs(pred - y))
        mse = torch.mean((pred - y) ** 2)
        loss = mae + mse + self.hparams.decouple_beta * decouple_loss

        self.log(f"{stage}_mae", mae, prog_bar=True)
        self.log(f"{stage}_mse", mse, prog_bar=True)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

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

    # NOTE: --num_layers/--num_hidden/--patch_size/--filter_size/--stride must
    # match the architecture the checkpoint was originally trained with, or
    # load_pretrained_weights' strict=False load will silently drop mismatched
    # tensors.
    module = PredRNNv2Finetune(
        lr=args.lr,
        num_layers=len(num_hidden),
        num_hidden=num_hidden,
        filter_size=args.filter_size,
        stride=args.stride,
        patch_size=args.patch_size,
        layer_norm=args.layer_norm,
        decouple_beta=args.decouple_beta,
        freeze_mode=args.freeze_mode,
        freeze_layers=args.freeze_layers,
        max_epochs=args.max_epochs,
    )

    module.model = load_pretrained_weights(module.model, args.checkpoint, device)
    module.model = apply_freeze(module.model, args.freeze_mode, args.freeze_layers)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_mse",
        filename="predrnnv2-finetune-best-{epoch}",
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
    parser.add_argument("--save_dir",      "-sd",  default="../../model_weights/predrnnv2", type=str)
    # Architecture args — must match the checkpoint being loaded
    parser.add_argument("--num_hidden",    default="64,64,64,64", type=str)
    parser.add_argument("--filter_size",   default=5,     type=int)
    parser.add_argument("--stride",        default=1,     type=int)
    parser.add_argument("--patch_size",    default=4,     type=int)
    parser.add_argument("--layer_norm",    action="store_true", default=False)
    parser.add_argument("--decouple_beta", default=0.1,   type=float)
    # Freeze strategy
    parser.add_argument("--freeze_mode",   choices=["none", "first_n", "all_but_last"], default="none")
    parser.add_argument("--freeze_layers", default=2,     type=int,
                        help="Used with --freeze_mode first_n: number of lower ST-LSTM layers to freeze")
    args = parser.parse_args()
    main(args)
