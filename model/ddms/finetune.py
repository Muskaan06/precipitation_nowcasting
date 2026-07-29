import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_arch import build_ddms_model
from patch_utils import mmhr_to_intensity_with_normalize, to_pm1

# ── Data utilities (identical to train.py) ───────────────────────────────────

class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:, 0, :, :], dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.expand_dims(arr, axis=1)
        clip = torch.from_numpy(arr).float()

        return mmhr_to_intensity_with_normalize(clip)

# ── Pretrained checkpoint loading ────────────────────────────────────────────

def load_pretrained_weights(model, checkpoint_path, device):
    """
    Loads a GaussianDiffusion's weights from a Lightning .ckpt, a raw DDMS
    checkpoint ({"model","ema"}, possibly "module."-prefixed), or a plain
    state_dict — non-strictly (in case architecture flags don't exactly match).
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
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        import collections
        state = collections.OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in checkpoint["model"].items()
        )
        model.load_state_dict(state, strict=False)
        print("Loaded raw DDMS checkpoint (strict=False).")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded plain state-dict checkpoint (strict=False).")

    return model

# ── Freeze strategy ───────────────────────────────────────────────────────────

def apply_freeze(model, freeze_mode="none"):
    """
    model: a GaussianDiffusion instance (denoise_fn / history_fn / transform_fn).

    - "none"             : everything trainable.
    - "freeze_context"   : freezes history_fn (CondNet) + transform_fn (HistoryNet),
                            adapting only the denoising U-Net to the new domain.
    - "freeze_denoiser"  : freezes denoise_fn (Unet), adapting only the
                            temporal/trend encoders.
    - "freeze_transform" : freezes only transform_fn, letting the U-Net and
                            context encoder adapt.
    """
    if freeze_mode == "none":
        return model

    if freeze_mode == "freeze_context":
        for p in model.history_fn.parameters():
            p.requires_grad = False
        for p in model.transform_fn.parameters():
            p.requires_grad = False

    elif freeze_mode == "freeze_denoiser":
        for p in model.denoise_fn.parameters():
            p.requires_grad = False

    elif freeze_mode == "freeze_transform":
        for p in model.transform_fn.parameters():
            p.requires_grad = False

    else:
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    return model

# ── Lightning module ──────────────────────────────────────────────────────────

class DDMSFinetune(pl.LightningModule):
    def __init__(self, lr=1e-5, dim=32, dim_mults=(1, 1, 2, 2, 4),
                 transform_dim_mults=(1, 2, 3, 4), backbone="resnet",
                 timesteps=500, sample_steps=50, freeze_mode="none", max_epochs=5):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_ddms_model(
            dim=dim, dim_mults=dim_mults, transform_dim_mults=transform_dim_mults,
            channels=1, backbone=backbone, timesteps=timesteps, sample_steps=sample_steps,
        )

    def _step(self, batch, stage):
        video = to_pm1(batch).transpose(0, 1)
        loss, d_loss, mse_loss = self.model(video)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_mse", mse_loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
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

    dim_mults = tuple(int(x) for x in args.dim_mults.split(","))
    transform_dim_mults = tuple(int(x) for x in args.transform_dim_mults.split(","))

    # NOTE: --dim/--dim_mults/--transform_dim_mults/--backbone/--timesteps must
    # match the architecture the checkpoint was originally trained with, or
    # load_pretrained_weights' strict=False load will silently drop mismatched
    # tensors.
    module = DDMSFinetune(
        lr=args.lr,
        dim=args.dim,
        dim_mults=dim_mults,
        transform_dim_mults=transform_dim_mults,
        backbone=args.backbone,
        timesteps=args.timesteps,
        sample_steps=args.sample_steps,
        freeze_mode=args.freeze_mode,
        max_epochs=args.max_epochs,
    )

    module.model = load_pretrained_weights(module.model, args.checkpoint, device)
    module.model = apply_freeze(module.model, args.freeze_mode)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_mse",
        filename="ddms-finetune-best-{epoch}",
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
    parser.add_argument("--batch_size",    "-bs",  default=2,     type=int)
    parser.add_argument("--num_workers",   "-nw",  default=1,     type=int)
    parser.add_argument("--max_epochs",    "-e",   default=5,     type=int)
    parser.add_argument("--devices",       "-dv",  default=1,     type=int)
    parser.add_argument("--lr",            "-lr",  default=1e-5,  type=float)
    parser.add_argument("--save_dir",      "-sd",  default="../../model_weights/ddms", type=str)
    # Architecture args — must match the checkpoint being loaded
    parser.add_argument("--dim",           default=32,    type=int)
    parser.add_argument("--dim_mults",     default="1,1,2,2,4", type=str)
    parser.add_argument("--transform_dim_mults", default="1,2,3,4", type=str)
    parser.add_argument("--backbone",      choices=["resnet", "convnext"], default="resnet")
    parser.add_argument("--timesteps",     default=500,   type=int)
    parser.add_argument("--sample_steps",  default=50,    type=int)
    # Freeze strategy
    parser.add_argument("--freeze_mode",   choices=["none", "freeze_context", "freeze_denoiser", "freeze_transform"], default="none")
    args = parser.parse_args()
    main(args)
