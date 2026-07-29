import os
import copy
import random
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from model_arch import build_ddms_model
from patch_utils import mmhr_to_intensity_with_normalize, to_pm1

# ── Data utilities ────────────────────────────────────────────────────────────

class NPZDataset(Dataset):
    """
    Returns the full 16-frame clip, batch-major (B, 16, 1, H, W); the
    LightningModule handles the [0,1]->[-1,1] rescale and the transpose to
    DDMS's native time-major (T, B, C, H, W) convention.
    """
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:, 0, :, :], dtype=np.float32)  # (16,H,W)

        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.expand_dims(arr, axis=1)  # (16,1,H,W)
        clip = torch.from_numpy(arr).float()

        return mmhr_to_intensity_with_normalize(clip)

# ── EMA callback ──────────────────────────────────────────────────────────────

class EMACallback(pl.Callback):
    """
    Replicates DDMS's own EMA class (DDMS/modules/trainer.py) as a Lightning
    callback: a deep-copied shadow model, exponential beta-averaging of
    parameters starting at `start_step`, updated every `update_every` steps.
    """
    def __init__(self, decay=0.99, start_step=2000, update_every=5):
        self.decay = decay
        self.start_step = start_step
        self.update_every = update_every
        self.ema_model = None

    def on_fit_start(self, trainer, pl_module):
        self.ema_model = copy.deepcopy(pl_module.model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.update_every != 0:
            return
        if step < self.start_step:
            self.ema_model.load_state_dict(pl_module.model.state_dict())
            return
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), pl_module.model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema_model is not None:
            checkpoint["ema"] = self.ema_model.state_dict()

# ── Lightning module ──────────────────────────────────────────────────────────

class DDMSLightning(pl.LightningModule):
    """
    input : (B, 16, 1, 112, 112), normalized [0,1] rain intensity
    Internally: rescaled [-1,1], transposed to (16, B, 1, 112, 112) — DDMS's
    native time-major convention. Trains the REAL diffusion loss (GaussianDiffusion.forward
    already implements forward-diffusion noise-prediction training) — nothing reimplemented here.
    """
    def __init__(self, lr=5e-5, dim=32, dim_mults=(1, 1, 2, 2, 4),
                 transform_dim_mults=(1, 2, 3, 4), backbone="resnet",
                 timesteps=500, sample_steps=50):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_ddms_model(
            dim=dim,
            dim_mults=dim_mults,
            transform_dim_mults=transform_dim_mults,
            channels=1,
            backbone=backbone,
            timesteps=timesteps,
            sample_steps=sample_steps,
        )

    def _step(self, batch, stage):
        video = to_pm1(batch).transpose(0, 1)  # (B,16,1,H,W) -> (16,B,1,H,W)
        loss, d_loss, mse_loss = self.model(video)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_diffusion_loss", d_loss, prog_bar=False)
        self.log(f"{stage}_mse", mse_loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # Same cheap teacher-forced forward() as training — no autoregressive
        # sample() here, so validation epochs stay fast. Expensive DDIM
        # sampling is confined to evaluate.py only.
        return self._step(batch, "val")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

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

    dim_mults = tuple(int(x) for x in args.dim_mults.split(","))
    transform_dim_mults = tuple(int(x) for x in args.transform_dim_mults.split(","))

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_mse",
        filename="ddms-best-{epoch}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    callbacks = [checkpoint_cb]
    if args.use_ema:
        callbacks.append(EMACallback())

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=callbacks,
    )

    trainer.fit(
        DDMSLightning(
            lr=args.lr,
            dim=args.dim,
            dim_mults=dim_mults,
            transform_dim_mults=transform_dim_mults,
            backbone=args.backbone,
            timesteps=args.timesteps,
            sample_steps=args.sample_steps,
        ),
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      "-d",   required=True, type=str)
    parser.add_argument("--train_split",   "-ts",  default=0.7,   type=float)
    parser.add_argument("--batch_size",    "-bs",  default=2,     type=int)
    parser.add_argument("--num_workers",   "-nw",  default=1,     type=int)
    parser.add_argument("--max_epochs",    "-e",   default=5,     type=int)
    parser.add_argument("--devices",       "-dv",  default=1,     type=int)
    parser.add_argument("--lr",            "-lr",  default=5e-5,  type=float)
    parser.add_argument("--save_dir",      "-sd",  default="../../model_weights/ddms", type=str)
    # DDMS architecture args
    parser.add_argument("--dim",           default=32,    type=int)
    parser.add_argument("--dim_mults",     default="1,1,2,2,4", type=str)
    parser.add_argument("--transform_dim_mults", default="1,2,3,4", type=str)
    parser.add_argument("--backbone",      choices=["resnet", "convnext"], default="resnet")
    parser.add_argument("--timesteps",     default=500,   type=int,
                        help="Diffusion training timesteps (cosine schedule).")
    parser.add_argument("--sample_steps",  default=50,    type=int,
                        help="DDIM sampling steps — only used if validation/eval calls .sample().")
    parser.add_argument("--use_ema",       action="store_true", default=False)
    args = parser.parse_args()
    main(args)
