import os
import random
import cv2
cv2.setNumThreads(0)
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

# Importing the actual architecture components
from model_arch import TUPANN, AutoencoderKL, make_grid, warp


# =============================================================================
# 1. Rain Preprocessing (IMERG Specs)
# =============================================================================

RAIN_MAX_MMHR: float = 128.0
_LOG_NORM_DENOM: float = float(np.log1p(RAIN_MAX_MMHR))

def preprocess_rain(x: np.ndarray) -> np.ndarray:
    """
    Normalises raw IMERG mm/hr to [0, 1] log-rain space.
    """
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    x = np.clip(x, 0.0, RAIN_MAX_MMHR)
    x = np.log1p(x) / _LOG_NORM_DENOM
    return x


# =============================================================================
# 2. Physics-Aligned Supervision (Lucas-Kanade Proxy)
# =============================================================================

def compute_lk_flow(prev_frame, curr_frame):
    """
    Computes dense optical flow (Farneback) as a physical motion proxy.
    Input frames are in [0, 1] log-space.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_frame,
        next=curr_frame,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow  # (H, W, 2)


def optical_flow_loss(flow_pred, flow_gt, of_cos_weight=0.33):
    """
    Implements L_OF = weight * L_cos + (1-weight) * L1_vec
    """
    l1_vec = nn.functional.l1_loss(flow_pred, flow_gt)

    B, _, H, W = flow_pred.shape
    fp_flat = flow_pred.permute(0, 2, 3, 1).reshape(B, H * W, 2)
    fg_flat = flow_gt.permute(0, 2, 3, 1).reshape(B, H * W, 2)

    cos_sim = nn.CosineSimilarity(dim=2, eps=1e-8)
    cos_raw = cos_sim(fp_flat, fg_flat)

    # Mask to avoid NaNs on dry/motionless pixels
    valid = (fg_flat.norm(dim=-1) > 1e-6) | (fp_flat.norm(dim=-1) > 1e-6)
    if valid.any():
        l_cos = -cos_raw[valid].mean()
    else:
        l_cos = flow_pred.new_zeros(1).squeeze()

    l_of = of_cos_weight * l_cos + (1.0 - of_cos_weight) * l1_vec
    return l_of, {"l1_vec": l1_vec.item(), "l_cos": l_cos.item()}


# =============================================================================
# 3. Dataset Implementation
# =============================================================================

class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # FIX: explicitly load 'array' key instead of f.files[0] which could
        # resolve to 'metadata' (a pickled dict) and print binary garbage.
        with np.load(self.files[idx], allow_pickle=True) as f:
            arr = np.array(f['array'], dtype=np.float32).copy()

        rain = arr[:, 0, :, :]
        rain = preprocess_rain(rain)

        # 10 frames total: 4 input, 6 target
        x = torch.from_numpy(rain[:6]).contiguous()
        y = torch.from_numpy(rain[6:16]).contiguous()

        # Physical Ground Truths
        # Flow is computed between the last two observed frames
        flow_np = compute_lk_flow(rain[2], rain[3])
        flow_gt = torch.from_numpy(flow_np).permute(2, 0, 1).contiguous()

        # Intensity GT for Autoencoder reconstruction is the current frame
        intensity_gt = x[-1].clone()

        return x, y, flow_gt, intensity_gt


# =============================================================================
# 4. Stage 1: Autoencoder (VED) Training
# =============================================================================

class AutoencoderLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, input_len=6, img_size=112, embed_dim=6,
                 reduc_factor=4, channels=64, dropout=0.0,
                 lambda1=1.0, lambda2=0.5, lambda3=1e-5, of_cos_weight=0.33):
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = AutoencoderKL(
            input_len=input_len, img_size=img_size, embed_dim=embed_dim,
            reduc_factor=reduc_factor, channels=channels, dropout=dropout,
        )

    def forward(self, x):
        return self.autoencoder(x)

    def _step(self, batch, split):
        x, y, flow_gt, intensity_gt = batch
        decoded, posterior = self.autoencoder(x)

        flow_pred      = decoded[:, :2]
        intensity_pred = decoded[:, 2]

        # L_target (Intensity Reconstruction)
        l_target = nn.functional.mse_loss(intensity_pred, intensity_gt)

        # L_OF (Physics Alignment)
        l_of, of_metrics = optical_flow_loss(
            flow_pred, flow_gt, of_cos_weight=self.hparams.of_cos_weight
        )

        # L_KL (Variational Bottleneck)
        l_kl = posterior.kl().mean()

        loss = (self.hparams.lambda1 * l_target +
                self.hparams.lambda2 * l_of +
                self.hparams.lambda3 * l_kl)

        self.log(f"{split}_loss", loss, prog_bar=True, sync_dist=False)
        self.log(f"{split}_l_of", l_of, prog_bar=True, sync_dist=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))


# =============================================================================
# 5. Stage 2: TUPANN (Latent Transformer) Training
# =============================================================================

class TUPANNLightning(pl.LightningModule):
    def __init__(self, autoenc_ckpt, lr=1e-4,
                 input_len=6, target_length=10, img_size=112,
                 embed_dim=6, reduc_factor=4, channels=64, dropout=0.0,
                 maxvit_dim=16, maxvit_depth=4,
                 lambda1=1.0, lambda2=0.5, lambda3=1e-5, of_cos_weight=0.33):
        super().__init__()
        self.save_hyperparameters()

        self.model = TUPANN(
            input_len=input_len, target_length=target_length, img_size=img_size,
            embed_dim=embed_dim, reduc_factor=reduc_factor, channels=channels,
            dropout=dropout, maxvit_dim=maxvit_dim, maxvit_depth=maxvit_depth,
        )

        # Loading and Freezing the VED from Stage 1
        state = torch.load(autoenc_ckpt, map_location="cpu")["state_dict"]
        state = {k.replace("autoencoder.", ""): v for k, v in state.items()}
        self.model.autoencoder.load_state_dict(state, strict=False)
        for p in self.model.autoencoder.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, split):
        x, y, flow_gt, intensity_gt = batch
        pred = self(x)  # Predicting future frames

        # L_target (Future Nowcast MSE)
        l_target = nn.functional.mse_loss(pred, y)

        # L_OF & L_KL (Grounding on frozen VED)
        with torch.no_grad():
            decoded, posterior = self.model.autoencoder(x)
        flow_pred = decoded[:, :2]
        l_of, _ = optical_flow_loss(flow_pred, flow_gt, self.hparams.of_cos_weight)
        l_kl = posterior.kl().mean()

        loss = (self.hparams.lambda1 * l_target +
                self.hparams.lambda2 * l_of +
                self.hparams.lambda3 * l_kl)

        self.log(f"{split}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.latent_model.parameters(), lr=self.hparams.lr)


# =============================================================================
# 6. Execution Block
# =============================================================================

def build_loaders(data_dir, train_split, batch_size, num_workers):
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_split)

    train_loader = DataLoader(NPZDataset(all_files[:split_idx]), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(NPZDataset(all_files[split_idx:]), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader = build_loaders(args.data_dir, args.train_split, args.batch_size, args.num_workers)

    if args.stage == 1:
        model = AutoencoderLightning(lr=args.lr, lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3)
        ckpt_cb = ModelCheckpoint(dirpath=args.save_dir, monitor="val_loss", filename="ae_best", mode="min",
                                  save_last=True, auto_insert_metric_name=False)
        trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu", devices=args.devices, callbacks=[ckpt_cb])
        trainer.fit(model, train_loader, val_loader)
        print(f"[Stage 1] Best checkpoint: {ckpt_cb.best_model_path}")

    elif args.stage == 2:
        model = TUPANNLightning(
                autoenc_ckpt=args.autoenc_ckpt,
                lr=args.lr,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                lambda3=args.lambda3,
                of_cos_weight=args.of_cos_weight
        )
        ckpt_cb = ModelCheckpoint(dirpath=args.save_dir, monitor="val_loss", filename="tupann_best", mode="min",
                                  save_last=True, auto_insert_metric_name=False)
        trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu", devices=args.devices, callbacks=[ckpt_cb])
        trainer.fit(model, train_loader, val_loader)
        print(f"[Stage 2] Best checkpoint: {ckpt_cb.best_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", "-s", required=True, type=int, choices=[1, 2])
    parser.add_argument("--data_dir", "-d", required=True, type=str)
    parser.add_argument("--autoenc_ckpt", "-ckpt", default = "/mnt/sda1/Muskaan/nowcast/precipitation_nowcasting/model/tupann/checkpoints/ae_best.ckpt", type=str)
    parser.add_argument("--save_dir", "-sd", default="./checkpoints")
    parser.add_argument("--train_split", "-ts", default=0.7, type=float)
    parser.add_argument("--batch_size", "-bs", default=16, type=int)
    parser.add_argument("--num_workers", "-nw", default=0, type=int)
    parser.add_argument("--max_epochs", "-e", default=5, type=int)
    parser.add_argument("--devices", "-dv", default=1, type=int)
    parser.add_argument("--lr", "-lr", default=1e-4, type=float)
    parser.add_argument("--lambda1", "-l1", default=1.0, type=float)
    parser.add_argument("--lambda2", "-l2", default=0.5, type=float)
    parser.add_argument("--lambda3", "-l3", default=1e-5, type=float)
    parser.add_argument("--of_cos_weight", "-ocw", default=0.33, type=float)
    args = parser.parse_args()
    main(args)