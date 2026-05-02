import os
import random
import matplotlib.colors as mcolors
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model_arch import EarthformerModel

# ── Scheduler utilities (from scheduler.py) ──────────────────────────────────

from packaging import version

if version.parse(torch.__version__) >= version.parse('1.11.0'):
    from torch.optim.lr_scheduler import SequentialLR, _LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor



class SequentialLR(_LRScheduler):
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        for scheduler_idx in range(1, len(schedulers)):
            if schedulers[scheduler_idx].optimizer != schedulers[0].optimizer:
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
            )
        self.optimizer = optimizer
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            self._schedulers[idx].step()

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)
        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        state_dict['_schedulers'] = _schedulers
        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda


# ── Normalization — identical to evaluate_earthformer_0_shot.py ───────────────

def mmhr_to_intensity_with_normalize(x, R_max=60.0):
    """
    Forward: mm/hr -> [0, 1]
    """
    norm = torch.log1p(x) / np.log1p(R_max)
    return torch.clamp(norm, 0.0, 1.0)


# ── Dataset — identical to evaluate_earthformer_0_shot.py ────────────────────

class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:,0,:,:], dtype=np.float32)

        arr = np.expand_dims(arr, axis=-1)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.concatenate((arr, [arr[-1]]))
        x_raw = torch.from_numpy(arr[:13]).float()     # T_in=13
        y_raw = torch.from_numpy(arr[13:25]).float()   # T_out=12

        return mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)


# ── Load model — identical to evaluate_earthformer_0_shot.py ─────────────────

def load_model(checkpoint_path, device, freeze_mode="none"):
    """
    Mirrors load_model from evaluate_earthformer_0_shot.py exactly.
    Extra: applies freeze after loading, and sets model to train mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = EarthformerModel().to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Lightning checkpoint — strip the "model." prefix added by LightningModule
        state = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.model.load_state_dict(state, strict=False)
        print("Loaded PyTorch Lightning checkpoint.")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded plain state-dict checkpoint.")

    # Apply freeze strategy after weights are loaded
    _apply_freeze(model, freeze_mode)

    model.train()
    return model


# ── Freeze helper ─────────────────────────────────────────────────────────────

def _apply_freeze(model, freeze_mode):
    """
    freeze_mode options:
      'none'    — all parameters trainable (full finetune)
      'encoder' — freeze encoder side, train decoder only
      'partial' — freeze early encoder blocks only
    """
    if freeze_mode == "none":
        for p in model.model.parameters():
            p.requires_grad = True
        print("Freeze mode: none — all parameters trainable.")

    elif freeze_mode == "encoder":
        frozen = [
            model.model.initial_encoder,
            model.model.encoder,
            model.model.enc_pos_embed,
        ]
        for m in frozen:
            for p in m.parameters():
                p.requires_grad = False
        model.model.init_global_vectors.requires_grad = False

        trainable = [
            model.model.decoder,
            model.model.final_decoder,
            model.model.dec_pos_embed,
            model.model.dec_final_proj,
            model.model.z_proj,
        ]
        for m in trainable:
            for p in m.parameters():
                p.requires_grad = True
        print("Freeze mode: encoder — encoder frozen, decoder trainable.")

    elif freeze_mode == "partial":
        frozen = [
            model.model.initial_encoder,
            model.model.encoder.blocks[0],
        ]
        for m in frozen:
            for p in m.parameters():
                p.requires_grad = False

        trainable = [
            model.model.encoder.blocks[1],
            model.model.encoder.down_layers,
            model.model.encoder.down_layer_global_proj,
            model.model.decoder,
            model.model.final_decoder,
            model.model.dec_pos_embed,
            model.model.dec_final_proj,
            model.model.z_proj,
        ]
        for m in trainable:
            for p in m.parameters():
                p.requires_grad = True
        print("Freeze mode: partial — encoder.blocks[0] + initial_encoder frozen.")

    else:
        raise ValueError(f"Unknown freeze_mode '{freeze_mode}'. Choose: none | encoder | partial")

    total     = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ── Finetuning Lightning Module ───────────────────────────────────────────────

class EarthformerFinetune(EarthformerModel):
    """
    Extends EarthformerModel with finetune-specific training logic.
    Weights are loaded externally via load_model() before trainer.fit().
    """
    def __init__(
        self,
        lr=1e-5,
        weight_decay=1e-5,
        threshold=1.0,
        warmup_epochs=2,
        min_lr_ratio=0.1,
        **kwargs
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        self.activation      = nn.Sigmoid()
        self._lr             = lr
        self._weight_decay   = weight_decay
        self._warmup_epochs  = warmup_epochs
        self._min_lr_ratio   = min_lr_ratio

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y   = batch
        pred   = self(x)                  # (B, 12, H, W, C)
        pred_11 = pred[:, :-1, ...]        # last 11 predicted frames
        y_11    = y[:, :-1, ...]           # last 11 target frames
        mae    = torch.mean(torch.abs(pred_11 - y_11))
        mse    = torch.mean((pred_11 - y_11) ** 2)
        loss   = mae + mse
        self.log("train_mae",  mae,  prog_bar=True)
        self.log("train_mse",  mse,  prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        pred   = self(x)                  # (B, 12, H, W, C)
        pred_11 = pred[:, :-1, ...]        # last 11 predicted frames
        y_11    = y[:, :-1, ...]           # last 11 target frames
        mae    = torch.mean(torch.abs(pred_11 - y_11))
        mse    = torch.mean((pred_11 - y_11) ** 2)
        loss   = mae + mse
        self.log("val_mae",  mae,  prog_bar=True)
        self.log("val_mse",  mse,  prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self._lr, weight_decay=self._weight_decay)

        warmup_scheduler   = LambdaLR(optimizer, lr_lambda=warmup_lambda(self._warmup_epochs, self._min_lr_ratio))
        constant_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[self._warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data — identical split logic to evaluate_earthformer_0_shot.py ────────
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

    # ── Load pretrained weights — identical to evaluate_earthformer_0_shot.py ─
    pretrained = load_model(args.checkpoint, device, freeze_mode=args.freeze_mode)

    # ── Build finetune model and copy loaded weights into it ──────────────────
    model = EarthformerFinetune(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        threshold=args.threshold,
    )
    model.load_state_dict(pretrained.state_dict())
    _apply_freeze(model, args.freeze_mode)
    print("Pretrained weights transferred to EarthformerFinetune.")

    # ── Trainer ───────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_mse",
        filename="earthformer-finetune-best-{epoch}",
        save_top_k=1,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[checkpoint_cb, lr_monitor],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir",      "-d",    required=True,          type=str)
    parser.add_argument("--train_split",   "-ts",   default=0.7,            type=float)
    parser.add_argument("--batch_size",    "-bs",   default=16,              type=int)
    parser.add_argument("--num_workers",   "-nw",   default=0,              type=int)

    # Checkpoint
    parser.add_argument("--checkpoint",    "-ckpt", default="earthformer_sevir.pt",          type=str,
                        help="Path to pretrained .pt file")
    parser.add_argument("--save_dir",      "-sd",   default="./checkpoints/finetune_11", type=str,
                        help="Directory to save finetuned checkpoints")

    # Training
    parser.add_argument("--max_epochs",    "-e",    default=5,             type=int)
    parser.add_argument("--devices",       "-dv",   default=1,              type=int)
    parser.add_argument("--lr",            "-lr",   default=1e-5,           type=float)
    parser.add_argument("--weight_decay",  "-wd",   default=1e-5,           type=float)
    parser.add_argument("--threshold",     "-t",    default=1.0,            type=float)
    parser.add_argument("--warmup_epochs", "-we",   default=2,              type=int)
    parser.add_argument("--min_lr_ratio",  "-mlr",  default=0.1,            type=float)

    # Freeze
    parser.add_argument("--freeze_mode",   "-fm",   default="none",         type=str,
                        choices=["none", "encoder", "partial"])

    args = parser.parse_args()
    main(args)