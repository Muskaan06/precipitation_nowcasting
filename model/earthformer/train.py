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


def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda

# ── Data utilities ────────────────────────────────────────────────────────────

def mmhr_to_intensity_with_normalize(x, R_max=60.0):
    """
    Forward: mm/hr -> [0, 1]
    """
    norm = torch.log1p(x) / np.log1p(R_max)
    return torch.clamp(norm, 0.0, 1.0)


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
        x_raw = torch.from_numpy(arr[:6]).float()    #change
        y_raw = torch.from_numpy(arr[6:16]).float()  #change

        return mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)

# ── Lightning module ──────────────────────────────────────────────────────────

class EarthformerLightning(EarthformerModel):
    def __init__(self, lr=1e-4, weight_decay=0.00001, threshold=1.0,
                 warmup_epochs=2, min_lr_ratio=0.1, **kwargs):
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        self.activation = nn.Sigmoid()
        # Explicitly store so configure_optimizers can always access them
        self._lr = lr
        self._weight_decay = weight_decay
        self._warmup_epochs = warmup_epochs
        self._min_lr_ratio = min_lr_ratio

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)  # [0, 1] range

        mae = torch.mean(torch.abs(pred - y))
        mse = torch.mean((pred - y) ** 2)
        loss = mae + mse

        self.log("train_mae",  mae,  prog_bar=True)
        self.log("train_mse",  mse,  prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)  # [0, 1] range

        mae = torch.mean(torch.abs(pred - y))
        mse = torch.mean((pred - y) ** 2)
        loss = mae + mse

        self.log("val_mae",  mae,  prog_bar=True)
        self.log("val_mse",  mse,  prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)

        # Warmup scheduler: ramps lr from min_lr_ratio → 1.0 over warmup_epochs
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda(self._warmup_epochs, self._min_lr_ratio)
        )

        # Post-warmup scheduler: holds lr at 1.0 multiplier (constant full lr)
        constant_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

        # Hand off to SequentialLR: warmup for warmup_epochs, then constant
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[self._warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # step once per epoch
                "frequency": 1,
            },
        }

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

    checkpoint_cb = ModelCheckpoint(
        monitor="val_mse",
        filename="earthformer-best-{epoch}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(
        EarthformerLightning(
            lr=args.lr,
            threshold=args.threshold,
            warmup_epochs=args.warmup_epochs,
            min_lr_ratio=args.min_lr_ratio,
        ),
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      "-d",   required=True, type=str)
    parser.add_argument("--train_split",   "-ts",  default=0.7,   type=float)
    parser.add_argument("--batch_size",    "-bs",  default=16,    type=int)
    parser.add_argument("--num_workers",   "-nw",  default=1,     type=int)
    parser.add_argument("--max_epochs",    "-e",   default=5,     type=int)
    parser.add_argument("--devices",       "-dv",  default=1,     type=int)
    parser.add_argument("--lr",            "-lr",  default=0.001, type=float)
    parser.add_argument("--threshold",     "-t",   default=1.0,   type=float)
    # Scheduler-specific args
    parser.add_argument("--warmup_epochs", "-we",  default=2,     type=int,
                        help="Number of epochs to linearly warm up the LR")
    parser.add_argument("--min_lr_ratio",  "-mlr", default=0.1,   type=float,
                        help="Starting LR as a fraction of the base LR during warmup")
    args = parser.parse_args()
    main(args)