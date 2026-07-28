# DiffCast

Integration of [DiffCast](https://arxiv.org/abs/2312.06734) (Yu et al., CVPR 2024) into this
repo's shared model-backend conventions (same data pipeline, `eval_metrics.py`, and
train/evaluate/finetune structure as `model/earthformer/`, `model/tupann/`, `model/predrnnv2/`).

Ported from the vendored official repo at `../../DiffCast/`.

## Architecture

DiffCast decomposes nowcasting into two stages:

1. **Deterministic backbone** (PhyDNet or SimVP) — predicts a coarse forecast `mu` for the
   full output horizon.
2. **Residual diffusion model** — a conditional U-Net that predicts the residual
   `y = frame - mu`, refining the backbone's coarse forecast with high-frequency detail.
   Final prediction per fragment = `mu + y`.

## Important: what's actually trainable here

The **official DiffCast repo never released diffusion-training code**. In
`diffusion.py` (a verbatim port of their `diffcast.py`), the training path is a
deliberate stub:

```python
def predict(self, frames_in, compute_loss=False, **kwargs):
    ...
    if compute_loss:
        raise NotImplementedError(
            "We are sorry that we do not support training process for now "
            "because of business limitation"
        )
```

This matches their README exactly: "Backbone Training" (`python run.py --backbone simvp`)
and "Evaluation" (`--use_diff --eval --ckpt_milestone <pretrained>.pt`) are the only two
documented workflows — diffusion is inference-only, against their pretrained checkpoint.

**Consequently, in this integration:**
- `train.py` / `finetune.py` train **only the backbone** (PhyDNet or SimVP). This is a
  complete, real training loop — nothing missing, standard practice.
- `evaluate.py` optionally runs the **full diffusion sampling pipeline**, but only against
  an externally downloaded pretrained checkpoint (see Mode B below). The diffusion weights
  are never updated by this code.

There is no way to train the diffusion component on your own data with the code currently
in this repo (or in the official DiffCast repo). Doing so would require implementing a
DDPM/v-parameterization training loss from scratch — a distinct, substantial undertaking,
intentionally out of scope for this integration.

## Files

| File | Purpose |
|---|---|
| `eval_metrics.py` | Verbatim copy of the shared metrics module (`hard_csi`, `hard_pod`, `hard_far`, `soft_csi_loss`, `compute_ssim`, `exp_weighted_temporal_ssim`) — identical across all backends. |
| `patch_utils.py` | `mmhr_to_intensity_with_normalize` / `to_pixel_intensity` — shared normalization helpers. |
| `backbones/phydnet.py` | Near-verbatim port of PhyDNet (physically-constrained ConvLSTM, moment-matrix regularization). |
| `backbones/simvp.py` | Near-verbatim port of SimVP (conv encoder → temporal translator → decoder). |
| `model_arch.py` | `DiffCastBackbone` — uniform wrapper exposing `.predict(x, frames_gt=None, compute_loss=False)` for either backbone. |
| `train.py` | Backbone-only Lightning training loop. |
| `finetune.py` | Backbone-only finetuning, with per-backbone freeze strategies. |
| `diffusion.py` | Verbatim port of the sampling-only `Unet` / `ContextNet` / `GaussianDiffusion` (residual diffusion stage). Training path intentionally left raising `NotImplementedError`, matching upstream. |
| `evaluate.py` | Two-mode evaluation script (see below). |

## Data convention

Unlike `predrnnv2` (channels-last `(B,T,H,W,C)`) or `tupann` (channel-less `(B,T,H,W)`),
DiffCast's backbones consume **`(B, T, C, H, W)`** — channel at axis 2, matching the ported
architecture's native convolution shape.

Same `.npz` source format as the rest of the repo: key `"array"` shaped
`(T_total, C_raw, H, W)`, channel 0 = rain rate (mm/hr), normalized via
`mmhr_to_intensity_with_normalize` (`log1p(x)/log1p(60)`, clamped to `[0,1]`).

Default split: `T_in=8, T_out=8` (fits the repo's established 16-frame `.npz` clip length).
This also satisfies **SimVP's hard constraint that `T_out` must be a multiple of `T_in`**
(it autoregresses in `T_in`-sized chunks) — PhyDNet doesn't need this, but both backbones
share the same default for consistency. Override via `--T_in`/`--T_out` if your data differs;
just keep the divisibility constraint in mind if using `--backbone simvp`.

## Usage

### Train (backbone only)

```bash
python train.py -d /path/to/npz_dir --backbone phydnet
# or: --backbone simvp   (requires T_out % T_in == 0)
```

Checkpoints saved to `../../model_weights/diffcast/diffcast-<backbone>-best-{epoch}.ckpt`.

### Evaluate — Mode A (backbone-only, default, fast)

Edit `CONFIG["checkpoint"]` / `CONFIG["data_dir"]` / `CONFIG["backbone"]` in `evaluate.py`,
then:

```bash
python evaluate.py
```

Runs one forward pass per batch, scores with the full shared metric suite (CSI/POD/FAR at
dynamic thresholds, soft-CSI, SSIM overall + per-timestep, EW-SSIM), writes
`validation_results_diffcast.csv` and GridSpec PNGs to `./plottings_diffcast/`.

### Evaluate — Mode B (full diffusion, opt-in, slow)

Requires manually downloading the **only publicly available** pretrained checkpoint,
`diffcast_phydnet_sevir128.pt`, from the Google Drive link in the official
[DiffCast README](../../DiffCast/README.md). Then set:

```python
CONFIG["diffusion_checkpoint"] = "/path/to/diffcast_phydnet_sevir128.pt"
```

Mode B uses its **own independent config** (`CONFIG["diff_*"]`), defaulting to the values
that checkpoint was actually trained with — `diff_T_in=5, diff_T_out=20, diff_img_size=128,
diff_backbone=phydnet, diff_dim=64` — separate from Mode A's `T_in=8/T_out=8/112px`.

**Known caveats, by design, not bugs:**
- The checkpoint was trained on **SEVIR VIL** (satellite reflectivity proxy), not this
  repo's rain-rate radar data — a domain mismatch.
- Requires **25 frames per clip** (`diff_T_in + diff_T_out`) at the official config; this
  repo's `.npz` clips only have 16 — `evaluate.py` raises a clear error if your data doesn't
  have enough frames rather than crashing obscurely. Lowering `diff_T_out` (kept a multiple
  of `diff_T_in`, also enforced) trades fidelity-to-the-original-checkpoint for compatibility.
- Input frames are resized 112×112 → 128×128 (bilinear) before sampling, and predictions
  resized back down to 112×112 for scoring — an interpolation round-trip that affects fidelity.
- Sampling is **~250 DDIM steps × (T_out/T_in) fragments ≈ 1000 sequential network passes per
  batch** — orders of magnitude slower than Mode A or any other backend's single-pass
  inference. `CONFIG["num_eval_samples"]` (default 16) caps how many validation examples are
  scored.

**Do not cite Mode B results as "DiffCast trained on \[your dataset\]."** Since the diffusion
weights are frozen, untouched pretrained weights from a different sensor/resolution/domain,
Mode B measures zero-shot transfer of the authors' SEVIR checkpoint to your (resized) data —
not the performance of DiffCast trained on your data. If included in a paper, it must be
explicitly labeled as such, not placed in a baseline comparison table alongside models that
were actually trained on your dataset.

### Finetune (backbone only)

```bash
python finetune.py -d /path/to/npz_dir -ckpt /path/to/checkpoint.ckpt \
    --backbone phydnet --freeze_mode first_n
```

`--freeze_mode`:
- `none` — everything trainable.
- `first_n` — freezes the backbone's lower/generic-dynamics submodule
  (PhyDNet: `.phycell`; SimVP: `.enc`).
- `all_but_last` — additionally freezes the next submodule
  (PhyDNet: `+ .convcell`; SimVP: `+ .hid`), leaving only the final decoder trainable.

`--backbone`/`--T_in`/`--T_out`/`--img_width` must match the checkpoint being loaded
(loaded with `strict=False`, so mismatches fail silently rather than erroring).

## What's citable

- **Mode A** results (backbone trained on your own data) — standard, legitimate, same
  methodology as every other backend in this repo. Report it as the backbone's performance
  (PhyDNet or SimVP), not as "DiffCast," since the diffusion refinement was never trained.
- **Mode B** results — exploratory only; requires explicit domain-transfer framing, not a
  fair baseline comparison (see caveats above).
