# DDMS

Integration of [DDMS](https://arxiv.org/abs/2404.10512) ("Four-Hour Thunderstorm Nowcasting
using Deep Diffusion Models of Satellite", Dai et al.) into this repo's shared model-backend
conventions (same data pipeline, `eval_metrics.py`, and train/evaluate/finetune structure as
`model/earthformer/`, `model/tupann/`, `model/predrnnv2/`, `model/diffcast/`).

Ported from the vendored official repo at `../../DDMS/`.

## Architecture

DDMS is a **conditional residual-diffusion** video model with three sub-networks:

1. **`denoise_fn`** (`Unet`) — a conditional U-Net that predicts noise at each diffusion step,
   conditioned on per-resolution context via channel-concatenation (not cross-attention).
2. **`history_fn`** (`CondNet`) — a ConvGRU-based recurrent encoder that scans past frames and
   produces the per-scale `context` list consumed by the U-Net.
3. **`transform_fn`** (`HistoryNet`) — a U-Net+ConvGRU "trend" network that predicts a
   deterministic next-frame estimate `mu`; the diffusion model only has to learn the residual
   `x_start - mu`.

All three are wrapped by `GaussianDiffusion`, which owns the noise schedule (cosine), the
DDIM sampler, and the training loss.

## Is this a faithful port? — exact scope

**Model code: architecturally identical.** `Unet`, `CondNet`, `HistoryNet`, `GaussianDiffusion`
in `modules/` are near-verbatim copies of the official repo (only change: dropped an unused
`kornia` import). Same layers, same `forward()`/`p_losses()`/`sample()` logic. If you pass in
the paper's original hyperparameters (`dim=64, dim_mults=(1,1,2,2,4,4), timesteps=1000,
sample_steps=200, img_size=256`), you get the identical architecture described in the paper —
this integration just ships smaller **defaults** (see table below), CLI-overridable.

**Training loss: real, not stubbed.** Unlike `model/diffcast/`, whose vendored repo only
shipped diffusion *inference* code, DDMS's `p_losses`/`forward` (in
`DDMS/modules/denoising_diffusion.py`) implement genuine forward-diffusion noise-prediction
training — verified by reading the code directly, not just the README. `train.py` calls
`GaussianDiffusion.forward(video)` directly; nothing about the loss is reimplemented.

**Training/evaluation harness: reimplemented, not copied.** The official repo drives training
with a hand-rolled multi-GPU (`torch.distributed.launch`, `gloo` backend) loop in
`DDMS/modules/trainer.py`. This integration replaces that with PyTorch Lightning, matching
every other backend in this repo. EMA is functionally reproduced (same beta-averaging formula,
same decay/start-step/update-interval) as a Lightning callback, not their literal `Trainer`
class. `evaluate.py` calls the same `GaussianDiffusion.sample()` method their `test_video.py`
does, but the surrounding script (metrics, CSV/plot output) is a full rewrite using this
repo's `eval_metrics.py`, not their `utils/metrics.py`.

**Data pipeline: fully replaced, different domain.** The official repo consumes FengYun-4A
satellite infrared brightness-temperature PNGs. This integration uses this repo's own
rain-rate `.npz` convention instead — different sensor, different physical variable.

**Explicitly excluded: `gate_unet/` (convection/thunderstorm detection).** A separate
secondary model in the official repo that consumes DDMS's saved image outputs for a
downstream binary-detection task. Not ported — it's architecturally unrelated to the
nowcasting model itself, ships no training code (inference-only via their checkpoint), and
its `GatedSpatialConv.py` is NVIDIA CC BY-NC-SA-licensed code embedded in an otherwise-MIT
repo (a license conflict). If you cite this integration, call it the **DDMS nowcasting
model**, not the full two-stage paper system.

## Files

| File | Purpose |
|---|---|
| `eval_metrics.py` | Verbatim copy of the shared metrics module. |
| `patch_utils.py` | `mmhr_to_intensity_with_normalize`/`to_pixel_intensity` (shared convention) plus `to_pm1`/`from_pm1` for DDMS's internal `[-1,1]` convention. |
| `modules/` | Near-verbatim port: `denoising_diffusion.py` (`GaussianDiffusion`), `unet.py` (`Unet`), `temporal_models.py` (`CondNet`, `HistoryNet`), `network_components.py`, `utils.py`. |
| `model_arch.py` | `build_ddms_model()` — wires the three sub-networks together; `T_CONTEXT=8`, `T_OUT=8` fixed constants. |
| `train.py` | Lightning training loop (real diffusion loss) + `EMACallback`. |
| `evaluate.py` | DDIM-sampling-based evaluation, full shared metric suite. |
| `finetune.py` | Finetuning with three freeze strategies. |

## Data convention

Same `.npz` source format as the rest of the repo: key `"array"` shaped `(16, C_raw, H, W)`,
channel 0 = rain rate (mm/hr), normalized via `mmhr_to_intensity_with_normalize` to `[0,1]`,
then rescaled to `[-1,1]` internally (`to_pm1`) to match DDMS's native convention.

**`T_CONTEXT=8, T_OUT=8` are fixed, not CLI-configurable.** The vendored `GaussianDiffusion.
forward()` hardcodes the context/target split at frame index 8 (`if i >= 8: compute loss`).
This exactly matches an 8+8 split of the repo's established 16-frame `.npz` clips, so no
changes to the diffusion math were needed — but it also means this split can't be changed
without patching `modules/denoising_diffusion.py`.

Tensor convention inside the model is **time-major** `(T, B, C, H, W)`, unlike predrnnv2's
channels-last or diffcast's `(B,T,C,H,W)` — `train.py`/`evaluate.py` handle the transpose from
this repo's batch-major `.npz` loading.

## Scaled-down defaults

The paper's setup (256×256, 8-GPU distributed, `n_step=3,000,000`) isn't reproducible on
modest hardware or at this repo's 112×112 resolution. Defaults here:

| param | paper | this integration's default |
|---|---|---|
| img size | 256 | 112 |
| `dim` | 64 | 32 |
| `dim_mults` (U-Net/CondNet) | (1,1,2,2,4,4) | (1,1,2,2,4) |
| `transform_dim_mults` (HistoryNet) | (1,2,3,4) | unchanged |
| diffusion `timesteps` (train) | 1000 | 500 |
| DDIM `sample_steps` (eval) | 200 | 50 |
| EMA | on | opt-in via `--use_ema`, off by default |

All CLI-overridable — pass the paper's values if you want the literal original architecture
size (needs more compute/memory).

## Usage

### Train

```bash
python train.py -d /path/to/npz_dir --dim 32 --dim_mults "1,1,2,2,4" \
    --transform_dim_mults "1,2,3,4" --timesteps 500 --sample_steps 50 --use_ema
```

`training_step` calls `GaussianDiffusion.forward(video)` directly — the real diffusion loss,
not a proxy. `validation_step` uses the same cheap teacher-forced call (no autoregressive
sampling), so validation epochs stay fast; expensive DDIM sampling only happens in
`evaluate.py`. Checkpoints saved to `../../model_weights/ddms/ddms-best-{epoch}.ckpt`.

### Evaluate

Edit `CONFIG["checkpoint"]`/`CONFIG["data_dir"]` (and the architecture fields — must match
what the checkpoint was trained with) in `evaluate.py`, then:

```bash
python evaluate.py
```

Each validation example costs `T_OUT × sample_steps` sequential U-Net passes (DDIM sampling
is inherently iterative — there's no fast shortcut here, unlike diffcast's backbone-only Mode
A). `CONFIG["num_eval_samples"]` (default 16) caps how many examples get scored. Outputs
`validation_results_ddms.csv` and GridSpec PNGs to `./plottings_ddms/`.

`load_model()` accepts three checkpoint formats: this repo's Lightning `.ckpt`, a raw DDMS
checkpoint (`{"model","ema"}`, `module.`-prefixed from DDP — set `CONFIG["use_ema"]=True` to
prefer the EMA weights if present), or a plain state_dict.

### Finetune

```bash
python finetune.py -d /path/to/npz_dir -ckpt /path/to/checkpoint.ckpt --freeze_mode freeze_context
```

`--freeze_mode`:
- `none` — everything trainable.
- `freeze_context` — freezes `history_fn` (CondNet) + `transform_fn` (HistoryNet), adapts only
  the denoising U-Net.
- `freeze_denoiser` — freezes `denoise_fn` (Unet), adapts only the temporal/trend encoders.
- `freeze_transform` — freezes only `transform_fn`, lets the U-Net and context encoder adapt.

Architecture flags (`--dim`, `--dim_mults`, `--transform_dim_mults`, `--timesteps`) must match
the checkpoint being loaded (loaded with `strict=False`, so mismatches fail silently rather
than erroring).

## What's citable

Training this on your own dataset produces a legitimate result — the real diffusion loss,
real gradients, real convergence, same standing as any other backend in this repo. To keep
the claim accurate:

1. **Report the actual config you trained at** (this integration's scaled-down defaults, or
   your own), not the paper's 256px/8-GPU setup, unless you actually matched it.
2. **Call it the DDMS nowcasting model**, not the full two-stage system — `gate_unet`
   (convection detection) isn't included.
3. **State the domain**: this applies DDMS's architecture to your dataset, not a reproduction
   of their FengYun-4A satellite results.
4. A trained-and-evaluated run on your real data is required before there's a result to cite —
   verification so far used tiny synthetic data for one epoch, confirming the code runs
   correctly end-to-end, not a real trained model.
