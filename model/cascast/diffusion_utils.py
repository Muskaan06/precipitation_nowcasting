"""
diffusion_utils.py
===================
Self-contained helpers that sit between the (unchanged) CasFormer architecture
in `model_arch.py` and the training / evaluation scripts:

  - GaussianDiffusion : minimal linear-beta-schedule DDPM (q_sample for
                        training, DDIM-style p_sample_loop for sampling).
                        No dependency on `diffusers`, matching the
                        dependency-light style of the tupann/earthformer
                        folders.
  - load_coarse_forecaster : loads a frozen Earthformer model to supply the
                        "coarse prediction" that CasFormer conditions on and
                        refines — matching the CasCast paper's design of
                        pairing the diffusion transformer with a
                        deterministic forecaster. Our dataset uses a 30-min
                        timestep (unlike SEVIR's 5-min), so instead of the
                        SEVIR-pretrained earthformer_sevir.pt we use our own
                        Earthformer checkpoint trained on this dataset at
                        4 input / 6 output frames (see
                        ../earthformer/checkpoints/lightning_logs_4_6).
                        That checkpoint's positional embeddings
                        (enc_pos_embed.T_embed: [4,128], dec_pos_embed.T_embed:
                        [6,256]) confirm it was trained with input_shape=
                        (4,112,112,1)/target_shape=(6,112,112,1) — neither
                        ../earthformer/model_arch.py nor model_arch2.py (both
                        hardcode 13/12) matches that, so the CuboidTransformerModel
                        is built here directly with the checkpoint's actual
                        shapes, reusing model_arch.py's other hyperparameters
                        (base_units=128, num_heads=4, num_global_vectors=8 —
                        confirmed by that run's hparams.yaml). Falls back to a
                        persistence baseline (repeat last observed frame) when
                        no checkpoint is given, so the pipeline stays runnable
                        standalone.
  - run_coarse_forecaster : adapts between cascast's channels-first
                        (B, T, H, W) rain tensors and Earthformer's
                        channels-last (B, T, H, W, C) interface.
"""

import pathlib
import sys

import torch
import torch.nn as nn


# =============================================================================
# Gaussian diffusion (linear beta schedule, DDPM training / DDIM sampling)
# =============================================================================

class GaussianDiffusion:
    """
    x: clean target, shape (B, T, C, H, W) — rain frames or any tensor
    CasFormer predicts the noise added to x at timestep t, conditioned on
    a coarse prediction of the same shape.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device):
        for name in ("betas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: sample x_t given x_0 and timestep t (B,)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        shape = [x_start.shape[0]] + [1] * (x_start.ndim - 1)
        sqrt_ac = self.sqrt_alphas_cumprod[t].reshape(shape)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].reshape(shape)
        return sqrt_ac * x_start + sqrt_1m_ac * noise, noise

    @torch.no_grad()
    def ddim_sample_loop(self, model, cond, shape, num_inference_steps=20, device="cpu"):
        """
        DDIM-style deterministic sampling (eta=0), conditioned on `cond`.
        model(x, t, cond) -> predicted noise, same shape as x.
        """
        step_indices = torch.linspace(self.timesteps - 1, 0, num_inference_steps, device=device).long()
        x = torch.randn(shape, device=device)

        for i, t in enumerate(step_indices):
            t_batch = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            noise_pred = model(x=x, timesteps=t_batch, cond=cond)

            alpha_cumprod_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i + 1 < len(step_indices):
                t_prev = step_indices[i + 1]
                alpha_cumprod_prev = self.alphas_cumprod[t_prev]
                x = torch.sqrt(alpha_cumprod_prev) * x0_pred + torch.sqrt(1 - alpha_cumprod_prev) * noise_pred
            else:
                x = x0_pred

        return x


# =============================================================================
# Frozen Earthformer coarse forecaster
# (reference: ../earthformer/model_arch.py + our own 4-in/6-out checkpoint)
# =============================================================================

EARTHFORMER_DIR = pathlib.Path(__file__).resolve().parent.parent / "earthformer"
EARTHFORMER_INPUT_LENGTH = 4    # matches enc_pos_embed.T_embed [4, 128] in our checkpoint
EARTHFORMER_TARGET_LENGTH = 10   # matches dec_pos_embed.T_embed [6, 256] in our checkpoint
DEFAULT_COARSE_CKPT = str(
    EARTHFORMER_DIR / "checkpoints" / "lightning_logs_4_10" / "version_2"
    / "checkpoints" / "earthformer-best-epoch=2.ckpt"
)


def _load_cuboid_transformer_model_cls():
    """
    Dynamically loads ../earthformer/earthformer_module.py for
    CuboidTransformerModel — the underlying network both
    ../earthformer/model_arch.py and model_arch2.py wrap. It plain-imports
    sibling modules (earthformer_patterns, earthformer_utils, registry) that
    only resolve once the earthformer folder itself is on sys.path.
    """
    earthformer_dir = str(EARTHFORMER_DIR)
    if earthformer_dir not in sys.path:
        sys.path.insert(0, earthformer_dir)

    import earthformer_module
    return earthformer_module.CuboidTransformerModel


def _build_earthformer_4_6(base_units=128, num_heads=4, num_global_vectors=8):
    """
    Builds the CuboidTransformerModel with input_shape=(4,112,112,1) /
    target_shape=(6,112,112,1) — the shapes our own checkpoint
    (checkpoints/lightning_logs_4_6) was actually trained with, confirmed by
    its enc/dec T_embed sizes. Same construction as
    ../earthformer/model_arch.py otherwise (that file currently hardcodes
    13/12, tailored for the unrelated SEVIR-pretrained earthformer_sevir.pt,
    so it can't be reused directly for this checkpoint).
    """
    CuboidTransformerModel = _load_cuboid_transformer_model_cls()

    input_shape = (EARTHFORMER_INPUT_LENGTH, 112, 112, 1)
    target_shape = (EARTHFORMER_TARGET_LENGTH, 112, 112, 1)
    num_blocks = 2

    return CuboidTransformerModel(
        input_shape=input_shape,
        target_shape=target_shape,
        base_units=base_units,
        block_units=None,
        scale_alpha=1.0,
        enc_depth=[1, 1],
        dec_depth=[1, 1],
        enc_use_inter_ffn=True,
        dec_use_inter_ffn=True,
        dec_hierarchical_pos_embed=False,
        downsample=2,
        downsample_type="patch_merge",
        enc_attn_patterns=["axial"] * num_blocks,
        dec_self_attn_patterns=["axial"] * num_blocks,
        dec_cross_attn_patterns=["cross_1x1"] * num_blocks,
        dec_cross_last_n_frames=None,
        dec_use_first_self_attn=False,
        num_heads=num_heads,
        attn_drop=0.1,
        proj_drop=0.1,
        ffn_drop=0.1,
        upsample_type="upsample",
        ffn_activation="gelu",
        gated_ffn=False,
        norm_layer="layer_norm",
        num_global_vectors=num_global_vectors,
        use_dec_self_global=False,
        dec_self_update_global=True,
        use_dec_cross_global=False,
        use_global_vector_ffn=False,
        use_global_self_attn=True,
        separate_global_qkv=True,
        global_dim_ratio=1,
        initial_downsample_type="stack_conv",
        initial_downsample_activation="leaky",
        initial_downsample_stack_conv_num_layers=3,
        initial_downsample_stack_conv_dim_list=[base_units // 4, base_units // 2, base_units],
        initial_downsample_stack_conv_downscale_list=[3, 2, 2],
        initial_downsample_stack_conv_num_conv_list=[2, 2, 2],
        padding_type="zeros",
        z_init_method="zeros",
        checkpoint_level=0,
        pos_embed_type="t+h+w",
        use_relative_pos=True,
        self_attn_use_final_proj=True,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        conv_init_mode="0",
        down_up_linear_init_mode="0",
        norm_init_mode="0",
    )


def load_coarse_forecaster(ckpt_path=None, device="cpu"):
    """
    Returns a frozen, eval-mode Earthformer (CuboidTransformerModel) producing
    the coarse prediction that CasFormer refines, or None if no checkpoint is
    given (callers should fall back to a persistence baseline in that case).
    """
    if ckpt_path is None:
        return None

    model = _build_earthformer_4_6().to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # PyTorch Lightning checkpoint — strip the "model." prefix
        prefix = "model."
        state = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
        model.load_state_dict(state, strict=True)
    else:
        # Raw net-only state-dict
        model.load_state_dict(ckpt, strict=True)

    model.eval().requires_grad_(False)
    return model


def run_coarse_forecaster(coarse_model, x):
    """
    x: (B, T_in, H, W) rain frames, channels-first, T_in == EARTHFORMER_INPUT_LENGTH.
    Earthformer expects/returns channels-last tensors: (B, T, H, W, 1).
    Returns: (B, T_out, H, W) with T_out == EARTHFORMER_TARGET_LENGTH.
    """
    x_ = x.unsqueeze(-1)          # (B, T_in, H, W, 1)
    out = coarse_model(x_)        # (B, T_out, H, W, 1)
    return out.squeeze(-1)        # (B, T_out, H, W)


def persistence_coarse(x, target_length):
    """Fallback coarse prediction: repeat the last observed frame."""
    last = x[:, -1:]
    return last.repeat(1, target_length, 1, 1)
