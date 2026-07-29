from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.temporal_models import CondNet, HistoryNet

# Fixed by the vendored GaussianDiffusion.forward()'s hardcoded context/target
# split (frame index 8) — not a constructor parameter. Matches this repo's
# established 16-frame .npz clip length. Changing these requires patching
# modules/denoising_diffusion.py's forward().
T_CONTEXT = 8
T_OUT = 8


def build_ddms_model(dim=32, dim_mults=(1, 1, 2, 2, 4), transform_dim_mults=(1, 2, 3, 4),
                      channels=1, backbone="resnet", context_dim_factor=1, transform_dim_factor=1,
                      timesteps=500, loss_type="l1", pred_mode="noise", clip_noise=True,
                      sample_steps=50, ddim=True):
    """
    Builds the full DDMS residual-diffusion model: a conditional U-Net
    denoiser (denoise_fn), a ConvGRU context encoder (history_fn), and a
    ConvGRU trend/residual network (transform_fn), wired together exactly as
    DDMS/train_video.py does.
    """
    denoise_fn = Unet(
        dim=dim,
        context_dim_factor=context_dim_factor,
        channels=channels,
        dim_mults=dim_mults,
        backbone=backbone,
    )

    history_fn = CondNet(
        dim=int(context_dim_factor * dim),
        channels=channels,
        backbone=backbone,
        dim_mults=dim_mults,
    )

    transform_fn = HistoryNet(
        dim=int(transform_dim_factor * dim),
        channels=channels,
        context_mode="residual",
        dim_mults=transform_dim_mults,
        backbone=backbone,
    )

    return GaussianDiffusion(
        denoise_fn=denoise_fn,
        history_fn=history_fn,
        transform_fn=transform_fn,
        channels=channels,
        timesteps=timesteps,
        loss_type=loss_type,
        pred_mode=pred_mode,
        clip_noise=clip_noise,
        sample_steps=sample_steps,
        ddim=ddim,
    )
