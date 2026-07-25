import math
import torch


def reserve_schedule_sampling_exp(step, batch_size, input_length, total_length,
                                   r_sampling_step_1=25000, r_sampling_step_2=50000,
                                   r_exp_alpha=5000, device="cpu"):
    """
    Reverse scheduled sampling mask, adapted from predrnn-pytorch/run.py's
    reserve_schedule_sampling_exp. Ported to torch (per-sample-per-timestep
    coin flips broadcast over the spatial/channel dims, rather than
    materializing full (h, w, C) blocks of ones/zeros as the numpy original
    does — mathematically identical, just avoids the extra allocation).

    Early in training (step < r_sampling_step_1) the model is fed mostly its
    own generated frames even during the "input" phase (r_eta=0.5, eta=0.5);
    as `step` grows past r_sampling_step_1 towards r_sampling_step_2, ground
    truth is used more during the input phase (r_eta -> 1) while ground
    truth in the output phase is phased OUT (eta -> 0) so that by
    r_sampling_step_2 the model trains under conditions identical to test
    time (pure autoregression past `input_length`).

    Returns
    -------
    mask : (batch_size, total_length - 2, 1, 1, 1) tensor of 0.0/1.0,
           to be broadcast-multiplied against patchified NCHW-time frames.
           mask[:, t-1] is used to blend at timestep t (t = 1 .. total_length-2).
    """
    if step < r_sampling_step_1:
        r_eta = 0.5
    elif step < r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(step - r_sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0

    if step < r_sampling_step_1:
        eta = 0.5
    elif step < r_sampling_step_2:
        eta = 0.5 - (0.5 / (r_sampling_step_2 - r_sampling_step_1)) * (step - r_sampling_step_1)
    else:
        eta = 0.0

    r_true_token = (torch.rand(batch_size, input_length - 1, device=device) < r_eta).float()
    true_token = (torch.rand(batch_size, total_length - input_length - 1, device=device) < eta).float()

    mask = torch.cat([r_true_token, true_token], dim=1)  # (B, total_length - 2)
    return mask.view(batch_size, total_length - 2, 1, 1, 1)
