import torch
import torch.nn as nn
import torch.nn.functional as F

from cell import SpatioTemporalLSTMCell
from patch_utils import reshape_patch, reshape_patch_back


class PredRNNv2Core(nn.Module):
    """
    PredRNN-V2 (Wang et al.) adapted from predrnn-pytorch/core/models/predrnn_v2.py
    to this repo's fixed-length nowcasting convention.

    forward(x) takes channels-last (B, T_in, H, W, img_channel) ground-truth
    input frames, teacher-forces the ST-LSTM stack for the first
    `input_length` steps, then autoregresses on its own generated frames for
    the remaining `total_length - input_length` steps. This is the path used
    at inference time (evaluate.py) and whenever training runs without
    reverse scheduled sampling.

    forward(x, y, mask_true) — used during training when reverse scheduled
    sampling is enabled (see scheduling.reserve_schedule_sampling_exp) — takes
    the full ground-truth sequence (x concatenated with the target y) plus a
    per-timestep 0/1 mask, and blends ground-truth vs. self-generated frames
    at every step past t=0: net = mask*gt_frame + (1-mask)*x_gen. This lets
    the model occasionally see its own generated frames even during the
    "input" phase, matching predrnn-pytorch/core/models/predrnn_v2.py's
    training procedure. At t=0 the mask is unused; validation/inference
    should never pass mask_true (always pure teacher-force-then-autoregress).

    Returns (pred, decouple_loss):
      pred          : (B, T_out, H, W, img_channel), T_out = total_length - input_length
      decouple_loss : scalar regularization term (mean |cosine_similarity(delta_c, delta_m)|
                      across layers/timesteps) — add `decouple_beta * decouple_loss`
                      to the training loss.
    """

    def __init__(self, num_layers=4, num_hidden=(64, 64, 64, 64), img_channel=1,
                 img_width=112, patch_size=4, filter_size=5, stride=1,
                 layer_norm=False, input_length=6, total_length=16,
                 decouple_beta=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.num_hidden = list(num_hidden)
        self.img_channel = img_channel
        self.patch_size = patch_size
        self.input_length = input_length
        self.total_length = total_length
        self.decouple_beta = decouple_beta

        self.frame_channel = patch_size * patch_size * img_channel
        width = img_width // patch_size
        self.width = width

        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], width,
                                       filter_size, stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[-1], self.frame_channel,
                                    kernel_size=1, stride=1, padding=0, bias=False)

        # shared adapter used for the decoupling loss
        adapter_num_hidden = self.num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1,
                                  stride=1, padding=0, bias=False)

    def forward(self, x, y=None, mask_true=None):
        # x: (B, T_in, H, W, img_channel) channels-last
        device = x.device
        batch = x.shape[0]

        full = torch.cat([x, y], dim=1) if y is not None else x
        frames = reshape_patch(full, self.patch_size)        # (B, T_in or total_length, h, w, Cp)
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()  # (..., Cp, h, w)

        h_t, c_t = [], []
        delta_c_list, delta_m_list = [], []
        for i in range(self.num_layers):
            zeros = torch.zeros(batch, self.num_hidden[i], self.width, self.width, device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros(batch, self.num_hidden[0], self.width, self.width, device=device)

        next_frames = []
        decouple_loss = []

        for t in range(self.total_length - 1):
            if t == 0:
                net = frames[:, 0]
            elif mask_true is not None:
                m = mask_true[:, t - 1]
                net = m * frames[:, t] + (1 - m) * x_gen
            elif t < self.input_length:
                net = frames[:, t]
            else:
                net = x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            x_gen = self.conv_last(h_t[-1])
            next_frames.append(x_gen)

            for i in range(self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)))
                )

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        # [total_length-1, B, Cp, h, w] -> [B, total_length-1, h, w, Cp]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = reshape_patch_back(next_frames, self.patch_size)  # (B, total_length-1, H, W, C)

        T_out = self.total_length - self.input_length
        pred = next_frames[:, -T_out:]

        return pred, decouple_loss
