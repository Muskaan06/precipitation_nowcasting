import torch.nn as nn

from backbones import get_phydnet_model, get_simvp_model


class DiffCastBackbone(nn.Module):
    """
    Uniform wrapper around DiffCast's deterministic backbones (PhyDNet or
    SimVP), adapted from DiffCast/models/{phydnet,simvp}. Both expose
    .predict(frames_in, frames_gt=None, compute_loss=False) -> (pred, loss).

    Data convention: (B, T, C, H, W) — channel at axis 2, matching the
    backbones' native shape (unlike predrnnv2's channels-last convention).
    """

    def __init__(self, backbone="phydnet", C=1, H=112, W=112, T_in=8, T_out=8, device="cuda"):
        super().__init__()

        if backbone == "simvp" and T_out % T_in != 0:
            raise ValueError(
                f"SimVP requires T_out to be a multiple of T_in (it autoregresses in "
                f"chunks of T_in), got T_in={T_in}, T_out={T_out}."
            )

        self.backbone_name = backbone
        self.T_in = T_in
        self.T_out = T_out
        in_shape = (C, H, W)

        if backbone == "phydnet":
            self.net = get_phydnet_model(in_shape, T_in, T_out, device=device)
        elif backbone == "simvp":
            self.net = get_simvp_model(in_shape, T_in, T_out)
        else:
            raise ValueError(f"Unknown backbone: {backbone!r} (expected 'phydnet' or 'simvp')")

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        return self.net.predict(frames_in, frames_gt=frames_gt, compute_loss=compute_loss)
