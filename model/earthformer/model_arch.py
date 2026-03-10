"""
Earthformer Lightning module — independent of repo's LModule base class.
Wraps CuboidTransformerModel for training on (4, 3, 112, 112) -> (6, 3, 112, 112) data.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn

from earthformer.earthformer_module import CuboidTransformerModel


class EarthformerModel(pl.LightningModule):
    """
    Earthformer expects channels-last input: (B, T, H, W, C)
    - input : (B, 4, 112, 112, 3)
    - output: (B, 6, 112, 112, 3)
    """
    def __init__(self, lr=1e-4, weight_decay=0.0, base_units=128, num_heads=4, num_global_vectors=8):
        super().__init__()
        self.save_hyperparameters()

        input_shape  = (4,  112, 112, 3)  # (T_in,  H, W, C)
        target_shape = (6,  112, 112, 3)  # (T_out, H, W, C)

        num_blocks = 2
        self.model = CuboidTransformerModel(
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
            initial_downsample_stack_conv_dim_list=[base_units//4, base_units//2, base_units],
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
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        # x: (B, T, H, W, C) — channels last
        return self.model(x)  # out: (B, T_out, H, W, C)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)