"""
Earthformer Lightning module — corrected to match saved checkpoint architecture.

Confirmed from state dict:
  - input_shape  : (T_in,  112, 112, 1)  — T_in  sized by enc_pos_embed.T_embed [13, 128]
  - target_shape : (T_out, 112, 112, 1)  — T_out sized by dec_pos_embed.T_embed [12, 256]
  - base_units = 128
  - num_global_vectors = 8
  - num_heads = 4
  - stack_conv_dims = [16, 64, 128]   (from conv_block_list weight shapes)
  - stack_conv_downscale = [3, 2, 2]  (from patch_merge_list norm shapes)
  - enc spatial after stem: 32x32     (from enc_pos_embed H/W embed [32, 128])
  - dec spatial: 16x16                (from dec_pos_embed H/W embed [16, 256])
  - encoder: 2 stages, dim 128 → 256  (blocks.0 ffn [512,128], blocks.1 ffn [1024,256])
  - decoder: 1 self-attn stage + 2 cross-attn stages
  - output channel: 1                 (dec_final_proj [1, 16])
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from earthformer_module import CuboidTransformerModel


class EarthformerModel(pl.LightningModule):
    """
    Earthformer expects channels-last input: (B, T, H, W, C)
      input : (B, 13, 112, 112, 1)   — T_in=13  from enc_pos_embed.T_embed [13, 128]
      output: (B, 12, 112, 112, 1)   — T_out=12 from dec_pos_embed.T_embed [12, 256]
    """

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        base_units: int = 128,
        num_heads: int = 4,
        num_global_vectors: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── shapes derived strictly from this checkpoint's state dict ──────
        input_shape  = (13, 112, 112, 1)   # T_in=13  ← enc_pos_embed.T_embed [13,128]
        target_shape = (12, 112, 112, 1)   # T_out=12 ← dec_pos_embed.T_embed [12,256]

        # ── CNN stem dims confirmed from conv_block_list weight shapes ──────
        #   block 0 conv: [16,  1, 3,3]  → out=16  = base//8
        #   block 1 conv: [64, 16, 3,3]  → out=64  = base//2
        #   block 2 conv: [128,64, 3,3]  → out=128 = base
        stack_conv_dims      = [base_units // 8, base_units // 2, base_units]  # [16, 64, 128]

        # ── downscale confirmed from patch_merge_list norm shapes ────────────
        #   patch_merge_list.0.norm [144]  = 16  * 3² → stride-3
        #   patch_merge_list.1.norm [256]  = 64  * 2² → stride-2
        #   patch_merge_list.2.norm [512]  = 128 * 2² → stride-2
        stack_conv_downscale = [3, 2, 2]

        # ── 2 convs per block confirmed from indices .0 and .3 in each block ─
        stack_conv_num_conv  = [2, 2, 2]

        self.model = CuboidTransformerModel(
            input_shape=input_shape,
            target_shape=target_shape,

            # ── channel / dimension ────────────────────────────────────────
            base_units=base_units,          # 128
            block_units=None,               # auto-scales: [128, 256]
            scale_alpha=1.0,

            # ── encoder: 2 depth-1 stages (blocks.0 and blocks.1) ──────────
            enc_depth=[1, 1],
            dec_depth=[1, 1],
            enc_use_inter_ffn=True,
            dec_use_inter_ffn=True,
            dec_hierarchical_pos_embed=False,

            # ── patch merge downsampling (patch_merge_list present) ─────────
            downsample=2,
            downsample_type="patch_merge",

            # ── attention patterns ──────────────────────────────────────────
            #   encoder attn_l:      qkv + l2g/g2l/g2g nets → axial + global
            #   decoder self_blocks: qkv only (no global nets) → axial
            #   decoder cross_blocks: q_proj + kv_proj → cross_1x1
            enc_attn_patterns=["axial"] * 2,
            dec_self_attn_patterns=["axial"] * 2,
            dec_cross_attn_patterns=["cross_1x1"] * 2,

            dec_cross_last_n_frames=None,
            dec_use_first_self_attn=False,

            # ── attention config ────────────────────────────────────────────
            #   qkv.weight [384, 128] → 3*128=384, heads=4, head_dim=32 ✓
            num_heads=num_heads,            # 4
            attn_drop=0.1,
            proj_drop=0.1,
            ffn_drop=0.1,

            # ── upsample: confirmed from upsample_list conv weights ──────────
            upsample_type="upsample",

            # ── FFN / norm ──────────────────────────────────────────────────
            ffn_activation="gelu",
            gated_ffn=False,
            norm_layer="layer_norm",

            # ── global vectors: init_global_vectors [8, 128] ────────────────
            num_global_vectors=num_global_vectors,   # 8

            # ── encoder HAS global attn (l2g/g2l/g2g nets present) ──────────
            use_global_self_attn=True,
            separate_global_qkv=True,
            global_dim_ratio=1,

            # ── decoder does NOT have global attn (no l2g/g2g in self_blocks) ─
            use_dec_self_global=False,
            dec_self_update_global=True,
            use_dec_cross_global=False,
            use_global_vector_ffn=False,

            # ── initial CNN stem ────────────────────────────────────────────
            initial_downsample_type="stack_conv",
            initial_downsample_activation="leaky",
            initial_downsample_stack_conv_num_layers=3,
            initial_downsample_stack_conv_dim_list=stack_conv_dims,            # [16, 64, 128]
            initial_downsample_stack_conv_downscale_list=stack_conv_downscale, # [3, 2, 2]
            initial_downsample_stack_conv_num_conv_list=stack_conv_num_conv,   # [2, 2, 2]

            # ── misc ────────────────────────────────────────────────────────
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
        """x: (B, T, H, W, C) channels-last → returns (B, T_out, H, W, C)"""
        return self.model(x)

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
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )