"""AGDiff: full Aligned-and-Guided Diffusion model (Section 3.3).

Extends MDM (transformer encoder variant) with a parallel GuideNet that injects
sketch-derived features layer-by-layer via zero-initialised linear projections.

Training stages (Section 4.2):
    Stage 1  – Train AlignNet alone (train/train_align_net.py).
    Stage 2  – Freeze / jointly fine-tune AlignNet; train AGDiff end-to-end.

Loss (Equations 5-7):
    L = L_diff  +  λ_s · L_sketch
where
    L_diff   = E[||ε − ε_θ(x_t, t, c)||²₂]
    L_sketch = (1/N_k) Σ_{i∈K} ||x̂_i − x^sketch_i||²₂
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mdm import MDM
from model.guide_net import GuideNet
from model.align_net import AlignNet


class AGDiff(MDM):
    """AG-Diff = MDM Generation Net + GuideNet (ControlNet-style injection).

    The Generation Net (this class) is the standard MDM transformer encoder.
    The GuideNet runs in parallel and injects per-layer residuals.

    Extra constructor arguments (on top of MDM's):
        sketch_latent_dim  – dimensionality of AlignNet's output (default 256)
        guide_num_layers   – number of GuideNet transformer layers
        guide_ff_size      – feed-forward size in GuideNet
        align_net          – optional pre-built AlignNet; if provided it is
                             registered as a sub-module (can be frozen externally)
    """

    def __init__(self,
                 *args,
                 sketch_latent_dim: int = 256,
                 guide_num_layers: int = 8,
                 guide_ff_size: int = 1024,
                 align_net: Optional[AlignNet] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.sketch_latent_dim = sketch_latent_dim

        # ── AlignNet (may be frozen after Stage-1 training) ───────────────── #
        self.align_net = align_net  # None is fine; can set later

        # ── Guide Net (parallel transformer, zero-init injection) ─────────── #
        self.guide_net = GuideNet(
            latent_dim=self.latent_dim,
            ff_size=guide_ff_size,
            num_layers=min(guide_num_layers, self.num_layers),
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        # ── Optional sketch-latent → model-latent projection ──────────────── #
        if sketch_latent_dim != self.latent_dim:
            self.sketch_proj: nn.Module = nn.Linear(sketch_latent_dim, self.latent_dim)
        else:
            self.sketch_proj = nn.Identity()

        # Initialise GuideNet with Generation Net weights after super().__init__
        # (seqTransEncoder is now fully built)
        self.guide_net.copy_from_generation_net(self.seqTransEncoder)

    # ── parameter helpers ─────────────────────────────────────────────────── #

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters()
                if not name.startswith('clip_model.')]

    def parameters_guide_only(self):
        """Parameters that only belong to the GuideNet + projection."""
        return (
            list(self.guide_net.parameters())
            + list(self.sketch_proj.parameters())
        )

    # ── forward ───────────────────────────────────────────────────────────── #

    def forward(self, x, timesteps, y=None, cond_val=None, cond_mask=None):
        """Extended forward pass with optional sketch guidance injection.

        Args:
            x          : [bs, njoints, nfeats, nframes]  – noisy motion x_t
            timesteps  : [bs]
            y          : conditioning dict; expected keys:
                           'text'                   – list[str] text prompts
                           'sketch_latent'          – [bs, sketch_latent_dim] or None
                           'sketch_keyframe_idx'    – list[int] (0-indexed, excluding
                                                      the timestep token offset)
                           'uncond'                 – bool (CFG unconditional pass)
            cond_val, cond_mask : for better_cond variants (passed through)

        Returns:
            [bs, njoints, nfeats, nframes]  – predicted noise / x₀
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, latent_dim]

        force_mask = y.get('uncond', False) if y is not None else False

        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb = emb + self.embed_text(
                self.mask_cond(enc_text, force_mask=force_mask)
            )
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb = emb + self.mask_cond(action_emb, force_mask=force_mask)

        # Skip-connection source (raw input features)
        src = x.permute(3, 0, 1, 2).reshape(nframes, bs, njoints * nfeats)

        # Project input to latent space
        x_proj = self.input_process(x)  # [nframes, bs, latent_dim]

        # Build sequence: [timestep_token, frame_0, …, frame_{T-1}]
        xseq = torch.cat((emb, x_proj), dim=0)   # [nframes+1, bs, latent_dim]
        xseq = self.sequence_pos_encoder(xseq)
        seqlen = xseq.shape[0]  # nframes + 1

        # ── Guide Net residuals ──────────────────────────────────────────── #
        guide_residuals = None
        if (
            not force_mask
            and y is not None
            and y.get('sketch_latent') is not None
        ):
            sketch_latent = self.sketch_proj(y['sketch_latent'])  # [bs, latent_dim]
            raw_idx = y.get('sketch_keyframe_idx', None)
            # Shift by +1 to account for the prepended timestep token
            kf_idx = [k + 1 for k in raw_idx] if raw_idx is not None else None
            guide_residuals = self.guide_net(sketch_latent, seqlen, kf_idx)

        # ── Generation Net transformer (layer-by-layer) ──────────────────── #
        output = xseq
        for i, layer in enumerate(self.seqTransEncoder.layers):
            output = layer(output)
            if guide_residuals is not None and i < len(guide_residuals):
                output = output + guide_residuals[i]

        # Apply final LayerNorm if present
        if self.seqTransEncoder.norm is not None:
            output = self.seqTransEncoder.norm(output)

        # Remove the timestep token
        output = output[1:]  # [nframes, bs, latent_dim]
        return self.output_process(output, skip=src)  # [bs, njoints, nfeats, nframes]

    # ── sketch-loss helper (Equation 6) ─────────────────────────────────── #

    @staticmethod
    def sketch_loss(pred_x0: torch.Tensor,
                    sketch_keyframes: torch.Tensor,
                    keyframe_mask: torch.Tensor) -> torch.Tensor:
        """Spatial consistency loss at sketch-derived keyframe positions.

        Args:
            pred_x0        : [bs, njoints, nfeats, nframes]  – predicted x₀
            sketch_keyframes: [bs, njoints, nfeats, nframes]  – sketch-derived target
            keyframe_mask  : [bs, njoints, nfeats, nframes]  bool  – True at keyframes

        Returns:
            scalar MSE loss
        """
        diff = (pred_x0 - sketch_keyframes) ** 2
        n_k = keyframe_mask.float().sum().clamp(min=1.0)
        return (diff * keyframe_mask.float()).sum() / n_k

    # ── convenience: encode sketch at inference ──────────────────────────── #

    @torch.no_grad()
    def encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """Encode a sketch image using the attached AlignNet.

        sketch : [bs, 1|3, H, W]
        Returns: [bs, sketch_latent_dim]
        """
        assert self.align_net is not None, "AlignNet not attached to AGDiff"
        return self.align_net.encode_sketch(sketch)

    @torch.no_grad()
    def sketch_to_keyframe(self, sketch: torch.Tensor,
                           mean: torch.Tensor, std: torch.Tensor,
                           joint_dim: int = 67) -> torch.Tensor:
        """Decode sketch → normalised joint features suitable for keyframe conditioning.

        As per Section 3.3: global root features (first 4 dims) are set to zero
        because a 2-D sketch cannot encode global trajectory.

        Args:
            sketch   : [bs, 1|3, H, W]
            mean     : [joint_dim]  – HumanML3D normalisation mean
            std      : [joint_dim]  – HumanML3D normalisation std
            joint_dim: number of joint features (67 for drop_redundant=True)

        Returns:
            [bs, joint_dim]  – normalised keyframe features (zero global root)
        """
        assert self.align_net is not None
        z_s = self.align_net.encode_sketch(sketch)
        j_raw = self.align_net.decode(z_s)          # [bs, joint_dim] raw

        # Normalise (the decoder outputs in the raw feature space)
        mean_ = mean.to(sketch.device)
        std_  = std.to(sketch.device)
        j_norm = (j_raw - mean_) / (std_ + 1e-8)

        # Zero global root features (dims 0:4 → ṙa, ṙxz, ṙh)
        j_norm[:, :4] = 0.0
        return j_norm  # [bs, joint_dim]
