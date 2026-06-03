"""GuideNet: Parallel sketch-guidance network for AG-Diff (Section 3.3).

Architecture (Fig. 2d):
    sketch_latent  → GlobalToLocal  → SelfAttention ×N
                                            ↓ Zero Linear
                               added to Generation Net layer-i output

The Guide Net is initialised with weights copied from the Generation Net's
transformer encoder, then fine-tuned jointly during diffusion training.
Zero-initialised linear layers ensure that at the start of training the Guide Net
contributes nothing (identity behaviour), allowing stable convergence.
"""

import copy
import torch
import torch.nn as nn


class GlobalToLocal(nn.Module):
    """Convert a single per-keyframe sketch latent into a temporal feature map.

    The sketch describes ONE frame; we project it to the model dimension and
    broadcast / place it at the specified keyframe positions.

    Input:
        sketch_latent : [bs, latent_dim]
        seqlen        : target sequence length (including the leading timestep token)
        keyframe_idx  : list[int] – positions (0-indexed) to place the sketch feature.
                        If None, the feature is broadcast to every position.
    Output: [seqlen, bs, latent_dim]
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self,
                sketch_latent: torch.Tensor,
                seqlen: int,
                keyframe_idx=None) -> torch.Tensor:
        bs, d = sketch_latent.shape
        feat = self.norm(self.proj(sketch_latent))  # [bs, d]

        if keyframe_idx is None:
            # Broadcast: every position carries the sketch signal
            return feat.unsqueeze(0).expand(seqlen, -1, -1)

        # Sparse placement: only keyframe positions are non-zero
        out = torch.zeros(seqlen, bs, d,
                          device=feat.device, dtype=feat.dtype)
        for idx in keyframe_idx:
            if 0 <= idx < seqlen:
                out[idx] = feat
        return out


class GuideNet(nn.Module):
    """Parallel transformer network that injects sketch guidance.

    For each of the N transformer encoder layers in the Generation Net:
        1.  Guide Net runs the SAME architecture on sketch-derived features.
        2.  A zero-initialised linear layer (Zero Linear) maps the Guide output.
        3.  The result is **added** to the corresponding Generation Net output.

    This ControlNet-style design (zero init) guarantees the model starts as a
    pure text+motion diffusion model and gradually learns sketch conditioning.
    """

    def __init__(self,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # ── Global-to-Local expansion ────────────────────────────────────── #
        self.global_to_local = GlobalToLocal(latent_dim)

        # ── Self-attention blocks (same architecture as MDM Generation Net) ─ #
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation='gelu',
                batch_first=False,   # expects [seqlen, bs, d]
            )
            for _ in range(num_layers)
        ])

        # ── Zero-initialised injection layers (Zero Linear in Fig. 2d) ───── #
        self.zero_linears = nn.ModuleList([
            self._zero_linear(latent_dim) for _ in range(num_layers)
        ])

    # ── helpers ───────────────────────────────────────────────────────────── #

    @staticmethod
    def _zero_linear(dim: int) -> nn.Linear:
        """Create a linear layer whose weight AND bias start at zero."""
        layer = nn.Linear(dim, dim)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def copy_from_generation_net(self, gen_encoder: nn.TransformerEncoder) -> None:
        """Initialise Guide Net layers with weights from the Generation Net.

        Called once after the Generation Net is built.  Layers beyond
        `len(gen_encoder.layers)` keep their default random initialisation.
        """
        for i, layer in enumerate(self.layers):
            if i < len(gen_encoder.layers):
                layer.load_state_dict(
                    copy.deepcopy(gen_encoder.layers[i].state_dict())
                )

    # ── forward ───────────────────────────────────────────────────────────── #

    def forward(self,
                sketch_latent: torch.Tensor,
                seqlen: int,
                keyframe_idx=None):
        """Compute per-layer residuals to be added to the Generation Net.

        Args:
            sketch_latent  : [bs, latent_dim]  – from AlignNet.encode_sketch()
            seqlen         : temporal length incl. prepended timestep token
            keyframe_idx   : list[int] | None  – sketch keyframe positions
                             (already shifted by +1 for the timestep token)

        Returns:
            residuals : list of N tensors, each [seqlen, bs, latent_dim]
        """
        # Place sketch latent at keyframe positions → temporal feature map
        x = self.global_to_local(sketch_latent, seqlen, keyframe_idx)
        # x: [seqlen, bs, latent_dim]

        residuals = []
        for layer, zero_lin in zip(self.layers, self.zero_linears):
            x = layer(x)                        # self-attention + FFN
            residuals.append(zero_lin(x))       # zero-init projection
        return residuals
