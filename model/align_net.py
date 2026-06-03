"""AlignNet: Sketch-Motion Shared Autoencoder (Section 3.2 of AG-Diff paper).

Maps sketch images and joint-feature vectors into a shared latent space via
InfoNCE contrastive alignment + MSE reconstruction losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── building blocks ──────────────────────────────── #

class ResidualMLP(nn.Module):
    """Pre-norm residual MLP block (acts as 1-D ResNet element)."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


# ──────────────────────────── encoders / decoder ──────────────────────────── #

class JointEncoder(nn.Module):
    """1-D ResNet for encoding a single-frame joint-feature vector.

    Input:  [bs, joint_dim]   – normalized HumanML3D features (up to 67 dims)
    Output: [bs, latent_dim]  – unit-normalized embedding
    """
    def __init__(self, joint_dim: int = 67, latent_dim: int = 256,
                 hidden_dim: int = 512, num_blocks: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(joint_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLP(hidden_dim) for _ in range(num_blocks)]
        )
        self.proj_out = nn.Linear(hidden_dim, latent_dim)
        self.norm_out = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [bs, joint_dim]
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(self.proj_out(x))


class SketchEncoder(nn.Module):
    """CNN encoder for binarised sketch images.

    Tries HRNet-W18 via timm first; falls back to ResNet-18 from torchvision.

    Input:  [bs, 1|3, H, W]  – grayscale or RGB sketch (normalised)
    Output: [bs, latent_dim]
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        backbone_dim: int = 512
        try:
            import timm  # type: ignore
            self.backbone = timm.create_model(
                'hrnet_w18', pretrained=False, num_classes=0, global_pool='avg'
            )
            backbone_dim = self.backbone.num_features
            print('SketchEncoder: using HRNet-W18 backbone')
        except Exception:
            import torchvision.models as tvm
            resnet = tvm.resnet18(pretrained=False)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 512
            print('SketchEncoder: using ResNet-18 backbone (timm not available)')

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [bs, 1|3, H, W]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        return self.proj(feat)  # [bs, latent_dim]


class SharedDecoder(nn.Module):
    """Symmetric 1-D ResNet decoder: latent → reconstructed joint features.

    Input:  [bs, latent_dim]
    Output: [bs, joint_dim]
    """
    def __init__(self, latent_dim: int = 256, joint_dim: int = 67,
                 hidden_dim: int = 512, num_blocks: int = 3):
        super().__init__()
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLP(hidden_dim) for _ in range(num_blocks)]
        )
        self.proj_out = nn.Linear(hidden_dim, joint_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [bs, latent_dim]
        x = self.proj_in(z)
        for blk in self.blocks:
            x = blk(x)
        return self.proj_out(x)  # [bs, joint_dim]


# ───────────────────────────── AlignNet ───────────────────────────────────── #

class AlignNet(nn.Module):
    """Sketch-Motion Shared Autoencoder (Section 3.2).

    Architecture:
        - SketchEncoder (HRNet-W18 / ResNet-18)  → sketch_latent
        - JointEncoder  (1-D ResNet MLP)          → joint_latent
        - SharedDecoder (symmetric 1-D ResNet)    → reconstructed joints

    Losses (Equations 1–3):
        L_rec = |J − Ĵ|_1 + |Ŝ_j − J|_1       (MSE-like L1 reconstruction)
        L_con = InfoNCE(z_s, z_j)                (inter-modal alignment)
        L     = L_rec + λ · L_con
    """

    def __init__(self, joint_dim: int = 67, latent_dim: int = 256,
                 hidden_dim: int = 512, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.latent_dim = latent_dim
        self.joint_dim = joint_dim

        self.sketch_encoder = SketchEncoder(latent_dim=latent_dim)
        self.joint_encoder = JointEncoder(
            joint_dim=joint_dim, latent_dim=latent_dim, hidden_dim=hidden_dim
        )
        self.decoder = SharedDecoder(
            latent_dim=latent_dim, joint_dim=joint_dim, hidden_dim=hidden_dim
        )

    # ── encode ────────────────────────────────────────────────────────────── #

    def encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """[bs, 1|3, H, W] → unit-normalised embedding [bs, D]"""
        return F.normalize(self.sketch_encoder(sketch), dim=-1)

    def encode_joint(self, joint: torch.Tensor) -> torch.Tensor:
        """[bs, joint_dim] → unit-normalised embedding [bs, D]"""
        return F.normalize(self.joint_encoder(joint), dim=-1)

    # ── decode ────────────────────────────────────────────────────────────── #

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[bs, D] → reconstructed joint features [bs, joint_dim]"""
        return self.decoder(z)

    # ── losses ────────────────────────────────────────────────────────────── #

    def infonce_loss(self, z_s: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE (Equation 2)."""
        N = z_s.shape[0]
        sim = torch.mm(z_s, z_j.T) / self.temperature  # [N, N]
        labels = torch.arange(N, device=z_s.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

    def forward(self, sketch: torch.Tensor, joint: torch.Tensor):
        """Full forward pass.

        Returns:
            z_s       – sketch embedding  [bs, D]
            z_j       – joint embedding   [bs, D]
            j_rec_s   – joint reconstructed from sketch latent  [bs, joint_dim]
            j_rec_j   – joint reconstructed from joint latent   [bs, joint_dim]
        """
        z_s = self.encode_sketch(sketch)
        z_j = self.encode_joint(joint)
        j_rec_s = self.decode(z_s)
        j_rec_j = self.decode(z_j)
        return z_s, z_j, j_rec_s, j_rec_j

    def compute_loss(self, sketch: torch.Tensor, joint: torch.Tensor,
                     lambda_con: float = 0.5):
        """Compute total training loss (Equation 3).

        Returns:
            loss   – scalar total loss
            info   – dict with per-component loss values
        """
        z_s, z_j, j_rec_s, j_rec_j = self.forward(sketch, joint)

        # Reconstruction loss (Equation 1) – L1 on joint space
        l_rec = F.l1_loss(j_rec_s, joint) + F.l1_loss(j_rec_j, joint)
        # Contrastive loss (Equation 2)
        l_con = self.infonce_loss(z_s, z_j)

        loss = l_rec + lambda_con * l_con
        return loss, {'rec': l_rec.item(), 'con': l_con.item(), 'total': loss.item()}

    # ── inference helper ──────────────────────────────────────────────────── #

    @torch.no_grad()
    def sketch_to_joints(self, sketch: torch.Tensor) -> torch.Tensor:
        """Convert a sketch image to reconstructed joint features.

        sketch: [bs, 1|3, H, W]
        Returns: [bs, joint_dim]  (un-normalised; in the normalised feature space)
        """
        z_s = self.encode_sketch(sketch)
        return self.decode(z_s)
