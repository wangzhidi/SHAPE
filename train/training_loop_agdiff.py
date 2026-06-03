"""AG-Diff training loop (Stage 2).

Extends the base TrainLoop to:
  1.  Encode sketch images via the AlignNet and attach sketch_latent to cond.
  2.  Compute the sketch-loss (L_sketch, Equation 6) at keyframe positions.
  3.  Combine with the standard diffusion loss (Equation 7).
  4.  Optionally freeze AlignNet parameters.
"""

import functools
import os
import torch
from torch import nn
from torch.cuda import amp

from diffusion import logger
from diffusion.resample import LossAwareSampler
from train.training_loop import TrainLoop
from train.training_loop import log_loss_dict
try:
    from train.training_loop import compute_norms
except ImportError:
    def compute_norms(params):  # fallback stub
        return 0.0, 0.0
from utils import dist_util
from utils.editing_util import get_keyframes_mask


class AGDiffTrainLoop(TrainLoop):
    """Training loop for AG-Diff (Stage 2: diffusion + sketch loss).

    Extra args in ``args`` (beyond those required by TrainLoop):
        freeze_align_net    bool  – whether to freeze AlignNet weights
        lambda_sketch       float – weight for L_sketch (Equation 7)
        sketch_keyframe_scheme str – how to choose sketch keyframes during
                                      training ('random' or 'inbetween')
    """

    def __init__(self, args, model, diffusion, data):
        super().__init__(args, model, diffusion, data)

        self.freeze_align_net = getattr(args, 'freeze_align_net', True)
        self.lambda_sketch    = getattr(args, 'lambda_sketch', 0.1)
        self.sketch_keyframe_scheme = getattr(
            args, 'sketch_keyframe_scheme', 'random'
        )

        if self.freeze_align_net and hasattr(self.model, 'align_net'):
            align_net = self.model.align_net
            if align_net is not None:
                for p in align_net.parameters():
                    p.requires_grad_(False)
                print('AlignNet parameters FROZEN.')

    # ── forward-backward with sketch loss ──────────────────────────────── #

    def forward_backward(self, batch, cond):
        if self.use_fp16:
            self.opt.zero_grad()
        else:
            self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            micro      = batch
            micro_cond = cond
            t, weights = self.schedule_sampler.sample(
                micro.shape[0], dist_util.dev()
            )

            # ── encode sketch → latent ────────────────────────────────────── #
            self._attach_sketch_latent(micro_cond, micro)

            # ── diffusion forward ─────────────────────────────────────────── #
            with amp.autocast(enabled=self.use_fp16, dtype=torch.float16):
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                    dataset=self.data.dataset,
                )
                losses = compute_losses()

                # ── sketch loss ──────────────────────────────────────────── #
                sketch_loss = self._compute_sketch_loss(losses, micro_cond)
                if sketch_loss is not None:
                    losses['sketch'] = sketch_loss.unsqueeze(0).expand(
                        micro.shape[0]
                    )

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses['loss'].detach()
                    )

                # total loss = diffusion_loss + λ_s * sketch_loss
                loss = (losses['loss'] * weights).mean()
                if sketch_loss is not None:
                    loss = loss + self.lambda_sketch * sketch_loss

            log_loss_dict(
                self.diffusion, t,
                {k: v * weights for k, v in losses.items()
                 if k != 'sketch'}
            )
            if sketch_loss is not None:
                logger.logkv_mean('sketch_loss', sketch_loss.item())

            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                self.mp_trainer.backward(loss)

    # ── helpers ───────────────────────────────────────────────────────────── #

    @torch.no_grad()
    def _attach_sketch_latent(self, cond: dict, motion: torch.Tensor) -> None:
        """Encode sketch images (if present) and attach latent to cond['y']."""
        align_net = getattr(self.model, 'align_net', None)
        if align_net is None:
            return
        if 'sketch' not in cond['y']:
            return

        sketches = cond['y']['sketch']           # [bs, 1, H, W]
        kf_indices = cond['y'].get('sketch_keyframe_idx', None)  # [bs]

        bs = sketches.shape[0]
        sketch_latent = align_net.encode_sketch(sketches)  # [bs, latent_dim]

        cond['y']['sketch_latent'] = sketch_latent

        # Build per-sample keyframe index as a Python list for the GuideNet
        if kf_indices is not None:
            cond['y']['sketch_keyframe_idx_list'] = kf_indices.tolist()

    def _compute_sketch_loss(self, losses: dict, cond: dict):
        """Compute L_sketch = MSE between predicted x₀ and sketch keyframes
        at the sketch-guided positions (Equation 6).

        Returns a scalar tensor or None if prediction / sketch data unavailable.
        """
        if 'pred_xstart' not in losses:
            return None
        if 'sketch' not in cond['y']:
            return None

        pred_x0 = losses['pred_xstart']  # [bs, njoints, nfeats, nframes]
        sketches = cond['y']['sketch']   # [bs, 1, H, W]
        kf_idx   = cond['y'].get('sketch_keyframe_idx', None)  # [bs]

        align_net = getattr(self.model, 'align_net', None)
        if align_net is None or kf_idx is None:
            return None

        bs, njoints, nfeats, nframes = pred_x0.shape

        # Decode sketch → normalised joint features
        with torch.no_grad():
            sketch_joints = align_net.sketch_to_joints(sketches)  # [bs, joint_dim]

        joint_dim = sketch_joints.shape[1]

        # Expand sketch joints to match pred_x0 shape
        # sketch_joints: [bs, joint_dim] → pad zeros to njoints if needed
        target = torch.zeros_like(pred_x0)
        target[:, :joint_dim, 0, :] = sketch_joints.unsqueeze(-1).expand(
            -1, -1, nframes
        )

        # Build keyframe mask: True at the sketch-guided frame index
        mask = torch.zeros_like(pred_x0, dtype=torch.bool)
        for b in range(bs):
            k = kf_idx[b].item()
            if 0 <= k < nframes:
                mask[b, :joint_dim, :, k] = True

        if not mask.any():
            return None

        diff = (pred_x0 - target) ** 2
        n_k  = mask.float().sum().clamp(min=1.0)
        return (diff * mask.float()).sum() / n_k

    # ── run-loop override (add sketch keyframing on top of standard masking) ─ #

    def run_loop(self):
        """Extended run-loop: injects sketch conditioning for each training step."""
        print('AG-Diff train steps:', self.num_steps)
        from tqdm import tqdm
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or
                        self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {
                    k: (v.to(self.device) if torch.is_tensor(v) else v)
                    for k, v in cond['y'].items()
                }

                # Standard keyframe masking (condMDI-style)
                if self.args.keyframe_conditioned:
                    cond['obs_x0']  = motion
                    cond['obs_mask'] = get_keyframes_mask(
                        data=motion,
                        lengths=cond['y']['lengths'],
                        edit_mode=self.args.keyframe_selection_scheme,
                    )

                self.run_step(motion, cond)

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                    for k, v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(
                                self.step + self.resume_step, v))

                if self.step > 0 and self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                self.step += 1

            if not (not self.lr_anneal_steps or
                    self.step + self.resume_step < self.lr_anneal_steps):
                break

        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()
