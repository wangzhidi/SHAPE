"""Stage-2 training: AG-Diff (diffusion + sketch guidance).

Builds the AGDiff model (Generation Net + Guide Net), loads a pre-trained
AlignNet checkpoint, and runs the AG-Diff training loop.

Usage example
─────────────
    python train/train_agdiff.py \\
        --dataset         humanml \\
        --abs_3d          \\
        --drop_redundant  \\
        --arch            trans_enc \\
        --layers          8 \\
        --latent_dim      512 \\
        --ff_size         1024 \\
        --keyframe_conditioned \\
        --keyframe_selection_scheme random_frames \\
        --batch_size      64 \\
        --num_steps       600000 \\
        --lr              1e-4 \\
        --align_net_path  save/align_net/align_net_best.pt \\
        --sketch_dir      dataset/sketches \\
        --lambda_sketch   0.1 \\
        --freeze_align_net \\
        --save_dir        save/agdiff
"""

import os
import json
from pprint import pprint

import torch

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.data.sketch_dataset import (
    SketchMotionDataset, sketch_t2m_collate, get_sketch_dataset
)
from utils.model_util import create_model_and_diffusion
from model.align_net import AlignNet
from model.ag_diff import AGDiff
from model.guide_net import GuideNet
from train.training_loop_agdiff import AGDiffTrainLoop
from configs import card

import wandb


def build_agdiff_model(args, data, align_net_path: str) -> AGDiff:
    """Create AGDiff model and load a pre-trained AlignNet.

    The Generation Net (MDM-transformer) weights are randomly initialised;
    the GuideNet is initialised with those same weights (ControlNet-style).
    AlignNet weights are loaded from ``align_net_path``.
    """
    from utils.model_util import get_model_args
    base_kwargs = get_model_args(args, data)

    joint_dim = 67 if args.drop_redundant else 263
    align_net = AlignNet(
        joint_dim=joint_dim,
        latent_dim=256,
        hidden_dim=512,
        temperature=0.1,
    )
    if align_net_path and os.path.exists(align_net_path):
        ckpt = torch.load(align_net_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        align_net.load_state_dict(state)
        print(f'Loaded AlignNet from {align_net_path}')
    else:
        print(f'WARNING: AlignNet checkpoint not found at {align_net_path}. '
              'Using random weights.')

    model = AGDiff(
        **base_kwargs,
        sketch_latent_dim=256,
        guide_num_layers=base_kwargs['num_layers'],
        guide_ff_size=base_kwargs['ff_size'],
        align_net=align_net,
    )
    return model


def main():
    # ── parse arguments (inherit from condMDI / GMD card) ─────────────────── #
    args = train_args(base_cls=card.motion_abs_unet_adagn_xl)

    # Extra AG-Diff specific arguments with defaults
    align_net_path       = getattr(args, 'align_net_path',  'save/align_net/align_net_best.pt')
    sketch_dir           = getattr(args, 'sketch_dir',      'dataset/sketches')
    lambda_sketch        = getattr(args, 'lambda_sketch',   0.1)
    freeze_align_net     = getattr(args, 'freeze_align_net', True)

    wandb.init(project='agdiff', config=vars(args))
    args.save_dir = os.path.join('save', wandb.run.id)
    pprint(args.__dict__)
    fixseed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    # ── data loader (with sketch support) ─────────────────────────────────── #
    print('Creating data loader…')
    if os.path.isdir(sketch_dir):
        # Use SketchMotionDataset
        from torch.utils.data import DataLoader
        dataset_root = os.path.join('dataset', 'HumanML3D')
        train_dataset = get_sketch_dataset(
            dataset_root=dataset_root,
            sketch_dir=sketch_dir,
            split='train',
            num_frames=args.num_frames,
            drop_redundant=args.drop_redundant,
        )
        data = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
            collate_fn=sketch_t2m_collate,
        )
    else:
        # Fall back to standard loader (no sketch conditioning)
        print('WARNING: sketch_dir not found; falling back to standard loader.')
        data_conf = DatasetConfig(
            name=args.dataset,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            use_abs3d=args.abs_3d,
            traject_only=args.traj_only,
            drop_redundant=args.drop_redundant,
        )
        data = get_dataset_loader(data_conf)

    # ── model & diffusion ─────────────────────────────────────────────────── #
    print('Creating model and diffusion…')
    from utils.model_util import create_gaussian_diffusion
    model = build_agdiff_model(args, data, align_net_path)
    diffusion = create_gaussian_diffusion(args)

    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print(f'Total params (excl. CLIP): '
          f'{sum(p.numel() for p in model.parameters_wo_clip()) / 1e6:.2f}M')

    # ── attach extra args needed by the training loop ─────────────────────── #
    args.freeze_align_net       = freeze_align_net
    args.lambda_sketch          = lambda_sketch
    args.sketch_keyframe_scheme = getattr(args, 'keyframe_selection_scheme', 'random_frames')

    # ── train ────────────────────────────────────────────────────────────────
    print('Training AG-Diff…')
    AGDiffTrainLoop(args, model, diffusion, data).run_loop()
    wandb.finish()


if __name__ == '__main__':
    main()
