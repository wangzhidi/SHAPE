"""Stage-1 training: AlignNet (sketch-motion shared autoencoder).

Usage example
─────────────
    python train/train_align_net.py \\
        --sketch_dir  dataset/sketches \\
        --motion_dir  dataset/HumanML3D/new_joint_vecs \\
        --mean_path   dataset/HumanML3D/Mean.npy \\
        --std_path    dataset/HumanML3D/Std.npy \\
        --train_split dataset/HumanML3D/train.txt \\
        --save_dir    save/align_net \\
        --epochs      1200 \\
        --batch_size  256 \\
        --lr          1e-4 \\
        --latent_dim  256 \\
        --lambda_con  0.5 \\
        --temperature 0.1

Outputs
───────
    save/align_net/align_net_{epoch}.pt   – checkpoint every --save_interval epochs
    save/align_net/align_net_best.pt      – best checkpoint (lowest val loss)
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from model.align_net import AlignNet
from data_loaders.humanml.data.sketch_dataset import SketchJointDataset


# ─────────────────────────── argument parsing ─────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser('Train AlignNet (Stage 1)')
    # Data
    p.add_argument('--sketch_dir',  required=True)
    p.add_argument('--motion_dir',  required=True)
    p.add_argument('--mean_path',   required=True)
    p.add_argument('--std_path',    required=True)
    p.add_argument('--train_split', required=True)
    p.add_argument('--val_split',   default=None,
                   help='If not given, 10 %% of train data is used for validation.')
    p.add_argument('--img_size',    type=int, default=256)
    p.add_argument('--joint_dim',   type=int, default=67)
    p.add_argument('--max_frames',  type=int, default=196)
    # Model
    p.add_argument('--latent_dim',  type=int, default=256)
    p.add_argument('--hidden_dim',  type=int, default=512)
    p.add_argument('--num_blocks',  type=int, default=4)
    p.add_argument('--temperature', type=float, default=0.1)
    # Training
    p.add_argument('--epochs',         type=int,   default=1200)
    p.add_argument('--batch_size',     type=int,   default=256)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--lambda_con',     type=float, default=0.5,
                   help='Weight of the contrastive loss term (Equation 3).')
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--save_dir',       default='save/align_net')
    p.add_argument('--save_interval',  type=int,   default=100)
    p.add_argument('--device',         default='cuda')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--resume',         default='',
                   help='Path to checkpoint to resume from.')
    return p.parse_args()


# ───────────────────────────── main ─────────────────────────────────────── #

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── dataset ───────────────────────────────────────────────────────────── #
    mean = np.load(args.mean_path).astype(np.float32)
    std  = np.load(args.std_path).astype(np.float32)

    full_dataset = SketchJointDataset(
        sketch_dir=args.sketch_dir,
        motion_dir=args.motion_dir,
        mean=mean,
        std=std,
        split_file=args.train_split,
        img_size=args.img_size,
        joint_dim=args.joint_dim,
        max_frames=args.max_frames,
    )

    if args.val_split is not None:
        val_dataset = SketchJointDataset(
            sketch_dir=args.sketch_dir,
            motion_dir=args.motion_dir,
            mean=mean,
            std=std,
            split_file=args.val_split,
            img_size=args.img_size,
            joint_dim=args.joint_dim,
            max_frames=args.max_frames,
        )
        train_dataset = full_dataset
    else:
        n_val = max(1, int(len(full_dataset) * 0.1))
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed)
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              drop_last=False, pin_memory=True)

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # ── model ────────────────────────────────────────────────────────────── #
    model = AlignNet(
        joint_dim=args.joint_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
    ).to(device)

    print(f'AlignNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'Resumed from {args.resume} at epoch {start_epoch}')

    # ── training loop ────────────────────────────────────────────────────── #
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_loss, train_rec, train_con = 0.0, 0.0, 0.0
        n_batches = 0

        for sketch, joint in train_loader:
            sketch = sketch.to(device)    # [bs, 1, H, W]
            joint  = joint.to(device)     # [bs, joint_dim]

            optimizer.zero_grad()
            loss, info = model.compute_loss(sketch, joint, lambda_con=args.lambda_con)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += info['total']
            train_rec  += info['rec']
            train_con  += info['con']
            n_batches  += 1

        scheduler.step()

        # ── validation ───────────────────────────────────────────────────── #
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for sketch, joint in val_loader:
                sketch = sketch.to(device)
                joint  = joint.to(device)
                loss, _ = model.compute_loss(sketch, joint, lambda_con=args.lambda_con)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        elapsed = time.time() - t0

        print(
            f'Epoch [{epoch+1:4d}/{args.epochs}]  '
            f'train_loss={train_loss/n_batches:.4f}  '
            f'(rec={train_rec/n_batches:.4f} con={train_con/n_batches:.4f})  '
            f'val_loss={val_loss:.4f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
            f't={elapsed:.1f}s'
        )

        # ── save checkpoints ─────────────────────────────────────────────── #
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }

        if (epoch + 1) % args.save_interval == 0:
            path = os.path.join(args.save_dir, f'align_net_{epoch+1:04d}.pt')
            torch.save(ckpt, path)
            print(f'  → Saved checkpoint: {path}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt['best_val_loss'] = best_val_loss
            best_path = os.path.join(args.save_dir, 'align_net_best.pt')
            torch.save(ckpt, best_path)
            print(f'  → New best val_loss={val_loss:.4f}, saved to {best_path}')

    print('AlignNet training complete.')


if __name__ == '__main__':
    main()
