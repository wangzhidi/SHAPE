# AG-Diff

## Pose Sketch Dataset
Synthetic and hand drawn sketch data are avaliable at https://drive.google.com/file/d/1thIQ_P7E9CoFKnqL7TP1LEGBU0yEMTtv/view?usp=drive_link

(a) prepare humanml3d dataset and human3.6m dataset at ./assets/datasets/*

(b) unzip sketch data into ./assets/datasets/sketch_data/*

## Getting started

This code was developed on `Ubuntu 20.04 LTS` with Python 3.7, CUDA 11.7 and PyTorch 1.13.1.


### 1. Setup environment
Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.


### 2. Install dependencies
This codebase shares a large part of its base dependencies with [GMD](https://github.com/korrawe/guided-motion-diffusion). We recommend installing our dependencies from scratch to avoid version differences.

Setup virtual env:
```shell
python3 -m venv .env_condmdi
source .env_condmdi/bin/activate
pip uninstall ffmpeg
pip install spacy
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:

<details>
  <summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```
</details>

<details>
  <summary><b>Unconstrained</b></summary>

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_recognition_unconstrained_models.sh
```
</details>

### 2. Get data
There are two paths to get the data:

(a) **Generation only** wtih pretrained text-to-motion model without training or evaluating

(b) **Get full data** to train and evaluate the model.


#### a. Generation only (text only)

**HumanML3D** - Clone HumanML3D, then copy the data dir to our repository:

```shell
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D diffusion-motion-inbetweening/dataset/HumanML3D
cd diffusion-motion-inbetweening
cp -a dataset/HumanML3D_abs/. dataset/HumanML3D/
```


#### b. Full data (text + motion capture)

**[Important !]**
Following GMD, the representation of the root joint has been changed from relative to absolute. Therefore, you need to replace the original files and run GMD's version of `motion_representation.ipynb` and `cal_mean_variance.ipynb` provided in `./HumanML3D_abs/` instead to get the absolute-root data.

**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```


## Training

AG-Diff is trained in **two stages**.

### Stage 1 — AlignNet (Sketch-Motion Shared Autoencoder)

Pre-generates XYZ sketch images from the dataset, then trains the AlignNet to align sketch and joint representations in a shared latent space.

```shell
# 1a. Pre-generate synthetic sketch images from joint XYZ data
python draw_sketches.py

# 1b. Train AlignNet (~1200 epochs)
python train/train_align_net.py \
    --sketch_dir  dataset/sketches \
    --motion_dir  dataset/HumanML3D/new_joint_vecs \
    --mean_path   dataset/HumanML3D/Mean.npy \
    --std_path    dataset/HumanML3D/Std.npy \
    --train_split dataset/HumanML3D/train.txt \
    --val_split   dataset/HumanML3D/val.txt \
    --save_dir    save/align_net \
    --epochs      1200 \
    --batch_size  256 \
    --latent_dim  256 \
    --lambda_con  0.5 \
    --temperature 0.1
```

Key arguments:
* `--lambda_con` — weight for InfoNCE contrastive loss (default: `0.5`)
* `--temperature` — temperature coefficient τ in InfoNCE (default: `0.1`)
* `--joint_dim` — number of joint feature dims to use; `67` for `drop_redundant` mode (default: `67`)

### Stage 2 — AG-Diff (Generation Net + Guide Net)

Loads the pre-trained AlignNet and trains the full AG-Diff model with the masked diffusion loss plus the sketch keyframe loss.

```shell
python train/train_agdiff.py \
    --dataset              humanml \
    --abs_3d \
    --drop_redundant \
    --keyframe_conditioned \
    --keyframe_selection_scheme random_frames \
    --arch                 trans_enc \
    --layers               8 \
    --latent_dim           512 \
    --ff_size              1024 \
    --batch_size           64 \
    --num_steps            600000 \
    --lr                   1e-4 \
    --align_net_path       save/align_net/align_net_best.pt \
    --sketch_dir           dataset/sketches \
    --lambda_sketch        0.1 \
    --freeze_align_net \
    --device               0
```

Key arguments:
* `--align_net_path` — path to the Stage-1 checkpoint (required)
* `--lambda_sketch` — weight for the sketch keyframe loss L_sketch (default: `0.1`)
* `--freeze_align_net` — freeze AlignNet weights during Stage-2 training (recommended)
* `--device` — GPU id

## Evaluate

All evaluations are done on the **HumanML3D** dataset augmented with sketch data.

### Motion Quality (Table 4) — FID, R-Precision, Diversity, Foot Skating

```shell
python eval/eval_humanml_agdiff.py \
    --model_path      save/agdiff/model_best.pt \
    --align_net_path  save/align_net/align_net_best.pt \
    --sketch_dir      dataset/sketches \
    --dataset         humanml \
    --abs_3d \
    --drop_redundant \
    --num_samples     1000 \
    --batch_size      32 \
    --guidance_param  2.5 \
    --output_dir      eval_results/agdiff
```

### Motion Editing / Controllability (Table 5) — Trajectory / Location / Average / Keyframe Error

Pass `--edit_mode` to select the editing benchmark:

```shell
python eval/eval_humanml_agdiff.py \
    --model_path      save/agdiff/model_best.pt \
    --align_net_path  save/align_net/align_net_best.pt \
    --sketch_dir      dataset/sketches \
    --edit_mode       in_between \
    --n_keyframes     5 \
    --num_samples     1000 \
    --output_dir      eval_results/agdiff_edit
```

* `--edit_mode` choices: `in_between` (default), `benchmark_sparse`, `benchmark_clip`
* `--n_keyframes` — number of sketch-guided keyframes per clip (default: `5`)
* Results (JSON) and motion arrays are saved to `--output_dir`


## Acknowledgments

We would like to thank the following contributors for the great foundation that we build upon:
[GMD](https://github.com/korrawe/guided-motion-diffusionhttps://github.com/korrawe/guided-motion-diffusion), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi).

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.