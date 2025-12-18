# SHAPE
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

### 3. Download the pretrained models

Download the model(s) you wish to use, then unzip and place them in `./save/`.

Our models are all trained on the HumanML3D dataset.

[Conditionally trained on randomly sampled frames and joints (CondMDI)](https://drive.google.com/file/d/1aP-z1JxSCTcUHhMqqdL2wbwQJUZWHT2j/view?usp=sharing)

[Conditionally trained on randomly sampled frames](https://drive.google.com/file/d/15mYPp2U0VamWfu1SnwCukUUHczY9RPIP/view?usp=sharing)

[Unconditionally (no keyframes) trained](https://drive.google.com/file/d/1B0PYpmCXXwV0a5mhkgea_J2pOwhYy-k5/view?usp=sharing)



## Motion Synthesis
<details>
  <summary><b>Text to Motion - <u>Without</u> spatial conditioning</b></summary>

This part is a standard text-to-motion generation.


## Training

Our model is trained on the **HumanML3D** dataset.
### Conditional Model
```shell
python -m train.train_condmdi --keyframe_conditioned
```
* You can ramove `--keyframe_conditioned` to train a unconditioned model.
* Use `--device` to define GPU id.

## Evaluate
All evaluation are done on the HumanML3D dataset.

### Text to Motion - <u>With</u> keyframe conditioning

* Takes about 20 hours (on a single GPU)
* The output of this script for the pre-trained models (as was reported in the paper) is provided in the checkpoints zip file.
* For each prompt, 5 keyframes are sampled from the ground truth motion. The ground locations of the root joint in those frames are used as conditions.


## Acknowledgments

We would like to thank the following contributors for the great foundation that we build upon:
[GMD](https://github.com/korrawe/guided-motion-diffusionhttps://github.com/korrawe/guided-motion-diffusion), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi).

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.