# DualFashion
This is the implementation of DualFashion.

Dual-Diffusional Generative Fashion Recommendation (Accepted by SIGIR 2026)

1. **Stage 1**: joint fashion image/text training with structured attribute captions.
2. **Stage 2**: outfit-conditioned recommendation with user preference and incomplete-outfit context.
3. **Stage 3**: text-only finetuning with Gemini-augmented captions.

## TODO List

- [x] Environment
- [x] Datasets
- [x] Release checkpoint
- [x] Inference code
- [x] Evaluation code
- [x] Train code

## Installation

Clone the repository:

```bash
git clone https://github.com/LinkMingzhe/DualFashion.git
cd Dual-Diffusion
```

Create a fresh Python environment and install dependencies:

```bash
pip install -r requirements.txt
```


## Data And Checkpoints

### 1. Dataset

We use the datasets of iFashion and Polyvore-U, which include the required data of both fashion outfit and user-fashion item interactions. 

To get user textual preference, we use Gemini to extract structured fashion item caption and further sample user preference in attribute level. 
All the captions and user preference data can be downloaded from [google drive](https://drive.google.com/drive/folders/1eigLDq2_3Jpyr2bWZgoXDZNJSFyB7I1x?usp=sharing) (`{your_path}/processed_info`).

- `{your_path}/processed_info/stage1_ifashion/`
- `{your_path}/processed_info/stage1_ifashion_val/`
- `{your_path}/processed_info/stage2_ifashion/`
- `{your_path}/processed_info/stage2_ifashion_test/`
- `{your_path}/processed_info/stage3_data_augmentation/`


### 2. Pre-trained Models and checkpoints

We release the checkpoint at [google drive](https://drive.google.com/drive/folders/1eigLDq2_3Jpyr2bWZgoXDZNJSFyB7I1x?usp=sharing) (`{your_path}/checkpoint`), and pre-trained model at (`{your_path}/pretrained_models`).

You further need to download the stable-diffusion-3-medium-diffusers and put it in `{your_path}/pretrained_models` for initialize the model structure.


## Project Structure

```text
Dual-Diffusion/
├── baselines/
│   └── DiFashion/
├── configs/
├── eval/
├── output/
├── pretrained_models/
├── processed_info/
├── sd3_modules/
├── Inference_ifashion_GOR.py
├── Inference_ifashion_PFITB.py
├── eval_utils.py
├── evaluate.py
├── stage1_train_fashion_pairs.py
├── stage2_train_ifashion_lrem.py
├── stage3_train_data_augmentation.py
├── requirements.txt
└── README.md
```

## Training

### Stage 1: joint fashion image/text training

Stage 1 uses the structured captions under `{your_path}/processed_info/stage1_ifashion/` and trains the base dual diffusion model.

```bash

python {your_path}/stage1_train_fashion_pairs.py \
  --config {your_path}/configs/stage1_config.py \
  --training single \
  --device 0
```

Default outputs:

- checkpoints: `{your_path}/output/ifashion/stage1_checkpoints_t5/`
- exported models: `{your_path}/output/ifashion/stage1_models_t5/`

### Stage 2: outfit-conditioned recommendation training

Stage 2 adds user-preference and incomplete-outfit conditioning.

```bash
export PYTHONPATH={your_path}/py_tools:$PYTHONPATH

python {your_path}/stage2_train_ifashion_lrem.py \
  --config {your_path}/configs/stage2_config_ifashion.py \
  --training single \
  --device 0
```

Default outputs:

- checkpoints: `{your_path}/output/ifashion/stage2_checkpoints/`
- exported models: `{your_path}/output/ifashion/stage2_models/`

### Stage 3: Gemini augmentation + text-only finetuning

First, optionally regenerate the augmented caption file:

```bash
export GOOGLE_API_KEY=<your-gemini-api-key>

python {your_path}/processed_info/stage3_data_augmentation/data_augmentation.py \
  --resume
```

Then finetune the text branch:

```bash
python {your_path}/stage3_train_data_augmentation.py \
  --config {your_path}/configs/stage3_config.py \
  --device cuda:0
```

Default outputs:

- checkpoints: `{your_path}/output/ifashion/stage3_checkpoints/`
- exported models: `{your_path}/output/ifashion/stage3_checkpoints/stage3_models/`

## Inference


You can run `{your_path}/Inference_ifashion_PFITB.py` and `{your_path}/Inference_ifashion_GOR.py` to test the model performance.

It's for the following two tasks: 1) Personalized Fill-in-the-Blank (PFITB) - generating one matching item to complete the outfit, 
and 2) Generative Outfit Recommendation (GOR) - building one completed outfit.


### Stage-2/3 PFITB inference

Generate missing items for the FITB task:

```bash
python {your_path}/Inference_ifashion_PFITB.py \
  --models_root {your_path}/output/ifashion/{stage2 or stage3 checkpoint} \
  --output_root {your_path}/output/QualitativeResults/{output_dir} \
```

Outputs:

- generated images per `(uid, oid)`
- generated captions in `generated_caption.json`
- packed `gen.npy` / `grd.npy` files for evaluation


### Stage-2/3 GOR inference

Generate complete outfits sequentially:

```bash
python {your_path}/Inference_ifashion_GOR.py \
  --models_root {your_path}/output/ifashion/{stage2 or stage3 checkpoint} \
  --output_root {your_path}/output/QualitativeResults/{output_dir} \
```


## Evaluation

`{your_path}/evaluate_PFITB.py` expects each checkpoint folder to contain `gen.npy` and `grd.npy`.

The evaluation stack includes:

- Inception Score / category accuracy
- LPIPS diversity
- personalization similarity
- outfit compatibility
- optional Gemini-based text compatibility scoring
