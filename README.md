# DualFashion
This is the implementation of DualFashion.
(Dual-Diffusional Generative Fashion Recommendation)


## TODO List
- [x] Environment
- [x] Datasets
- [x] Release checkpoint
- [x] Inference code
- [x] Evaluation code
      
## Installation
Clone this repository:
```
git clone https://github.com/LinkMingzhe/DualFashion.git
cd ./DualFashion/
```

Install PyTorch and other dependencies:
```
pip install -r requirements.txt
```

## Project Structure
After download the datasets, pre-trained models and checkpoins, the project structure should be:

DualFashion/\\
├── configs/\\
├── sd3_modules/
├── dataset/
├── eval/
├── processed_info/
├── pretrained_models/
├── checkpoint/
├── Inference_ifashion_GOR.py
├── Inference_ifashion_PFITB.py
├── eval_utils.py
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore


## Datasets

We follow the previous work [DiFashion](https://github.com/YiyanXu/DiFashion?tab=readme-ov-file) and use the datasets of iFashion and Polyvore-U, which include the required data of both fashion outfit and user-fashion item interactions. 

To get user textual preference, we use Gemini to extract structured fashion item caption and further sample user preference in attribute level. All the captions and user preference data can be downloaded from [google drive](https://drive.google.com/drive/folders/1eigLDq2_3Jpyr2bWZgoXDZNJSFyB7I1x?usp=sharing) (DualFashion/processed_info).

# Pre-trained Models

We release the checkpoint at [google drive](https://drive.google.com/drive/folders/1eigLDq2_3Jpyr2bWZgoXDZNJSFyB7I1x?usp=sharing) (DualFashion/checkpoint), and pre-trained model at (DualFashion/pretrained_models).

You further need to download the stable-diffusion-3-medium-diffusers and put it in the DualFashion/pretrained_models for initialize the model structure.

## Inference 

You can run the Inference_ifashion_PFITB.py and Inference_ifashion_GOR.py to test the model performance.

It's for the following two tasks:
1) Personalized Fill-in-the-Blank (PFITB) - generating one matching item to complete the outfit, 
and 2) Generative Outfit Recommendation (GOR) - building one completed outfit.

## Evaliuation

You can run the evaluate.py to get the quantitative results.


