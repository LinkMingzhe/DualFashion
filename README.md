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


## Datasets

We follow the previous work [DiFashion](https://github.com/YiyanXu/DiFashion?tab=readme-ov-file) and use the datasets of iFashion and Polyvore-U, which include the required data of both fashion outfit and user-fashion item interactions. 

To get user textual preference, we use Gemini to extract structured fashion item caption and further sample user preference in attribute level. All the captions and user preference data can be downloaded from google drive (DualFashion/processed_info).




