# Retinex-Guided Low-Light Image Enhancement

This repository contains the official PyTorch implementation of the paper: **"[Insert Paper Title Here]"**, submitted to **IEEE ICIP 2026**.

> **Note:** This code is for academic research use only.

## 1. Dependencies and Installation

- Python >= 3.9
- PyTorch >= 2.7
- CUDA >= 10.2

### Installation Steps

1. Clone this repository (or unzip the supplementary material).
2. Install dependent packages:

```bash
pip install -r requirements.txt
python setup.py develop
```

## 2. Dataset Preparation
We used the [Insert Dataset Name, e.g., LOL / VE-LOL] dataset for training and testing.

Organize the dataset structure as follows:

## 3. Training
Unlike the multi-stage training in the original CUE baseline, our method simplifies the process into a single end-to-end training stage for the enhancement network.

### Prerequisites
Ensure your dataset paths are correctly configured in the configuration file:
`options/train/retinex_guided_llie/retinex_guided_llie.yml` (Check your specific YAML path)

### Run Training
To train the Retinex-Guided model, run the following command:

```bash
# Basic usage
python train.py -opt options/train/retinex_guided_llie/retinex_guided_llie.yml
```

## 4. Testing / Inference
To evaluate the model using pre-trained weights:

Check the test config file: options/test/learnedPrior/LearnablePrior.yml

Run the testing command:

```Bash
python test.py -opt options/test/retinex_guided_llie/retinex_guided_llie.yml
```
