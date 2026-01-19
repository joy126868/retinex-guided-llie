# Retinex-Guided Low-Light Image Enhancement

This repository contains the official PyTorch implementation of the paper: **"[Insert Paper Title Here]"**, submitted to **IEEE ICIP 2026**.

> **Note:** This code is for academic research use only.

## 1. Dependencies and Installation

- Python >= 3.9
- PyTorch >= 2.7 (Tested with 2.7.1+cu118)
- CUDA >= 11.8

### Installation Steps

1. Clone this repository (or unzip the supplementary material).
2. Install dependent packages:

```bash
pip install -r requirements.txt
python setup.py develop
```

## 2. Dataset Preparation
We used the [LOL / VE-LOL] dataset for training and testing.

## 3. Training
Unlike the multi-stage training in the original CUE baseline, our method simplifies the process into a **single end-to-end training stage** for the enhancement network.

### Prerequisites
The training configuration is located at: options/train/retinex_guided_llie/retinex_guided_llie.yml

### Run Training
To train the Retinex-Guided model, run the following command:

```bash
# Basic usage
python train.py -opt options/train/retinex_guided_llie/retinex_guided_llie.yml
```
Checkpoints: Model weights are saved in experiments/.

## 4. Testing / Inference
To evaluate the model using pre-trained weights:

1. Check the test config file: options/test/retinex_guided_llie/retinex_guided_llie.yml
2. Run the testing command:

```Bash
python test.py -opt options/test/retinex_guided_llie/retinex_guided_llie.yml
```
The visual results will be saved in results/.

