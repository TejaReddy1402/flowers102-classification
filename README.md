# flowers102-classification
# Flower Classification using Deep Learning

This project implements an image classification pipeline using PyTorch and the Oxford 102 Flower Dataset. It leverages pretrained convolutional neural networks for transfer learning to accurately classify 102 different species of flowers.

## Features
- Dataset preprocessing and augmentation
- Model training with pretrained architectures (e.g., ResNet)
- Evaluation on validation and test sets
- GPU acceleration support

## Dataset
- [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- 102 flower categories with 8,189 labeled images
- Predefined training, validation, and test splits

## Requirements
- Python 3.x
- PyTorch
- torchvision
- scipy, PIL, numpy

## How to Run
1. Download and extract the dataset into `102flowers/`.
2. Place `imagelabels.mat` and `setid.mat` in the working directory.
3. Run the notebook or Python script to train and evaluate the model.

## Results
Achieved high classification accuracy using transfer learning techniques. Further improvements can include hyperparameter tuning and model ensembling.
