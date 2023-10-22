# Fruits Classifier using TensorFlow

This code demonstrates how to train a fruits classifier using TensorFlow and the Fruits-360 dataset. The model utilizes a pre-trained MobileNetV2 model as a feature extractor, combined with a dense layer to perform classification.

## Table of Contents
- [Requirements](#requirements)
- [Overview](#overview)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting a Fruit](#predicting-a-fruit)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Future Work](#future-work)

## Requirements
- TensorFlow 2.x
- Fruits-360 dataset placed in a directory named `fruits-360` with subdirectories `Training` and `Test`.

## Overview

1. **Dataset Acquisition**: Loads the images from the `Training` and `Test` directories of the Fruits-360 dataset.
2. **Model Selection**: Uses the MobileNetV2 model pre-trained on the ImageNet dataset. Only the top layers are added for this specific classification task.
3. **Training**: Trains the model using the training set and validates using the test set.
4. **Inference**: Provides a function `predict_fruit` that takes an image path and returns the predicted fruit.

## Usage

### Training the Model

Simply run the script to load the data and start training the model. Training configurations such as data augmentation, optimizer, and number of epochs can be adjusted in the code.


### Predicting a Fruit
To predict a fruit from an image, use the predict_fruit function:

image_path = "path_to_your_image.jpg"
print(predict_fruit(image_path))
Replace path_to_your_image.jpg with the path to your image.

### Data Augmentation
The training data generator uses several data augmentation techniques to improve model generalization:
Rescaling
Random rotation (up to 20 degrees)
Width and height shifts
Zoom
Horizontal flip

### Model Architecture
Base Model: MobileNetV2 with weights pre-trained on ImageNet.
Pooling Layer: Global Average Pooling layer to reduce spatial dimensions.
Output Layer: Dense layer with a number of units equal to the number of fruit classes and a softmax activation for classification.

## Future Work
Save and load the trained model for future usage.
Integrate a more advanced augmentation library like albumentations for better data augmentation.
Experiment with different architectures and hyperparameters for improved accuracy.
