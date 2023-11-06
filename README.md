# CIFAR-10 Image Classification Using TensorFlow
This project is an implementation of an image classification model using the CIFAR-10 dataset. It utilizes TensorFlow and Keras to build and train a Convolutional Neural Network (CNN) to classify images into one of the 10 different classes present in the dataset.

## Project Description
The CIFAR-10 dataset contains 60,000 32x32 color images divided into ten classes, each with 6,000 images. There are 50,000 training photos and 10,000 testing images in the dataset. For classification, the model used in this study is a sequential CNN with numerous convolutional and max pooling layers, followed by dense layers with a softmax activation function.

## Requirements
- Python 3.x
- Matplotlib
- Pandas
- TensorFlow 2.x

## Installation

To set up the project environment, run the following commands:

```bash
!pip install tensorflow matplotlib pandas
```

## Application

Run the notebook *CIFAR10_CLASSIFICATION_USING_TENSORFLOW.ipynb* to train the model with the CIFAR-10 dataset. The notebook contains code for loading and preprocessing data, building the model architecture, configuring callbacks for early halting and model checkpointing, compiling the model, and fitting the model to training data.

## Attributes
- CNN with 4 convolutional layers and max pooling.
- Data normalization to scale image pixel values to a range of 0 to 1.
- Early stopping to prevent overfitting.
- Model checkpointing to save the best model based on validation accuracy.
- Detailed model summary and training history.


