# Self-Driving-Car-Simulation
Implemented CV techniques with Deep Learning to interpret road scenarios, aiding perception of lane markings, obstacles, and traffic signs.  
Utilized CNNs for feature extraction and pattern recognition, essential for decision-making. 
Enhanced model accuracy and efficiency using gradient descent and fine-tuning method

# Behavioral Cloning Project

This project demonstrates a behavioral cloning approach using a deep learning model to autonomously drive a car in a simulator. The model is trained using images from the car's cameras and corresponding steering angles.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Overview

The goal of this project is to train a neural network to drive a car in a simulator using the NVIDIA model architecture. The model learns from a dataset of images captured from a car's cameras and corresponding steering angles.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- imgaug
- pandas
- numpy
- matplotlib
- PIL

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/BehavioralCloning.git
    cd BehavioralCloning
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

1. Place the simulator's driving log (`driving_log.csv`) and images in a directory named `track/`.

2. Run the preprocessing and training script:
    ```bash
    python BehaviouralCloning.ipynb
    ```

### Preprocessing and Augmentation

The data is preprocessed and augmented to improve model robustness:
- **Cropping**: Remove irrelevant parts of the images.
- **Color Space Conversion**: Convert images to YUV color space.
- **Gaussian Blur**: Apply Gaussian blur to reduce noise.
- **Resize**: Resize images to 200x66.
- **Normalization**: Normalize pixel values.

### Model Architecture

The model uses the NVIDIA architecture:
- Convolutional layers with `elu` activation
- Fully connected layers
- Mean Squared Error (MSE) loss function
- Adam optimizer

### Training

The model is trained using the augmented dataset. The training and validation loss are plotted to monitor performance.

### Evaluation

Evaluate the model's performance using the validation dataset.

## Results

- Training and validation loss curves
- Model evaluation on new data
- Example images of preprocessed and augmented data

## References

- NVIDIA End-to-End Deep Learning for Self-Driving Cars
- Udacity Self-Driving Car Engineer Nanodegree

## Example Code Snippets

### Data Loading and Preprocessing

```python
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering('track/IMG', data)

