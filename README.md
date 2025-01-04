# EMG Hand Gesture Classification

## Introduction

This project focuses on classifying hand gestures using Electromyography (EMG) data. The data is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures) and consists of recordings from 36 participants performing 8 distinct gestures. Each gesture lasts for 3 seconds, and the aim is to compare the performance of windowing versus non-windowing data processing techniques for gesture classification. This work evaluates deep learning and machine learning models, leveraging hardware acceleration for efficient signal processing, which has significant implications for real-time applications like robotic prosthetics.

## Contents
1. Introduction
  - Overview of the project and objectives.
2. Data Description
  - Source, structure, and characteristics of the dataset.
3. Data Preprocessing
  - Handling class imbalance, feature scaling, and data transformations.
4. Modeling Approaches
  - Windowing and non-windowing pipelines.
5. Evaluation Metrics
  - Accuracy, precision, recall, and AUC-ROC for performance comparison.
6. Visualization
  - Proportions of classes, confusion matrices, and other relevant plots.
7. Conclusion
  - Insights and future directions.


## Algorithms Used

### Deep Learning
  - Windowing Technique:
      A deep learning pipeline utilizing:
    - Convolutional Neural Networks (CNNs) for feature extraction.
    - Deep Cross Networks (DCNs) for efficient feature interaction.
    - Multilayer Perceptron (MLP) for classification.
    - Custom windowing to capture temporal dependencies over a sequence of 16 time steps.
    - Hardware acceleration using Apple’s Metallic Performance Shaders (MPS).
    - Non-Windowing Technique:
A feedforward neural network designed to process individual time steps as separate samples. This approach avoids temporal aggregation and focuses on leveraging raw feature values.
