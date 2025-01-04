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


## Code Dependencies

To install the dependencies, run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch pytorch-lightning tqdm
```

For hardware acceleration on macOS (using MPS), ensure you have macOS 12.3+ and a compatible Apple Silicon device.


## Data Description

The dataset contains EMG signals from 8 channels for 8 gestures. The gestures include:
1. Hand at Rest
2. Hand Clenched in a Fist
3. Wrist Flexion
4. Wrist Extension
5. Radial Deviations
6. Ulnar Deviations
7. Extended Palm
8. Unmarked (no gesture)

Each gesture is performed for 3 seconds per trial, repeated twice by each participant.

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

### Machine Learning
Baseline models used for comparison:
  - Logistic Regression (with and without class balancing).
  - Random Forest Classifier (with and without class balancing).
  - Linear Support Vector Machines (SVMs, calibrated for probabilistic outputs).


## How to Use the Script

1. Clone this repository:

```bash
git clone https://github.com/your-username/emg-classification.git
cd emg-classification
```

2. Place the dataset in the EMG_data folder.

3. Open the main notebook:

```bash
jupyter notebook EMG_Classification.ipynb
```

4. Run the notebook step by step:
    - Preprocess the data.
    - Train the models using the windowing or non-windowing pipelines.
    - Evaluate the results with the provided visualization functions.

5. For training and testing specific models:
    - Use scripts in the windowing and non-windowing folders:
    - data_processing.py for splitting sequences.
    - model.py for defining and training the models.
    - To visualize results, call the visualize_results function from the respective scripts.


## Applications

This project has several real-world applications:
1. Robotic Prosthetics: Real-time gesture classification for prosthetic control.
2. Healthcare: EMG-based monitoring and rehabilitation systems for patients.
3. Human-Machine Interaction: Enhancing gesture-based interfaces in wearable technology.
4. Gaming and VR/AR: Gesture recognition for immersive user experiences.

Conclusion

The study demonstrates the potential of modern deep learning techniques, particularly windowing-based methods, to outperform traditional machine learning models in terms of accuracy and testing speed. While the non-windowing approach with deep learning achieves comparable results, windowing captures temporal dependencies more effectively. This project underscores the importance of leveraging hardware acceleration for real-time applications and sets the foundation for optimizing EMG signal processing pipelines in advanced systems.
