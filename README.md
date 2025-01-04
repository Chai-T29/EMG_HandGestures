# EMG Hand Gesture Classification

## Introduction

This project focuses on classifying hand gestures using Electromyography (EMG) data. The data is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures) and consists of recordings from 36 participants performing 8 distinct gestures. Each gesture lasts for 3 seconds, and the aim is to compare the performance of windowing versus non-windowing data processing techniques for gesture classification. This work evaluates deep learning and machine learning models, leveraging hardware acceleration for efficient signal processing, which has significant implications for real-time applications like robotic prosthetics.

<br>

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

<br>

## Code Dependencies

To install the dependencies, run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch pytorch-lightning tqdm
```

For hardware acceleration on macOS (using MPS), ensure you have macOS 12.3+ and a compatible Apple Silicon device.

<br>

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

<br>

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

<br>

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

<br>

## Applications

This project has several real-world applications:
1. Robotic Prosthetics: Real-time gesture classification for prosthetic control.
2. Healthcare: EMG-based monitoring and rehabilitation systems for patients.
3. Human-Machine Interaction: Enhancing gesture-based interfaces in wearable technology.
4. Gaming and VR/AR: Gesture recognition for immersive user experiences.

<br>

## Conclusion

The study demonstrates the potential of modern deep learning techniques, particularly windowing-based methods, to outperform traditional machine learning models in terms of accuracy and testing speed. While the non-windowing approach with deep learning achieves comparable results, windowing captures temporal dependencies more effectively. This project underscores the importance of leveraging hardware acceleration for real-time applications and sets the foundation for optimizing EMG signal processing pipelines in advanced systems.

<br>

## References
1. Olmo, M. D., & Domingo, R. (2020). EMG Characterization and Processing in Production Engineering. Materials (Basel, Switzerland), 13(24), 5815. https://doi.org/10.3390/ma13245815
2. Raez, M. B., Hussain, M. S., & Mohd-Yasin, F. (2006). Techniques of EMG signal analysis: detection, processing, classification and applications. Biological procedures online, 8, 11– 35. https://doi.org/10.1251/bpo115
3. Rani, G. J., Hashmi, M. F., & Gupta, A. (2023). Surface Electromyography and Artificial Intelligence for Human Activity Recognition-A Systematic Review on Methods, Emerging Trends Applications, Challenges, and Future Implementation. IEEE Access, 11, 105140–105169. https://doi.org/10.1109/ACCESS.2023.3316509
4. Asogbon, Mojisola Grace, et al. "Effect of window conditioning parameters on the classification performance and stability of EMG-based feature extraction methods." 2018 IEEE International Conference on Cyborg and Bionic Systems (CBS). IEEE, 2018. https://doi.org/10.1109/CBS.2018.8612246
5. Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V., & Lobov, S. (2018). EMG Data for Gestures [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZP5C.
6. Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001). https://doi.org/10.1023/A:1010933404324
7. MPS backend — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/notes/mps.html
8. Bracewell, R. N. (1989). The Fourier Transform. Scientific American, 260(6), 86–95. http://www.jstor.org/stable/24987290
9. Lai, E. (2003). Frequency-domain representation of discrete-time signals. In Elsevier eBooks (pp. 61–78). https://doi.org/10.1016/b978-075065798-3/50004-7
10. Conv1d — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
11. LeakyReLU — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
12. ReLU — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
13. Dureja, A., & Pahwa, P. (2019). Analysis of nonlinear activation functions for classification tasks using convolutional neural networks. In Lecture notes in electrical engineering (pp. 1179–1190). https://doi.org/10.1007/978-981-13-6772-4_103
14. BatchNorm1d — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
15. MaxPool1d — PyTorch 2.5 documentation. (n.d.). 13 https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
16. Dropout1d — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.nn.Dropout1d.html
17. Wang, R., Shivanna, R., Cheng, D., Jain, S., Lin, D., Hong, L., & Chi, E. (2021, April). Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the web conference 2021 (pp. 1785-1797). https://arxiv.org/abs/2008.13535
18. R. Johnson, C. (1974). Hadamard products of matrices. Linear and Multilinear Algebra, 1(4), 295–307. https://doi.org/10.1080/03081087408817030
19. LayerNorm — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
20. Linear — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
21. RandomForestClassifier. (n.d.). Scikit-learn. https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
22. He, X., Du, X., Wang, X., Tian, F., Tang, J., & Chua, T. (2018, August 12). Outer product-based neural collaborative filtering. arXiv.org. https://arxiv.org/abs/1808.03912
23. CrossEntropyLoss — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
24. Adam — PyTorch 2.5 documentation. (n.d.). https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
25. accuracy_score. (n.d.). Scikit-learn. https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.accuracy_score.html
26. confusion_matrix. (n.d.). Scikit-learn. https://scikit-learn.org/dev/modules/generated/sklearn.metrics.confusion_matrix.html
27. Zhuang, F., Qi, Z., Duan, K., Xi, D., Zhu, Y., Zhu, H., Xiong, H., & He, Q. (2019, November 7). A comprehensive survey on transfer learning. arXiv.org. https://arxiv.org/abs/1911.02685
28. Malešević, N., Olsson, A., Sager, P., Andersson, E., Cipriani, C., Controzzi, M., Björkman, A., & Antfolk, C. (2021). A database of high-density surface electromyogram signals comprising 65 isometric hand gestures. Scientific Data, 8(1). https://doi.org/10.1038/s41597-021-00843-9
