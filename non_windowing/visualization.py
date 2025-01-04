import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

def visualize_results(X_test, y_test, model):
    '''
    Function that plots a performance dashboard for a multi-class classification model.
    '''
    
    # Data Preparation
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    testing_time = end - start
    print(f'Testing Time: {testing_time:.3f} seconds')
    
    # Predict probabilities for each class
    probabilities = model.predict_proba(X_test)
    
    # Data Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: \033[1m{accuracy}\033[0m')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC / AUC (One-vs-Rest for multi-class)
    # binarize the labels for ROC curve computation
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Precision-Recall Curve / Average Precision Score
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], probabilities[:, i])
        avg_precision[i] = average_precision_score(y_test_bin[:, i], probabilities[:, i])

    # Classification Report
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    clf_df = pd.DataFrame(clf_report).transpose()
    
    # Visualizing the performance metrics
    plt.style.use("dark_background")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

    # Confusion Matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='magma', ax=axes[0, 0], norm=LogNorm())
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')

    # ROC Curve (One-vs-Rest)
    for i in range(n_classes):
        axes[0, 1].plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve (One-vs-Rest)')
    axes[0, 1].legend(loc='lower right')

    # Precision-Recall Curve (One-vs-Rest)
    for i in range(n_classes):
        axes[1, 0].step(recall[i], precision[i], where='post', label=f'Class {i} AP = {avg_precision[i]:.2f}')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_title('Precision-Recall Curve (One-vs-Rest)')
    axes[1, 0].legend(loc='lower left')

    # Classification Report Heatmap
    sns.heatmap(clf_df.iloc[:-1, :-1], annot=True, cmap='magma', fmt='.2f', ax=axes[1, 1])
    axes[1, 1].set_title('Classification Report')

    plt.tight_layout()
    plt.show()