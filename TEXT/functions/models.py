import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenization(tokenizer: keras.preprocessing.text.Tokenizer, X_train: pd.Series, X_test: pd.Series, col: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Tokenizes a column of a dataframe by applying: tokenization, sequencing and padding. 

    Args:
    -------
    tokenizer: keras.preprocessing.text.Tokenizer
        Tokenizer to use.
    X_train: pd.Series
        Train dataframe.
    X_test: pd.Series
        Test dataframe.
    col: str
        Name of the column to tokenize.

    Returns:
    -------
    train_padded: np.array
        Padded train sequences.
    test_padded: np.array
        Padded test sequences.
    max_seq_len: int
        Length of the longest sequence.
    vocab_size: int
        Size of the vocabulary.
    """
    # Fit tokenizer on train set
    tokenizer.fit_on_texts(X_train[col])

    # Conver text to sequences for both train and test sets
    train_sequences = tokenizer.texts_to_sequences(X_train[col])
    test_sequences = tokenizer.texts_to_sequences(X_test[col])

    # Get lenght of the longest sequence
    max_seq_len = max([len(seq) for seq in train_sequences])
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    
    # Applying padding to both train and test sets
    train_padded = pad_sequences(train_sequences, maxlen=max_seq_len, padding="post")
    test_padded = pad_sequences(test_sequences, maxlen=max_seq_len, padding="post")

    return train_padded, test_padded, max_seq_len, vocab_size, tokenizer

def plot_metric_curves(epochs, train_curve, val_curve, train_color, val_color, metric, epochs_interval: int = 1):
    """
    Plots the training and validation curves for a specific metric.

    Args:
    -------
    epochs: list
        List of epochs.
    train_curve: list
        List of training metric values.
    val_curve: list
        List of validation metric values.
    train_color: str
        Color of the training curve.
    val_color: str
        Color of the validation curve.
    metric: str
        Metric to plot.
    epochs_interval: int
        Interval of epochs to show in the x-axis. Default is 1.
    
    Returns:
    -------
    None
    """
    # Set graphics format as svg
    set_matplotlib_formats('svg')
    # Add grid
    sns.set_style("whitegrid")
    # Set figure size
    plt.figure(figsize=(12,8), dpi=200)
    # Plot curves
    plt.plot(epochs, train_curve, color=train_color, linewidth=2, label=f'Training {metric.lower()}')
    plt.plot(epochs, val_curve, color=val_color, linewidth=2, label=f'Validation {metric.lower()}')
    # Set title and labels
    plt.title(f'Training and validation {metric.lower()}', fontsize=20)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel(metric.capitalize(), fontsize=15)
    # Show epoch values in x-axis in the specified interval
    plt.xticks(epochs[::epochs_interval])
    # Set max x-axis as the last epoch + 1
    plt.xlim(0, epochs[-1] + 1)
    # Remove top and right spines
    sns.despine(left=True, bottom=True)
    # Put the legend outside of the plot
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=13)
    plt.show()
    return

def plot_confusion_matrix(confusion_matrix: np.ndarray, labels: list[str], cmap: str = "Blues"):
    """
    Plots the confusion matrix.

    Args:
    -------
    confusion_matrix: np.ndarray
        Confusion matrix.
    labels: list[str]
        List of labels.
    cmap: str
        Color map to be used. Default is "Blues".

    Returns:
    -------
    None
    """
    # Set graphics format as svg
    set_matplotlib_formats('svg')
    # Plot confusion matrix
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=cmap, vmin=0)
    # Set title
    plt.title("Confusion matrix", fontsize=20)
    # Set labels
    plt.xlabel("Predicted label", fontsize=15)
    plt.ylabel("True label", fontsize=15)
    plt.xticks([0.5, 1.5], labels, fontsize=11)
    plt.yticks([0.5, 1.5], labels, fontsize=11)
    plt.show()