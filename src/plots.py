import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_numerical_distributions(X):
    numerical_col = X.select_dtypes(include=['int64']).columns

    num_cols = 3  
    num_rows = (len(numerical_col) + num_cols - 1) // num_cols  # Calculate required rows

    plt.figure(figsize=(20, num_rows * 5))

    for i, feature in enumerate(numerical_col):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(X[feature], bins=30, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show(block=True)

def plot_categorical_distributions(X):
    categorical_col = X.select_dtypes(include=['object']).columns

    num_cols = 3  
    num_rows = (len(categorical_col) + num_cols - 1) // num_cols  # Calculate required rows

    plt.figure(figsize=(20, num_rows * 5))

    for i, feature in enumerate(categorical_col):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(x=X[feature])
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show(block=True)

def plot_heatmap_with_target(X, y):
    numerical_cor = X.select_dtypes(include=['int64']).corr()
    plt.figure(figsize=(20,20)) 
    sns.heatmap(numerical_cor, cmap='viridis', fmt='.2f', annot=True)
    plt.title('Correlation Heatmap')
    plt.show(block=True)

def plot_target_distribution(y):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Damage Grade')
    plt.xlabel('Damage Grade')
    plt.ylabel('Count')
    plt.show(block=True)
