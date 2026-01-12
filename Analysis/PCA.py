# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 18:05:13 2026

@author: mrrec
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import percentileofscore

def optimize_microbiome_pca(data_matrix, var_threshold):
    """
    Function to perform PCA with feature selection based on variance threshold
    
    Parameters:
    -----------
    data_matrix : numpy array
        Input data matrix (samples x features)
    var_threshold : float
        Threshold for variance (e.g., 0.1 means keep top 90% features)
    
    Returns:
    --------
    Y : numpy array
        PCA scores (first 2 components)
    explained : numpy array
        Explained variance ratio for first 2 components
    selected_features : numpy array
        Boolean mask of selected features
    """
    # Calculate variance of each feature
    variances = np.var(data_matrix, axis=0)
    
    # Calculate threshold (keep features above this percentile)
    threshold = np.percentile(variances, var_threshold * 100)
    selected_features = variances > threshold
    
    # Filter data
    data_filtered = data_matrix[:, selected_features]
    
    # Add pseudo count (half of minimum positive value)
    pseudo_count = 0.5 * np.min(data_filtered[data_filtered > 0])
    data_pseudo = data_filtered + pseudo_count
    
    # Center log-ratio (CLR) transformation
    # Calculate geometric mean for each sample
    geo_mean = np.exp(np.mean(np.log(data_pseudo), axis=1))
    # CLR transform
    data_clr = np.log(data_pseudo / geo_mean[:, np.newaxis])
    
    # Center the data
    data_centered = data_clr - np.mean(data_clr, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(data_centered, rowvar=False)
    
    # Regularization (add small value to diagonal)
    reg_param = 0.01 * np.trace(cov_matrix) / cov_matrix.shape[0]
    cov_reg = cov_matrix + reg_param * np.eye(cov_matrix.shape[0])
    
    # Eigen decomposition
    eigenvals, V = np.linalg.eigh(cov_reg)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    V_sorted = V[:, idx]
    
    # Calculate scores
    score = data_centered @ V_sorted
    
    # Get first 2 components
    Y = score[:, :2]
    
    # Calculate explained variance
    total_variance = np.sum(eigenvals)
    explained = 100 * eigenvals[:2] / total_variance
    
    print(f'PC1 explained rate: {explained[0]:.1f}%')
    print(f'PC2 explained rate: {explained[1]:.1f}%')
    
    return Y, explained, selected_features

def main():
    # Read Input Data
    data = pd.read_excel('microbiome.xlsx')
    # 1-515 rows represent features of pre-FMT recipient
    # 516-1030 rows represent features of post-FMT recipient
    # 1031-1545 rows represent features of donor
    
    labels = pd.read_excel('label.xlsx')
    
    # Filter Features
    data_matrix = data.values
    diseases_class = labels['Response'].values
    
    # PCA Method
    Y_optimized, explained_opt, selected = optimize_microbiome_pca(data_matrix, 0.1)
    Y = Y_optimized[:, :2]
    
    # Alternative: t-SNE (commented out)
    # Note: t-SNE results may vary slightly due to randomization
    # The overall trends will remain the same
    # tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    # Y = tsne.fit_transform(data_matrix)
    
    # For PCA figure of different categories (Response/Non-Response/Unknown)
    # Calculate Post-FMT Recipient - Pre-FMT Recipient
    point = Y[515:1030, :] - Y[:515, :]  # Python uses 0-based indexing
    
    # Alternative: Post-FMT Recipient - Donor
    # point = Y[515:1030, :] - Y[1030:1545, :]
    
    new_class = diseases_class[:515]  # Use only pre-FMT recipient labels
    
    # Create colors for different groups
    # Using matplotlib's tab20 colormap for more colors
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Adjust colors as needed (similar to MATLAB lines colormap)
    # For the first 6 colors, we can use tab20
    # For the next 7, we can adjust (similar to MATLAB code)
    # Create custom colormap if needed
    
    # Figure 1: Different categories (Response/Non-Response/Unknown)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Plot points for each class
    for i in range(3):
        idx = (new_class == i)
        ax1.scatter(point[idx, 0], point[idx, 1], 
                   s=40, marker='o',
                   color=cmap[i], 
                   alpha=0.6,
                   edgecolors=cmap[i],
                   linewidths=1,
                   label=f'Class {i}')
    
    # Calculate center points
    centers = np.zeros((2, 2))
    for i in range(2):
        class_points = point[new_class == i]
        if len(class_points) > 0:
            centers[i, :] = np.mean(class_points, axis=0)
    
    # Draw arrows from origin to center points
    for i in range(1, -1, -1):  # Reverse order as in MATLAB
        ax1.arrow(0, 0, centers[i, 0], centers[i, 1],
                 head_width=0.05, head_length=0.1, 
                 fc=cmap[i], ec=cmap[i],
                 linewidth=1.5,
                 length_includes_head=True)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA: Post-FMT vs Pre-FMT Differences')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Figure 2: Different diseases categories
    # Note: Assuming Diseases_Class column exists in labels
    if 'Diseases_Class' in labels.columns:
        diseases_class_full = labels['Diseases_Class'].values
        diseases_new_class = diseases_class_full[:515]  # Pre-FMT labels
        
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        
        # Plot points for each disease class (1-13)
        for i in range(1, 14):  # 1 to 13 inclusive
            idx = (diseases_new_class == i)
            if np.any(idx):
                ax2.scatter(point[idx, 0], point[idx, 1],
                           s=40, marker='o',
                           color=cmap[i-1],
                           alpha=0.3,
                           edgecolors=cmap[i-1],
                           linewidths=1,
                           label=f'Disease {i}')
        
        # Calculate center points for each disease class
        centers_diseases = np.zeros((13, 2))
        for i in range(1, 14):
            class_points = point[diseases_new_class == i]
            if len(class_points) > 0:
                centers_diseases[i-1, :] = np.mean(class_points, axis=0)
        
        # Draw arrows from origin to center points
        for i in range(13):
            ax2.arrow(0, 0, centers_diseases[i, 0], centers_diseases[i, 1],
                     head_width=0.02, head_length=0.04,
                     fc=cmap[i], ec=cmap[i],
                     linewidth=1.5,
                     length_includes_head=True)
        
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('PCA: Disease Categories Differences')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
