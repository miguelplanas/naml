"""
Visualization Module
====================

Plotting utilities for PCA, optimization, and classification visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple, Optional, Union
from mpl_toolkits.mplot3d import Axes3D


def plot_eigenfaces(eigenfaces: np.ndarray, 
                   img_shape: Tuple[int, int] = (32, 32),
                   n_cols: int = 5,
                   figsize: Tuple[int, int] = (10, 10),
                   title: str = "Eigenfaces") -> Figure:
    """
    Plot eigenfaces in a grid.
    
    Parameters
    ----------
    eigenfaces : np.ndarray
        Array of eigenfaces, shape (n_faces, n_pixels)
    img_shape : Tuple[int, int]
        Shape of individual images
    n_cols : int
        Number of columns in the grid
    figsize : Tuple[int, int]
        Figure size
    title : str
        Plot title
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    n_faces = len(eigenfaces)
    n_rows = (n_faces + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n_faces:
            img = eigenfaces[i].reshape(img_shape).T
            ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_image_grid(images: np.ndarray,
                   img_shape: Tuple[int, int] = (32, 32),
                   n_cols: int = 10,
                   figsize: Tuple[int, int] = (15, 15),
                   title: str = "Images") -> Figure:
    """
    Plot images in a grid.
    
    Parameters
    ----------
    images : np.ndarray
        Array of images, shape (n_images, n_pixels)
    img_shape : Tuple[int, int]
        Shape of individual images
    n_cols : int
        Number of columns
    figsize : Tuple[int, int]
        Figure size
    title : str
        Plot title
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < n_images:
                axes[i, j].imshow(images[idx].reshape(img_shape).T, cmap='gray')
            axes[i, j].axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_reconstruction_comparison(original: np.ndarray,
                                  reconstructed: np.ndarray,
                                  img_shape: Tuple[int, int] = (32, 32),
                                  n_samples: int = 5,
                                  figsize: Tuple[int, int] = (15, 6)) -> Figure:
    """
    Compare original and reconstructed images side by side.
    
    Parameters
    ----------
    original : np.ndarray
        Original images
    reconstructed : np.ndarray
        Reconstructed images
    img_shape : Tuple[int, int]
        Shape of images
    n_samples : int
        Number of samples to show
    figsize : Tuple[int, int]
        Figure size
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original[i].reshape(img_shape).T, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].reshape(img_shape).T, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed')
        
        # Difference
        diff = np.abs(original[i] - reconstructed[i])
        axes[2, i].imshow(diff.reshape(img_shape).T, cmap='hot')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Difference')
    
    plt.suptitle('Original vs Reconstructed Images', fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_reconstruction_error(errors: np.ndarray,
                             figsize: Tuple[int, int] = (10, 5)) -> Figure:
    """
    Plot reconstruction error per sample.
    
    Parameters
    ----------
    errors : np.ndarray
        Per-sample reconstruction errors
    figsize : Tuple[int, int]
        Figure size
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(errors, marker='o', markersize=3, linewidth=1)
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.set_title('Reconstruction Error per Image')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_explained_variance(cumulative_var: np.ndarray,
                           figsize: Tuple[int, int] = (10, 5)) -> Figure:
    """
    Plot cumulative explained variance.
    
    Parameters
    ----------
    cumulative_var : np.ndarray
        Cumulative explained variance ratio
    figsize : Tuple[int, int]
        Figure size
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_components = len(cumulative_var)
    ax.plot(range(1, n_components + 1), cumulative_var, 'b-', linewidth=2)
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Explained Variance vs. Number of Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_quadratic_surface(A: np.ndarray, d: np.ndarray, c: float,
                          xlim: Tuple[float, float] = (-1, 2),
                          ylim: Tuple[float, float] = (-1, 2),
                          n_points: int = 100,
                          figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    Plot 3D surface of a quadratic function.
    
    Parameters
    ----------
    A : np.ndarray
        Quadratic matrix (2x2)
    d : np.ndarray
        Linear term
    c : float
        Constant term
    xlim, ylim : Tuple[float, float]
        Plot limits
    n_points : int
        Grid resolution
    figsize : Tuple[int, int]
        Figure size
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    w0 = np.linspace(xlim[0], xlim[1], n_points)
    w1 = np.linspace(ylim[0], ylim[1], n_points)
    W0, W1 = np.meshgrid(w0, w1)
    
    # Compute J(w) for the quadratic form
    J = (A[0, 0] * W0**2 + 2 * A[0, 1] * W0 * W1 + A[1, 1] * W1**2 +
         d[0] * W0 + d[1] * W1 + c)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W0, W1, J, cmap='viridis', alpha=0.8)
    ax.set_xlabel('$w_0$')
    ax.set_ylabel('$w_1$')
    ax.set_zlabel('$J(w)$')
    ax.set_title('Cost Function Surface')
    
    return fig


def plot_gradient_descent_trajectory(trajectory: List[np.ndarray],
                                    A: np.ndarray, d: np.ndarray, c: float,
                                    xlim: Tuple[float, float] = (0, 1),
                                    ylim: Tuple[float, float] = (0, 1),
                                    n_levels: int = 50,
                                    figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    Plot gradient descent trajectory on contour plot.
    
    Parameters
    ----------
    trajectory : List[np.ndarray]
        List of parameter vectors along the optimization path
    A, d, c : quadratic form parameters
    xlim, ylim : Tuple[float, float]
        Plot limits
    n_levels : int
        Number of contour levels
    figsize : Tuple[int, int]
        Figure size
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    trajectory = np.array(trajectory)
    
    w0 = np.linspace(xlim[0], xlim[1], 100)
    w1 = np.linspace(ylim[0], ylim[1], 100)
    W0, W1 = np.meshgrid(w0, w1)
    
    J = (A[0, 0] * W0**2 + 2 * A[0, 1] * W0 * W1 + A[1, 1] * W1**2 +
         d[0] * W0 + d[1] * W1 + c)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Contour plot
    contour = ax.contour(W0, W1, J, levels=n_levels, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', 
            markersize=4, linewidth=1.5, label='GD Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], 
               color='green', s=100, zorder=5, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
               color='blue', s=100, zorder=5, label='End')
    
    ax.set_xlabel('$w_0$')
    ax.set_ylabel('$w_1$')
    ax.set_title('Gradient Descent Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_classification_data(datasets: dict, figsize: Tuple[int, int] = (15, 5)) -> Figure:
    """
    Plot classification datasets.
    
    Parameters
    ----------
    datasets : dict
        Dictionary with dataset names as keys, containing 'x1', 'x2', 'labels'
    figsize : Tuple[int, int]
        Figure size
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize)
    
    if n_datasets == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, datasets.items()):
        x1 = np.array(data['x1'])
        x2 = np.array(data['x2'])
        labels = np.array(data['labels'])
        
        # Plot class 0
        ax.scatter(x1[labels == 0], x2[labels == 0], 
                   c='blue', marker='o', s=100, label='Class 0')
        # Plot class 1
        ax.scatter(x1[labels == 1], x2[labels == 1], 
                   c='red', marker='x', s=100, label='Class 1')
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, 
                          weights: np.ndarray,
                          xlim: Tuple[float, float] = (-2, 2),
                          ylim: Tuple[float, float] = (-2, 2),
                          figsize: Tuple[int, int] = (8, 8)) -> Figure:
    """
    Plot data points with logistic regression decision boundary.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, 2)
    y : np.ndarray
        Labels
    weights : np.ndarray
        Logistic regression weights [bias, w1, w2]
    xlim, ylim : Tuple[float, float]
        Plot limits
    figsize : Tuple[int, int]
        Figure size
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o', 
               s=100, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='x', 
               s=100, label='Class 1')
    
    # Plot decision boundary: w0 + w1*x1 + w2*x2 = 0
    # => x2 = -(w0 + w1*x1) / w2
    if abs(weights[2]) > 1e-6:
        x1_boundary = np.linspace(xlim[0], xlim[1], 100)
        x2_boundary = -(weights[0] + weights[1] * x1_boundary) / weights[2]
        ax.plot(x1_boundary, x2_boundary, 'g-', linewidth=2, label='Decision Boundary')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Classification with Decision Boundary')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
