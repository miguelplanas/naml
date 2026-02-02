"""
PCA Utilities Module
====================

Functions for Principal Component Analysis and Eigenfaces computation.
"""

import numpy as np
from typing import Tuple, Optional


def normalize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize data by subtracting the mean.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    
    Returns
    -------
    X_normalized : np.ndarray
        Centered data matrix
    X_mean : np.ndarray
        Mean vector of shape (n_features,)
    """
    X_mean = np.mean(X, axis=0)
    X_normalized = X - X_mean
    return X_normalized, X_mean


def compute_pca(X: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform PCA using Singular Value Decomposition.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    normalize : bool
        Whether to center the data before PCA
    
    Returns
    -------
    U : np.ndarray
        Left singular vectors (principal component scores)
    S : np.ndarray
        Singular values
    VT : np.ndarray
        Right singular vectors (principal components)
    X_mean : np.ndarray or None
        Mean vector if normalization was applied
    """
    if normalize:
        X_centered, X_mean = normalize_data(X)
    else:
        X_centered = X
        X_mean = None
    
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    
    return U, S, VT, X_mean


def get_eigenfaces(VT: np.ndarray, n_components: int = 25) -> np.ndarray:
    """
    Extract the first n eigenfaces from the right singular vectors.
    
    Parameters
    ----------
    VT : np.ndarray
        Right singular vectors from SVD
    n_components : int
        Number of eigenfaces to extract
    
    Returns
    -------
    eigenfaces : np.ndarray
        First n_components eigenfaces
    """
    return VT[:n_components]


def project_data(U: np.ndarray, S: np.ndarray, n_components: int) -> np.ndarray:
    """
    Project data onto the first n principal components.
    
    Parameters
    ----------
    U : np.ndarray
        Left singular vectors
    S : np.ndarray
        Singular values
    n_components : int
        Number of components to use
    
    Returns
    -------
    X_reduced : np.ndarray
        Projected data of shape (n_samples, n_components)
    """
    return U[:, :n_components] @ np.diag(S[:n_components])


def reconstruct_data(U: np.ndarray, S: np.ndarray, VT: np.ndarray, 
                     n_components: int, X_mean: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct data from the first n principal components.
    
    Parameters
    ----------
    U : np.ndarray
        Left singular vectors
    S : np.ndarray
        Singular values
    VT : np.ndarray
        Right singular vectors
    n_components : int
        Number of components to use for reconstruction
    X_mean : np.ndarray, optional
        Mean to add back if data was centered
    
    Returns
    -------
    X_reconstructed : np.ndarray
        Reconstructed data
    """
    X_reconstructed = U[:, :n_components] @ np.diag(S[:n_components]) @ VT[:n_components]
    
    if X_mean is not None:
        X_reconstructed += X_mean
    
    return X_reconstructed


def compute_reconstruction_error(X_original: np.ndarray, X_reconstructed: np.ndarray, 
                                  per_sample: bool = True) -> np.ndarray:
    """
    Compute mean squared reconstruction error.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original data
    X_reconstructed : np.ndarray
        Reconstructed data
    per_sample : bool
        If True, return error per sample; if False, return total error
    
    Returns
    -------
    error : np.ndarray or float
        Reconstruction error
    """
    if per_sample:
        return np.mean((X_original - X_reconstructed)**2, axis=1)
    else:
        return np.mean((X_original - X_reconstructed)**2)


def explained_variance_ratio(S: np.ndarray) -> np.ndarray:
    """
    Compute the explained variance ratio for each component.
    
    Parameters
    ----------
    S : np.ndarray
        Singular values
    
    Returns
    -------
    ratio : np.ndarray
        Explained variance ratio for each component
    """
    variance = S**2
    return variance / np.sum(variance)


def cumulative_explained_variance(S: np.ndarray) -> np.ndarray:
    """
    Compute cumulative explained variance.
    
    Parameters
    ----------
    S : np.ndarray
        Singular values
    
    Returns
    -------
    cumulative : np.ndarray
        Cumulative explained variance
    """
    return np.cumsum(explained_variance_ratio(S))
