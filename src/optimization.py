"""
Optimization Module
===================

Implementation of optimization algorithms including gradient descent.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional

# Try to import JAX, fall back to numpy if not available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np


def quadratic_form_from_least_squares(V: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert least squares problem to quadratic form.
    
    Given min_w sum_i (y_i - w^T v_i)^2, compute A, d, c such that
    J(w) = w^T A w + d^T w + c
    
    Parameters
    ----------
    V : np.ndarray
        Feature vectors, shape (n_samples, n_features)
    y : np.ndarray
        Target values, shape (n_samples,)
    
    Returns
    -------
    A : np.ndarray
        Quadratic term matrix
    d : np.ndarray
        Linear term vector
    c : float
        Constant term
    """
    A = V.T @ V
    d = -2 * V.T @ y
    c = np.dot(y, y)
    
    return A, d, c


def compute_exact_minimizer(A: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Compute the exact minimizer of a quadratic function.
    
    For J(w) = w^T A w + d^T w + c, the minimizer is:
    w* = -0.5 * A^(-1) * d
    
    Parameters
    ----------
    A : np.ndarray
        Positive definite matrix
    d : np.ndarray
        Linear term vector
    
    Returns
    -------
    w_star : np.ndarray
        Optimal parameter vector
    """
    return -0.5 * np.linalg.solve(A, d)


def compute_max_learning_rate(A: np.ndarray) -> float:
    """
    Compute the maximum learning rate for gradient descent convergence.
    
    For a quadratic function with Hessian H = 2A, the maximum learning rate
    is eta_max = 2 / lambda_max, where lambda_max is the largest eigenvalue of H.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix from quadratic form
    
    Returns
    -------
    eta_max : float
        Maximum learning rate for convergence
    """
    # The Hessian is 2A, so eigenvalues of Hessian are 2 * eigenvalues of A
    eigenvalues = np.linalg.eigvals(A)
    lambda_max = np.max(np.real(eigenvalues))
    
    # For stability: eta < 2 / (2 * lambda_max) = 1 / lambda_max
    # But commonly we use eta < 2 / L where L is Lipschitz constant of gradient
    return 2.0 / (2.0 * lambda_max)


def gradient_descent_numpy(x0: np.ndarray, 
                           grad_f: Callable[[np.ndarray], np.ndarray],
                           lr: float = 0.1, 
                           tol: float = 1e-6, 
                           max_iter: int = 1000) -> Tuple[List[np.ndarray], bool]:
    """
    Gradient descent optimization using NumPy.
    
    Parameters
    ----------
    x0 : np.ndarray
        Initial guess
    grad_f : Callable
        Function that computes the gradient
    lr : float
        Learning rate
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    trajectory : List[np.ndarray]
        Optimization trajectory
    converged : bool
        Whether algorithm converged
    """
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            return trajectory, True
        x = x - lr * grad
        trajectory.append(x.copy())
    
    return trajectory, False


def gradient_descent_jax(x0, f: Callable, lr: float = 0.1, 
                         tol: float = 1e-6, max_iter: int = 1000):
    """
    Gradient descent optimization using JAX automatic differentiation.
    
    Parameters
    ----------
    x0 : jax.numpy.ndarray
        Initial guess
    f : Callable
        Cost function to minimize
    lr : float
        Learning rate
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    trajectory : List
        Optimization trajectory
    converged : bool
        Whether algorithm converged
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for this function")
    
    grad_f = jax.grad(jax.jit(f))
    x = x0
    trajectory = [x]
    
    for i in range(max_iter):
        grad = grad_f(x)
        if jnp.linalg.norm(grad) < tol:
            return trajectory, True
        x = x - lr * grad
        trajectory.append(x)
    
    return trajectory, False


def quadratic_cost(w: np.ndarray, A: np.ndarray, d: np.ndarray, c: float) -> float:
    """
    Evaluate quadratic cost function.
    
    J(w) = w^T A w + d^T w + c
    
    Parameters
    ----------
    w : np.ndarray
        Parameter vector
    A : np.ndarray
        Quadratic term matrix
    d : np.ndarray
        Linear term vector
    c : float
        Constant term
    
    Returns
    -------
    cost : float
        Cost function value
    """
    return float(w.T @ A @ w + d.T @ w + c)


def quadratic_gradient(w: np.ndarray, A: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Compute gradient of quadratic cost function.
    
    ∇J(w) = 2Aw + d
    
    Parameters
    ----------
    w : np.ndarray
        Parameter vector
    A : np.ndarray
        Quadratic term matrix
    d : np.ndarray
        Linear term vector
    
    Returns
    -------
    grad : np.ndarray
        Gradient vector
    """
    return 2 * A @ w + d


class GradientDescentOptimizer:
    """
    Gradient Descent optimizer with configurable options.
    """
    
    def __init__(self, lr: float = 0.01, tol: float = 1e-6, 
                 max_iter: int = 1000, momentum: float = 0.0):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        lr : float
            Learning rate
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
        momentum : float
            Momentum coefficient (0 for vanilla GD)
        """
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.momentum = momentum
        self.trajectory = []
        self.converged = False
        
    def optimize(self, x0: np.ndarray, 
                 grad_f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Run optimization.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial guess
        grad_f : Callable
            Gradient function
        
        Returns
        -------
        x_opt : np.ndarray
            Optimized parameters
        """
        x = x0.copy()
        v = np.zeros_like(x)  # velocity for momentum
        self.trajectory = [x.copy()]
        
        for i in range(self.max_iter):
            grad = grad_f(x)
            
            if np.linalg.norm(grad) < self.tol:
                self.converged = True
                break
            
            # Momentum update
            v = self.momentum * v - self.lr * grad
            x = x + v
            
            self.trajectory.append(x.copy())
        
        return x
