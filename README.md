# Numerical Analysis and Machine Learning Projects

**Author:** Miguel Planas Díaz (Erasmus+ Student)  
**Personal Code:** 11071870  
**Matricola:** 276442

## Project Overview

This repository contains a collection of projects exploring fundamental concepts in **Numerical Analysis** and **Machine Learning**. The projects cover dimensionality reduction techniques, optimization algorithms, and neural network classification.

---

## Repository Structure

```
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── data/
│   └── faces.mat                       # Faces dataset for PCA analysis
├── notebooks/
│   ├── 00_project_overview.ipynb       # Main index notebook
│   ├── 01_pca_eigenfaces.ipynb         # PCA and Eigenfaces analysis
│   ├── 02_gradient_descent.ipynb       # Gradient Descent optimization
│   └── 03_neural_classification.ipynb  # Neural Network classification
├── src/
│   ├── __init__.py
│   ├── pca_utils.py                    # PCA utility functions
│   ├── optimization.py                 # Optimization algorithms
│   └── visualization.py                # Plotting utilities
└── results/
    └── figures/                        # Generated figures
```

---

## Projects

### Project 1: PCA and Eigenfaces Analysis
**Notebook:** `notebooks/01_pca_eigenfaces.ipynb`

Principal Component Analysis (PCA) applied to facial recognition. This project covers:
- Data normalization and preprocessing
- Singular Value Decomposition (SVD)
- Eigenfaces visualization
- Dimensionality reduction from 1024 to 100 dimensions
- Image reconstruction and error analysis

### Project 2: Gradient Descent Optimization
**Notebook:** `notebooks/02_gradient_descent.ipynb`

Implementation and analysis of gradient descent for quadratic optimization:
- Quadratic form derivation of cost functions
- Analytical solution computation
- Gradient descent implementation with JAX
- Learning rate analysis and convergence bounds
- Visualization of optimization trajectories

### Project 3: Neural Network Classification
**Notebook:** `notebooks/03_neural_classification.ipynb`

Classification of linearly and non-linearly separable datasets:
- Linear separability analysis
- Logistic regression decision boundaries
- Feature transformation for non-linear problems
- Neural network architecture design

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NAML_Projects.git
cd NAML_Projects
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- SciPy
- JAX
- tqdm

---

## Quick Start

```python
# Start with the overview notebook
jupyter notebook notebooks/00_project_overview.ipynb
```

Or run individual project notebooks directly.

---

## Key Results

### Eigenfaces Analysis
The PCA analysis successfully reduces facial images from 1024 dimensions to 100 while preserving essential features for reconstruction.

### Gradient Descent
Convergence achieved with learning rate η = 0.05, with theoretical maximum η_max ≈ 0.3636.

### Neural Classification
Demonstrated the need for hidden layers when dealing with non-linearly separable data (XOR-like patterns).

---

## Theoretical Background

### Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that identifies the directions of maximum variance in high-dimensional data through eigenvalue decomposition.

### Gradient Descent
An iterative optimization algorithm that moves toward the minimum of a function by following the negative gradient direction.

### Neural Networks
Computational models inspired by biological neural networks, capable of learning complex non-linear decision boundaries.

---

## License

This project is for educational purposes.

---

## Acknowledgments

- Course: Numerical Analysis and Machine Learning
- University: [Your University Name]
- Academic Year: 2024-2025
