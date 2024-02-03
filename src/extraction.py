"""
FactorExtractor: A Python class for factor analysis using Maximum Likelihood Estimation (MLE).

The FactorExtractor class is designed for performing factor analysis on datasets to identify latent
variables that explain patterns in the observed variables. It uses the Maximum Likelihood Estimation
method to extract the underlying factors, compute the factor loadings, uniqueness, and communality of
each feature. This class is particularly useful in exploratory data analysis, where the underlying
structure of the data is not known beforehand.

Usage:
1. Import the Extractor class.
2. Create an instance of FactorExtractor.
3. Call the mle method with your data matrix, specifying the number of factors and other optional parameters.

Example:
    from extractor import FactorExtractor
    ex = FactorExtractor()
    loadings, psi, communality = ex.mle(data_matrix, n_factors=4)

Date: 2023.2.3
Version: 1.0
"""

import numpy as np

class FactorExtractor:
    """
    Method:
        - mle (Maximum Likelihood Estimation)
    """
    def __init__(self):
        pass

    def mle(self, X: np.ndarray, n_factors: int, max_iter: int = 100, tol: float = 1e-5) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Perform Factor Analysis using the Maximum Likelihood Estimation (MLE) method.

        Args:
        X (np.ndarray): The input data matrix with shape (n_samples, n_features).
        n_factors (int): The number of latent factors to extract.
        max_iter (int): The maximum number of iterations for the convergence of the algorithm.
        tol (float): The tolerance level for convergence. The algorithm stops if the change in estimates is below this level.

        Returns:
        (np.ndarray, np.ndarray, np.ndarray):
            - loadings (np.ndarray): The factor loadings matrix of shape (n_features, n_factors).
            - psi (np.ndarray): The uniqueness (unique variances) of the variables, array of length n_features.
            - communality (np.ndarray): The communality of each feature, array of length n_features.
        """
        # Standardize the data by subtracting the mean of each feature
        X_std = X - np.mean(X, axis=0)
        n_samples, n_features = X_std.shape

        # Calculate the covariance matrix of the standardized data
        cov_matrix = np.cov(X_std, rowvar=False)

        # Initialize the factor loadings (Lambda) matrix with random values
        loadings = np.random.rand(n_features, n_factors)

        # Initialize the unique variances (Psi) vector with random values
        psi = np.random.rand(n_features)

        # Iterate until convergence or maximum iterations
        for i in range(max_iter):
            # E-step: Compute the inverse of the expected factor covariance matrix
            # Convert Psi to a diagonal matrix and add it to Lambda Lambda^T
            psi_diag = np.diag(psi)
            factor_cov = np.dot(loadings, loadings.T) + psi_diag
            factor_cov_inv = np.linalg.inv(factor_cov)

            # M-step: Update the factor loadings (Lambda) and unique variances (Psi)
            # Update loadings using the formula: Lambda_new = S * Lambda * factor_cov_inv
            loadings_new = np.dot(np.dot(cov_matrix, loadings), factor_cov_inv)

            # Update unique variances (Psi)
            # The new Psi is the diagonal of the covariance matrix minus the diagonal of Lambda_new * Lambda_new^T
            psi_new = np.diag(cov_matrix) - np.diag(np.dot(np.dot(loadings_new, factor_cov_inv), loadings_new.T))

            # Check for convergence
            # If the maximum change in loadings and unique variances is below the tolerance, stop the iteration
            if np.max(np.abs(loadings_new - loadings)) < tol and np.max(np.abs(psi_new - psi)) < tol:
                break

            # Update the parameters for the next iteration
            loadings = loadings_new
            psi = psi_new

        # Calculate communalities
        # Communalities are the sum of squared loadings for each feature
        communality = np.sum(loadings**2, axis=1)

        return loadings, psi, communality
