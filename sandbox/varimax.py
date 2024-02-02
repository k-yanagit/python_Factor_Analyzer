import numpy as np


def varimax(Phi, gamma: float = 1.0, max_itr: int = 100, tol: float = 1e-6):
    """
    Perform Varimax (orthogonal) rotation, with an option to adjust the gamma parameter for
    different types of rotations (e.g., Quartimax when gamma = 0).

    Args:
        Phi (ndarray): Initial factor loading matrix (2D array: variables x factors).
        gamma (float, optional): The weight of gradient for the rotation.
        max_itr (int, optional): Maximum number of iterations for convergence.
        tol (float, optional): Tolerance for convergence.

    Returns:
        ndarray: Rotation matrix.
    """

    # Get the number of variables (p) and factors (k)
    p, k = Phi.shape

    # Initialize the rotation matrix to the identity matrix
    R = np.eye(k)

    # Initialize the cumulative explained variance
    d = 0

    # Iterate up to max_itr times
    for i in range(max_itr):
        # Store the old explained variance
        d_old = d

        # Compute the rotated factor loadings
        Lambda = np.dot(Phi, R)

        # Compute the matrix for singular value decomposition (SVD)
        # The matrix is the difference between the cubic transformation of the loadings and
        # a normalization term based on gamma
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))
            )
        )

        # Compute the rotation matrix from the SVD
        R = np.dot(u, vh)

        # Update the cumulative explained variance
        d = np.sum(s)

        # Check for convergence (if the change in explained variance is less than the tolerance)
        if d_old != 0 and d / d_old < 1 + tol:
            # If converged, exit the loop
            break

    # Return the rotation matrix
    return R
