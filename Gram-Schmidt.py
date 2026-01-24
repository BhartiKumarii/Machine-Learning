import numpy as np

def gram_schmidt(A):
    """
    This function performs the Gram-Schmidt process on a given matrix A.

    Parameters:
    A : numpy.ndarray
        A matrix whose columns are vectors to be orthonormalized.

    Returns:
    Q : numpy.ndarray
        A matrix with orthonormal columns.
    """

    # Convert input to NumPy array of float type
    A = np.array(A, dtype=float)

    # Get number of rows (m) and columns (n)
    m, n = A.shape

    # Initialize matrix Q with zeros
    Q = np.zeros((m, n))

    # Loop over each column vector in A
    for j in range(n):

        # Take the j-th column of A
        v = A[:, j]

        # Subtract projections onto previously computed orthonormal vectors
        for i in range(j):

            # Compute projection coefficient
            projection = np.dot(Q[:, i], A[:, j])

            # Subtract the projection
            v = v - projection * Q[:, i]

        # Compute the norm (length) of the vector
        norm_v = np.linalg.norm(v)

        # Check for zero norm to avoid division by zero
        if norm_v == 0:
            # If vector is zero, keep it as zero
            Q[:, j] = v
        else:
            # Normalize the vector
            Q[:, j] = v / norm_v

    # Return the orthonormal matrix
    return Q
