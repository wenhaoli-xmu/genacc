import numpy as np

def nystrom_approximation(K, m):
    """
    Perform Nystrom approximation on a given kernel matrix K.

    Parameters:
    - K: The original kernel matrix (n x n)
    - m: The number of columns to sample for approximation

    Returns:
    - K_approx: The approximated kernel matrix
    """

    n = K.shape[0]
    
    # Step 1: Randomly select m indices from the kernel matrix
    indices = np.random.choice(n, m, replace=False)
    
    # Step 2: Construct C by selecting the columns indexed by indices
    C = K[:, indices]
    
    # Step 3: Construct W by selecting the rows and columns indexed by indices
    W = C[indices, :]
    
    # Step 4: Perform eigen decomposition on W
    eigvals, eigvecs = np.linalg.eigh(W)
    
    # Step 5: Compute the pseudo-inverse of the diagonal matrix of eigenvalues
    eigvals_inv = np.diag(1.0 / eigvals)
    
    # Step 6: Calculate W+ (pseudo-inverse of W)
    W_plus = eigvecs @ eigvals_inv @ eigvecs.T
    
    # Step 7: Compute the Nystrom approximation
    K_approx = C @ W_plus @ C.T
    
    return K_approx

# Example usage:
# Assume we have an original kernel matrix K
np.random.seed(42)  # For reproducibility
K = np.random.rand(100, 100)
K = (K + K.T) / 2  # Make it symmetric

# Approximate the kernel matrix using Nystrom method
K_approx = nystrom_approximation(K, 20)

print("Original Kernel Matrix Shape:", K.shape)
print("Approximated Kernel Matrix Shape:", K_approx.shape)
