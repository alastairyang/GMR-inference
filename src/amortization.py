import numpy as np
from gmr.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.spatial import cKDTree
from gmr.mvn import regression_coefficients
from gmr.gmm import _safe_probability_density
import scipy.io
from scipy.linalg import pinvh
import time


def form_obs_cov_col(H, x, y, mask, colnum):
    """ 
    Forming the column of the covariance matrix for the observation Y
    This can be used later for computing the V^T * Cov * V term, specifically for computing V^T * Cov 

    Parameters
        H: 1xN array, ice thickness array. Here we assume the covariance is a function of ice thickness
        x: 1xN array, x coordinates of the locations
        y: 1xN array, y coordinates of the locations
        mask: 1xN array, mask array indicating present radar observation 
        colnum: column number (i.e. location number)

    Return:
        cov_col: 1D array, the column of the covariance matrix for the observation Y

    """

    # if H and mask not flatten, flatten them
    if len(H.shape) > 1:
        H = H.flatten()
    if len(mask.shape) > 1:
        mask = mask.flatten()

    # characteristic length scale of correlation
    length_scale = 100e3 

    # standardize H 
    H_standardized = (H - np.mean(H)) / np.std(H)

    if not mask[colnum]:
        # no data present: return all zero
        cov_col = np.transpose(np.zeros_like(H))
    else:
        # first compute the distance between the location and every other location
        dist = np.sqrt((x - x[colnum])**2 + (y - y[colnum])**2)
        # covariance is an exponential decay function of inter-point distance weighted by ice thickness
        # where high ice thickness means lower covariance
        cov_col = np.exp(-dist / length_scale) * (1 - H_standardized / np.max(H_standardized))
        cov_col[~mask] = 0.0

    return cov_col


def build_spatial_covariance_operator(H, x, y, mask, length_scale=200e3, cutoff_factor=5.0, verbose=True):
    """
    Constructs a LinearOperator representing the spatial covariance matrix.
    
    Returns:
        Sigma_op: LinearOperator (n, n)
        n: integer, dimension of the problem
    """
    # Flatten inputs
    H = H.flatten()
    x = x.flatten()
    y = y.flatten()
    mask = mask.flatten().astype(bool)
    n = len(H)
    
    cutoff_dist = cutoff_factor * length_scale
    
    if verbose:
        print(f"Building Operator | Size: n = {n:,}")
        print(f"Valid points: {mask.sum():,} | Length scale: {length_scale/1e3:.1f} km")
    
    # Precompute ice thickness weights
    # Note: Standardizing ensures weights are relative
    if H.std() == 0:
        H_std = np.zeros_like(H)
    else:
        H_std = (H - H.mean()) / H.std()
        
    H_weight = 1 - H_std / (H_std.max() + 1e-8) # Avoid div by zero
    H_weight[~mask] = 0.0
    
    # Build KD-tree for masked points
    if verbose: print("Building spatial index (KD-tree)...")
    
    mask_indices = np.where(mask)[0]
    masked_coords = np.column_stack([x[mask], y[mask]])
    tree = cKDTree(masked_coords)
    
    # Precompute neighbor lists
    if verbose: print("Precomputing neighbor lists...")
    
    neighbors_list = []
    neighbor_weights_list = []
    
    # Loop only over valid pixels to save memory/time
    for idx, j in enumerate(mask_indices):
        # Query neighbors within cutoff distance
        neighbor_indices = tree.query_ball_point([x[j], y[j]], r=cutoff_dist)
        
        # Convert to global indices
        neighbors_global = mask_indices[neighbor_indices]
        
        # Compute distances
        dists = np.sqrt((x[neighbors_global] - x[j])**2 + (y[neighbors_global] - y[j])**2)
        
        # --- CRITICAL COVARIANCE DEFINITION ---
        # Symetric weighting: w_i * w_j * exp(-dist)
        # This ensures the operator is symmetric and positive semi-definite
        cov_weights = np.exp(-dists / length_scale) * H_weight[neighbors_global] * H_weight[j]
        
        neighbors_list.append(neighbors_global)
        neighbor_weights_list.append(cov_weights)
    
    if verbose:
        avg_neighbors = np.mean([len(n) for n in neighbors_list])
        print(f"Avg neighbors: {avg_neighbors:.0f} | Sparsity: {100 * avg_neighbors / n:.2f}%")

    # Define matvec function using precomputed neighbors
    def matvec(v):
        result = np.zeros(n)
        
        # We only iterate over valid pixels 'j', but 'v' is size n
        for idx, j in enumerate(mask_indices):
            v_j = v[j]
            
            if abs(v_j) < 1e-15: continue
            
            neighbors_j = neighbors_list[idx]
            weights_j = neighbor_weights_list[idx]
            
            # Scatter add: result[neighbors] += weight * v[j]
            result[neighbors_j] += weights_j * v_j
        
        return result
    
    # Create LinearOperator
    Sigma_op = LinearOperator((n, n), matvec=matvec, dtype=float)
    return Sigma_op, H_weight

def sample_from_spatial_cov(H, x, y, mask, mean=None, sigma_op=None, 
                            num_samples=1, rank=100, length_scale=200e3, 
                            cutoff_factor=5.0, tol=1e-3, verbose=True):
    """
    Sample from N(mean, Sigma) using low-rank Lanczos approximation.
    """
    # 1. Build or Retrieve Operator
    if sigma_op is None:
        sigma_op, H_weight = build_spatial_covariance_operator(
            H, x, y, mask, length_scale, cutoff_factor, verbose
        )
    else:
        # We still need H_weight for the starting vector v0 in eigsh
        # Re-calculating just the weight vector is cheap
        H_flat = H.flatten()
        H_std = (H_flat - H_flat.mean()) / H_flat.std()
        H_weight = 1 - H_std / H_std.max()
        H_weight[~mask.flatten().astype(bool)] = 0.0

    n = sigma_op.shape[0]
    
    # Handle mean vector
    if mean is not None:
        mean = mean.flatten()
    else:
        mean = np.zeros(n)

    # 2. Estimate Trace (Optional, for stats)
    if verbose:
        print("\nEstimating total variance (trace)...")
        num_probes = 20
        trace_estimates = []
        for _ in range(num_probes):
            z = np.random.randn(n)
            z = z / np.linalg.norm(z)
            trace_estimates.append(z @ sigma_op.matvec(z) * n)
        total_trace = np.mean(trace_estimates)
        print(f"Estimated total variance: {total_trace:.2e}")

    # 3. Eigendecomposition (Lanczos)
    print(f"\nComputing top {rank} eigenvectors...")
    start_time = time.time()
    
    eigenvalues, eigenvectors = eigsh(
        sigma_op, 
        k=rank, 
        which='LM',
        tol=tol,
        maxiter=1000,
        v0=H_weight / np.linalg.norm(H_weight) # Use H_weight as hint
    )

    if verbose:
        print(f"Eigendecomposition done in {time.time() - start_time:.1f}s")
    
    # Ensure positive eigenvalues (numerical noise floor)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # 4. Generate Samples
    if verbose: print(f"Generating {num_samples} samples...")
        
    z = np.random.randn(rank, num_samples)
    sqrt_Lambda = np.sqrt(eigenvalues)

    # x = mu + V * sqrt(L) * z
    zero_mean_samples = eigenvectors @ (sqrt_Lambda[:, None] * z)
    samples = zero_mean_samples + mean[:, None]
    
    # 5. Compute Log Probs
    logprobs = compute_logprob(samples, mean, eigenvalues, eigenvectors, verbose=verbose)
    
    return samples, zero_mean_samples, logprobs

def compute_logprob(samples, mean, eigenvalues, eigenvectors, verbose=False):
    """
    Compute log probability for samples from low-rank approximation of N(mean, Sigma)
    
    Parameters:
        samples: (n, num_samples) array of samples
        mean: (n,) mean vector
        eigenvalues: (k,) top k eigenvalues
        eigenvectors: (n, k) top k eigenvectors
        verbose: print progress
    
    Returns:
        logprobs: (num_samples,) log probabilities
    """
    n, num_samples = samples.shape
    k = len(eigenvalues)
    
    # Center the samples
    centered = samples - mean[:, None]  # (n, num_samples)
    
    # Project onto eigenvector subspace: Q^T (x - mu)
    # Shape: (k, num_samples)
    projections = eigenvectors.T @ centered
    
    # Compute Mahalanobis distance in subspace: sum_i (q_i^T(x-mu))^2 / lambda_i
    # Shape: (num_samples,)
    mahalanobis = np.sum(projections**2 / eigenvalues[:, None], axis=0)
    
    # Log determinant: log|Sigma| ≈ sum log(lambda_i)
    log_det = np.sum(np.log(eigenvalues))
    
    # Normalization constant for k-dimensional Gaussian
    log_norm = -0.5 * k * np.log(2 * np.pi)
    
    # Log probability
    logprobs = log_norm - 0.5 * log_det - 0.5 * mahalanobis
    
    return logprobs



def pushforward(md, mean, covariance, mean_p, covariance_p, i_in, i_out):
    """    
    Pushforward a gaussian distribution through a conditional distribution:
    i.e.: 
    \int p(x|y) p(z) dz
    where p(z) ~ N(mean_mi, covariance_mi)
          p(x|y) ~ N(mean_ma, covariance_ma)
    
    Theory:
    if p(x|y) = N(y|Ay + b, cov_ma)
       p(z)   = N(z|mu, cov_mi)
    therefore:
    \int p(x|y) p(z) dz = N(x| A*mu + b, cov_ma + A@cov_mi@A.T)

    Parameters
    ----------
    md : GMM or MVN model object
        This is to get the random state seed to ensure consistency.

    mean : array, shape (n_features,)
        Mean of MVN

    covariance : array, shape (n_features, n_features)
        Covariance of MVN

    mean_p: array, shape (n_features_in,)
        Mean of the MVN to be pushed
    
    covariance_p: array, shape (n_features_in, n_features_in)
        Covariance of the MVN to be pushed

    i_out : array, shape (n_features_out,)
        Output feature indices

    i_in : array, shape (n_features_in,)
        Input feature indices

    Returns
    -------
    MVN model object
    """
    cov_12 = covariance[np.ix_(i_out, i_in)]
    cov_11 = covariance[np.ix_(i_out, i_out)]
    regression_coeffs = regression_coefficients(
        covariance, i_out, i_in, cov_12=cov_12)
    
    # print("i_out:", i_out)
    # print("i_in:", i_in)

    # print("size of mean[i_out]:", mean[i_out].shape)
    # print("size of mean[i_in]:", mean[i_in].reshape(-1,1).shape)
    # print("size of X:", X.shape)
    # print("size of mean_p:", mean_p.shape)
    # print("size of regression_coeffs:", regression_coeffs.shape)
    mean_target = mean[i_out] + regression_coeffs.dot(mean_p.squeeze() - mean[i_in])
    # print("size of mean_target:", mean_target.shape)
    covariance = cov_11 - regression_coeffs.dot(cov_12.T)

    covariance_target = covariance + regression_coeffs.dot(covariance_p).dot(regression_coeffs.T)
    return MVN(mean=mean_target, covariance=covariance_target,
               random_state=md.random_state)

def propagate_uncertainty(mean_p, covariance_p, gmr_md, indices, X):
    """  
    Propagate the observational uncertainty P(y_obs) through the learned distribution of P(x|y) marginalized from P(x,y)
    using the pushforward operation
    i.e.: 
    \int p(x|y) p(y_obs) dy_obs = \int p(x,y)/p(y) p(y_obs) dy_obs
    where p(y_obs) ~ N(mean_mi, covariance_mi)
          p(x|y) ~ N(mean_ma, covariance_ma)


    Notation: ma stands for major, mi stands for minor. However this is just to distinguish the two distributions.

    Parameters
    ----------
    mean_p: array, shape (n_features_in,)
        Mean of the MVN to be pushed (observational uncertainty)
    covariance_p: array, shape (n_features_in, n_features_in)
        Covariance of the MVN to be pushed (observational uncertainty)
    gmr_md: GMM model object
        The Gaussian mixture model which learned P(x,y)
    indices: array-like, shape (n_new_features,)
        Indices of dimensions to condition on.
    X : array, shape (n_samples, n_features_in)
        Inputs to the major MVN

    Returns
    -----------
    GMM model class
    """
    print("conditioned indices are: ", indices)

    indices = np.asarray(indices, dtype=int)
    X = np.asarray(X)

    n_features = gmr_md.means.shape[1] - len(indices)
    means = np.empty((gmr_md.n_components, n_features))
    covariances = np.empty((gmr_md.n_components, n_features, n_features))

    marginal_norm_factors = np.empty(gmr_md.n_components)
    marginal_prior_exponents = np.empty(gmr_md.n_components)

    # iterate through each Gaussian component
    y_indices = indices
    x_indices = np.setdiff1d(np.arange(gmr_md.means.shape[1]), indices)
    print("x indices are: ", x_indices)
    for k in range(gmr_md.n_components):
        mvn = MVN(mean=gmr_md.means[k], covariance=gmr_md.covariances[k],
                    random_state=gmr_md.random_state)
        # ---
        pushed = pushforward(mvn, mvn.mean, mvn.covariance, mean_p, covariance_p,
                             y_indices, x_indices)
        means[k] = pushed.mean
        covariances[k] = pushed.covariance

        marginal_norm_factors[k], exponents = \
            mvn.marginalize(y_indices).to_norm_factor_and_exponents(mean_p.reshape(1, -1))
        marginal_prior_exponents[k] = exponents[0]

    # priors = gmr_md.priors # unchanged during pushforward operation

    priors = _safe_probability_density(
        gmr_md.priors * marginal_norm_factors,
        marginal_prior_exponents[np.newaxis])[0]
    
    return GMM(n_components=gmr_md.n_components, priors=priors, means=means,
                covariances=covariances, random_state=gmr_md.random_state)
