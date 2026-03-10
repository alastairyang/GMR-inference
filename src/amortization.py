import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.spatial import cKDTree
import scipy as sp
import scipy.io
from scipy.linalg import pinvh
import time

from gmr.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses
from gmr.mvn import regression_coefficients
from gmr.gmm import _safe_probability_density

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
    if mean is not None:
        mean = mean.flatten()
    else:
        mean = np.zeros(n)

    # Estimate Trace
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

    # Eigendecomposition
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
    
    # Generate Samples
    if verbose: print(f"Generating {num_samples} samples...")
        
    z = np.random.randn(rank, num_samples)
    sqrt_Lambda = np.sqrt(eigenvalues)

    # x = mu + V * sqrt(L) * z
    zero_mean_samples = eigenvectors @ (sqrt_Lambda[:, None] * z)
    samples = zero_mean_samples + mean[:, None]
    
    return samples, zero_mean_samples

def compute_conditional_expected_val(gmm, index, condition_val, sample_size=200):
    Y_n_samples = condition_val.shape[1]
    Y_n_dim = condition_val.shape[0]
    
    X_n_dim = gmm.means.shape[1] - Y_n_dim
    inner_samples_mean_sum = np.zeros(X_n_dim)  # Note: 1D array
    
    print(f"Propagating {Y_n_samples} samples")
    
    for i in range(Y_n_samples):
        pred_gmm = gmm.condition(index, condition_val[:,i])
        inner_samples = pred_gmm.sample(sample_size)  # Shape: (sample_size, X_n_dim)
        
        # Simply take the mean across samples
        inner_samples_mean_sum += np.mean(inner_samples, axis=0)
    
    total_mean = inner_samples_mean_sum / Y_n_samples
    return total_mean

def compute_conditional_std_val_latent(gmm, index, condition_val, mean_data, pca, sample_size=200):
    """  
    Compute the conditional standard deviation.
    PCA is necessary since variance/std is not a linear operation
    We need to do the std computation in the data space, not in the latent space

    Parameters:
    ----------
    gmm: GMM model object
    index: array-like, shape (n_new_features,)
        Indices of dimensions to condition on.
    condition_val: array, shape (n_new_features, n_samples)
        Values of the features to condition on. Each column is a sample.
    mean_data: array, shape (n_data_space_features,)
        Mean of the data in the original data space.
    pca: PCA object

    sample_size: int
        Number of samples to draw from the conditional distribution for each condition_val sample.
    """
    Y_n_samples = condition_val.shape[1]
    Y_n_dim = condition_val.shape[0]
    
    X_n_dim = gmm.means.shape[1] - Y_n_dim
    inner_samples_mean_sum = np.zeros(X_n_dim)  # Note: 1D array
    
    print(f"Propagating {Y_n_samples} samples")
    mean_data = mean_data.flatten()  # Ensure mean_data is 1D

    # second pass to get the variance
    inner_samples_var_sum = np.zeros(mean_data.shape[0])  # Note: 1D array
    for i in range(Y_n_samples):
        pred_gmm = gmm.condition(index, condition_val[:,i])
        inner_samples = pred_gmm.sample(sample_size)  # Shape: (sample_size, X_n_dim)
        
        inner_samples_ori = np.zeros((256*256, inner_samples.shape[0]))
        # Inverse transform to original space
        # then the uq samples
        for j, sample in enumerate(inner_samples):
            inner_samples_ori[:,j] = pca.inverse_transform(sample)
        # print('shape of inner_samples_ori:', inner_samples_ori.shape)
        # print('shape of mean_data:', mean_data.flatten().reshape(-1,1).shape)
        inner_samples_var_sum += np.sum((inner_samples_ori - mean_data.flatten().reshape(-1,1))**2, axis=1)
        # print every 50 samples
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{Y_n_samples} samples")
    total_var = inner_samples_var_sum / (Y_n_samples * sample_size)
    return np.sqrt(total_var)

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

    priors = _safe_probability_density(
        gmr_md.priors * marginal_norm_factors,
        marginal_prior_exponents[np.newaxis])[0]
    
    return GMM(n_components=gmr_md.n_components, priors=priors, means=means,
                covariances=covariances, random_state=gmr_md.random_state)

def to_log_probability_density(gmm, X):
    """
    Compute the log probability density for each sample in X.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data.
    
    Returns
    -------
    log_prob : array, shape (n_samples,)
        Log probability density for each sample.
    """
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape
    
    # Store log probabilities for each component and sample
    log_prob_components = np.zeros((n_samples, gmm.n_components))
    
    for k in range(gmm.n_components):
        mean = gmm.means[k]
        covariance = gmm.covariances[k]
        
        # Cholesky decomposition
        try:
            L = sp.linalg.cholesky(covariance, lower=True)
        except np.linalg.LinAlgError:
            L = sp.linalg.cholesky(
                covariance + 1e-3 * np.eye(n_features), lower=True)
        
        # Log normalization constant: log(1/sqrt((2π)^d * |Σ|))
        log_det_L = np.sum(np.log(np.diag(L)))  # log|L| = sum(log(L_ii))
        log_norm = -0.5 * n_features * np.log(2.0 * np.pi) - log_det_L
        
        # Mahalanobis distance
        X_minus_mean = X - mean
        X_normalized = sp.linalg.solve_triangular(
            L, X_minus_mean.T, lower=True).T
        log_exponent = -0.5 * np.sum(X_normalized ** 2, axis=1)
        
        # Log probability for component k: log(π_k) + log(N(x|μ_k, Σ_k))
        log_prob_components[:, k] = np.log(gmm.priors[k]) + log_norm + log_exponent
    
    # Log-sum-exp trick: log(Σ exp(x_i)) = c + log(Σ exp(x_i - c))
    c = np.max(log_prob_components, axis=1, keepdims=True)
    log_prob = c.squeeze() + np.log(np.sum(np.exp(log_prob_components - c), axis=1))
    
    return log_prob
