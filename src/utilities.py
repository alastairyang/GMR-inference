import numpy as np

# ---------- NOT ACTIVELY USED ----------
def build_distance_matrix(x, y):
    """
    Build a distance matrix

    Parameters:
    -------
    x: array, shape (n,)
        1D array of x coordinates
    y: array, shape (n,)
        1D array of y coordinates

    Returns:
    -------
    dist_mtx: array, shape (n, n)
        distance matrix where dist_mtx[i,j] is the distance between (x[i], y[i]) and (x[j], y[j])
    
    """
    n = len(x)
    dist_mtx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mtx[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    return dist_mtx


def spatial_density_weighting(coord):
    """
    Compute spatial and density weights for the evidence points based on their coordinates and pairwise distances.
    Originally intended to find a weighting scheme for the evidence poitns
    to reduce overcounting the closely spaced points

    Parameters:
    -------
    coord: list of arrays
        list of coordinate arrays (e.g., [x, y]) for the evidence points
    """
    from scipy.stats import gaussian_kde
    # Normalize both to [0, 1]
    def minmax(x):
        return (x - x.min()) / (x.max() - x.min())

    coords = np.vstack(coord)
    kde = gaussian_kde(coords)
    density = kde(coords)  # estimated density at each point

    density_weight_kde = 1.0 / (density + 1e-10)
    density_weight_kde = minmax(density_weight_kde)

    return density_weight_kde


def build_decorr_weight_matrix(X_vec, combined_idx):
    """   
    Build a decorrelation weight matrix based covariance (similarity measure)

    Parameters:
    -------
    X_vec: array, shape (n_samples, n_features)
        Input data vectors
    combined_idx: array, shape (n_samples,)
        Indices of the combined data points

    Returns:
    -------
    weight_mtx: array, shape (n_samples, n_samples)
        Decorrelation weight matrix
    sim_mtx: array, shape (n_samples, n_samples)
        Similarity matrix
    """
    def redundancy_weights(sim_mtx):
        sim_abs = np.abs(sim_mtx)
        sim_min = sim_abs.min()
        sim_max = sim_abs.max()
        sim_norm = (sim_abs - sim_min) / (sim_max - sim_min)
        return 1.0 - sim_norm

    n = combined_idx.shape[0]
    combined_idx_mtx = np.zeros((n, n))
    sim_mtx = np.zeros((n, n))

    for loc1, idx1 in enumerate(combined_idx):   # idx1 = actual spatial index
        data_idx1 = X_vec[idx1]                  # ← correct: index by VALUE
        remaining = combined_idx[loc1:]           # ← renamed to avoid collision

        combined_idx_mtx[loc1, loc1:loc1+len(remaining)] = remaining.flatten()


        for loc2, idx2 in enumerate(remaining):
            data_idx2 = X_vec[idx2]
            covariance = np.cov(data_idx1, data_idx2)[0, 1]
            sim_mtx[loc1, loc1+loc2] = covariance
            sim_mtx[loc1+loc2, loc1] = covariance

    weight_mtx = redundancy_weights(sim_mtx)
    return weight_mtx, sim_mtx