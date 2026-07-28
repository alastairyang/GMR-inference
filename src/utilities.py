import numpy as np
import os
import scipy
import xarray as xr
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def shape_check(*arrays):
    """  
    Check that all input arrays have the same shape.

    Parameters:
    -------
    *arrays: list of arrays
        list of arrays to check

    Raises:
    -------
    ValueError: if any two arrays have different shapes
    """
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        raise ValueError("All input arrays must have the same shape, but got shapes: {}".format(shapes))
    
def shape_check(X, mean, std):
    """
    Checks that mean and std are compatible with X for broadcasting.
    Allows:
      - X shape (n_features,)     with mean/std shape (n_features,)
      - X shape (N, n_features)   with mean/std shape (n_features,)
      - X shape (N, n_features)   with mean/std shape (N, n_features)
    """
    if mean.shape != std.shape:
        raise ValueError(
            f"mean and std must have the same shape. "
            f"Got mean={mean.shape}, std={std.shape}"
        )
    # Allow exact match or broadcasting along first dimension
    if X.shape != mean.shape and X.shape[-1] != mean.shape[-1]:
        raise ValueError(
            f"Shape mismatch: X={X.shape} is not compatible with "
            f"mean/std={mean.shape}. Expected X.shape[-1] == mean.shape[-1]."
        )


def reverse_standardize(X, mean, std, method='standard', epsilon=None):
    """
    Reverse z-score standardization.
    Vectorized across the first dimension of X.

    Parameters
    ----------
    X : array, shape (n_features,) or (N, n_features)
    mean : array, shape (n_features,)
    std  : array, shape (n_features,)
    method : 'standard' or 'relaxation'
    epsilon : float, required if method='relaxation'
    """
    shape_check(X, mean, std)

    if method == 'standard':
        return X * std + mean

    elif method == 'relaxation':
        if epsilon is None:
            raise ValueError("Epsilon must be provided for relaxation method")
        return X * (std + epsilon) + mean

    else:
        raise ValueError(f"Unknown method: {method}")


def standardize(X, mean, std, method='standard', epsilon=None):
    """
    Z-score standardization.
    Vectorized across the first dimension of X.

    Parameters
    ----------
    X : array, shape (n_features,) or (N, n_features)
    mean : array, shape (n_features,)
    std  : array, shape (n_features,)
    method : 'standard' or 'relaxation'
    epsilon : float, required if method='relaxation'
    """
    shape_check(X, mean, std)

    if method == 'standard':
        return (X - mean) / std

    elif method == 'relaxation':
        if epsilon is None:
            raise ValueError("Epsilon must be provided for relaxation method")
        return (X - mean) / (std + epsilon)

    else:
        raise ValueError(f"Unknown method: {method}")

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

def low_high_joint_percentile(samples, log_probs, low_percentile=5, high_percentile=95):
    """
    Compute the low and high percentiles of the samples based on their log probabilities.

    Parameters:
    -------
    samples: array, shape (n_samples, n_features)
        MCMC samples
    log_probs: array, shape (n_samples,)
        Log probabilities of the samples
    low_percentile: float
        Percentile for the lower bound (default: 5)
    high_percentile: float
        Percentile for the upper bound (default: 95)

    Returns:
    -------
    low_bound: array, shape (n_features,)
        Low percentile bound of the samples
    high_bound: array, shape (n_features,)
        High percentile bound of the samples
    """
    # Sort samples by log probability (ascending)
    sorted_indices = np.argsort(log_probs) 
    samples_sorted = samples[sorted_indices]
    log_probs_sorted = log_probs[sorted_indices]

    lp_low  = np.percentile(log_probs_sorted, low_percentile)
    lp_high = np.percentile(log_probs_sorted, high_percentile)

    # the single representative samples closest to each threshold
    idx_low  = np.argmin(np.abs(log_probs_sorted - lp_low))
    idx_high = np.argmin(np.abs(log_probs_sorted - lp_high))

    low_bound = samples_sorted[idx_low]  
    high_bound = samples_sorted[idx_high]  
    return low_bound, high_bound

def low_high_marginal_percentile(samples, low_percentile=5, high_percentile=95):
    """
    Compute marginal percentile bounds across all features independently.

    Parameters
    ----------
    samples : array, shape (n_samples, n_features)
        MCMC samples drawn from the posterior.
    low_percentile : float
        Lower percentile (default: 5) → lower bound of 90% credible interval.
    high_percentile : float
        Upper percentile (default: 95) → upper bound of 90% credible interval.

    Returns
    -------
    low_bound : array, shape (n_features,)
        Marginal low_percentile bound per feature.
    high_bound : array, shape (n_features,)
        Marginal high_percentile bound per feature.
    """
    low_bound  = np.percentile(samples, low_percentile,  axis=0)
    high_bound = np.percentile(samples, high_percentile, axis=0)
    return low_bound, high_bound


def read_exp(filepath):
    """
    Parse an Elmer/Ice .exp file.
    Handles multi-segment files (segments separated by NaN NaN rows).
    Returns x, y arrays for the first (or only) segment, with NaNs stripped.
    """
    xs, ys = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # skip header/comment lines
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x_val, y_val = float(parts[0]), float(parts[1])
                xs.append(x_val)
                ys.append(y_val)
            except ValueError:
                continue

    xs = np.array(xs)
    ys = np.array(ys)

    # strip NaN rows (segment separators)
    valid = ~(np.isnan(xs) | np.isnan(ys))
    return xs[valid], ys[valid]


def cluster_points(x_coords, y_coords, n, random_state=42, show_plot=False):
    """
    Clusters 2D points into n groups based on spatial proximity.

    Args:
        x_coords   : list or array of x coordinates
        y_coords   : list or array of y coordinates
        n          : number of clusters (groups)
        random_state: seed for reproducibility

    Returns:
        labels     : array of cluster IDs (0 to n-1) for each point
        centroids  : array of shape (n, 2) — the center of each cluster
        groups     : dict mapping cluster_id -> list of (x, y) points
    """
    # Stack into (num_points, 2) array
    points = np.column_stack((x_coords, y_coords))

    # Fit K-Means
    kmeans = KMeans(n_clusters=n, random_state=random_state, n_init="auto")
    kmeans.fit(points)

    labels    = kmeans.labels_          # cluster ID per point
    centroids = kmeans.cluster_centers_ # (n, 2) centroid positions

    # Build a readable dict: {cluster_id: [(x1,y1), (x2,y2), ...]}
    groups = {i: [] for i in range(n)}
    for point, label in zip(points, labels):
        groups[label].append(tuple(point))

    if show_plot:
        def plot_clusters(x_coords, y_coords, labels, centroids):
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(x_coords, y_coords, c=labels, cmap="tab10", s=60, zorder=2)
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        c="black", marker="X", s=200, label="Centroids", zorder=3)
            plt.colorbar(scatter, label="Cluster ID")
            plt.legend()
            plt.title("Point Clusters")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.tight_layout()
            plt.show()
        
        plot_clusters(x_coords, y_coords, labels, centroids)


    return labels, centroids, groups


def build_masks_from_points(point_list, idx_thawed, idx_frozen,
                             flat_x, flat_y, grid_shape):
    """
    Given a list of (x, y) tuples (from cluster_groups),
    return thawed_mask and frozen_mask of shape grid_shape.

    Strategy: match each (x,y) back to its flat grid index,
    then check whether that index lives in idx_thawed or idx_frozen.
    """
    # Build a lookup: (x, y) -> flat grid index for thawed and frozen points
    thawed_lookup = {(flat_x[i], flat_y[i]): i for i in idx_thawed}
    frozen_lookup = {(flat_x[i], flat_y[i]): i for i in idx_frozen}

    thawed_mask_flat = np.zeros(len(flat_x), dtype=bool)
    frozen_mask_flat = np.zeros(len(flat_x), dtype=bool)

    for (px, py) in point_list:
        if (px, py) in thawed_lookup:
            thawed_mask_flat[thawed_lookup[(px, py)]] = True
        elif (px, py) in frozen_lookup:
            frozen_mask_flat[frozen_lookup[(px, py)]] = True

    return (thawed_mask_flat.reshape(grid_shape),
            frozen_mask_flat.reshape(grid_shape))


def load_ase_datasets(verbose=True):
    """
    Load and regrid all standard Amundsen Sea Embayment (ASE) datasets onto the ASE coordinate grid
    (defined by the basal temperature posterior_mean_Tb.tif).

    Datasets prepared
    -----------------
    Geometry   : ice thickness (H), bed topography (bed), surface topography (sdem)
    Thermal    : pressure-melting point (Tpmp), basal temperature (Tb)
    Kinematics : surface velocity magnitude + components (vel, vx, vy)
    Slope      : surface slope magnitude + components (slope_mag, slope_x, slope_y)

    Returns
    -------
    dict with keys:
        coords  : dict(X, Y, xs, ys)          – 2-D meshgrid + 1-D axes
        H       : ice thickness                (m)
        bed     : bed topography               (m)
        sdem    : surface topography           (m)
        Tb      : basal temperature            (K)
        Tpmp    : pressure-melting point       (K)
        dTpmp   : Tpmp - Tb                    (K)  positive = below melting
        vel     : velocity magnitude           (m/yr)
        vx      : x-component of velocity     (m/yr)
        vy      : y-component of velocity     (m/yr)
        slope_mag : surface slope magnitude   (m/m)
        slope_x   : slope in x-direction      (m/m)
        slope_y   : slope in y-direction      (m/m)
        smb     : surface mass balance (m/yr)
        mask    : boolean mask of the continuous ASE domain
    """

    # ─────────────────────────────────────────────────────────────────
    # 0.  Paths  (edit here if anything moves)
    # ─────────────────────────────────────────────────────────────────
    BASE   = '/home/donglaiyang/Documents/Georgia-Tech/Research'

    PATHS = dict(
        bedmap  = os.path.join(BASE, 'common-data-set/bed-topography/bedmap3AIS.nc'),
        vel_mag = os.path.join(BASE, 'common-data-set/velocity/antarctica_ice_velocity_450m_v2.nc'),
        H_mat   = os.path.join(BASE,
                    'thermal-model/Amundsen-thermal-output-Yang/'
                    'thermal-training-data/Thwaites-PIG/training/gridded/'
                    'H_gridded.mat'),
        coord   = os.path.join(BASE,
                    'thermal-model/Amundsen-thermal-output-Yang/'
                    'thermal-training-data/Thwaites-PIG/training/gridded/'
                    'trainingAll_image_coord.mat'),
        domain_mask = os.path.join(BASE,
                    'thermal-model/Amundsen-thermal-output-Yang/thermal-training-data/Thwaites-PIG/training/gridded/',
                    'training_mask_domain_continuous.mat'),
        smb    = os.path.join(BASE,
                              'thermal-model/data/smb/AIS_smb_avg_1980_2022.mat')

    )

    # ─────────────────────────────────────────────────────────────────
    # 1.  Helper utilities
    # ─────────────────────────────────────────────────────────────────
    def _world_grid(src):
        """1-D axes + 2-D meshgrid from an open rasterio dataset."""
        cols = np.arange(src.width)
        rows = np.arange(src.height)
        xs = np.array([src.xy(0, c)[0] for c in cols])
        ys = np.array([src.xy(r, 0)[1] for r in rows])
        X, Y = np.meshgrid(xs, ys)
        return X, Y, xs, ys

    def _make_interp(ys_src, xs_src, data):
        """
        Build a RegularGridInterpolator, flipping axes to ascending order
        if the source grid is descending (standard rasterio convention).
        """
        if ys_src[0] > ys_src[-1]:
            return RegularGridInterpolator(
                (ys_src[::-1], xs_src), data[::-1, :],
                method='linear', bounds_error=False, fill_value=np.nan)
        return RegularGridInterpolator(
            (ys_src, xs_src), data,
            method='linear', bounds_error=False, fill_value=np.nan)

    def _regrid(ys_src, xs_src, data, Y_tgt, X_tgt):
        """Interpolate data onto target (Y_tgt, X_tgt) meshgrid."""
        fn  = _make_interp(ys_src, xs_src, data)
        pts = np.column_stack([Y_tgt.ravel(), X_tgt.ravel()])
        return fn(pts).reshape(X_tgt.shape)

    def _log(msg):
        if verbose:
            print(f'  [load_ase_datasets] {msg}')

    # ─────────────────────────────────────────────────────────────────
    # 2.  reference coordinates
    # ─────────────────────────────────────────────────────────────────
    _log('Loading reference coordinates …')
    coord_data = sio.loadmat(PATHS['coord'])
    xs = coord_data['training_coord']['xs'][0][0].flatten()
    ys = coord_data['training_coord']['ys'][0][0].flatten()
    X, Y = np.meshgrid(xs, ys)

    out = dict(coords=dict(X=X, Y=Y, xs=xs, ys=ys))

    # ─────────────────────────────────────────────────────────────────
    # 3.  Bedmap3  →  H, bed, sdem  (crop + regrid)
    # ─────────────────────────────────────────────────────────────────
    _log('Loading Bedmap3 …')
    buffer = 50_000   # 50 km padding around ASE domain

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bm = xr.open_dataset(PATHS['bedmap'])
    bm_x = bm['x'].values
    bm_y = bm['y'].values

    mx = (bm_x >= x_min - buffer) & (bm_x <= x_max + buffer)
    my = (bm_y >= y_min - buffer) & (bm_y <= y_max + buffer)

    x_crop = bm_x[mx]
    y_crop = bm_y[my]

    def _bm_crop_regrid(var_name):
        arr = bm[var_name].values[np.ix_(my, mx)].astype(np.float64)
        return _regrid(y_crop, x_crop, arr, Y, X)

    out['sdem'] = _bm_crop_regrid('surface_topography')
    out['bed']  = _bm_crop_regrid('bed_topography')
    out['H']    = _bm_crop_regrid('ice_thickness')
    bm.close()

    # ─────────────────────────────────────────────────────────────────
    # 4.  Pressure-melting point
    # ─────────────────────────────────────────────────────────────────
    _log('Computing Tpmp …')
    BETA = 9.8e-8   # K Pa⁻¹  Clausius–Clapeyron
    RHO  = 917.0    # kg m⁻³  ice density
    G    = 9.81     # m s⁻²

    out['Tpmp']  = 273.15 - BETA * RHO * G * out['H']

    # ─────────────────────────────────────────────────────────────────
    # 5.  Surface slope  (from sdem on the ASE grid)
    # ─────────────────────────────────────────────────────────────────
    _log('Computing surface slope …')
    # np.gradient respects non-uniform spacing when axes are supplied
    slope_y, slope_x = np.gradient(out['sdem'], ys, xs)
    out['slope_x']   = slope_x
    out['slope_y']   = slope_y
    out['slope_mag'] = np.sqrt(slope_x**2 + slope_y**2)

    # ─────────────────────────────────────────────────────────────────
    # 6.  Ice velocity  (MEaSUREs / ITSLIVE NetCDF)
    # ─────────────────────────────────────────────────────────────────
    _log('Loading velocity …')
    vel_ds  = xr.open_dataset(PATHS['vel_mag'])

    # ── normalise variable names across products ──────────────────
    # MEaSUREs Phase Map uses 'VX'/'VY'; ITSLIVE uses 'vx'/'vy'
    vx_key = 'VX' if 'VX' in vel_ds else 'vx'
    vy_key = 'VY' if 'VY' in vel_ds else 'vy'
    x_key  = 'x'  if 'x'  in vel_ds.coords else 'X'
    y_key  = 'y'  if 'y'  in vel_ds.coords else 'Y'

    vel_x_raw = vel_ds[x_key].values
    vel_y_raw = vel_ds[y_key].values

    mvx = (vel_x_raw >= x_min - buffer) & (vel_x_raw <= x_max + buffer)
    mvy = (vel_y_raw >= y_min - buffer) & (vel_y_raw <= y_max + buffer)

    vx_crop = vel_ds[vx_key].values[np.ix_(mvy, mvx)].astype(np.float64)
    vy_crop = vel_ds[vy_key].values[np.ix_(mvy, mvx)].astype(np.float64)

    out['vx']  = _regrid(vel_y_raw[mvy], vel_x_raw[mvx], vx_crop, Y, X)
    out['vy']  = _regrid(vel_y_raw[mvy], vel_x_raw[mvx], vy_crop, Y, X)
    out['vel'] = np.sqrt(out['vx']**2 + out['vy']**2)
    vel_ds.close()

    # ─────────────────────────────────────────────────────────────────
    # 7.  domain mask
    # ─────────────────────────────────────────────────────────────────
    _log('Loading domain mask …')
    mask_data = sio.loadmat(PATHS['domain_mask'])
    mask = mask_data['in_domain_mask']
    out['mask'] = mask
    # no need to interpolate; domain mask was saved with coord

    # get the domain outline from domain outline
    ny, nx = mask.shape

    # Use contour at level 0.5 to find the boolean boundary
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Y, mask.astype(float), levels=[0.5])
    plt.close(fig_tmp)  # don't display the temp figure

    segments = []
    for path in cs.collections[0].get_paths():
        verts = path.vertices
        segments.append((verts[:, 0], verts[:, 1]))
            
    domain_bound_line = segments[0]
    # # flip y wrt to the midpoint of the domain extent to allow origin='upper' in imshow
    # extent = np.array([xs.min(), xs.max(), ys.max(), ys.min()])/1e3 # convert to km (left, right, bottom, top), imshow by default is inverted in y-axis

    # y_mid = (extent[2] + extent[3]) / 2 
    # corrected_y = 2 * y_mid - domain_bound_line[1]
    # domain_bound_line = (domain_bound_line[0], corrected_y)
    out['domain_bound_line'] = domain_bound_line

    # ────────────────────────────────────────────────────────────────
    # 8.  Surface mass balance (SMB)
    # ────────────────────────────────────────────────────────────────
    _log('Loading surface mass balance (SMB) …')

    smb_data = scipy.io.loadmat(PATHS['smb'])
    smb_val = smb_data['smbAvgXY']['data'][0][0]
    smb_x   = smb_data['smbAvgXY']['x'][0][0].flatten()
    smb_y   = smb_data['smbAvgXY']['y'][0][0].flatten()

    # interpolate
    out['smb'] = _regrid(smb_y, smb_x, smb_val, Y, X)

    _log('All datasets ready.')
    return out
