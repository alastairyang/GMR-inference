import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

def grid_laplacian(mask, hx=1.0, hy=1.0):
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape

    ids = np.arange(ny * nx).reshape(ny, nx)

    # Horizontal edges
    horizontal = mask[:, :-1] & mask[:, 1:]
    left = ids[:, :-1][horizontal]
    right = ids[:, 1:][horizontal]

    # Vertical edges
    vertical = mask[:-1, :] & mask[1:, :]
    lower = ids[:-1, :][vertical]
    upper = ids[1:, :][vertical]

    rows = np.concatenate([
        left, right,
        lower, upper
    ])

    cols = np.concatenate([
        right, left,
        upper, lower
    ])

    weights = np.concatenate([
        np.full(left.size, 1.0 / hx**2),
        np.full(right.size, 1.0 / hx**2),
        np.full(lower.size, 1.0 / hy**2),
        np.full(upper.size, 1.0 / hy**2),
    ])

    A_full = sparse.coo_matrix(
        (weights, (rows, cols)),
        shape=(nx * ny, nx * ny)
    ).tocsr()

    active = mask.ravel()
    full_ids = np.arange(nx * ny)

    A = A_full[active][:, active]

    degree = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(degree) - A

    return L, np.flatnonzero(active), full_ids

def eigen_laplacian(mask, k=50, hx=1.0, hy=1.0):
    L, active_ids, full_ids = grid_laplacian(mask, hx=hx, hy=hy)

    # Smallest eigenvalues/eigenvectors of a symmetric Laplacian
    eigenvalues, eigenvectors = spla.eigsh(
        L,
        k=k,
        which="SM",
        tol=1e-6,
        maxiter=10000
    )

    return eigenvalues, eigenvectors, active_ids, full_ids

def sample_graph_matern(
    eigenvalues,
    eigenvectors,
    sigma=1.0,
    nu=1.0,
    rho=10.0,
    dimension=2,
    n_samples=1,
    remove_constant=False,
    rng=None,
):
    """
    Generate truncated graph-Matern samples using Laplacian eigenpairs.

    Parameters
    ----------
    eigenvalues : (k,) ndarray
        Laplacian eigenvalues.
    eigenvectors : (n_active, k) ndarray
        Corresponding orthonormal eigenvectors.
    sigma : float
        Target average marginal standard deviation.
    nu : float
        Matern smoothness.
    rho : float
        Approximate practical range, in units consistent with the Laplacian.
        With an unscaled grid Laplacian, this is approximately in grid cells.
    dimension : int
        Spatial dimension. Use 2 for a two-dimensional grid.
    n_samples : int
        Number of independent fields.
    remove_constant : bool
        If True, remove zero/near-zero Laplacian modes.
    rng : np.random.Generator or None
        Random-number generator.

    Returns
    -------
    samples : (n_active, n_samples) ndarray
        Random-field samples on active nodes.
    spectral_variance : (k_used,) ndarray
        Variance assigned to each retained Laplacian mode.
    retained : (k,) boolean ndarray
        Indicator for retained modes.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)

    # eigsh does not always return eigenpairs in sorted order
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Remove tiny negative values caused by numerical roundoff
    eigenvalues = np.maximum(eigenvalues, 0.0)

    retained = np.ones(eigenvalues.size, dtype=bool)

    if remove_constant:
        tolerance = 1e-10 * max(1.0, eigenvalues.max())
        retained = eigenvalues > tolerance

    lam = eigenvalues[retained]
    U = eigenvectors[:, retained]

    if lam.size == 0:
        raise ValueError("No Laplacian modes remain after filtering.")

    alpha = nu + dimension / 2.0
    kappa = np.sqrt(8.0 * nu) / rho

    # Unnormalized Matern spectral variances
    raw_variance = (kappa**2 + lam) ** (-alpha)

    # For orthonormal U:
    # trace(U diag(w) U.T) = sum(w).
    # Normalize so average marginal variance is sigma**2.
    n_active = U.shape[0]
    normalization = n_active / raw_variance.sum()

    spectral_variance = (
        sigma**2 * normalization * raw_variance
    )

    if rng is None:
        rng = np.random.default_rng()

    z = rng.standard_normal((lam.size, n_samples))

    samples = U @ (
        np.sqrt(spectral_variance)[:, None] * z
    )

    return samples, spectral_variance, retained