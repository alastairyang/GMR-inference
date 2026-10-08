import numpy as np
import torch
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

class lowRankGP:
    """
    A class representing a low-rank Gaussian Process (GP) model
    built on graph laplacian. The basis are formed in the null space
    of the simulation PCA basis.
    """

    def __init__(
        self,
        domain_mask,
        hx,
        hy,
        n_mode,
        pca_basis, 
    ):
        self.domain_mask = domain_mask
        self.hx = hx
        self.hy = hy
        self.n_mode = n_mode
        self.pca_basis = pca_basis 

        # initialize (saving the eigenpairs)
        self.discrepancy_basis(
            hx = self.hx,
            hy = self.hy,
            n_mode = self.n_mode,
            V = self.pca_basis
        )
    
    def grid_laplacian(self, hx=1.0, hy=1.0):
        mask = np.asarray(self.domain_mask, dtype=bool)
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

    def eigen_laplacian(self, k=50, hx=1.0, hy=1.0):
        """
        Compute the smallest k eigenvalues and 
        corresponding eigenvectors of the grid Laplacian.
        """

        L, active_ids, full_ids = self.grid_laplacian(hx=hx, hy=hy)

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
        self,
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
        raw_variance : (k_used,) ndarray
            Unnormalized Matern spectral variances for each retained mode.
        """
        eigenvalues = np.asarray(self.eigenvalues, dtype=float)
        eigenvectors = np.asarray(self.eigenvectors, dtype=float)

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

        return samples, spectral_variance, retained, raw_variance

    @staticmethod
    def _project_away(F, V):
        """
        project F away from the subspace spanned by the columns of V.

        Parameters
        ----------
        F : (n, m) ndarray
            Matrix whose columns are to be projected.
        V : (n, k) ndarray
            Matrix whose columns span the subspace to project away from.

        Returns
        -------
        F_projected : (n, m) ndarray
            The columns of F after projection.
        """
        return F - V @ (V.T @ F)

    def discrepancy_basis(self, hx, hy, n_mode, V, plotting=False):
        """
        Create basis function built on Matern kernel Gaussian Process and
        the basis are orthogonal to the columns of V (i.e. PCA basis from simulation)

        Parameters
        ----------
        hx : float
            Grid spacing in the x direction.
        hy : float
            Grid spacing in the y direction.
        n_mode : int
            Number of discrepancy basis functions to generate.
        V : (n, k) ndarray
            Matrix whose columns span the subspace to project away from.

        Returns
        -------
        U_F : (n, n_mode) ndarray
            Orthonormal basis for the discrepancy subspace.
        S_F : (n_mode,) ndarray
            Singular values corresponding to the basis vectors.
        VT_F : (n_mode, m) ndarray
            Right singular vectors of the projected basis.

        """
        print("Step 1: Eigen-decomposition on graph Laplacian...")
        eigenval, eigenvec, active_ids, full_ids = self.eigen_laplacian(
            k = n_mode,
            hx = hx, 
            hy = hy
        )
        print("...Done.")

        nx = self.domain_mask.shape[0]
        ny = self.domain_mask.shape[1]

        print("Step 2: Get mode covariance under Matern kernel...")
        samples_active, mode_variance, retained, raw_variance = self.sample_graph_matern(
            eigenvalues=eigenval,
            eigenvectors=eigenvec,
            sigma=1.0,
            nu=0.1,
            rho=1.0,          # approximately 10 cells with your current L
            n_samples=5,
            remove_constant=False,
            rng=np.random.default_rng(42),
        )
        print("...Done.")

        print("Step 3: Projecting away and performing SVD...")
        # construct the covariance-scaled graph basis
        eigenvec_full = np.zeros((nx * ny, eigenvec.shape[1]))
        eigenvec_full[active_ids, :] = eigenvec
        eigenvec_full = eigenvec_full.reshape(ny, nx, eigenvec.shape[1])
        eigenvec_flat = eigenvec_full.reshape(ny * nx, eigenvec_full.shape[2])
        F = eigenvec_flat @ np.diag(np.sqrt(mode_variance))

        # project F away from the PCA subspace spanned by V
        F_normal = F - V @ (V.T @ F)

        # do SVD on F_normal: coordinate transformation
        U_F, S_F, VT_F = np.linalg.svd(F_normal, full_matrices=False)
        print("...Done.")

        self.left_singular_vectors  = U_F
        self.singular_values        = S_F
        self.right_singular_vectors = VT_F

        return U_F, S_F, VT_F

    def plot_discrepancy_modes(self):
        import matplotlib.pyplot as plt

        # plot every 10 eigenvectors
        plt.figure(figsize=(25, 3))
        for i in range(0, self.left_singular_vectors.shape[1], 10):
            plt.subplot(1, self.left_singular_vectors.shape[1] // 10, i // 10 + 1)
            plt.imshow(self.left_singular_vectors[:, i].reshape(self.domain_mask.shape[1], self.domain_mask.shape[0]), 
                    cmap='RdBu_r',
                    vmin=-0.03, vmax=0.03)
            plt.title(f"Mode {i}")
            plt.gca().invert_yaxis()
        plt.show()

        return


    def generate_random_field(self, sigma, U, S, n):
        """
        Generate random fields from eigenvectors and singular values

        Parameters
        ----------
        sigma : float
            Amplitude pre-factor.
        U : (n, n_mode) ndarray
            Orthonormal basis for the discrepancy subspace.
        S : (n_mode,) ndarray
            Singular values corresponding to the basis vectors.
        n : int
            Number of random fields to generate.

        Returns
        -------
        fields : (n, n_mode) ndarray
            Generated random fields.

        """
        # make input torch tensor, if not already
        if not torch.is_tensor(U):
            U = torch.tensor(U, dtype=torch.float32)
        if not torch.is_tensor(S):
            S = torch.tensor(S, dtype=torch.float32)
        return sigma * (U @ (torch.diag(S) @ torch.randn(S.shape[0], n)))
