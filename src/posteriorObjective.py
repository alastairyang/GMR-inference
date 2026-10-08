class posteriorObjective:

    """
    Bridging enthalpyPosterior and scipy.optimize
    """

    def __init__(self, posterior, gp_dimension, gmm_dimension):
        self.posterior = posterior
        self.gp_dimension = gp_dimension
        self.gmm_dimension = gmm_dimension

    def unpack(self, theta):
        expected = self.gp_dimension + self.gmm_dimension

        if theta.size != expected:
            raise ValueError(f"Expected theta of size {expected}, got {theta.size}")

        z_gp = theta[:self.gp_dimension]
        z_gmm = theta[self.gp_dimension:]
        return z_gp, z_gmm

    def value(self, theta):
        z_gp, z_gmm = self.unpack(theta)

        value = self.posterior.log_prob(z_gp, z_gmm)
        return -float(value.detach().cpu())

    def gradient(self, theta):
        z_gp, z_gmm = self.unpack(theta)

        return -self.posterior.packed_gradient(z_gp, z_gmm)

    def value_and_gradient(self, theta):
        return self.value(theta), self.gradient(theta)

    @staticmethod
    def finite_difference_gradient(objective, theta, epsilon=1e-6):
        import numpy as np
        
        theta = np.asarray(theta, dtype=float)
        gradient = np.zeros_like(theta)

        for i in range(theta.size):
            theta_plus = theta.copy()
            theta_minus = theta.copy()

            theta_plus[i] += epsilon
            theta_minus[i] -= epsilon

            gradient[i] = (
                objective(theta_plus)
                - objective(theta_minus)
            ) / (2.0 * epsilon)

        return gradient
