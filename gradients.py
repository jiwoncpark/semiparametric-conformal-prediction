import numpy as np


def reflect_perturbations(x):
    """
    Reflect the values of x to keep them within [0, 1] bounds.
    """
    x_reflected = np.where(x > 1, 2 - x, x)
    x_reflected = np.where(x_reflected < 0, -x_reflected, x_reflected)
    return x_reflected

def get_stochastic_gradient(F, x0, sigma=1e-3, num_samples=100):
    """
    Vectorized version of estimating the gradient of a noisy function F at point x0 using Gaussian smoothing,
    with the perturbed values reflected to the [0, 1] bound.

    Parameters:
    - F: The noisy scalar function F(x).
    - x0: A numpy array representing the point at which to compute the gradient.
    - sigma: The standard deviation for the Gaussian perturbations.
    - num_samples: The number of random perturbations to use for gradient estimation.

    Returns:
    - grad: A numpy array representing the estimated gradient of F at x0.
    """
    d = x0.shape[0]  # Dimension of the input

    # Generate random perturbations for all samples at once
    perturbations = np.random.normal(0, 1, (num_samples, d))

    # Compute positive and negative perturbed points for all samples
    x_pos = reflect_perturbations(x0 + sigma * perturbations)
    x_neg = reflect_perturbations(x0 - sigma * perturbations)

    # Evaluate the function F for all perturbed points
    f_pos = F(x_pos)
    f_neg = F(x_neg)

    # Compute the gradient estimate as the average over all samples
    grad = np.mean((f_pos - f_neg).reshape(-1, 1) * perturbations, axis=0) / (2 * sigma)

    return grad


if __name__ == "__main__":
    # Define a noisy function F(x)
    def F(x):
        # Simple quadratic function with added noise
        noise = np.random.normal(0, 0.01)
        return np.sum(x**2, axis=1) + noise

    # Point at which to evaluate the gradient
    x0 = np.array([0.5, 0.25, 0.75])

    # Compute the stochastic gradient at x0 with reflection and vectorization
    gradient = get_stochastic_gradient(
        F, x0, sigma=1.e-2, num_samples=1000)
    print("Estimated Gradient at x0 (reflected, vectorized):", gradient)

    from scipy.optimize import approx_fprime
    # Compute the true gradient at x0 using finite differences
    def F_scipy(x):
        return F(x.reshape(1, -1)).squeeze()

    true_gradient = approx_fprime(x0, F_scipy, 1e-6)
    print("True Gradient at x0:", true_gradient)
