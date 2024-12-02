import os
import random
import numpy as np
from scipy.stats import truncnorm, norm, multivariate_normal


def compute_truncated_normal_mean(mean, sd, lower, upper):
    """
    Compute the mean of a truncated normal distribution.
    """
    a, b = (lower - mean) / sd, (upper - mean) / sd
    trunc_mean = truncnorm.mean(a, b, loc=mean, scale=sd)
    return trunc_mean


def em_algorithm_with_missing(Y, num_iters=100):
    """
    Run EM algorithm for copula estimation with missing data, mirroring
    the Gibbs sampler in Algorithm 1 of Feldman et al. 2023.

    Parameters:
    - Y: np.array of shape (n, p), the observed data matrix (with NaNs for
        missing data).
    - num_iters: int, number of iterations for the EM algorithm.

    Returns:
    - Z: np.array of shape (n, p), the estimated latent variables.
    - Sigma: np.array of shape (p, p), the estimated correlation matrix.
    """
    n, p = Y.shape  # Number of observations and variables

    # Initialize latent variables Z
    Z = np.where(~np.isnan(Y), norm.ppf((np.argsort(np.argsort(Y, axis=0), axis=0) + 1) / (n + 1)), np.random.normal(size=(n, p)))

    # Initialize correlation matrix (identity matrix for simplicity)
    Sigma = np.eye(p)

    for it in range(num_iters):
        # E-step: Update the latent variables Z
        for j in range(p):  # Loop over each variable
            for i in range(n):  # Loop over each observation
                if np.isnan(Y[i, j]):
                    # If Y_ij is missing, use the conditional mean for imputation
                    Z_minus_j = np.delete(Z[i, :], j)
                    Sigma_jj = Sigma[j, j]
                    Sigma_j_minusj = Sigma[j, np.arange(p) != j]
                    Sigma_minusj_minusj = np.delete(
                        np.delete(Sigma, j, axis=0), j, axis=1)

                    # Compute the conditional mean
                    cond_mean = Sigma_j_minusj @ np.linalg.inv(
                        Sigma_minusj_minusj) @ Z_minus_j
                    Z[i, j] = cond_mean
                else:
                    # Compute the conditional mean and variance for Z_ij
                    Z_minus_j = np.delete(Z[i, :], j)
                    Sigma_jj = Sigma[j, j]
                    Sigma_j_minusj = Sigma[j, np.arange(p) != j]
                    Sigma_minusj_minusj = np.delete(
                        np.delete(Sigma, j, axis=0), j, axis=1)

                    # Compute the conditional mean and variance
                    cond_mean = Sigma_j_minusj @ np.linalg.inv(
                        Sigma_minusj_minusj) @ Z_minus_j
                    cond_var = Sigma_jj - Sigma_j_minusj @ np.linalg.inv(
                        Sigma_minusj_minusj) @ Sigma_j_minusj.T
                    cond_sd = np.sqrt(cond_var)

                    # Determine truncation bounds using only observed values
                    lower_bound = -np.inf
                    upper_bound = np.inf

                    for k in range(n):
                        if k != i and not np.isnan(Y[k, j]):
                            if Y[k, j] < Y[i, j]:
                                lower_bound = max(lower_bound, Z[k, j])
                            elif Y[k, j] > Y[i, j]:
                                upper_bound = min(upper_bound, Z[k, j])

                    # Compute the expected value of the truncated normal distribution
                    Z[i, j] = compute_truncated_normal_mean(
                        cond_mean, cond_sd, lower_bound, upper_bound)

        # M-step: Update the correlation matrix Sigma
        Sigma = np.corrcoef(Z, rowvar=False)
        if it % 20 == 0:
            print(f"EM iteration {it}: {np.linalg.norm(Sigma - np.eye(p))}")

    return Z, Sigma


def run_joint_cp_with_missingness(
        underlying_model, X_val, Y_val, alpha=0.1,
        maxiter=64, popsize=32, sigma0=0.02,
        reg_weight=0.01, init_u_val=0.9999,
        Y_lower=None, Y_upper=None,
        limit_val2=None,
        X_val2=None, Y_val2=None):
    """
    Compute the 1-alpha quantile of the calib-set residuals jointly.
    """
    from cdf import add_jitter
    from run_experiments import get_corrected_quantile
    import time
    import cma
    # Get the validation-set residuals
    uncalib_pred = underlying_model.predict(X_val)
    residuals = np.abs(Y_val - uncalib_pred)
    to_fit_cdf_residuals = add_jitter(residuals)
    num_calib, target_dim = residuals.shape

    print("Fitting the Gaussian copula...")
    start = time.time()
    Z, Sigma = em_algorithm_with_missing(to_fit_cdf_residuals, num_iters=200)
    end = time.time()
    print(f"Time taken: {end - start:.2f}")
    joint_cdf_vals = get_latent_copula_cdf(Z, Sigma)
    adjusted_level = 1 - alpha

    # Find reasonable initialization for CMA-ES
    diff = np.abs(joint_cdf_vals - adjusted_level)
    # diff[diff < 0] = np.inf  # only consider residuals greater than level
    closest_idx = np.argmin(diff[:-1])  # exclude the infinity point
    init_Z = Z[closest_idx]
    print(f"Initial Z: {init_Z}, diff: {diff[closest_idx]}")

    # Optimize the copula using CMA-ES
    optimization_path = []
    optimization_path.append(np.ones(target_dim)*init_u_val)
    es = cma.CMAEvolutionStrategy(
        x0=init_Z,  # simulated_u[sigma_joint_i],
        sigma0=sigma0,
        # tolupsigma=(init_data.shape[-1])**0.5,  # sqrt(d)
        inopts={
            "bounds": [-5.0, 5.0],
            "popsize": popsize})
    # List of options: https://github.com/CMA-ES/pycma/issues/171
    es.opts.set({
        'maxiter': maxiter  #  100 + 150 * (target_dim+3)**2 // popsize**0.5,
        })
    print("Optimizing the copula...")
    start = time.time()
    log_data = np.log(residuals)
    y_max = np.nanmax(log_data)
    y_min = np.nanmin(log_data)
    while not es.stop():
        z_spawn = es.ask()  # ask for new points to evaluate
        # xs ~ list of np array candidates
        z_stacked = np.stack(z_spawn, axis=0)  # [popsize, x_dim]
        # Evaluate the CDF
        cdf_stacked = get_latent_copula_cdf(z_stacked, Sigma)  # [popsize,]
        # Minimization objective
        cdf_objective = (cdf_stacked - adjusted_level)**2.0
        # cdf_objective[cdf_stacked < adjusted_level] = target_dim*2
        y_stacked = map_z_to_y(
            z_stacked, Z_data=Z, Y_data=to_fit_cdf_residuals,
            Y_lower=Y_lower, Y_upper=Y_upper)
        y_stacked = np.log(y_stacked)
        y_stacked = (y_stacked - y_min)/(y_max - y_min)
        regularization = np.sum(y_stacked, axis=1)
        y = 10000*(cdf_objective + reg_weight*regularization)
        # y[cdf_stacked < adjusted_level] = 2.0
        es.tell(z_spawn, y)  # return the result to the optimizer
        optimization_path.append(es.best.x)
        best_i = y.argmin()
        print(es.best.f, cdf_objective[best_i], regularization[best_i])

    end = time.time()
    print(f"Time taken for CMA-ES: {end - start:.2f}")
    z_opt = es.best.x
    sigma_joint = map_z_to_y(
        z_opt.reshape(1, -1),
        Z_data=Z, Y_data=to_fit_cdf_residuals,
        Y_lower=Y_lower, Y_upper=Y_upper).squeeze(0)

    class GaussianCopula:
        def __init__(self, Sigma):
            self.Sigma = Sigma
        def cdf(self, z):
            return get_latent_copula_cdf(z, self.Sigma)

    cop = GaussianCopula(Sigma)

    z_opt_corrected = get_corrected_quantile(
        initial_estimate=z_opt, residuals=Z, alpha=alpha,
        copula=cop)
    corrected_sigma_joint = map_z_to_y(
        z_opt_corrected.reshape(1, -1),
        Z_data=Z, Y_data=to_fit_cdf_residuals,
        Y_lower=Y_lower, Y_upper=Y_upper).squeeze(0)

    log = {
        "Sigma": Sigma,
        "copula": lambda z: get_latent_copula_cdf(z, Sigma),
        "sigma_joint": sigma_joint,
        "scores": to_fit_cdf_residuals,
        "z_scores": Z,
        "optimization_path": optimization_path,
        "corrected_sigma_joint": corrected_sigma_joint,
        "z_opt": z_opt,
        "z_opt_corrected": z_opt_corrected,
        }
    return sigma_joint, log


def map_z_to_y(z, Z_data, Y_data, Y_lower, Y_upper):
    """
    Map the latent variables Z to the observed residuals Y.

    Parameters:
    - z: np.array of shape (num_query, p), the query latent variables.
    - Z_data: np.array of shape (n, p), the latent variables in the data.
    - Y_data: np.array of shape (n, p), the observed residuals in the data.

    """
    n, p = Y_data.shape
    y = np.zeros_like(z)

    for j in range(p):
        Y_j_grid = np.linspace(Y_lower[j], Y_upper[j], num=1000)
        adjusted_F_j = compute_margin_adjustment(
            Y_j_grid, Z_data[:, j], Y_data[:, j])
        y[:, j] = np.interp(norm.cdf(z[:, j]), adjusted_F_j, Y_j_grid)

    return y


def compute_margin_adjustment(Y_j_query, Z_j_data, Y_j_data):
    """
    Compute the margin adjustment $\tilde{F}_j$ using Equation (10).

    Parameters:
    - Y_j_query: np.array of shape (num_query,), the query values for the j-th variable.
    - Z_j_data: np.array of shape (n,), the latent variables for the j-th variable.
    - Y_j_data: np.array of shape (n,), the observed data for the j-th variable.

    Returns:
    - margin_adjustment: np.array of shape (num_query,), the adjusted marginal
        CDF values.
    """
    num_query = len(Y_j_query)
    margin_adjustment = np.zeros_like(Y_j_query)
    min_Z_j_data = Z_j_data[np.argmin(Y_j_data)]
    is_observed = ~np.isnan(Y_j_data)
    Y_j_data = Y_j_data[is_observed]
    Z_j_data = Z_j_data[is_observed]

    for i in range(num_query):  # For each observation
        # Get the maximum of Z_ij for values where Y_ij <= Y_ij (Equation 10)
        if np.sum(Y_j_data <= Y_j_query[i]) > 0:
            Z_max = max(np.max(Z_j_data[Y_j_data <= Y_j_query[i]]), min_Z_j_data)
        else:
            Z_max = min_Z_j_data
        margin_adjustment[i] = norm.cdf(Z_max)

    return margin_adjustment


def get_latent_copula_cdf(Z, Sigma):
    """
    Compute the copula CDF given the latent variables Z and correlation matrix Sigma.

    Parameters:
    - Z: np.array of shape (n, p), the latent variables.
    - Sigma: np.array of shape (p, p), the correlation matrix.

    Returns:
    - copula_cdf: np.array of shape (n, p), the copula CDF values.
    """
    d = Z.shape[1]
    mvn = multivariate_normal(mean=np.zeros(d), cov=Sigma)
    copula_cdf = mvn.cdf(Z)
    return copula_cdf


def generate_synthetic_data(num_data, seed=0, mcar=False):
    rng = np.random.RandomState(seed)
    # Example observed data matrix Y with missing values (NaN)
    Y = rng.standard_normal(num_data, 2)
    Y_obs = Y.copy()
    if mcar:
        # MCAR: Randomly set 20% of entries in Y to NaN
        num_missing = int(0.2 * Y.size)
        missing_indices = np.random.choice(Y.size, num_missing, replace=False)
        Y_obs.ravel()[missing_indices] = np.nan
    else:
        # MAR: Set the second variable to NaN if the first variable is
        # below -0.5
        Y_obs[Y_obs[:, 0] < -0.5, 1] = np.nan
    return Y, Y_obs


class UnderlyingModel:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train, seed):
        from sklearn.linear_model import LassoCV
        lassos = []
        for j in range(Y_train.shape[1]):
            # Remove nans
            nan_indices = np.isnan(Y_train[:, j])
            X_train_j = X_train[~nan_indices]
            Y_train_j = Y_train[~nan_indices, j]
            # Fit a separate Lasso model for each variable
            lasso = LassoCV(cv=5, random_state=seed)
            lasso.fit(X_train_j, Y_train_j)
            lassos.append(lasso)
        self.lassos = lassos

    def predict(self, X):
        return np.array([lasso.predict(X) for lasso in self.lassos]).T


def run_mar_experiment(synthetic=False, seed=1, mcar=False, alpha=0.1):
    from data_utils import load_penicillin, get_indices
    from run_experiments import (
        evaluate_efficiency,
        evaluate_coverage,
        summarize_metrics_across_splits
    )

    if synthetic:
        # Generate synthetic data
        scores, scores_obs = generate_synthetic_data(
            num_data=100, seed=seed, mcar=mcar)
    else:
        # Load real data
        X, Y, _ = load_penicillin(num_samples=2000, seed=seed)
        indices = get_indices(
            X.shape[0], seed=seed,
            frac_train=0.15, frac_test=0.294, frac_val2=0.9)
        Y = Y[:, [0, 1]]
        X_train = X[indices["train"]]
        Y_train = Y[indices["train"]]
        set_Y0_to_nan = Y_train[:, 1] > 50.0
        print(f"Setting {np.mean(set_Y0_to_nan)} of values of Y0 to NaN")
        Y_train[set_Y0_to_nan, 0] = np.nan
        # Validation set
        X_val = X[indices["val"]]
        Y_val = Y[indices["val"]]
        Y_val_full = Y_val.copy()
        X_val_full = X_val.copy()  # fully observed val set
        set_Y0_to_nan_val = Y_val[:, 1] > 50.0
        Y_val[set_Y0_to_nan_val, 0] = np.nan
        Y_val_with_nans = Y_val.copy()
        X_val_with_nans = X_val.copy()  # val set with missingness
        both_observed = ~np.isnan(Y_val).any(axis=1)
        Y_val = Y_val[both_observed]
        X_val = X_val[both_observed]  # both variables observed
        print(f"MAR validation set size: {len(Y_val_with_nans)}")
        print(f"Both observed validation set size: {len(Y_val)}")

        # Fit the underlying model
        underlying_model = UnderlyingModel()
        underlying_model.fit(X_train, Y_train, seed)

        # Compute the residuals
        pred = underlying_model.predict(X_val_full)
        scores = np.abs(Y_val_full - pred)  # [n_val, 2]
        Y_lower = scores.min(axis=0)
        Y_upper = scores.max(axis=0)

    # 1. Run all methods on Y_obs with both variables observed
    from run_experiments import (
        run_standard_cp,
        run_scalar_score_cp,
        run_empirical_cp,
        run_joint_cp)

    # Run the standard CP
    sigma_standard, _ = run_standard_cp(underlying_model, X_val, Y_val, alpha)
    print(f"Independent: {sigma_standard}")

    # Run the scalar-score CP
    sigma_scalar_score, _ = run_scalar_score_cp(
        underlying_model, X_val, Y_val, alpha, order=2)
    print(f"Scalar score: {sigma_scalar_score}")

    # Run the empirical CP
    sigma_emp, _ = run_empirical_cp(underlying_model, X_val, Y_val, alpha)
    print(f"Empirical: {sigma_emp}")

    # Run the joint CP
    joint_cp_sigmas = {}
    sigma_joint, log  = run_joint_cp(
        underlying_model, X_val, Y_val, alpha,
        cdf_type="nonparam_vine_copula",
        maxiter=64)
    joint_cp_sigmas["Plug-in (ours)"] = sigma_joint
    joint_cp_sigmas["Corrected (ours)"] = log["corrected_sigma_joint"]
    print(f"Plug-in (ours): {sigma_joint}")
    print(f"Corrected (ours): {log['corrected_sigma_joint']}")

    out_sigmas = {
        "Independent": sigma_standard,
        "Scalar score": sigma_scalar_score,
        "Empirical": sigma_emp,
    }
    out_sigmas.update(joint_cp_sigmas)

    # 2. Run plug-in and corrected on Y_obs with missingness imputation
    # Optimize copula to find Z at 1-alpha quantile
    sigma_joint, log = run_joint_cp_with_missingness(
        underlying_model,
        X_val_with_nans, Y_val_with_nans, alpha=alpha, maxiter=64, popsize=32,
        sigma0=0.001, reg_weight=0.01, Y_lower=Y_lower, Y_upper=Y_upper)
    out_sigmas.update(
        {
            "Plug-in with imputation (ours)": sigma_joint,
            "Corrected with imputation (ours)": log["corrected_sigma_joint"]
        }
    )

    # 3. Evaluate metrics
    X_test = X[indices["test"]]
    Y_test = Y[indices["test"]]
    efficiency = evaluate_efficiency(out_sigmas, None)
    coverage = evaluate_coverage(underlying_model, X_test, Y_test, out_sigmas)

    # Plot the copula CDF
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    cdf_values = log["copula"](log["z_scores"])
    scores_obs = scores.copy()
    scores_obs[set_Y0_to_nan_val, 0] = np.nan
    ax.scatter(scores[:, 0], scores[:, 1], c=cdf_values, cmap="viridis")
    ax.scatter(scores_obs[:, 0], scores_obs[:, 1], c="red", marker="x")
    # plt.show()
    fig.colorbar(ax.collections[0], ax=ax, label='Copula CDF')
    fig.savefig("mar_example.png")
    plotting_data = dict(
        cdf_values=cdf_values,
        scores=scores,
        scores_obs=scores_obs,
    )
    np.save("mar_example_plotting_data.npy", plotting_data)

    return out_sigmas, efficiency, coverage





if __name__ == "__main__":
    from run_experiments import summarize_metrics_across_splits

    metrics = {}
    metrics["efficiency"] = []
    metrics["coverage"] = []
    metrics["sigmas"] = []

    for seed in range(1):
        out_sigmas, efficiency, coverage = run_mar_experiment(
            synthetic=False, seed=seed, mcar=False)
        print(out_sigmas)
        metrics["efficiency"].append(efficiency)
        metrics["coverage"].append(coverage)
        metrics["sigmas"].append(out_sigmas)
    mean_metrics, stderr_metrics = summarize_metrics_across_splits(metrics)
    for metric_name in mean_metrics.keys():
        for method_name in mean_metrics[metric_name].keys():
            print(f"{metric_name} ({method_name}): "
                f"{mean_metrics[metric_name][method_name]:.3f} "
                f"± {stderr_metrics[metric_name][method_name]:.3f}")
    # np.save("mar_example_metrics.npy", metrics)
