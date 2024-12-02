import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
import cma
# Add path
import sys
sys.path.append("./")
from cdf import get_empirical_cdf, plot_level_curves, get_vine_copula_cdf, get_kde_cdf, simulate_from_copula, evaluate_copula_cdf, evaluate_copula_pdf, get_gradient_copula_cdf, map_u_to_y, get_empirical_copula_cdf, add_jitter, get_pit
from data_utils import load_data, get_indices
from underlying_models import fit_gaussian_cde, fit_multitask_lasso
import pyvinecopulib as pv
import lightning as L


def run_standard_cp(underlying_model, X_val, Y_val, alpha=0.1):
    """
    Compute the 1-alpha quantile of the calib-set residuals for each target.
    """
    # Get the validation-set residuals
    if isinstance(underlying_model, L.Trainer):
        uncalib_pred = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X_val))
        mu, logvar = tuple(map(np.stack, zip(*uncalib_pred)))
        residuals = np.abs(mu - Y_val) / np.exp(0.5*logvar)
    else:
        uncalib_pred = underlying_model.predict(X_val)
        residuals = np.abs(Y_val - uncalib_pred)
    residuals = add_jitter(residuals)
    num_calib, target_dim = residuals.shape

    # Split CP
    sigma = np.empty(target_dim)
    new_alpha = 1.0 - (1.0 - alpha)**(1.0/target_dim)
    adjusted_level = math.ceil(
        (1 - new_alpha) * (num_calib + 1)
        )/(num_calib + 1)
    for prop_idx in range(target_dim):
        sigma[prop_idx] = np.quantile(
            np.concatenate([residuals[:, prop_idx], [np.inf]]),
            adjusted_level,
            method="inverted_cdf")
    log = {}
    return sigma, log


def run_scalar_score_cp(underlying_model, X_val, Y_val, alpha=0.1, order=1):
    """
    Compute the 1-alpha quantile of the calib-set residuals for each target.
    """
    # Get the validation-set residuals
    if isinstance(underlying_model, L.Trainer):
        uncalib_pred = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X_val))
        mu, logvar = tuple(map(np.stack, zip(*uncalib_pred)))
        residuals = (mu - Y_val) / np.exp(0.5*logvar)  # shape (n, d)
    else:
        uncalib_pred = underlying_model.predict(X_val)  # shape (n, d)
        residuals = np.abs(Y_val - uncalib_pred)  # shape (n, d)
    residuals = np.linalg.norm(
        residuals, ord=order, axis=1)  # shape (n,)
    num_calib, target_dim = Y_val.shape
    residuals = add_jitter(residuals)

    # Split CP
    adjusted_level = np.ceil((1 - alpha) * (num_calib + 1))/(num_calib + 1)
    dist_to_corner = np.quantile(
        np.concatenate([residuals, [np.inf]]), adjusted_level, method="inverted_cdf")
    sigma = np.ones(target_dim) * dist_to_corner
    # dist_to_corner is radius if order=2
    # and distance from origin (original pred) to corner if order=1
    log = {}
    return sigma, log


def run_scalar_score_cp_split(underlying_model, X_val, Y_val, alpha=0.1,
        order=1,
        limit_val2=None,
        X_val2=None, Y_val2=None):
    """
    Compute the 1-alpha quantile of the calib-set residuals for each target.
    """
    # Get the validation-set residuals
    if isinstance(underlying_model, L.Trainer):
        uncalib_pred = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X_val))
        mu, logvar = tuple(map(np.stack, zip(*uncalib_pred)))
        residuals = (mu - Y_val) / np.exp(0.5*logvar)  # shape (n, d)
    else:
        uncalib_pred = underlying_model.predict(X_val)  # shape (n, d)
        residuals = (Y_val - uncalib_pred)  # shape (n, d)

    # Get the validation-set residuals used to fit the copula
    if limit_val2 is not None:
        X_val2 = X_val2[:limit_val2]
        Y_val2 = Y_val2[:limit_val2]
        print(f"Limiting val2 to {limit_val2}")
    if isinstance(underlying_model, L.Trainer):
        uncalib_pred_2 = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X_val2))
        mu_2, logvar_2 = tuple(map(np.stack, zip(*uncalib_pred_2)))
        residuals_2 = (mu_2 - Y_val2) / np.exp(0.5*logvar_2)
    else:
        uncalib_pred_2 = underlying_model.predict(X_val2)
        residuals_2 = (Y_val2 - uncalib_pred_2)
    rescale_mu = np.mean(residuals_2, axis=0, keepdims=True)
    rescale_sigma = np.std(residuals_2, axis=0, keepdims=True)

    # Rescale the residuals
    original_residuals = np.copy(residuals)
    residuals = (residuals - rescale_mu)/rescale_sigma
    residuals = np.linalg.norm(
        residuals, ord=order, axis=1)  # shape (n,)
    num_calib, target_dim = Y_val.shape
    residuals = add_jitter(residuals)

    # Compute original scalar scores
    original_residuals = np.linalg.norm(
        original_residuals, ord=order, axis=1)  # shape (n,)
    original_residuals = add_jitter(original_residuals)

    # Split CP
    residuals_with_inf = np.concatenate([residuals, [np.inf]])
    # orig_residuals_with_inf = np.concatenate([original_residuals, [np.inf]])
    adjusted_level = np.ceil((1 - alpha) * (num_calib + 1))/(num_calib + 1)
    dist_to_corner_scaled = np.quantile(residuals_with_inf, adjusted_level, method="inverted_cdf")

    # index = np.where(np.abs(residuals_with_inf - dist_to_corner_scaled) < 1.e-7)[0][0]
    # dist_to_corner = orig_residuals_with_inf[index]

    sigma = (rescale_sigma.squeeze(0) * dist_to_corner_scaled, rescale_mu)

    # dist_to_corner is radius if order=2
    # and distance from origin (original pred) to corner if order=1
    log = {
        "rescale_mu": rescale_mu.squeeze(0),
        "rescale_sigma": rescale_sigma.squeeze(0),
        "dist_to_corner_scaled": dist_to_corner_scaled,
    }
    return sigma, log


def run_empirical_cp(
        underlying_model, X_val, Y_val, alpha=0.1, cdf_type="empirical"):
    """
    Empirical CP method of Messoudi et al. 2021

    """
    from scipy.optimize import minimize

    # Get the validation-set residuals

    if isinstance(underlying_model, L.Trainer):
        uncalib_pred = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X_val))
        mu, logvar = tuple(map(np.stack, zip(*uncalib_pred)))
        residuals = np.abs(mu - Y_val) / np.exp(0.5*logvar)
    else:
        uncalib_pred = underlying_model.predict(X_val)
        residuals = np.abs(Y_val - uncalib_pred)
    to_fit_cdf_residuals = add_jitter(residuals)
    num_calib, target_dim = residuals.shape

    # Get the empirical CDF
    to_fit_cdf_residuals = np.concatenate(
            [residuals, np.ones((1, target_dim))*np.inf], axis=0)
    joint_cdf_vals, out = get_empirical_copula_cdf(
        to_fit_cdf_residuals, query_points=to_fit_cdf_residuals)

    # Find reasonable initialization for optimization
    adjusted_level = math.ceil((1 - alpha)*(num_calib + 1))/(num_calib + 1)
    diff = np.abs(joint_cdf_vals - adjusted_level)
    # diff[diff < 0] = np.inf  # only consider residuals greater than level
    sigma_joint_i = np.argmin(diff[:-1])  # exclude the infinity point
    sigma_joint = to_fit_cdf_residuals[sigma_joint_i]

    # Optimize the copula assuming u1 = ... = ud = u
    init_u = out["u"][sigma_joint_i]

    def minimize_fn(u, level, target_dim):
        cdf = out["copula"].cdf(
                np.ones((1, target_dim))*u).squeeze()
        loss = (cdf - level)**2.0
        # Below only hurts the optimization
        # if cdf < level:
        #     return target_dim*2
        return loss

    results = minimize(minimize_fn, init_u, args=(adjusted_level, target_dim))
    u_opt = np.ones(target_dim)*results.x
    sigma_joint = map_u_to_y(u_opt.reshape(1, -1), out["data"]).squeeze(0)

    log = {
        "u_opt": u_opt,
        "optimization_result": results,
        }
    return sigma_joint, log


def run_joint_cp(
        underlying_model, X_val, Y_val, alpha=0.1, cdf_type="empirical",
        maxiter=64, popsize=32, sigma0=0.001,
        reg_weight=0.01, init_u_val=0.9999,
        limit_val2=None,
        X_val2=None, Y_val2=None):
    """
    Compute the 1-alpha quantile of the calib-set residuals jointly.
    """
    # Get the validation-set residuals
    if isinstance(underlying_model, L.Trainer):
        uncalib_pred = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X_val))
        mu, logvar = tuple(map(np.stack, zip(*uncalib_pred)))
        residuals = np.abs(mu - Y_val) / np.exp(0.5*logvar)
    else:
        uncalib_pred = underlying_model.predict(X_val)
        residuals = np.abs(Y_val - uncalib_pred)
    to_fit_cdf_residuals = add_jitter(residuals)
    num_calib, target_dim = residuals.shape
    # Plot
    if target_dim == 2:
        fig, ax = plot_level_curves(residuals, get_empirical_cdf)
        fig.savefig("level_curves_empirical_joint_cp.png")
        fig, ax = plot_level_curves(residuals, get_vine_copula_cdf)
        fig.savefig("level_curves_vine_copula_joint_cp.png")
        fig, ax = plot_level_curves(residuals, get_kde_cdf)
        fig.savefig("level_curves_kde_joint_cp.png")

    # joint CP
    if cdf_type == "kde":
        joint_cdf_vals, out = get_kde_cdf(to_fit_cdf_residuals, query_points=None)
        adjusted_level = 1 - alpha
    elif cdf_type == "nonparam_vine_copula":
        print("Fitting the vine...")
        start = time.time()
        joint_cdf_vals, out = get_vine_copula_cdf(to_fit_cdf_residuals, query_points=None)
        end = time.time()
        print(f"Time taken: {end - start:.2f}")
        simulated_u = out["copula"].simulate(
            min(max(1000, 4**target_dim), 100000))
        joint_cdf_vals = out["copula"].cdf(
            simulated_u, N=10000, seeds=list(range(10000)))
        adjusted_level = 1 - alpha
    else:
        raise ValueError(f"Unknown cdf_type: {cdf_type}")

    # Find reasonable initialization for CMA-ES
    diff = np.abs(joint_cdf_vals - adjusted_level)
    # diff[diff < 0] = np.inf  # only consider residuals greater than level
    sigma_joint_i = np.argmin(diff[:-1])  # exclude the infinity point
    candidate_sigmas = to_fit_cdf_residuals
    if cdf_type == "nonparam_vine_copula":
        candidate_sigmas = map_u_to_y(simulated_u, out["data"])
    sigma_joint = candidate_sigmas[sigma_joint_i]

    # Optimize the copula using CMA-ES
    optimization_path = []
    optimization_path.append(np.ones(target_dim)*init_u_val)
    if cdf_type in ["nonparam_vine_copula"]:
        es = cma.CMAEvolutionStrategy(
            x0=np.ones(target_dim)*init_u_val,  # simulated_u[sigma_joint_i],
            sigma0=sigma0,
            # tolupsigma=(init_data.shape[-1])**0.5,  # sqrt(d)
            inopts={
                "bounds": [0, 1],
                "popsize": popsize})
        # List of options: https://github.com/CMA-ES/pycma/issues/171
        es.opts.set({
            'maxiter': maxiter  #  100 + 150 * (target_dim+3)**2 // popsize**0.5,
            })
        print("Optimizing the copula...")
        start = time.time()
        log_data = np.log(out["data"] + 1.e-7)
        y_max = log_data.max()
        y_min = log_data.min()
        while not es.stop():
            u_spawn = es.ask()  # ask for new points to evaluate
            # xs ~ list of np array candidates
            u_stacked = np.stack(u_spawn, axis=0)  # [popsize, x_dim]
            # Evaluate the CDF
            cdf_stacked = out["copula"].cdf(
                u_stacked,
                N=10000, seeds=list(range(10000))
            )  # [popsize,]
            # Minimization objective
            cdf_objective = (cdf_stacked - adjusted_level)**2.0
            cdf_objective[cdf_stacked < adjusted_level] = target_dim*2
            y_stacked = np.log(map_u_to_y(u_stacked, out["data"]))
            y_stacked = (y_stacked - y_min)/(y_max - y_min)
            regularization = np.sum(y_stacked, axis=1)
            y = 10000*(cdf_objective + reg_weight*regularization)
            # y[cdf_stacked < adjusted_level] = 2.0
            es.tell(u_spawn, y)  # return the result to the optimizer
            optimization_path.append(es.best.x)
            # best_i = y.argmin()
            # print(es.best.f, cdf_objective[best_i], regularization[best_i])
        end = time.time()
        print(f"Time taken for CMA-ES: {end - start:.2f}")
        u_opt = es.best.x
        sigma_joint = map_u_to_y(u_opt.reshape(1, -1), out["data"]).squeeze(0)

    # Correct the optimized u
    if X_val2 is not None:
        if limit_val2 is not None:
            X_val2 = X_val2[:limit_val2]
            Y_val2 = Y_val2[:limit_val2]
        uncalib_pred_val2 = underlying_model.predict(X_val2)
        residuals_val2 = np.abs(Y_val2 - uncalib_pred_val2)
        residuals_val2 = add_jitter(residuals_val2)
        u_for_eif = get_pit(
            data=to_fit_cdf_residuals,
            query_points=residuals_val2)
    else:
        u_for_eif = out["u"]
    u_opt_corrected = get_corrected_quantile(
        initial_estimate=u_opt, residuals=u_for_eif, alpha=alpha,
        copula=out["copula"])
    corrected_sigma_joint = map_u_to_y(
        u_opt_corrected.reshape(1, -1), out["data"]).squeeze(0)

    log = {
        "copula": out["copula"],
        "sigma_joint": sigma_joint,
        "scores": out["data"],
        "u_scores": out["u"],
        "optimization_path": optimization_path,
        "corrected_sigma_joint": corrected_sigma_joint,
        "u_opt": u_opt,
        "u_opt_corrected": u_opt_corrected,
        }
    return sigma_joint, log


def run_joint_cp_extra_split(
        underlying_model, X_val, Y_val, alpha=0.1, cdf_type="empirical",
        maxiter=64, popsize=32, sigma0=0.001,
        reg_weight=0.01, init_u_val=0.9999,
        limit_val2=None,
        X_val2=None, Y_val2=None):
    """
    Compute the 1-alpha quantile of the calib-set residuals jointly.
    """
    assert X_val2 is not None

    # Get the validation-set residuals used to fit the marginal ECDF
    if isinstance(underlying_model, L.Trainer):
        uncalib_pred = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X_val))
        mu, logvar = tuple(map(np.stack, zip(*uncalib_pred)))
        residuals = np.abs(mu - Y_val) / np.exp(0.5*logvar)
    else:
        uncalib_pred = underlying_model.predict(X_val)
        residuals = np.abs(Y_val - uncalib_pred)
    to_get_marginal_ecdf_residuals = add_jitter(residuals)
    _, target_dim = residuals.shape

    # Get the validation-set residuals used to fit the copula
    if limit_val2 is not None:
        X_val2 = X_val2[:limit_val2]
        Y_val2 = Y_val2[:limit_val2]
        print(f"Limiting val2 to {limit_val2}")
    if isinstance(underlying_model, L.Trainer):
        uncalib_pred_2 = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X_val2))
        mu_2, logvar_2 = tuple(map(np.stack, zip(*uncalib_pred_2)))
        residuals_2 = np.abs(mu_2 - Y_val2) / np.exp(0.5*logvar_2)
    else:
        uncalib_pred_2 = underlying_model.predict(X_val2)
        residuals_2 = np.abs(Y_val2 - uncalib_pred_2)
    to_fit_copula_residuals = add_jitter(residuals_2)

    # Plot
    if target_dim == 2:
        fig, ax = plot_level_curves(residuals, get_empirical_cdf)
        fig.savefig("level_curves_empirical_joint_cp.png")
        fig, ax = plot_level_curves(residuals, get_vine_copula_cdf)
        fig.savefig("level_curves_vine_copula_joint_cp.png")
        fig, ax = plot_level_curves(residuals, get_kde_cdf)
        fig.savefig("level_curves_kde_joint_cp.png")

    # joint CP
    if cdf_type == "kde":
        raise ValueError("KDE not supported for extra-split setting")
    elif cdf_type == "nonparam_vine_copula":
        print("Fitting the vine...")
        start = time.time()
        joint_cdf_vals, out = get_vine_copula_cdf(
            to_fit_copula_residuals,
            query_points=None,
            residuals_for_pit=to_get_marginal_ecdf_residuals)
        end = time.time()
        print(f"Time taken: {end - start:.2f}")
        simulated_u = out["copula"].simulate(
            min(max(1000, 4**target_dim), 100000))
        joint_cdf_vals = out["copula"].cdf(
            simulated_u, N=10000, seeds=list(range(10000)))
        adjusted_level = 1 - alpha
    else:
        raise ValueError(f"Unknown cdf_type: {cdf_type}")

    # Find reasonable initialization for CMA-ES
    diff = np.abs(joint_cdf_vals - adjusted_level)
    # diff[diff < 0] = np.inf  # only consider residuals greater than level
    sigma_joint_i = np.argmin(diff[:-1])  # exclude the infinity point
    candidate_sigmas = to_fit_copula_residuals
    if cdf_type == "nonparam_vine_copula":
        candidate_sigmas = map_u_to_y(simulated_u, out["data"])
    sigma_joint = candidate_sigmas[sigma_joint_i]

    # Optimize the copula using CMA-ES
    optimization_path = []
    optimization_path.append(np.ones(target_dim)*init_u_val)
    if cdf_type in ["nonparam_vine_copula"]:
        es = cma.CMAEvolutionStrategy(
            x0=np.ones(target_dim)*init_u_val,  # simulated_u[sigma_joint_i],
            sigma0=sigma0,
            # tolupsigma=(init_data.shape[-1])**0.5,  # sqrt(d)
            inopts={
                "bounds": [0, 1],
                "popsize": popsize})
        # List of options: https://github.com/CMA-ES/pycma/issues/171
        es.opts.set({
            'maxiter': maxiter  #  100 + 150 * (target_dim+3)**2 // popsize**0.5,
            })
        print("Optimizing the copula...")
        start = time.time()
        log_data = np.log(out["data"] + 1.e-7)
        y_max = log_data.max()
        y_min = log_data.min()
        while not es.stop():
            u_spawn = es.ask()  # ask for new points to evaluate
            # xs ~ list of np array candidates
            u_stacked = np.stack(u_spawn, axis=0)  # [popsize, x_dim]
            # Evaluate the CDF
            cdf_stacked = out["copula"].cdf(
                u_stacked,
                N=10000, seeds=list(range(10000))
            )  # [popsize,]
            # Minimization objective
            cdf_objective = (cdf_stacked - adjusted_level)**2.0
            cdf_objective[cdf_stacked < adjusted_level] = target_dim*2
            y_stacked = np.log(map_u_to_y(u_stacked, out["data"]))
            y_stacked = (y_stacked - y_min)/(y_max - y_min)
            regularization = np.sum(y_stacked, axis=1)
            y = 10000*(cdf_objective + reg_weight*regularization)
            # y[cdf_stacked < adjusted_level] = 2.0
            es.tell(u_spawn, y)  # return the result to the optimizer
            optimization_path.append(es.best.x)
            # best_i = y.argmin()
            # print(es.best.f, cdf_objective[best_i], regularization[best_i])
        end = time.time()
        print(f"Time taken for CMA-ES: {end - start:.2f}")
        u_opt = es.best.x
        sigma_joint = map_u_to_y(u_opt.reshape(1, -1), out["data"]).squeeze(0)

    # Correct the optimized u using EIF on the entire val set
    u_for_eif = get_pit(
        data=to_get_marginal_ecdf_residuals,
        query_points=np.concatenate(
            [to_fit_copula_residuals, to_get_marginal_ecdf_residuals],
            axis=0))

    u_opt_corrected = get_corrected_quantile(
        initial_estimate=u_opt, residuals=u_for_eif, alpha=alpha,
        copula=out["copula"])
    corrected_sigma_joint = map_u_to_y(
        u_opt_corrected.reshape(1, -1), out["data"]).squeeze(0)

    log = {
        "copula": out["copula"],
        "sigma_joint": sigma_joint,
        "scores": out["data"],
        "u_scores": out["u"],
        "optimization_path": optimization_path,
        "corrected_sigma_joint": corrected_sigma_joint,
        "u_opt": u_opt,
        "u_opt_corrected": u_opt_corrected,
        }
    return sigma_joint, log


def get_corrected_quantile(
        initial_estimate, residuals, alpha, copula,
        num_samples=10000, perturb_sigma=1e-2):
    """
    Apply the one-step correction to the initial estimate of the
    (1-alpha)-quantile of the residuals.

    Parameters
    ----------
    initial_estimate : float
        The initial estimate of the (1-alpha)-quantile in copula space.
    residuals : array-like
        The residuals or pseudo-obs of residuals, of shape (n, d).
    alpha : float
        The miscoverage level.
    copula : object
        The fitted copula object.
    """
    from gradients import get_stochastic_gradient
    num_calib, target_dim = residuals.shape
    indicator = np.all(residuals <= initial_estimate, axis=1).astype(float)  # shape (n,)
    eif = (1.0 - alpha - indicator).reshape(num_calib, 1)  # shape (n, 1)
    # grad = get_gradient_copula_cdf(
    #     initial_estimate, copula).reshape(1, target_dim)  # shape (1, d)
    grad = get_stochastic_gradient(
        lambda x: copula.cdf(np.atleast_2d(x)).squeeze(),
        initial_estimate,
        sigma=perturb_sigma,
        num_samples=num_samples).reshape(1, target_dim)
    grad_norm_sq = np.sum(grad**2, axis=1).squeeze()  # scalar
    eif = eif * grad / grad_norm_sq  # shape (n, d)
    correction_term = np.mean(eif, axis=0)  # shape (d,)
    print("grad", grad)
    print("correction_term", correction_term)
    return initial_estimate + correction_term


def evaluate_efficiency(out_sigmas, metadata):
    """
    Evaluate the efficiency of the CPs.
    """
    from scipy.special import gamma
    efficiency = {}
    for method_name, method_sigma in out_sigmas.items():
        if method_name in ["Scalar score", "Scalar score, split"]:
            if "split" in method_name:
                method_sigma, _ = method_sigma
                scaled_sigma = method_sigma
                dim = len(scaled_sigma)  # target_dim
                log_vol = (dim*0.5)*np.log(np.pi) + np.log(scaled_sigma).sum() \
                - np.log(gamma(dim*0.5 + 1))
            else:
                scaled_sigma = method_sigma
                dim = len(scaled_sigma)  # target_dim
                # Log-volume of a d-ball
                log_vol = (dim*0.5)*np.log(np.pi) + dim*np.log(scaled_sigma[0]) \
                    - np.log(gamma(dim*0.5 + 1))
                # # Log-volume of cross-polytope
                # log_vol = dim*np.log(2.0) + dim*np.log(scaled_sigma[0]) \
                #     - np.log(math.factorial(dim))
            efficiency[method_name] = log_vol
        elif method_name in ["Scalar score (order=1)", "Scalar score (order=1), split"]:
            if "split" in method_name:
                method_sigma, _ = method_sigma
                scaled_sigma = method_sigma
                dim = len(scaled_sigma)  # target_dim
                log_vol = dim*np.log(2.0) + np.log(scaled_sigma).sum() \
                    - np.log(math.factorial(dim))
            else:
                scaled_sigma = method_sigma
                dim = len(scaled_sigma)  # target_dim
                # Log-volume of cross-polytope
                log_vol = dim*np.log(2.0) + dim*np.log(scaled_sigma[0]) \
                    - np.log(math.factorial(dim))
            efficiency[method_name] = log_vol
        elif method_name in ["Scalar score (order=inf)", "Scalar score (order=inf), split"]:
            if "split" in method_name:
                method_sigma, _ = method_sigma
                scaled_sigma = method_sigma
                dim = len(scaled_sigma)  # target_dim
                log_vol = np.log(2.0*scaled_sigma).sum()
            else:
                scaled_sigma = method_sigma
                dim = len(scaled_sigma)  # target_dim
                # Log-volume of hypercube
                log_vol = dim*np.log(2.0*scaled_sigma[0])
            efficiency[method_name] = log_vol
        else:
            scaled_sigma = method_sigma
            eff = np.sum(np.log(2.0*scaled_sigma))
            efficiency[method_name] = eff
    return efficiency


def evaluate_coverage(underlying_model, X, Y, out_sigmas):
    """
    Evaluate the coverage of the CPs.

    Parameters
    ----------
    underlying_model : sklearn estimator
        The fitted model.
    X : array-like, shape (n, p)
        The input data.
    Y : array-like, shape (n, d)
        The target data.
    out_sigmas : dict
        The sigmas for each method.

    """
    coverage = {}
    if isinstance(underlying_model, L.Trainer):
        pred = underlying_model.predict(
            underlying_model.fit_model, torch.tensor(X))
        mu, logvar = tuple(map(np.stack, zip(*pred)))
        for method_name, method_sigma in out_sigmas.items():
            eff_sigma = method_sigma[0]*np.exp(0.5*logvar)
            if method_name == "Scalar score":
                sq_dist = np.sum((Y - mu)**2, axis=1)  # shape (n,)
                is_covered = (sq_dist <= eff_sigma**2).astype(float)  # shape (n)
            else:
                lower = mu - eff_sigma  # shape (n, d)
                upper = mu + eff_sigma  # shape (n, d)
                is_covered = np.all((lower <= Y) & (Y <= upper), axis=1)  # shape (n)
            coverage[method_name] = np.mean(is_covered)
    else:
        pred = underlying_model.predict(X)
        for method_name, method_sigma in out_sigmas.items():
            if method_name in ["Scalar score", "Scalar score, split"]:
                if "split" in method_name:
                    method_sigma, rescale_mu = method_sigma
                    pred = pred + rescale_mu
                sq_dist = np.sum((Y - pred)**2, axis=1)  # shape (n,)
                is_covered = (sq_dist <= method_sigma[0]**2).astype(float)  # shape (n)
            elif method_name in ["Scalar score (order=1)", "Scalar score (order=1), split"]:
                if "split" in method_name:
                    method_sigma, rescale_mu = method_sigma
                    pred = pred + rescale_mu
                l1_dist = np.linalg.norm(Y - pred, ord=1, axis=1)  # shape (n,)
                is_covered = (l1_dist <= method_sigma[0]).astype(float)  # shape (n)
            elif method_name in ["Scalar score (order=inf)", "Scalar score (order=inf), split"]:
                if "split" in method_name:
                    method_sigma, rescale_mu = method_sigma
                    pred = pred + rescale_mu
                max_dist = np.linalg.norm(Y - pred, ord=np.inf, axis=1)  # shape (n,)
                is_covered = (max_dist <= method_sigma[0]).astype(float)  # shape (n)
            else:
                lower = pred - method_sigma  # shape (n, d)
                upper = pred + method_sigma  # shape (n, d)
                is_covered = np.all((lower <= Y) & (Y <= upper), axis=1)  # shape (n)
            coverage[method_name] = np.mean(is_covered)
    return coverage


def summarize_metrics_across_splits(metrics_dict, include_quantiles=False):
    """
    Summarize the metrics across differents.

    Parameters
    ----------
    metrics_dict : dict
        A dictionary with the metrics for each. The keys are the metric
        names, and the values are lists of dictionaries. Each dictionary
        contains the metrics for each method.
    """
    metrics_mean = {}
    metrics_stderr = {}
    metrics_med = {}
    metrics_lower = {}
    metrics_upper = {}
    for metric_name, metric_list in metrics_dict.items():
        if metric_name not in ["efficiency", "coverage"]:
            continue
        metrics_mean[metric_name] = {}
        metrics_stderr[metric_name] = {}
        if include_quantiles:
            metrics_med[metric_name] = {}
            metrics_lower[metric_name] = {}
            metrics_upper[metric_name] = {}
        for method_name in metric_list[0].keys():
            num_seeds = len(metrics_dict[metric_name])
            metric_vals = [
                metric_list[i][method_name] for i in range(num_seeds)]
            metrics_mean[metric_name][method_name] = np.mean(metric_vals)
            metrics_stderr[metric_name][method_name] = np.std(metric_vals) / np.sqrt(num_seeds)
            if include_quantiles:
                metrics_med[metric_name][method_name] = np.quantile(
                    metric_vals, 0.5)
                metrics_lower[metric_name][method_name] = np.quantile(
                    metric_vals, 0.5 - 0.34)
                metrics_upper[metric_name][method_name] = np.quantile(
                    metric_vals, 0.5 + 0.34)
    if include_quantiles:
        return metrics_mean, metrics_stderr, metrics_med, metrics_lower, metrics_upper
    return metrics_mean, metrics_stderr


def run_experiments(
        dataset_names, out_dir, alpha=0.1,
        override_data=None,
        discard_val2=True,
        frac_test=0.5, frac_train=0.95, frac_val2=0.0, limit_val2=None,
        standardize_Y=False, init_u_val = 0.99, reg_weight=0.01,
        underlying_algorithm="lasso",
        num_seeds=5):

    for dataset_name in dataset_names:
        metrics = {}
        metrics["efficiency"] = []
        metrics["coverage"] = []
        metrics["sigmas"] = []

        if dataset_name in ["penicillin"]:
            # frac_train = 0.6
            frac_train = 0.15  # 300
            frac_test = 0.294  # 500
            # frac_val2 = 0.92  # val size 103, val + val2 = 1200
            # frac_val2 = 0.92 - 0.08
            # limit_val2 = 100
            init_u_val = 0.99
            # reg_weight = 0.04
        if dataset_name in ["caco2_plus"]:
            frac_train = 0.7
            # frac_val2 = 0.5
        elif dataset_name in ["rf1", "rf2"]:
            frac_train = 0.95
            frac_test = 1-0.5012
            frac_val2 = 0.5
        elif dataset_name in ["scm1d", "scm20d"]:
            frac_train = 0.9
            init_u_val = 0.9999
            frac_test = 1 - 0.5002
            # frac_test = 0.7
        elif dataset_name in ["stock_portfolio"]:
            frac_train = 0.3
            frac_test = 0.4
            reg_weight = 0.00001
        print(f"Dataset: {dataset_name}")
        if override_data is None:
            X, y, metadata = load_data(dataset_name, standardize_Y=standardize_Y)
        else:
            X, y, metadata = override_data

        print(f"Metadata: {metadata['Y_max'] - metadata['Y_min']}")
        for seed in range(num_seeds):
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            indices = get_indices(
                X.shape[0], seed=seed,
                frac_train=frac_train, frac_test=frac_test, frac_val2=frac_val2)
            X_train = X[indices["train"]]
            Y_train = y[indices["train"]]
            X_val = X[indices["val"]]
            Y_val = y[indices["val"]]
            print(f"Seed: {seed}")
            print(f"Train size: {X_train.shape[0]}")
            print(f"Val size: {X_val.shape[0]}")
            if (indices["val2"] is None) or discard_val2:
                X_val2 = None
                Y_val2 = None
            else:
                X_val2 = X[indices["val2"]]
                Y_val2 = y[indices["val2"]]
                print(f"Val2 size: {X[indices['val2']].shape[0]}")

            # Fit the model
            if underlying_algorithm == "lasso":
                underlying_model = fit_multitask_lasso(X_train, Y_train)
            elif underlying_algorithm == "gaussian_cde":
                underlying_model = fit_gaussian_cde(X_train, Y_train)
            else:
                raise ValueError(
                    f"Unknown underlying algorithm: {underlying_algorithm}")

            # Run the standard CP
            sigma_standard, _ = run_standard_cp(underlying_model, X_val, Y_val, alpha)
            print(f"Independent: {sigma_standard}")

            # Run the scalar-score CP
            sigma_scalar_score, _ = run_scalar_score_cp(
                underlying_model, X_val, Y_val, alpha, order=2)
            print(f"Scalar score: {sigma_scalar_score}")

            # Run the scalar-score CP
            sigma_scalar_score_1, _ = run_scalar_score_cp(
                underlying_model, X_val, Y_val, alpha, order=1)
            print(f"Scalar score (order=1): {sigma_scalar_score_1}")

            # Run the scalar-score CP with inf norm
            sigma_scalar_score_inf, _ = run_scalar_score_cp(
                underlying_model, X_val, Y_val, alpha, order=np.inf)
            print(f"Scalar score (order=inf): {sigma_scalar_score_inf}")

            if X_val2 is not None:

                # Run the scalar-score CP with rescaling split
                sigma_scalar_score_split, _ = run_scalar_score_cp_split(
                    underlying_model, X_val, Y_val, alpha=alpha,
                    order=2,
                    limit_val2=limit_val2,
                    X_val2=X_val2, Y_val2=Y_val2)
                print(f"Scalar score, split: {sigma_scalar_score_split}")

                # Run the scalar-score CP with rescaling split
                sigma_scalar_score_1_split, _ = run_scalar_score_cp_split(
                    underlying_model, X_val, Y_val, alpha=alpha,
                    order=1,
                    limit_val2=limit_val2,
                    X_val2=X_val2, Y_val2=Y_val2)
                print(f"Scalar score (order=1), split: {sigma_scalar_score_1_split}")

                # Run the scalar-score CP with rescaling split
                sigma_scalar_score_inf_split, _ = run_scalar_score_cp_split(
                    underlying_model, X_val, Y_val, alpha=alpha,
                    order=np.inf,
                    limit_val2=limit_val2,
                    X_val2=X_val2, Y_val2=Y_val2)
                print(f"Scalar score (order=inf), split: {sigma_scalar_score_inf_split}")

            # Run the empirical CP
            sigma_emp, _ = run_empirical_cp(underlying_model, X_val, Y_val, alpha)
            print(f"Empirical: {sigma_emp}")

            # # Run the joint CP
            # joint_cp_sigmas = {}
            # sigma_joint, log  = run_joint_cp(
            #     underlying_model, X_val, Y_val, alpha,
            #     cdf_type="nonparam_vine_copula",
            #     init_u_val=init_u_val,
            #     reg_weight=reg_weight,
            #     maxiter=64,
            #     X_val2=X_val2, Y_val2=Y_val2, limit_val2=limit_val2)
            # joint_cp_sigmas["Plug-in (ours)"] = sigma_joint
            # joint_cp_sigmas["Corrected (ours)"] = log["corrected_sigma_joint"]
            # print(f"Plug-in (ours): {sigma_joint}")
            # print(f"Corrected (ours): {log['corrected_sigma_joint']}")

            # joint_cp_sigmas_split = {}
            # sigma_joint_split, log_split  = run_joint_cp_extra_split(
            #     underlying_model, X_val, Y_val, alpha,
            #     cdf_type="nonparam_vine_copula",
            #     init_u_val=init_u_val,
            #     reg_weight=reg_weight,
            #     maxiter=64,
            #     X_val2=X_val2, Y_val2=Y_val2, limit_val2=limit_val2)
            # joint_cp_sigmas_split["Plug-in, split (ours)"] = sigma_joint_split
            # joint_cp_sigmas_split["Corrected, split (ours)"] = log_split["corrected_sigma_joint"]
            # print(f"Plug-in split (ours): {sigma_joint_split}")
            # print(f"Corrected split (ours): {log_split['corrected_sigma_joint']}")

            out_sigmas = {
                "Independent": sigma_standard,
                "Scalar score": sigma_scalar_score,
                "Scalar score (order=1)": sigma_scalar_score_1,
                "Scalar score (order=inf)": sigma_scalar_score_inf,
                "Empirical": sigma_emp,
            }
            if X_val2 is not None:
                out_sigmas.update({
                "Scalar score, split": sigma_scalar_score_split,
                "Scalar score (order=1), split": sigma_scalar_score_1_split,
                "Scalar score (order=inf), split": sigma_scalar_score_inf_split,
                })
            # out_sigmas.update(joint_cp_sigmas)
            # out_sigmas.update(joint_cp_sigmas_split)

            # Evaluation
            X_test = X[indices["test"]]
            Y_test = y[indices["test"]]

            # Evaluate the efficiency of the CPs
            efficiency = evaluate_efficiency(out_sigmas, metadata)
            metrics["efficiency"].append(efficiency)

            # Evaluate the coverage of the CPs
            coverage = evaluate_coverage(underlying_model, X_test, Y_test, out_sigmas)
            metrics["coverage"].append(coverage)
            metrics["sigmas"].append(out_sigmas)

        mean_metrics, stderr_metrics = summarize_metrics_across_splits(metrics)
        for metric_name in mean_metrics.keys():
            for method_name in mean_metrics[metric_name].keys():
                print(f"{metric_name} ({method_name}): "
                    f"{mean_metrics[metric_name][method_name]:.3f} "
                    f"± {stderr_metrics[metric_name][method_name]:.3f}")

        np.save(
            os.path.join(
                out_dir,
                f"{dataset_name}_standardize_Y_{standardize_Y}.npy"),
            metrics, allow_pickle=True)
        # hyper_metrics[frac_test] = mean_metrics, stderr_metrics


if __name__ == "__main__":
    global_dir = "rebuttals_1201"
    os.makedirs(global_dir, exist_ok=True)
    dataset_names = [
        # "penicillin",
        # "caco2_plus",
        # "solar_flare",
        # "energy_efficiency",
        # "stock_portfolio",
        # "rf1",
        # "rf2",
        "scm1d",
        "scm20d",
        # "scpf"
        ]
    out_dir = global_dir
    alpha = 0.1
    run_experiments(dataset_names, out_dir,
        alpha=alpha, standardize_Y=False,
        num_seeds=5)
    # run_experiments(dataset_names, out_dir,
    #     alpha=alpha, standardize_Y=True,
    #     discard_val2=True,
    #     num_seeds=5)
    # run_experiments(dataset_names, out_dir,
    #     alpha=alpha, standardize_Y=False,
    #     underlying_algorithm="gaussian_cde",
    #     num_seeds=5)

    # Emp vs. nominal coverage with varying alpha
    # dataset_names = ["rf1"]
    # for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     out_dir = os.path.join(
    #         global_dir, f"coverage_vs_alpha_{alpha:.1f}_rf1")
    #     os.makedirs(out_dir, exist_ok=True)
    #     run_experiments(dataset_names, out_dir,
    #         alpha=alpha, standardize_Y=False,
    #         # frac_val2=0.92,
    #         # discard_val2=True,
    #         num_seeds=5)

    # Emp vs. nominal coverage with varying n
    # dataset_names = ["penicillin"]
    # for alpha in [0.1]:
    #     for frac_val2 in [0.8, 0.85, 0.9, 0.95]:
    #         out_dir = os.path.join(
    #             global_dir, f"coverage_vs_alpha_{alpha:.1f}_frac_val2_{frac_val2:.1f}")
    #         os.makedirs(out_dir, exist_ok=True)
    #         run_experiments(dataset_names, out_dir,
    #             alpha=alpha, frac_val2=frac_val2, standardize_Y=True,
    #             discard_val2=True, num_seeds=10)
