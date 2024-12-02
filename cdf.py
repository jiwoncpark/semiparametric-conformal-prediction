import numpy as np
import matplotlib.pyplot as plt
import pyvinecopulib as pv


def get_empirical_cdf(residuals, query_points):
    """Return the empirical CDF of the data.

    Parameters
    ----------
    residuals : array-like
        Data to be used to compute the empirical CDF, of shape (n, d).
        These will correspond to the calibration set residuals.
    query_points : array-like
        Points at which to evaluate the empirical CDF, of shape (m, d).

    Returns
    -------
    joint_cdf_vals : array-like
        Empirical CDF values, of shape (m,) and taking values in [0, 1].
    """
    num_calib = residuals.shape[0]
    num_query = query_points.shape[0]
    joint_cdf_vals = np.empty(num_query)
    for i in range(num_query):
        joint_cdf_vals[i] = np.mean(np.all(residuals <= query_points[i], axis=1))
    out = {"data": residuals}
    return joint_cdf_vals, out


def get_empirical_copula_cdf(residuals, query_points=None):
    """Return the empirical copula CDF of the data.

    Parameters
    ----------
    residuals : array-like
        Data to be used to compute the empirical CDF, of shape (n, d).
        These will correspond to the calibration set residuals.
    query_points : array-like
        Points at which to evaluate the empirical CDF, of shape (m, d).

    Returns
    -------
    joint_cdf_vals : array-like
        Empirical CDF values, of shape (m,) and taking values in [0, 1].
    """

    u = get_pit(residuals, residuals)
    if query_points is None:
        eval_u = u
    else:
        # eval_u = pv.to_pseudo_obs(query_points, ties_method="random")
        eval_u = get_pit(residuals, query_points)

    class EmpiricalCopula:
        def cdf(self, pseudo_obs):
            num_query = pseudo_obs.shape[0]
            joint_cdf_vals = np.empty(num_query)
            for i in range(num_query):
                joint_cdf_vals[i] = np.mean(np.all(u <= pseudo_obs[i], axis=1))
            return joint_cdf_vals

    epmirical_copula = EmpiricalCopula()
    joint_cdf_vals = epmirical_copula.cdf(eval_u)

    out = {
        "copula": epmirical_copula,
        "data": residuals,
        "u": u,  # pseudo-observations on which the copula was fit
    }
    return joint_cdf_vals, out


def get_vine_copula_cdf(residuals, query_points=None, residuals_for_pit=None):
    """
    Get the copula CDF using a vine copula.

    Parameters
    ----------
    residuals : array-like
        Data to be used to fit the copula, of shape (n, d).
    query_points : array-like
        Points at which to evaluate the copula CDF, of shape (m, d).
    residuals_for_pit : array-like
        Data to be used to compute the PIT (build the marginal ECDFs),
        of shape (l, d).

    """
    if residuals_for_pit is None:
        residuals_for_pit = residuals
    target_dim = residuals.shape[1]
    controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.tll, pv.BicopFamily.indep],
            trunc_lvl=8 if target_dim > 3 else 999,
            # nonparametric_mult=1.0,
        )
    # u = pv.to_pseudo_obs(residuals, ties_method="random")
    u = get_pit(data=residuals_for_pit, query_points=residuals)
    if query_points is None:
        eval_u = u
    else:
        # eval_u = pv.to_pseudo_obs(query_points, ties_method="random")
        eval_u = get_pit(data=residuals_for_pit, query_points=query_points)
    copula = pv.Vinecop(u, controls=controls)
    joint_cdf_vals = copula.cdf(eval_u, N=10000, seeds=list(range(10000)))
    out = {
        "copula": copula,
        "data_for_pit": residuals_for_pit,
        "data": residuals,
        "u": u,  # pseudo-observations on which the copula was fit
    }
    return joint_cdf_vals, out


def get_pit(data, query_points):
    """Compute the probability integral transform (PIT) of the data."""
    n, d = query_points.shape
    pit = np.empty_like(query_points)
    for i in range(n):
        pit[i] = np.mean(data <= query_points[i], axis=0)
    return pit


def simulate_from_copula(copula, data, num_points):
    """
    Simulate points from a copula in the original data space.

    Parameters
    ----------
    copula : object
        A fitted copula object.
    data : array-like
        The original data used to fit the copula, of shape (n, d).
    num_points : int
        Number of points to simulate.
    """
    simulated_u = copula.simulate(num_points)
    simulated_y = map_u_to_y(simulated_u, data)
    return simulated_y


def map_u_to_y(u, data):
    target_dim = data.shape[1]
    # Get the marginal inverse CDF evaluated at u_i
    y = np.empty_like(u)
    for i in range(target_dim):
        min_y = np.min(data[:-1, i])
        max_y = np.max(data[:-1, i])
        y_grid = np.linspace(min_y, max_y, 100)  # shape (100,)
        emp_cdf, _ = get_empirical_cdf(
            data[:, [i]], y_grid[:, np.newaxis])  # shape (100,)
        sorted_idx = np.argsort(emp_cdf)  # shape (100,)
        y_i = np.interp(
            u[:, i],
            emp_cdf[sorted_idx],
            y_grid[sorted_idx])  # shape (num_points,)
        y[:, i] = y_i
    return y


def evaluate_copula_cdf(copula, data, query_points, marginal_cdf_fn=None):
    """
    Evaluate the copula CDF at query points.

    Parameters
    ----------
    copula : object
        A fitted copula object.
    data : array-like
        The original data used to fit the copula, of shape (n, d).
    query_points : array-like
        Points at which to evaluate the copula CDF, of shape (m, d).
    """
    if marginal_cdf_fn is not None:
        eval_u = marginal_cdf_fn(query_points)
    else:
        eval_u = get_pit(data, query_points)
    joint_cdf_vals = copula.cdf(eval_u)
    return joint_cdf_vals


def evaluate_copula_pdf(copula, data, query_points):
    """
    Evaluate the copula PDF at query points.

    Parameters
    ----------
    copula : object
        A fitted copula object.
    data : array-like
        The original data used to fit the copula, of shape (n, d).
    query_points : array-like
        Points at which to evaluate the copula CDF, of shape (m, d).
    """
    eval_u = get_pit(data, query_points)
    joint_cdf_vals = copula.pdf(eval_u)
    return joint_cdf_vals


# def get_gradient_cdf(eval_point, copula, data):
#     """
#     Get the gradient of the CDF at a point in the data space

#     Parameters
#     ----------
#     eval_point : array-like
#         The point at which to evaluate the gradient.
#     copula : object
#         The fitted copula object.
#     data : array-like
#         The original data used to fit the copula, of shape (n, d).
#     """
#     from scipy.optimize import approx_fprime
#     def fn(x):
#         x = np.atleast_2d(x)
#         return evaluate_copula_cdf(copula=copula, data=data, query_points=x)
#     grad = approx_fprime(eval_point, fn, 1.e-3)
#     return grad


def get_gradient_copula_cdf(eval_point, copula):
    """
    Get the gradient of the copula CDF at a point.

    Parameters
    ----------
    eval_point : array-like
        The point at which to evaluate the gradient.
    copula : object
        The fitted copula object.
    data : array-like
        The original data used to fit the copula, of shape (n, d).
    """
    from scipy.optimize import approx_fprime

    def fn(x):
        x = np.atleast_2d(x)
        x = np.clip(x, a_min=1.e-7, a_max=1 - 1.e-7)
        return copula.cdf(x, N=10000, num_threads=4, seeds=list(range(10000)))
    grad = approx_fprime(eval_point, fn, 1.e-3)
    return grad


# def get_vine_copula_cdf(residuals, query_points=None):
#     from copulas.multivariate import VineCopula
#     copula = VineCopula("regular")
#     copula.fit(residuals)
#     joint_cdf_vals = copula.cdf(eval_u)
#     return joint_cdf_vals


def get_kde_cdf(residuals, query_points=None):
    import statsmodels.api as sm
    target_dim = residuals.shape[1]
    if query_points is None:
        query_points = residuals
    kde_obj = sm.nonparametric.KDEMultivariate(
        residuals, var_type="c"*target_dim, bw="cv_ml")
    joint_cdf_vals = kde_obj.cdf(query_points)
    out = {}
    return joint_cdf_vals, out


def to_pseudo_obs(y, ties_method="average"):
    """Converts data to pseudo observations."""
    out = np.zeros_like(y)
    for i in range(y.shape[1]):
        x = y[:, i]
        n = x.size
        if ties_method == "average":
            pseudo_i = np.argsort(np.argsort(x)) / (n - 1)
        elif ties_method == "min":
            pseudo_i = np.argsort(x) / (n - 1)
        elif ties_method == "max":
            pseudo_i = np.argsort(x) / n
        elif ties_method == "dense":
            pseudo_i = np.linspace(0, 1, n)
        elif ties_method == "ordinal":
            pseudo_i = np.unique(x, return_inverse=True)[1] / (n - 1)
        elif ties_method == "random":
            pseudo_i = (np.argsort(x)/n + np.random.uniform(0, 1, n)*1.e-7)
        else:
            raise ValueError(f"Unknown ties_method: {ties_method}")
        out[:, i] = pseudo_i
    out = np.clip(out, 1.e-5, 1 - 1.e-5)
    return out


def plot_level_curves(
        random_vals, get_cdf_fn, levels=[0.5, 0.85, 0.9, 0.95],
        fig=None, ax=None, plotting_kwargs={"cmap": "jet"}):
    """Plot level curves of the empirical CDF."""
    lower = np.min(random_vals, axis=0)
    upper = np.max(random_vals, axis=0)
    # print(f"Plot level curves for range [{lower}, {upper}]")
    sx, sy = np.meshgrid(
        np.linspace(lower[0], upper[0], 100),
        np.linspace(lower[-1], upper[-1], 100))
    s = np.vstack([sx.ravel(), sy.ravel()]).T  # shape (10000, 2)
    # random_vals = np.random.randn(10000, 2)
    # print(random_vals.min(), random_vals.max())
    joint_cdf_vals, _ = get_cdf_fn(random_vals, query_points=s)
    joint_cdf_vals = joint_cdf_vals.reshape(100, 100)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    img = ax.contour(sx, sy, joint_cdf_vals, levels=levels, cmap=plotting_kwargs["cmap"])
    ax.clabel(img, colors='black', inline=True, inline_spacing=0, fontsize=8)
    # observed_cdf_vals, _ = get_cdf_fn(random_vals, query_points=random_vals)
    ax.scatter(
            random_vals[:, 0], random_vals[:, 1],
            color="tab:gray",
            marker=".",
            # c=observed_cdf_vals,
            alpha=0.2)
    ax.set_xlim([lower[0], upper[0]])
    ax.set_ylim([lower[1], upper[1]])
    # ax.set_xlabel("$S_1$")
    # ax.set_ylabel("$S_2$")
    # ax.set_title("Level curves of the empirical CDF")
    # fig.colorbar(img)
    # If we want colorbar for the level lines
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar = fig.colorbar(img, cax=cax, ax=ax, extend="min")
    # cbar.lines[0].set_linewidth(10)
    return fig, ax


def add_jitter(residuals, eps=1.e-7):
    """
    Add jitter to make the CDF continuous
    """
    unif = np.random.rand(*residuals.shape)*2.0 - 1.0
    return residuals + unif*eps


if __name__ == "__main__":
    plot_level_curves()

