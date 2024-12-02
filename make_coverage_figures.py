import numpy as np
import matplotlib.pyplot as plt
import random
import scienceplots
plt.style.use('science')


pretty_method_names = {
    "Independent": "Independent",
    "Scalar score": "Scalar score ($L_2$)",
    "Scalar score (order=1)": "Scalar score ($L_1$)",
    "Scalar score (order=inf)": "Scalar score ($L_\infty$)",
    "Empirical": "Empirical copula",
    "Plug-in (ours)": "Plug-in (ours)",
    "Corrected (ours)": "Corrected (ours)"
}

method_colors = {
    "Independent": "tab:green",
    "Scalar score": "tab:blue",
    "Scalar score (order=1)": "tab:orange",
    "Scalar score (order=inf)": "tab:red",
    "Empirical": "k",
    "Plug-in (ours)": "tab:purple",
    "Corrected (ours)": "tab:pink"
}


def get_coverage_vs_alpha_plot(alpha_metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    offset = np.linspace(-0.04, 0.04, len(pretty_method_names))
    for alpha, (mean, stderr) in alpha_metrics.items():
        i = 0
        for _, (method, _) in enumerate(mean["coverage"].items()):
            # if method not in ["Plug-in (ours)", "Corrected (ours)"]:
            #     continue
            if alpha == 0.1:
                ax.errorbar(
                    1 - alpha + offset[i],
                    # med["coverage"][method],
                    mean["coverage"][method],
                    yerr=stderr["coverage"][method]*np.sqrt(10),
                    fmt=".", label=pretty_method_names[method],
                    color=method_colors[method],
                    )
            else:
                ax.errorbar(
                    1 - alpha + offset[i],
                    mean["coverage"][method],
                    yerr=stderr["coverage"][method]*np.sqrt(10),
                    fmt=".",
                    color=method_colors[method],
                    )
            i += 1
    ax.set_xlabel(r"Nominal coverage, 1 - $\alpha$")
    ax.set_ylabel("Empirical coverage")
    # grid = np.linspace(0, 1, 20)
    # ax.plot(grid, grid, "--", color="tab:gray")
    for alpha in alpha_metrics.keys():
        ax.plot(
            [1 - alpha + offset[0] - 0.01,
             1 - alpha + offset[-1] + 0.01],
            [1 - alpha, 1 - alpha], "--", color="tab:gray")
    ax.legend()
    ax.set_xticks([1 - alpha for alpha in alpha_metrics.keys()])
    ax.set_xticklabels([f"{1 - alpha:.1f}" for alpha in alpha_metrics.keys()])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    return fig, ax


def get_coverage_vs_alpha_plot_only_ours(alpha_metrics):
    fig, ax = plt.subplots()
    offset = np.linspace(-0.01, 0.01, 2)
    for alpha, (mean, stderr) in alpha_metrics.items():
        i = 0
        for _, (method, _) in enumerate(mean["coverage"].items()):
            if method not in ["Plug-in (ours)", "Corrected (ours)"]:
                continue
            if alpha == 0.1:
                ax.errorbar(
                    1 - alpha + offset[i],
                    # med["coverage"][method],
                    mean["coverage"][method],
                    yerr=stderr["coverage"][method]*np.sqrt(10),
                    fmt=".", label=pretty_method_names[method],
                    color=method_colors[method],
                    )
            else:
                ax.errorbar(
                    1 - alpha + offset[i],
                    mean["coverage"][method],
                    yerr=stderr["coverage"][method]*np.sqrt(10),
                    fmt=".",
                    color=method_colors[method],
                    )
            i += 1
    ax.set_xlabel(r"Nominal coverage, 1 - $\alpha$")
    ax.set_ylabel("Empirical coverage")
    # grid = np.linspace(0, 1, 20)
    # ax.plot(grid, grid, "--", color="tab:gray")
    for alpha in alpha_metrics.keys():
        if alpha == 0.1:
            ax.plot(
            [1 - alpha + offset[0] - 0.02,
             1 - alpha + offset[-1] + 0.02],
            [1 - alpha, 1 - alpha], "--", color="tab:gray",
            label="Exact coverage")
        else:
            ax.plot(
                [1 - alpha + offset[0] - 0.02,
                1 - alpha + offset[-1] + 0.02],
                [1 - alpha, 1 - alpha], "--", color="tab:gray")
    ax.legend()
    ax.set_yticks([1 - alpha for alpha in alpha_metrics.keys()])
    ax.set_yticklabels([f"{1 - alpha:.1f}" for alpha in alpha_metrics.keys()])
    ax.set_xticks([1 - alpha for alpha in alpha_metrics.keys()])
    ax.set_xticklabels([f"{1 - alpha:.1f}" for alpha in alpha_metrics.keys()])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    return fig, ax


def get_coverage_diff_vs_alpha_plot(alpha_metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    # offset = np.linspace(-0.04, 0.04, len(pretty_method_names))
    offset = np.linspace(-0.01, 0.01, 2)
    for alpha, (mean, stderr, med, lower, upper) in alpha_metrics.items():
        i = 0
        for _, (method, _) in enumerate(mean["coverage"].items()):
            if method not in ["Plug-in (ours)", "Corrected (ours)"]:
                continue
            if alpha == 0.1:
                ax.errorbar(
                    1 - alpha + offset[i],
                    # med["coverage"][method],
                    mean["coverage"][method] - (1 - alpha),
                    yerr=stderr["coverage"][method]*np.sqrt(10),
                    fmt="o", label=pretty_method_names[method],
                    color=method_colors[method],
                    )
            else:
                ax.errorbar(
                    1 - alpha + offset[i],
                    mean["coverage"][method] - (1 - alpha),
                    yerr=stderr["coverage"][method]*np.sqrt(10),
                    fmt="o",
                    color=method_colors[method],
                    )
            i += 1
    ax.set_xlabel(r"Nominal coverage, 1 - $\alpha$")
    ax.set_ylabel("Empirical - nominal coverage")
    # grid = np.linspace(0, 1, 20)
    # ax.plot(grid, grid, "--", color="tab:gray")
    ax.axhline(0, linestyle="--", color="tab:gray")
    ax.legend(loc="upper left")
    ax.set_xticks([1 - alpha for alpha in alpha_metrics.keys()])
    ax.set_xticklabels([f"{1 - alpha:.1f}" for alpha in alpha_metrics.keys()])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    return fig, ax


def get_coverage_vs_n_plot(n_metrics):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    offset = np.linspace(-5, 5, len(pretty_method_names))
    for val2_ratio, (mean, stderr, med, lower, upper) in n_metrics.items():
        for i, (method, _) in enumerate(mean["coverage"].items()):
            n = int(1150 * (1 - val2_ratio))
            if val2_ratio == 0.8:
                ax.errorbar(
                    n + offset[i],
                    mean["coverage"][method],
                    yerr=stderr["coverage"][method],
                    # med["coverage"][method],
                    # yerr=[
                    #     [med["coverage"][method] - lower["coverage"][method]],
                    #     [upper["coverage"][method] - med["coverage"][method]]
                    #     ],
                    fmt="o", label=pretty_method_names[method],
                    color=method_colors[method])
            else:
                ax.errorbar(
                    n + offset[i],
                    mean["coverage"][method],
                    yerr=stderr["coverage"][method],
                    # med["coverage"][method],
                    # yerr=[
                    #     [med["coverage"][method] - lower["coverage"][method]],
                    #     [upper["coverage"][method] - med["coverage"][method]]
                    #     ],
                    fmt="o",
                    color=method_colors[method])
    ax.set_xticks([int(1150 * (1 - val2_ratio)) for val2_ratio in n_metrics.keys()])
    ax.set_xticklabels([str(int(1150 * (1 - val2_ratio))) for val2_ratio in n_metrics.keys()])
    ax.set_xlabel("Calibration set size $n$")
    ax.set_ylabel("Empirical coverage")
    ax.axhline(0.9, linestyle="--", color="tab:gray", label="Target level")
    ax.legend(ncol=2, loc=(0.2, 0.55))
    return fig, ax


if __name__ == "__main__":
    import os
    from run_experiments import summarize_metrics_across_splits

    # # Plot emp vs. nominal coverage with varying alpha
    # alpha_metrics = {}
    # for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     metrics = np.load(
    #         os.path.join(
    #             "results_20241007",
    #             # f"coverage_vs_alpha_{alpha:.1f}_rf1",
    #             # "rf1_standardize_Y_False.npy"),
    #             f"coverage_vs_alpha_{alpha:.1f}",
    #             "penicillin_standardize_Y_False.npy"),
    #         allow_pickle=True).item()
    #     alpha_metrics[alpha] = summarize_metrics_across_splits(metrics)

    # fig, ax = get_coverage_vs_alpha_plot_only_ours(alpha_metrics)
    # fig.savefig("figures/coverage_varying_alpha_pen_v5.pdf", bbox_inches="tight")
    # fig.savefig("figures/coverage_varying_alpha_pen_v5.png", bbox_inches="tight")
    # # plt.show()

    # # Plot emp vs. nominal coverage with varying alpha
    # alpha_metrics = {}
    # for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     metrics = np.load(
    #         os.path.join(
    #             "results_20241007",
    #             f"coverage_vs_alpha_{alpha:.1f}",
    #             "penicillin_standardize_Y_False.npy"),
    #             # f"coverage_vs_alpha_{alpha:.1f}_rf1",
    #             # "rf1_standardize_Y_False.npy"),
    #         allow_pickle=True).item()
    #     alpha_metrics[alpha] = summarize_metrics_across_splits(
    #         metrics, include_quantiles=True)

    # fig, ax = get_coverage_diff_vs_alpha_plot(alpha_metrics)
    # fig.savefig("figures/coverage_varying_alpha_diff.pdf", bbox_inches="tight")
    # fig.savefig("figures/coverage_varying_alpha_diff.png", bbox_inches="tight")
    # # plt.show()

    # Plot emp vs. nominal coverage with varying n
    n_metrics = {}
    alpha = 0.1
    for frac_val2 in [0.8, 0.85, 0.9, 0.95]:
        metrics = np.load(
            os.path.join(
                "rebuttals_1201",
                f"coverage_vs_alpha_{alpha:.1f}_frac_val2_{frac_val2:.1f}",
                "penicillin_standardize_Y_True.npy"),
            allow_pickle=True).item()
        n_metrics[frac_val2] = summarize_metrics_across_splits(
            metrics, include_quantiles=True)

    fig, ax = get_coverage_vs_n_plot(n_metrics)
    fig.savefig("figures/coverage_varying_n_v3.pdf", bbox_inches="tight")
    fig.savefig("figures/coverage_varying_n_v3.png", bbox_inches="tight")
