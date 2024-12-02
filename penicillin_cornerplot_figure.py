import numpy as np
import matplotlib.pyplot as plt
import random
import scienceplots
plt.style.use('science')


def get_cornerplot_3d(data, labels, title=None, figsize=(6, 6), **kwargs):
    fig, ax = plt.subplots(3, 3, figsize=figsize, **kwargs)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(3):
        for j in range(3):
            if i == j:
                ax[i, j].hist(data[:, i], bins=30, density=True, color="tab:gray", histtype="step")
                # ax[i, j].set_xlabel(labels[i])
                # ax[i, j].set_ylabel("Density")
            elif i > j:
                ax[i, j].scatter(data[:, j], data[:, i], color="tab:gray", marker=".", alpha=0.2)
                # ax[i, j].set_xlabel(labels[j])
                # ax[i, j].set_ylabel(labels[i])
    ax[0, 0].set_ylabel("Density")
    ax[1, 0].set_ylabel(labels[1])
    ax[2, 0].set_ylabel(labels[2])
    ax[2, 0].set_xlabel(labels[0])
    ax[2, 1].set_xlabel(labels[1])
    ax[2, 2].set_xlabel(labels[2])
    ax[1, 2].remove()
    ax[0, 1].remove()
    ax[0, 2].remove()
    if title is not None:
        fig.suptitle(title)
    return fig, ax


if __name__ == "__main__":
    import torch
    from data_utils import load_penicillin, get_indices
    from run_experiments import fit_multitask_lasso

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Prepare data
    X, Y, metadata = load_penicillin(num_samples=1024)
    indices = get_indices(
        X.shape[0], seed=seed,
        frac_train=0.6, frac_test=0.5, frac_val2=0.0)
    train_indices = indices["train"]
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    val_indices = indices["val"]
    X_val = X[val_indices]
    Y_val = Y[val_indices]
    labels = ["Yield", "Time", "Byproduct"]
    residuals_labels = [f"Score ({c})" for c in labels]

    # Fit the model
    clf = fit_multitask_lasso(X_train, Y_train)
    uncalib_pred = clf.predict(X_val)
    residuals = np.abs(uncalib_pred - Y_val)

    fig, ax = get_cornerplot_3d(Y_val, labels)
    fig.savefig("figures/cornerplot.png", bbox_inches="tight")
    fig.savefig("figures/cornerplot.pdf", bbox_inches="tight")

    plt.close("all")
    fig, ax = get_cornerplot_3d(residuals, residuals_labels)
    fig.savefig("figures/cornerplot_residuals.png", bbox_inches="tight")
    fig.savefig("figures/cornerplot_residuals.pdf", bbox_inches="tight")
