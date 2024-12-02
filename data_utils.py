import os
import numpy as np
import pandas as pd
import openml
from ucimlrepo import fetch_ucirepo
from gauche.representations.fingerprints import fragments


def load_openml_data(name, standardize_Y=False):
    if name == "andro":
        # Dataset is too small -- only 49 instances
        id = 41474
        if not os.path.exists(os.path.join("data", f"{name}.csv")):
            dataset = openml.datasets.get_dataset(id)
            df = dataset.get_data()[0]
        input_cols = [col for col in df.columns if "Att" in col]
        input_cols = np.sort(input_cols)
        target_cols = [col for col in df.columns if "Target" in col]
    elif name in ["rf1", "rf2"]:
        # d = 8
        id = 41483
        if os.path.exists(os.path.join("data", f"{name}.csv")):
            df = pd.read_csv(os.path.join("data", f"{name}.csv"))
        else:
            dataset = openml.datasets.get_dataset(id)
            df = dataset.get_data()[0]
            df = df.dropna()
        input_cols = [col for col in df.columns if "48H" not in col]
        target_cols = [col for col in df.columns if "48H" in col]
    elif name in ["scm20d", "scm1d"]:
        # d = 16
        id = 41404
        if os.path.exists(os.path.join("data", f"{name}.csv")):
            df = pd.read_csv(os.path.join("data", f"{name}.csv"))
        else:
            dataset = openml.datasets.get_dataset(id)
            df = dataset.get_data()[0]
        input_cols = [col for col in df.columns if "MTLp" not in col and "LBL" not in col]
        input_cols.sort()
        target_cols = [col for col in df.columns if "MTLp" in col or "LBL" in col]
        target_cols.sort()
    elif name == "scpf":  # MAR dataset, d=3
        id = 41487
        if os.path.exists(os.path.join("data", f"{name}.csv")):
            df = pd.read_csv(os.path.join("data", f"{name}.csv"))
        else:
            dataset = openml.datasets.get_dataset(id)
            df = dataset.get_data()[0]
        target_cols = ["num_views", "num_votes", "num_comments"]
        input_cols = [col for col in df.columns if col not in target_cols]
        input_cols.sort()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    df.to_csv(os.path.join("data", f"{name}.csv"), index=False)
    X = df[input_cols].values.astype(np.float32)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = df[target_cols].values.astype(np.float32)
    metadata = get_metadata(Y)
    if standardize_Y:
        Y = (Y - metadata["Y_mean"]) / metadata["Y_std"]
    return X, Y, metadata


def load_uci_data(name, standardize_Y=False):
    """Load a dataset from the UCI repository."""

    # Fetch dataset
    if name == "solar_flare":
        id = 89
        # d = 2
    elif name == "energy_efficiency":
        id = 242
        # d = 2
    elif name == "stock_portfolio":
        id = 390
        # d = 6
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if (os.path.exists(os.path.join("data", f"{name}_X.csv")) and \
        os.path.exists(os.path.join("data", f"{name}_Y.csv"))):
        X = pd.read_csv(os.path.join("data", f"{name}_X.csv"))
        Y = pd.read_csv(os.path.join("data", f"{name}_Y.csv"))
    else:
        dataset = fetch_ucirepo(id=id)
        # Data (as pandas dataframes)
        X = dataset.data.features
        Y = dataset.data.targets

        for col in X.columns:
            if X[col].dtype == "object":
                X.loc[:, col] = X[col].astype("category").cat.codes

        for col in Y.columns:
            if Y[col].dtype == "object":
                Y.loc[:, col] = Y[col].astype("category").cat.codes

        X.to_csv(os.path.join("data", f"{name}_X.csv"), index=False)
        Y.to_csv(os.path.join("data", f"{name}_Y.csv"), index=False)

    # # variable information
    # print(dataset.variables)
    X = X.values.astype(np.float32)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = Y.values.astype(np.float32)
    # (Y.values - Y.values.mean(axis=0))/Y.values.std(axis=0)
    metadata = get_metadata(Y)
    if standardize_Y:
        Y = (Y - metadata["Y_mean"]) / metadata["Y_std"]
    return X, Y, metadata


def load_caco2_plus_data(standardize_Y=False):
    """Load the augmented Caco-2 dataset."""
    df = pd.read_csv(os.path.join("data", "caco2_plus.csv"))
    X = fragments(df["SMILES"].values).astype(np.float32)
    prop_names = ['Y', 'CrippenClogP', 'TPSA', 'QED',
       'ExactMolWt', 'FractionCSP3']  # , 'NumRotatableBonds']
    prop_names = prop_names
    Y = df[prop_names].values
    metadata = get_metadata(Y)
    if standardize_Y:
        Y = (Y - metadata["Y_mean"]) / metadata["Y_std"]
    return X, Y, metadata


def get_metadata(Y):
    """
    Get standardizing metadata
    """
    metadata = {}
    metadata["Y_mean"] = Y.mean(axis=0)
    metadata["Y_std"] = Y.std(axis=0)
    metadata["Y_max"] = Y.max(axis=0)
    metadata["Y_min"] = Y.min(axis=0)
    return metadata


def currin(X):
        x_0 = X[..., 0]
        x_1 = X[..., 1]
        factor1 = 1 - np.exp(-1 / (2 * x_1))
        numer = 2300 * x_0**3 + 1900 * x_0**2 + 2092 * x_0 + 60
        denom = 100 * x_0**3 + 500 * x_0**2 + 4 * x_0 + 20
        return factor1 * numer / denom


def branin_currin(X):
    def branin(X):
        t1 = (
            X[..., 1]
            - 5.1 / (4 * np.pi**2) * X[..., 0]**2.0
            + 5 / np.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(X[..., 0])
        return t1**2.0 + t2 + 10

    def rescaled_branin(X):
        # return to Branin bounds
        x_0 = 15 * X[..., 0] - 5
        x_1 = 15 * X[..., 1]
        return branin(np.stack([x_0, x_1], axis=-1))

    def evaluate_true(X):
        # branin rescaled with inputsto [0,1]^2
        branin_out = rescaled_branin(X=X)
        currin_out = currin(X=X)
        return np.stack([branin_out, currin_out], axis=-1)
    return evaluate_true(X)


def load_branin_currin(num_samples=200, seed=0, standardize_Y=False):
    """
    Load synthetic data
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(num_samples, 10)
    clean_Y = branin_currin(X)
    Y = clean_Y
    metadata = get_metadata(Y)
    if standardize_Y:
        Y = (Y - metadata["Y_mean"]) / metadata["Y_std"]
    return X, Y, metadata


def load_penicillin(num_samples=2000, seed=0, standardize_Y=False):
    """
    Load the Penicillin simulator data

    Parameters
    ----------
    num_samples : int
        Number of samples to generate

    """
    import torch
    import random
    from botorch.test_functions import Penicillin
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    penicillin = Penicillin()
    X = torch.rand(num_samples, penicillin.dim)
    X_bounds = penicillin.bounds.clone().detach()  # (2, d)
    X_input = X * (X_bounds[1] - X_bounds[0]) + X_bounds[0]  # (num_samples, 7)
    Y = penicillin(X_input)  # (num_samples, 3)
    X = X.numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = Y.numpy()
    Y = Y + 0.1 * Y * rng.standard_normal(Y.shape)
    metadata = get_metadata(Y)
    if standardize_Y:
        Y = (Y - metadata["Y_mean"]) / metadata["Y_std"]
    return X, Y, metadata


def load_data(dataset_name, seed=0, standardize_Y=False):
    """Load the data."""
    # Use:
    # ["caco2_plus", "solar_flare", "energy_efficiency", "stock_portfolio",
    # "scm1d", "scm20d", "rf1", "rf2"]
    os.makedirs("data", exist_ok=True)
    if dataset_name == "caco2_plus":
        return load_caco2_plus_data(standardize_Y=standardize_Y)
    elif dataset_name in ["solar_flare", "energy_efficiency", "stock_portfolio"]:
        return load_uci_data(dataset_name, standardize_Y=standardize_Y)
    elif dataset_name in ["andro", "scm1d", "scm20d", "rf1", "rf2"]:
        return load_openml_data(dataset_name, standardize_Y=standardize_Y)
    elif dataset_name == "branin_currin":
        return load_branin_currin(seed=seed, standardize_Y=standardize_Y)
    elif dataset_name == "penicillin":
        return load_penicillin(seed=seed, standardize_Y=standardize_Y)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_indices(num_data, seed=42, frac_train=0.33, frac_test=0.5, frac_val2=0.0):
    from sklearn.model_selection import train_test_split

    # Split data into training, val, and test
    train_indices, valtest_indices = train_test_split(
        range(num_data), test_size=1.0-frac_train, random_state=seed)
    val_indices, test_indices = train_test_split(
        valtest_indices, test_size=frac_test, random_state=seed)
    if frac_val2 > 0.0:
        val_indices, val2_indices = train_test_split(
            val_indices, test_size=frac_val2, random_state=seed)
    else:
        val2_indices = None
    indices = {
        "train": train_indices,
        "val": val_indices,
        "val2": val2_indices,
        "test": test_indices
    }
    return indices


if __name__ == "__main__":
    # OpenML
    load_openml_data("scm20d")
