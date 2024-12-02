import numpy as np
from run_experiments import summarize_metrics_across_splits


if __name__ == "__main__":
    dataset_names = [
        "penicillin",
        # "solar_flare",
        # "energy_efficiency",
        "stock_portfolio",
        "caco2_plus",
        "rf1",
        "rf2",
        "scm1d",
        "scm20d",
        # "scpf"
        ]

    for dataset_name in dataset_names:
        print(dataset_name)
        for standardize_Y in [True]:
            for alpha in [0.1]:
                # metrics = np.load(
                #     f"results_20241010/{dataset_name}_standardize_Y_{standardize_Y}.npy",
                #         allow_pickle=True).item()
                metrics = np.load(
                    f"gaussian_cde_results/{dataset_name}_standardize_Y_{standardize_Y}.npy",
                        allow_pickle=True).item()
                mean_metrics, stderr_metrics = summarize_metrics_across_splits(metrics)
                for metric_name in mean_metrics.keys():
                    for method_name in mean_metrics[metric_name].keys():
                        print(f"{metric_name} ({method_name}): "
                            f"{mean_metrics[metric_name][method_name]:.3f} "
                            f"± {stderr_metrics[metric_name][method_name]:.3f}")
