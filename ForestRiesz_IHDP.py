#!/usr/bin/env python3

import argparse
import glob
import os
from pathlib import Path

from joblib import dump, load
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.forestriesz import ForestRieszATE
from utils.ihdp_data import load_and_format_covariates, load_other_stuff


MAE_METHODS = ["dr", "direct", "ips", "plugin"]
COVERAGE_METHODS = ["dr", "direct", "ips"]

MAE_BASE_DIR = Path("./results/IHDP/ForestRiesz/MAE")
MAE_SHARD_DIR = MAE_BASE_DIR / "shards"
COVERAGE_BASE_DIR = Path("./results/IHDP/ForestRiesz/coverage")
COVERAGE_SHARD_DIR = COVERAGE_BASE_DIR / "shards"


def rmse_fn(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def forest_riesz_ate_options(n_estimators, n_jobs):
    return {
        "criterion": "het",
        "n_estimators": n_estimators,
        "min_samples_leaf": 2,
        "min_var_fraction_leaf": 0.001,
        "min_var_leaf_on_val": True,
        "min_impurity_decrease": 0.01,
        "max_samples": 0.8,
        "max_depth": None,
        "warm_start": False,
        "inference": False,
        "subforest_size": 1,
        "honest": True,
        "verbose": 0,
        "n_jobs": n_jobs,
        "random_state": 123,
    }


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def select_simulation_files(data_base_dir, nsims, *, double_draw=False):
    simulation_files = sorted(glob.glob(f"{data_base_dir}/*.csv"))
    if nsims > len(simulation_files):
        raise ValueError(
            f"Requested nsims={nsims}, but only found {len(simulation_files)} files in {data_base_dir}."
        )

    rng = np.random.RandomState(123)
    sim_ids = rng.choice(len(simulation_files), nsims, replace=False)
    if double_draw:
        # Preserve the notebook's second sampling pass in the coverage section.
        sim_ids = rng.choice(len(simulation_files), nsims, replace=False)
    return [simulation_files[i] for i in sim_ids]


def get_shard_members(total_items, n_shards, shard):
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}.")
    if shard < 0 or shard >= n_shards:
        raise ValueError(f"shard must be in [0, {n_shards - 1}], got {shard}.")
    members = np.array_split(np.arange(total_items), n_shards)[shard]
    return [int(i) for i in members.tolist()]


def resolve_shards(shards, shard_start, shard_end, n_shards):
    if shards is None:
        resolved = list(range(shard_start, shard_end + 1))
    else:
        resolved = shards

    if not resolved:
        raise ValueError("No shards selected.")

    for shard in resolved:
        if shard < 0 or shard >= n_shards:
            raise ValueError(f"Shard {shard} is outside [0, {n_shards - 1}].")
    return resolved


def run_single_simulation(simulation_file, methods, est_options):
    x = load_and_format_covariates(simulation_file, delimiter=" ")
    t, y, _, mu_0, mu_1 = load_other_stuff(simulation_file, delimiter=" ")
    X = np.c_[t, x]
    true_ate = float(np.mean(mu_1 - mu_0))

    y_scaler = StandardScaler(with_mean=True).fit(y)
    y_scaled = y_scaler.transform(y).reshape(-1, 1)

    est = ForestRieszATE(**est_options)
    est.fit(X[:, 1:], X[:, [0]], y_scaled)

    params = (
        tuple(
            value * y_scaler.scale_[0]
            for method in methods
            for value in est.predict_ate(X, y_scaled, method=method)
        )
        + (true_ate,)
    )
    return params


def run_shard(simulation_files, methods, est_options, *, shard, n_shards, label, output_path):
    shard_members = get_shard_members(len(simulation_files), n_shards, shard)
    print(f"[{label}] shard {shard + 1}/{n_shards} running {len(shard_members)} datasets")

    results = []
    shard_files = [simulation_files[i] for i in shard_members]
    for idx, simulation_file in enumerate(shard_files, start=1):
        print(
            f"[{label}] shard {shard + 1}/{n_shards} dataset {idx}/{len(shard_files)}: "
            f"{Path(simulation_file).name}"
        )
        results.append(run_single_simulation(simulation_file, methods, est_options))

    ensure_dir(Path(output_path).parent)
    dump(
        {
            "label": label,
            "shard": shard,
            "n_shards": n_shards,
            "methods": methods,
            "simulation_files": shard_files,
            "simulation_file_indices": shard_members,
            "results": results,
        },
        output_path,
    )
    print(f"[{label}] wrote shard output to {output_path}")


def summarize_results(results, methods, metric_mode):
    if not results:
        raise ValueError("No results available to summarize.")

    res = tuple(np.array(x) for x in zip(*results))
    truth = np.asarray(res[-1], dtype=float)
    res_dict = {}

    for it, method in enumerate(methods):
        point, lb, ub = (np.asarray(x, dtype=float) for x in res[it * 3 : (it + 1) * 3])
        method_result = {"point": point, "lb": lb, "ub": ub}
        if metric_mode == "mae":
            abs_err = np.abs(point - truth)
            method_result["MAE"] = float(np.mean(abs_err))
            method_result["std. err."] = float(np.std(abs_err) / np.sqrt(len(abs_err)))
        elif metric_mode == "coverage":
            method_result["cov"] = float(np.mean(np.logical_and(truth >= lb, truth <= ub)))
            method_result["bias"] = float(np.mean(point - truth))
            method_result["rmse"] = float(rmse_fn(point, truth))
        else:
            raise ValueError(f"Unknown metric_mode={metric_mode}.")
        res_dict[method] = method_result

    return res_dict, truth


def load_shard_results(shard_dir, prefix, shards):
    results = []
    for shard in shards:
        shard_path = Path(shard_dir) / f"{prefix}_shard_{shard}.joblib"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard output: {shard_path}")
        payload = load(shard_path)
        results.extend(payload["results"])
    return results


def write_mae_table(res_dict):
    ensure_dir(MAE_BASE_DIR)
    methods_str = [
        "DR",
        "Direct",
        "IPS",
        "\\midrule \n\\multicolumn{2}{l}{\\textbf{Benchmark:}} \\\\ \nRF Plug-in",
    ]

    with open(MAE_BASE_DIR / "IHDP_MAE_RF.tex", "w") as f:
        f.write(
            "\\begin{tabular}{lc} \n"
            "\\toprule \n"
            "& MAE $\\pm$ std. err. \\\\ \n"
            "\\midrule \n"
            "\\multicolumn{2}{l}{\\textbf{Auto-DML:}} \\\\ \n"
        )

        for i, method in enumerate(MAE_METHODS):
            f.write(
                " & ".join(
                    [
                        methods_str[i],
                        "{:.3f} $\\pm$ {:.3f}".format(
                            res_dict[method]["MAE"], res_dict[method]["std. err."]
                        ),
                    ]
                )
                + " \\\\ \n"
            )

        f.write("\\bottomrule \n \\end{tabular}")


def write_coverage_plot(res_dict, truth):
    ensure_dir(COVERAGE_BASE_DIR)
    plt.figure()
    method_strs = [
        "{}. Bias: {:.3f}, RMSE: {:.3f}, Coverage: {:.3f}".format(
            method,
            d["bias"],
            d["rmse"],
            d["cov"],
        )
        for method, d in res_dict.items()
    ]
    plt.title("\n".join(method_strs))
    for method, d in res_dict.items():
        plt.hist(np.array(d["point"]), alpha=0.5, label=method)
    plt.axvline(x=float(np.mean(truth)), label="true", color="red")
    plt.legend()
    plt.savefig(COVERAGE_BASE_DIR / "IHDP_coverage_RF.pdf", bbox_inches="tight")
    plt.close()


def simulate_mae(shards, n_shards, nsims, n_estimators, n_jobs):
    selected_files = select_simulation_files("./data/IHDP/sim_data", nsims)
    est_options = forest_riesz_ate_options(n_estimators=n_estimators, n_jobs=n_jobs)
    ensure_dir(MAE_SHARD_DIR)

    for shard in shards:
        run_shard(
            selected_files,
            MAE_METHODS,
            est_options,
            shard=shard,
            n_shards=n_shards,
            label="MAE",
            output_path=MAE_SHARD_DIR / f"mae_shard_{shard}.joblib",
        )


def simulate_coverage(shards, n_shards, nsims, n_estimators, n_jobs):
    selected_files = select_simulation_files(
        "./data/IHDP/sim_data_redraw_T", nsims, double_draw=True
    )
    est_options = forest_riesz_ate_options(n_estimators=n_estimators, n_jobs=n_jobs)
    ensure_dir(COVERAGE_SHARD_DIR)

    for shard in shards:
        run_shard(
            selected_files,
            COVERAGE_METHODS,
            est_options,
            shard=shard,
            n_shards=n_shards,
            label="coverage",
            output_path=COVERAGE_SHARD_DIR / f"coverage_shard_{shard}.joblib",
        )


def summarize_mae(shards):
    results = load_shard_results(MAE_SHARD_DIR, "mae", shards)
    res_dict, _ = summarize_results(results, MAE_METHODS, metric_mode="mae")
    ensure_dir(MAE_BASE_DIR)
    dump(res_dict, MAE_BASE_DIR / "IHDP_MAE_RF.joblib")
    write_mae_table(res_dict)
    for method in MAE_METHODS:
        print(
            "{} : MAE = {:.3f} +/- {:.3f}".format(
                method, res_dict[method]["MAE"], res_dict[method]["std. err."]
            )
        )


def summarize_coverage(shards, plot=True):
    results = load_shard_results(COVERAGE_SHARD_DIR, "coverage", shards)
    res_dict, truth = summarize_results(results, COVERAGE_METHODS, metric_mode="coverage")
    ensure_dir(COVERAGE_BASE_DIR)
    dump(res_dict, COVERAGE_BASE_DIR / "IHDP_coverage_RF.joblib")
    if plot:
        write_coverage_plot(res_dict, truth)
    for method in COVERAGE_METHODS:
        print(
            "{} : bias = {:.3f}, rmse = {:.3f}, cov = {:.3f}".format(
                method,
                res_dict[method]["bias"],
                res_dict[method]["rmse"],
                res_dict[method]["cov"],
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster-friendly Python script version of ForestRiesz_IHDP.ipynb."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "simulate", "summarize"],
        default="all",
        help="all: run shard simulations then summarize; simulate: run only selected shards; summarize: build final outputs from shard files.",
    )
    parser.add_argument(
        "--shard",
        type=int,
        action="append",
        default=None,
        help="Shard to run. Repeat to pass multiple shards. If omitted, uses --shard-start..--shard-end.",
    )
    parser.add_argument(
        "--shard-start", type=int, default=0, help="Start shard (inclusive) for simulation."
    )
    parser.add_argument(
        "--shard-end", type=int, default=9, help="End shard (inclusive) for simulation."
    )
    parser.add_argument(
        "--aggregate-shard-start",
        type=int,
        default=0,
        help="Start shard (inclusive) used when building summary outputs.",
    )
    parser.add_argument(
        "--aggregate-shard-end",
        type=int,
        default=9,
        help="End shard (inclusive) used when building summary outputs.",
    )
    parser.add_argument(
        "--n-shards", type=int, default=10, help="Total number of shards used for the simulation."
    )
    parser.add_argument(
        "--mae-nsims",
        type=int,
        default=1000,
        help="Number of IHDP sim_data datasets used for the MAE experiment.",
    )
    parser.add_argument(
        "--coverage-nsims",
        type=int,
        default=100,
        help="Number of IHDP sim_data_redraw_T datasets used for the coverage experiment.",
    )
    parser.add_argument(
        "--mae-n-estimators",
        type=int,
        default=1000,
        help="Number of trees used for the MAE experiment.",
    )
    parser.add_argument(
        "--coverage-n-estimators",
        type=int,
        default=100,
        help="Number of trees used for the coverage experiment.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of worker threads used inside each forest fit. Defaults to SLURM_CPUS_PER_TASK or 1.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip histogram generation during summarize/all runs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast end-to-end check with tiny simulation counts and smaller forests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.n_jobs is None:
        args.n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    if args.smoke_test:
        if args.mae_nsims == 1000:
            args.mae_nsims = 2
        if args.coverage_nsims == 100:
            args.coverage_nsims = 2
        if args.mae_n_estimators == 1000:
            args.mae_n_estimators = 20
        if args.coverage_n_estimators == 100:
            args.coverage_n_estimators = 20
        if not args.no_plot:
            args.no_plot = True
        if args.shard is None and args.shard_end == 9 and args.n_shards == 1:
            args.shard_end = 0
        if args.aggregate_shard_end == 9 and args.n_shards == 1:
            args.aggregate_shard_end = 0

    if args.mode in ["all", "simulate"]:
        simulate_shards = resolve_shards(
            args.shard, args.shard_start, args.shard_end, args.n_shards
        )
        simulate_mae(
            simulate_shards,
            args.n_shards,
            args.mae_nsims,
            args.mae_n_estimators,
            args.n_jobs,
        )
        simulate_coverage(
            simulate_shards,
            args.n_shards,
            args.coverage_nsims,
            args.coverage_n_estimators,
            args.n_jobs,
        )

    if args.mode in ["all", "summarize"]:
        aggregate_shards = resolve_shards(
            None, args.aggregate_shard_start, args.aggregate_shard_end, args.n_shards
        )
        summarize_mae(aggregate_shards)
        summarize_coverage(aggregate_shards, plot=not args.no_plot)


if __name__ == "__main__":
    main()
