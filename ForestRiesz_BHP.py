#!/usr/bin/env python3

import argparse
import os
import glob
from joblib import dump, load
import pandas as pd
import scipy
import scipy.stats
import scipy.special
import numpy as np
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

from utils.forestriesz import poly_feature_fns
from utils.RF_avgmom_sim import sim_fun


def moment_fn(X, test_fn):
    epsilon = 0.01
    t1 = np.hstack([X[:, [0]] + epsilon, X[:, 1:]])
    t0 = np.hstack([X[:, [0]] - epsilon, X[:, 1:]])
    return (test_fn(t1) - test_fn(t0)) / (2 * epsilon)


# RFRiesz Settings
ForestRiesz_opt = {
    "reg_feature_fns": poly_feature_fns(1),
    "riesz_feature_fns": poly_feature_fns(3),
    "moment_fn": moment_fn,
    "l2": 1e-3,
    "criterion": "mse",
    "n_estimators": 100,
    "min_samples_leaf": 50,
    "min_var_fraction_leaf": 0.1,
    "min_var_leaf_on_val": True,
    "min_impurity_decrease": 0.001,
    "max_samples": 0.65,
    "max_depth": None,
    "warm_start": False,
    "inference": False,
    "subforest_size": 1,
    "honest": True,
    "verbose": 0,
    "n_jobs": -1,
    "random_state": 572,
}

# RFreg Settings
RFreg_opt = {
    "reg_feature_fns": poly_feature_fns(1),
    "l2": 1e-3,
    "criterion": "mse",
    "n_estimators": 100,
    "min_samples_leaf": 50,
    "min_var_fraction_leaf": 0.1,
    "min_var_leaf_on_val": True,
    "min_impurity_decrease": 0.001,
    "max_samples": 0.65,
    "max_depth": None,
    "warm_start": False,
    "inference": False,
    "subforest_size": 1,
    "honest": True,
    "verbose": 0,
    "n_jobs": -1,
    "random_state": 572,
}

# RFrr Settings
RFrr_opt = {
    "riesz_feature_fns": poly_feature_fns(3),
    "moment_fn": moment_fn,
    "l2": 1e-3,
    "criterion": "mse",
    "n_estimators": 100,
    "min_samples_leaf": 50,
    "min_var_fraction_leaf": 0.1,
    "min_var_leaf_on_val": True,
    "min_impurity_decrease": 0.001,
    "max_samples": 0.65,
    "max_depth": None,
    "warm_start": False,
    "inference": False,
    "subforest_size": 1,
    "honest": True,
    "verbose": 0,
    "n_jobs": -1,
    "random_state": 572,
}


F_STRING = [
    "1. Simple $f$",
    "2. Simple $f$ with linear confound.",
    "3. Simple $f$ with linear and non-linear confound.",
    "4. Complex $f$",
    "5. Complex $f$ with linear confound.",
    "6. Complex $f$ with linear and non-linear confound.",
]

TRUE_FS = [
    "true_f_simple",
    "true_f_simple_lin_conf",
    "true_f_simple_nonlin_conf",
    "true_f_compl",
    "true_f_compl_lin_conf",
    "true_f_compl_nonlin_conf",
]

METHODS = ["reg", "ips", "dr", "tmle"]
XFIT_MULT = [(0, True), (0, False), (1, True), (1, False), (2, False)]
CONFIGS_PER_SEED = len(TRUE_FS) * len(XFIT_MULT)


def load_bhp_data():
    df = pd.read_csv("./data/BHP/data_BHP2.csv")
    df = df[df["log_p"] > math.log(1.2)]
    df = df[df["log_y"] > math.log(15000)]
    Xdf = df.iloc[:, 1:]
    X_nostatedum = Xdf.drop(["distance_oil1000", "share"], axis=1).values
    columns = Xdf.columns
    state_dum = pd.get_dummies(Xdf["state_fips"], prefix="state", dtype=float)
    Xdf = pd.concat([Xdf, state_dum], axis=1)
    Xdf = Xdf.drop(["distance_oil1000", "state_fips", "share"], axis=1)
    W = Xdf.drop(["log_p"], axis=1).to_numpy(dtype=float)
    T = Xdf["log_p"].to_numpy(dtype=float)

    # Keep notebook variables untouched in behavior.
    _ = X_nostatedum
    _ = columns
    return W, T


def fit_treatment_models(W, T, n_estimators=100, n_jobs=None):
    # Conditional Mean
    mu_T = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=50,
        random_state=572,
        n_jobs=n_jobs,
    )
    mu_T.fit(W, T)

    # Conditional Variance
    sigma2_T = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=50,
        max_depth=5,
        random_state=572,
        n_jobs=n_jobs,
    )
    e_T = T - cross_val_predict(mu_T, W, T)
    sigma2_T.fit(W, e_T ** 2)
    return mu_T, sigma2_T


def build_seed_context(seed, mu_T, sigma2_T):
    rng = np.random.RandomState(seed)
    b = rng.uniform(-0.5, 0.5, size=(20, 1))
    c = rng.uniform(-0.2, 0.2, size=(8, 1))

    def gen_T(W_):
        n = W_.shape[0]
        return (
            mu_T.predict(W_)
            + np.sqrt(sigma2_T.predict(W_)) * np.random.normal(size=(n,))
        ).reshape(-1, 1)

    def true_rr(X):
        return (X[:, 0] - mu_T.predict(X[:, 1:])) / sigma2_T.predict(X[:, 1:])

    def nonlin(X):
        return 1.5 * scipy.special.expit(10 * X[:, 6]) + 1.5 * scipy.special.expit(10 * X[:, 8])

    def true_f_simple(X):
        return -0.6 * X[:, 0]

    def true_f_simple_lin_conf(X):
        return true_f_simple(X) + np.matmul(X[:, 1:21], b).flatten()

    def true_f_simple_nonlin_conf(X):
        return true_f_simple_lin_conf(X) + nonlin(X)

    def true_f_compl(X):
        return -0.5 * (X[:, 1] ** 2 / 10 + 0.5) * X[:, 0] ** 3 / 3

    def true_f_compl_lin_conf(X):
        return (
            -0.5
            * (X[:, 1] ** 2 / 10 + np.matmul(X[:, 1:9], c).flatten() + 0.5)
            * X[:, 0] ** 3
            / 3
            + np.matmul(X[:, 1:21], b).flatten()
        )

    def true_f_compl_nonlin_conf(X):
        return true_f_compl_lin_conf(X) + nonlin(X)

    true_fs = [
        true_f_simple,
        true_f_simple_lin_conf,
        true_f_simple_nonlin_conf,
        true_f_compl,
        true_f_compl_lin_conf,
        true_f_compl_nonlin_conf,
    ]
    return gen_T, true_rr, true_fs


def run_configuration(
    seed,
    W,
    mu_T,
    sigma2_T,
    *,
    design_index,
    config_index,
    n_sim=100,
    plot=True,
    forest_riesz_opt,
    rfreg_opt,
    rfrr_opt,
    sim_n_jobs=-1,
):
    gen_T, true_rr, true_fs = build_seed_context(seed, mu_T, sigma2_T)
    true_f = true_fs[design_index]
    xfit, mult = XFIT_MULT[config_index]

    print(
        "Running seed {}, design {}, xfit {}, multitasking {}".format(
            seed, true_f.__name__, xfit, int(mult)
        )
    )

    def gen_y(X):
        n = X.shape[0]
        return true_f(X) + np.random.normal(
            0, np.sqrt(5.6 * np.var(true_f(X))), size=(n,)
        )

    path = "./results/BHP/ForestRiesz/" + true_f.__name__
    if not os.path.exists(path):
        os.makedirs(path)

    namedata = (
        path
        + "/xfit_"
        + str(xfit)
        + "_mult_"
        + str(int(mult))
        + "_seed_"
        + str(seed)
        + ".joblib"
    )
    nameplot = (
        path
        + "/xfit_"
        + str(xfit)
        + "_mult_"
        + str(int(mult))
        + "_seed_"
        + str(seed)
        + ".pdf"
    )
    sim_fun(
        W,
        moment_fn=moment_fn,
        true_reg=true_f,
        true_rr=true_rr,
        gen_y=gen_y,
        gen_T=gen_T,
        N_sim=n_sim,
        xfit=xfit,
        multitasking=mult,
        ForestRiesz_opt=forest_riesz_opt,
        RFreg_opt=rfreg_opt,
        RFrr_opt=rfrr_opt,
        seed=seed,
        verbose=0,
        plot=plot,
        save=namedata,
        saveplot=nameplot,
        n_jobs=sim_n_jobs,
    )


def run_seed(
    seed,
    W,
    mu_T,
    sigma2_T,
    *,
    n_sim=100,
    plot=True,
    max_designs=None,
    max_xfit_mult=None,
    forest_riesz_opt,
    rfreg_opt,
    rfrr_opt,
    sim_n_jobs=-1,
):
    design_indices = list(range(len(TRUE_FS)))
    config_indices = list(range(len(XFIT_MULT)))
    if max_designs is not None:
        design_indices = design_indices[:max_designs]
    if max_xfit_mult is not None:
        config_indices = config_indices[:max_xfit_mult]

    for design_index in design_indices:
        for config_index in config_indices:
            run_configuration(
                seed,
                W,
                mu_T,
                sigma2_T,
                design_index=design_index,
                config_index=config_index,
                n_sim=n_sim,
                plot=plot,
                forest_riesz_opt=forest_riesz_opt,
                rfreg_opt=rfreg_opt,
                rfrr_opt=rfrr_opt,
                sim_n_jobs=sim_n_jobs,
            )


def get_total_task_count(seed_start, seed_end):
    return (seed_end - seed_start + 1) * CONFIGS_PER_SEED


def decode_task_index(task_index, seed_start, seed_end):
    total_tasks = get_total_task_count(seed_start, seed_end)
    if task_index < 0 or task_index >= total_tasks:
        raise ValueError(
            f"task_index must be in [0, {total_tasks - 1}], got {task_index}."
        )

    seed_offset, within_seed = divmod(task_index, CONFIGS_PER_SEED)
    design_index, config_index = divmod(within_seed, len(XFIT_MULT))
    return seed_start + seed_offset, design_index, config_index


def resolve_task_indices(task_indices, task_start, task_end, seed_start, seed_end):
    total_tasks = get_total_task_count(seed_start, seed_end)
    if task_indices is None:
        resolved_end = total_tasks - 1 if task_end is None else task_end
        resolved = list(range(task_start, resolved_end + 1))
    else:
        resolved = task_indices

    if not resolved:
        raise ValueError("No task indices selected.")

    for task_index in resolved:
        if task_index < 0 or task_index >= total_tasks:
            raise ValueError(
                f"Task index {task_index} is outside [0, {total_tasks - 1}]."
            )
    return resolved


def write_latex_table(aggregate_seeds):
    with open("./results/BHP/ForestRiesz/res_avg_der_RF.tex", "w") as f:
        f.write(
            "\\begin{tabular}{*{16}{r}} \n"
            + "\\toprule \n"
            + "&&&& \\multicolumn{3}{c}{Direct} & \\multicolumn{3}{c}{IPS} & \\multicolumn{3}{c}{DR} & \\multicolumn{3}{c}{DR + post-TMLE} \\\\ \n"
            + "\\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16} \n"
            + "x-fit & multit. & reg $R^2$ &  rr $R^2$ &  Bias &  RMSE &  Cov. &  Bias &  RMSE &  Cov. &  Bias &  RMSE &  Cov. &  Bias &  RMSE &  Cov. \\\\ \n"
            + "\\midrule \n"
        )

        for f_i, true_f in enumerate(TRUE_FS):
            path = "./results/BHP/ForestRiesz/" + true_f
            f.write("\\addlinespace \n \\multicolumn{16}{l}{\\textbf{" + F_STRING[f_i] + "}} \\\\ \n")

            for xfit, mult in XFIT_MULT:
                mult_str = "Yes" if mult else "No"

                f.write(" & ".join([str(xfit), mult_str]) + " & ")

                r2_reg, r2_rr = [], []
                res = {}
                for method in METHODS:
                    res[method] = {"bias": [], "rmse": [], "cov": []}

                for i in aggregate_seeds:
                    namedata = (
                        path
                        + "/xfit_"
                        + str(xfit)
                        + "_mult_"
                        + str(int(mult))
                        + "_seed_"
                        + str(i)
                        + ".joblib"
                    )
                    loaded = load(namedata)
                    r2_reg = np.append(r2_reg, loaded[2])
                    r2_rr = np.append(r2_rr, loaded[4])

                    for method in METHODS:
                        res[method]["bias"].append(loaded[0][method]["bias"])
                        res[method]["rmse"].append(loaded[0][method]["rmse"])
                        res[method]["cov"].append(loaded[0][method]["cov"])

                f.write(" & ".join(["{:.3f}".format(np.mean(x)) for x in [r2_reg, r2_rr]]) + " & ")
                f.write(
                    " & ".join(
                        [
                            "{:.3f}".format(np.mean(res[method][x]))
                            for method in METHODS
                            for x in ["bias", "rmse", "cov"]
                        ]
                    )
                    + " \\\\ \n"
                )

        f.write("\\bottomrule \n \\end{tabular}")


def write_plugin_table(aggregate_seeds):
    with open("./results/BHP/plugin.tex", "w") as f:
        f.write(
            "\\begin{tabular}{*{5}{r}} \n"
            + "\\toprule \n"
            + "&& \\multicolumn{3}{c}{RF Plug-in} \\\\ \n"
            + "\\cmidrule(lr){3-5} \n"
            + "reg $R^2$ &  rr $R^2$ &  Bias &  RMSE &  Cov. \\\\ \n"
            + "\\midrule \n"
        )

        for f_i, true_f in enumerate(TRUE_FS):
            path = "./results/BHP/ForestRiesz/" + true_f
            f.write("\\addlinespace \n \\multicolumn{5}{l}{\\textbf{" + F_STRING[f_i] + "}} \\\\ \n")

            r2_reg, r2_rr = [], []
            plugin = {"bias": [], "rmse": [], "cov": []}

            for i in aggregate_seeds:
                namedata = path + "/xfit_0_mult_0_seed_" + str(i) + ".joblib"
                loaded = load(namedata)
                r2_reg = np.append(r2_reg, loaded[2])
                r2_rr = np.append(r2_rr, loaded[4])
                plugin["bias"].append(loaded[0]["plugin"]["bias"])
                plugin["rmse"].append(loaded[0]["plugin"]["rmse"])
                plugin["cov"].append(loaded[0]["plugin"]["cov"])

            f.write(" & ".join(["{:.3f}".format(np.mean(x)) for x in [r2_reg, r2_rr]]) + " & ")
            f.write(" & ".join(["{:.3f}".format(np.mean(plugin[x])) for x in ["bias", "rmse", "cov"]]) + " \\\\ \n")

        f.write("\\bottomrule \n \\end{tabular}")


def write_histograms(aggregate_seeds):
    for true_f in TRUE_FS:
        path = "./results/BHP/ForestRiesz/" + true_f

        for xfit, mult in XFIT_MULT:
            rmse_reg, r2_reg, rmse_rr, r2_rr, ipsbias, drbias, truth = [], [], [], [], [], [], []
            res = {}

            for method in METHODS:
                res[method] = {"point": [], "bias": [], "rmse": [], "cov": []}

            for i in aggregate_seeds:
                namedata = (
                    path
                    + "/xfit_"
                    + str(xfit)
                    + "_mult_"
                    + str(int(mult))
                    + "_seed_"
                    + str(i)
                    + ".joblib"
                )
                loaded = load(namedata)
                rmse_reg = np.append(rmse_reg, loaded[1])
                r2_reg = np.append(r2_reg, loaded[2])
                rmse_rr = np.append(rmse_rr, loaded[3])
                r2_rr = np.append(r2_rr, loaded[4])
                ipsbias = np.append(ipsbias, loaded[5])
                drbias = np.append(drbias, loaded[6])
                truth = np.append(truth, loaded[7])

                for method in METHODS:
                    res[method]["point"] = np.append(res[method]["point"], loaded[0][method]["point"])
                    res[method]["bias"].append(loaded[0][method]["bias"])
                    res[method]["rmse"].append(loaded[0][method]["rmse"])
                    res[method]["cov"].append(loaded[0][method]["cov"])

            plot_methods = ["dr", "reg", "ips", "tmle"]
            method_strs = [
                "{}. Bias: {:.3f}, RMSE: {:.3f}, Coverage: {:.3f}".format(
                    method,
                    np.mean(res[method]["bias"]),
                    np.mean(res[method]["rmse"]),
                    np.mean(res[method]["cov"]),
                )
                for method in plot_methods
            ]
            fig = plt.figure()
            plt.title("\n".join(method_strs))
            for method in plot_methods:
                d = res[method]
                plt.hist(np.array(d["point"]), alpha=0.5, label=method)
            plt.axvline(x=np.mean(truth), label="true", color="red")
            handles, labels = plt.gca().get_legend_handles_labels()
            label_to_handle = dict(zip(labels, handles))
            legend_order = ["true"] + plot_methods
            plt.legend(
                [label_to_handle[label] for label in legend_order],
                legend_order,
            )
            nameplot = path + "/xfit_" + str(xfit) + "_mult_" + str(int(mult)) + "_all.pdf"
            plt.savefig(nameplot, bbox_inches="tight")
            plt.show()
            plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Python script version of ForestRiesz_BHP.ipynb with cluster-friendly seed selection."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "simulate", "summarize"],
        default="all",
        help="all: run simulation then summaries; simulate: run only selected seeds; summarize: run summary outputs only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Seed to simulate. Repeat option to pass multiple seeds. If omitted, uses --seed-start..--seed-end.",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed (inclusive) for simulation.")
    parser.add_argument("--seed-end", type=int, default=9, help="End seed (inclusive) for simulation.")
    parser.add_argument(
        "--aggregate-seed-start",
        type=int,
        default=0,
        help="Start seed (inclusive) used when building summary outputs.",
    )
    parser.add_argument(
        "--aggregate-seed-end",
        type=int,
        default=9,
        help="End seed (inclusive) used when building summary outputs.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        action="append",
        default=None,
        help="Tiny simulation task index. Repeat to pass multiple tasks. If omitted, uses --task-start..--task-end when task mode is enabled.",
    )
    parser.add_argument(
        "--task-start",
        type=int,
        default=0,
        help="Start task index (inclusive) for tiny simulation mode.",
    )
    parser.add_argument(
        "--task-end",
        type=int,
        default=None,
        help="End task index (inclusive) for tiny simulation mode. Defaults to the last available task.",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=100,
        help="Number of Monte Carlo replications per design/configuration.",
    )
    parser.add_argument(
        "--max-designs",
        type=int,
        default=None,
        help="Optional cap on how many response designs to run, in the notebook order.",
    )
    parser.add_argument(
        "--max-xfit-mult",
        type=int,
        default=None,
        help="Optional cap on how many xfit/multitasking configurations to run, in the notebook order.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip histogram generation during simulation runs.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override the number of trees used by the random-forest-based estimators.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for Monte Carlo replications. Positive values also pin each forest fit to one worker to avoid nested oversubscription.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    use_task_mode = args.task_index is not None or args.task_end is not None

    if args.seed is None:
        run_seeds = list(range(args.seed_start, args.seed_end + 1))
    else:
        run_seeds = args.seed

    if use_task_mode and args.seed is not None:
        raise ValueError("--task-index/--task-start/--task-end cannot be combined with --seed.")

    aggregate_seeds = list(range(args.aggregate_seed_start, args.aggregate_seed_end + 1))
    task_indices = resolve_task_indices(
        args.task_index, args.task_start, args.task_end, args.seed_start, args.seed_end
    ) if use_task_mode else None

    forest_riesz_opt = ForestRiesz_opt.copy()
    rfreg_opt = RFreg_opt.copy()
    rfrr_opt = RFrr_opt.copy()
    treatment_n_estimators = 100
    treatment_n_jobs = None

    if args.n_estimators is not None:
        forest_riesz_opt["n_estimators"] = args.n_estimators
        rfreg_opt["n_estimators"] = args.n_estimators
        rfrr_opt["n_estimators"] = args.n_estimators
        treatment_n_estimators = args.n_estimators

    if args.n_jobs > 0:
        forest_riesz_opt["n_jobs"] = 1
        rfreg_opt["n_jobs"] = 1
        rfrr_opt["n_jobs"] = 1
        treatment_n_jobs = 1

    if args.mode in ["all", "simulate"]:
        W, T = load_bhp_data()
        mu_T, sigma2_T = fit_treatment_models(
            W,
            T,
            n_estimators=treatment_n_estimators,
            n_jobs=treatment_n_jobs,
        )
        if use_task_mode:
            for task_index in task_indices:
                seed, design_index, config_index = decode_task_index(
                    task_index, args.seed_start, args.seed_end
                )
                run_configuration(
                    seed,
                    W,
                    mu_T,
                    sigma2_T,
                    design_index=design_index,
                    config_index=config_index,
                    n_sim=args.n_sim,
                    plot=not args.no_plot,
                    forest_riesz_opt=forest_riesz_opt,
                    rfreg_opt=rfreg_opt,
                    rfrr_opt=rfrr_opt,
                    sim_n_jobs=args.n_jobs,
                )
        else:
            for i in run_seeds:
                run_seed(
                    i,
                    W,
                    mu_T,
                    sigma2_T,
                    n_sim=args.n_sim,
                    plot=not args.no_plot,
                    max_designs=args.max_designs,
                    max_xfit_mult=args.max_xfit_mult,
                    forest_riesz_opt=forest_riesz_opt,
                    rfreg_opt=rfreg_opt,
                    rfrr_opt=rfrr_opt,
                    sim_n_jobs=args.n_jobs,
                )

    if args.mode in ["all", "summarize"]:
        write_latex_table(aggregate_seeds)
        write_plugin_table(aggregate_seeds)
        write_histograms(aggregate_seeds)


if __name__ == "__main__":
    main()
