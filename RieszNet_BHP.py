#!/usr/bin/env python3

import argparse
import math
import os

from joblib import load
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.special
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

from utils.NN_avgmom_sim import sim_fun
from utils.moments import avg_small_diff


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

METHODS = ["direct", "ips", "dr"]


def load_bhp_data():
    df = pd.read_csv("./data/BHP/data_BHP2.csv")
    df = df[df["log_p"] > math.log(1.2)]
    df = df[df["log_y"] > math.log(15000)]
    xdf = df.iloc[:, 1:]
    x_nostatedum = xdf.drop(["distance_oil1000", "share"], axis=1).values
    columns = xdf.columns
    state_dum = pd.get_dummies(xdf["state_fips"], prefix="state", dtype=float)
    xdf = pd.concat([xdf, state_dum], axis=1)
    xdf = xdf.drop(["distance_oil1000", "state_fips", "share"], axis=1)
    w = xdf.drop(["log_p"], axis=1).to_numpy(dtype=float)
    t = xdf["log_p"].to_numpy(dtype=float)

    _ = x_nostatedum
    _ = columns
    return w, t


def fit_treatment_models(w, t, n_estimators=100):
    mu_t = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=50, random_state=123
    )
    mu_t.fit(w, t)

    sigma2_t = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=50,
        max_depth=5,
        random_state=123,
    )
    e_t = t - cross_val_predict(mu_t, w, t)
    sigma2_t.fit(w, e_t ** 2)
    return mu_t, sigma2_t


def build_training_options(args):
    fast_train_opt = {
        "earlystop_rounds": args.fast_earlystop_rounds,
        "earlystop_delta": args.earlystop_delta,
        "learner_lr": args.fast_learner_lr,
        "learner_l2": args.learner_l2,
        "learner_l1": 0.0,
        "n_epochs": args.fast_n_epochs,
        "bs": args.batch_size,
        "target_reg": 1,
        "riesz_weight": 0.1,
        "optimizer": "adam",
    }
    train_opt = {
        "earlystop_rounds": args.train_earlystop_rounds,
        "earlystop_delta": args.earlystop_delta,
        "learner_lr": args.learner_lr,
        "learner_l2": args.learner_l2,
        "learner_l1": 0.0,
        "n_epochs": args.train_n_epochs,
        "bs": args.batch_size,
        "target_reg": 1,
        "riesz_weight": 0.1,
        "optimizer": "adam",
    }
    return fast_train_opt, train_opt


def run_seed(
    seed,
    w,
    mu_t,
    sigma2_t,
    *,
    n_sim,
    plot,
    max_designs,
    n_hidden,
    drop_prob,
    fast_train_opt,
    train_opt,
    parallel_jobs,
    parallel_verbose,
    torch_num_threads,
):
    np.random.seed(seed)

    b = np.random.uniform(-0.5, 0.5, size=(20, 1))
    c = np.random.uniform(-0.2, 0.2, size=(8, 1))

    def gen_t(w_):
        n = w_.shape[0]
        return (
            mu_t.predict(w_)
            + np.sqrt(sigma2_t.predict(w_)) * np.random.normal(size=(n,))
        ).reshape(-1, 1)

    def true_rr(x):
        return (x[:, 0] - mu_t.predict(x[:, 1:])) / (sigma2_t.predict(x[:, 1:]))

    def nonlin(x):
        return 1.5 * scipy.special.expit(10 * x[:, 6]) + 1.5 * scipy.special.expit(
            10 * x[:, 8]
        )

    def true_f_simple(x):
        return -0.6 * x[:, 0]

    def true_f_simple_lin_conf(x):
        return true_f_simple(x) + np.matmul(x[:, 1:21], b).flatten()

    def true_f_simple_nonlin_conf(x):
        return true_f_simple_lin_conf(x) + nonlin(x)

    def true_f_compl(x):
        return -0.5 * (x[:, 1] ** 2 / 10 + 0.5) * x[:, 0] ** 3 / 3

    def true_f_compl_lin_conf(x):
        return -0.5 * (
            x[:, 1] ** 2 / 10 + np.matmul(x[:, 1:9], c).flatten() + 0.5
        ) * x[:, 0] ** 3 / 3 + np.matmul(x[:, 1:21], b).flatten()

    def true_f_compl_nonlin_conf(x):
        return true_f_compl_lin_conf(x) + nonlin(x)

    true_fs = [
        true_f_simple,
        true_f_simple_lin_conf,
        true_f_simple_nonlin_conf,
        true_f_compl,
        true_f_compl_lin_conf,
        true_f_compl_nonlin_conf,
    ]
    if max_designs is not None:
        true_fs = true_fs[:max_designs]

    for true_f in true_fs:
        print("Now trying " + true_f.__name__)

        def gen_y(x):
            n = x.shape[0]
            return true_f(x) + np.random.normal(
                0, np.sqrt(5.6 * np.var(true_f(x))), size=(n,)
            )

        path = "./results/BHP/RieszNet/" + true_f.__name__
        if not os.path.exists(path):
            os.makedirs(path)

        namedata = path + "/seed_" + str(seed) + ".joblib"
        nameplot = path + "/seed_" + str(seed) + ".pdf"
        sim_fun(
            w,
            moment_fn=avg_small_diff,
            n_hidden=n_hidden,
            drop_prob=drop_prob,
            true_reg=true_f,
            true_rr=true_rr,
            gen_y=gen_y,
            gen_T=gen_t,
            N_sim=n_sim,
            fast_train_opt=fast_train_opt,
            train_opt=train_opt,
            seed=seed,
            verbose=parallel_verbose,
            plot=plot,
            save=namedata,
            saveplot=nameplot,
            parallel_n_jobs=parallel_jobs,
            torch_num_threads=torch_num_threads,
        )


def write_latex_table(aggregate_seeds):
    with open("./results/BHP/RieszNet/res_avg_der_NN.tex", "w") as f:
        f.write(
            "\\begin{tabular}{*{11}{r}} \n"
            + "\\toprule \n"
            + "&& \\multicolumn{3}{c}{Direct} & \\multicolumn{3}{c}{IPS} & \\multicolumn{3}{c}{DR} \\\\ \n"
            + "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \n"
            + "reg $R^2$ &  rr $R^2$ &  Bias &  RMSE &  Cov. &  Bias &  RMSE &  Cov. &  Bias &  RMSE &  Cov. \\\\ \n"
            + "\\midrule \n"
        )

        for f_i, true_f in enumerate(TRUE_FS):
            path = "./results/BHP/RieszNet/" + true_f
            f.write(
                "\\addlinespace \n \\multicolumn{11}{l}{\\textbf{"
                + F_STRING[f_i]
                + "}} \\\\ \n"
            )

            r2_reg, r2_rr = [], []
            res = {method: {"bias": [], "rmse": [], "cov": []} for method in METHODS}

            for i in aggregate_seeds:
                namedata = path + "/seed_" + str(i) + ".joblib"
                loaded = load(namedata)
                r2_reg = np.append(r2_reg, loaded[2])
                r2_rr = np.append(r2_rr, loaded[4])

                for method in METHODS:
                    res[method]["bias"].append(loaded[0][method]["bias"])
                    res[method]["rmse"].append(loaded[0][method]["rmse"])
                    res[method]["cov"].append(loaded[0][method]["cov"])

            f.write(
                " & ".join(["{:.3f}".format(np.mean(x)) for x in [r2_reg, r2_rr]])
                + " & "
            )
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


def write_histograms(aggregate_seeds):
    for true_f in TRUE_FS:
        path = "./results/BHP/RieszNet/" + true_f

        rmse_reg, r2_reg, rmse_rr, r2_rr, ipsbias, drbias, truth = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        res = {
            method: {"point": [], "bias": [], "rmse": [], "cov": []}
            for method in METHODS
        }

        for i in aggregate_seeds:
            namedata = path + "/seed_" + str(i) + ".joblib"
            loaded = load(namedata)
            rmse_reg = np.append(rmse_reg, loaded[1])
            r2_reg = np.append(r2_reg, loaded[2])
            rmse_rr = np.append(rmse_rr, loaded[3])
            r2_rr = np.append(r2_rr, loaded[4])
            ipsbias = np.append(ipsbias, loaded[5])
            drbias = np.append(drbias, loaded[6])
            truth = np.append(truth, loaded[7])

            for method in METHODS:
                res[method]["point"] = np.append(
                    res[method]["point"], loaded[0][method]["point"]
                )
                res[method]["bias"].append(loaded[0][method]["bias"])
                res[method]["rmse"].append(loaded[0][method]["rmse"])
                res[method]["cov"].append(loaded[0][method]["cov"])

        nuisance_str = (
            "reg RMSE: {:.3f}, R2: {:.3f}, rr RMSE: {:.3f}, R2: {:.3f}\n"
            "IPS orthogonality: {:.3f}, DR orthogonality: {:.3f}"
        ).format(
            np.mean(rmse_reg),
            np.mean(r2_reg),
            np.mean(rmse_rr),
            np.mean(r2_rr),
            np.mean(ipsbias),
            np.mean(drbias),
        )
        method_strs = [
            "{}. Bias: {:.3f}, RMSE: {:.3f}, Coverage: {:.3f}".format(
                method, np.mean(d["bias"]), np.mean(d["rmse"]), np.mean(d["cov"])
            )
            for method, d in res.items()
        ]
        plt.figure()
        plt.title("\n".join([nuisance_str] + method_strs))
        for method, d in res.items():
            plt.hist(np.array(d["point"]), alpha=0.5, label=method)
        plt.axvline(x=np.mean(truth), label="true", color="red")
        plt.legend()
        nameplot = path + "/all.pdf"
        plt.savefig(nameplot, bbox_inches="tight")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster-friendly Python script version of RieszNet_BHP.ipynb."
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
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=9)
    parser.add_argument("--aggregate-seed-start", type=int, default=0)
    parser.add_argument("--aggregate-seed-end", type=int, default=9)
    parser.add_argument("--n-sim", type=int, default=100)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--n-hidden", type=int, default=100)
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learner-lr", type=float, default=1e-4)
    parser.add_argument("--fast-learner-lr", type=float, default=1e-3)
    parser.add_argument("--learner-l2", type=float, default=1e-3)
    parser.add_argument("--earlystop-delta", type=float, default=1e-3)
    parser.add_argument("--fast-earlystop-rounds", type=int, default=2)
    parser.add_argument("--train-earlystop-rounds", type=int, default=20)
    parser.add_argument("--fast-n-epochs", type=int, default=100)
    parser.add_argument("--train-n-epochs", type=int, default=300)
    parser.add_argument("--treatment-n-estimators", type=int, default=100)
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=None,
        help="Number of worker processes used for Monte Carlo reps. Defaults to SLURM_CPUS_PER_TASK or 1.",
    )
    parser.add_argument(
        "--parallel-verbose",
        type=int,
        default=1,
        help="Verbosity level passed to joblib.Parallel inside the simulation helper.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Number of CPU threads used inside each PyTorch worker. Defaults to 1 when running under Slurm, otherwise leaves PyTorch defaults unchanged.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.parallel_jobs is None:
        args.parallel_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    if args.torch_num_threads is None and "SLURM_CPUS_PER_TASK" in os.environ:
        args.torch_num_threads = 1

    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)

    if args.smoke_test:
        if args.n_sim == 100:
            args.n_sim = 1
        if args.max_designs is None:
            args.max_designs = 1
        if not args.no_plot:
            args.no_plot = True
        if args.n_hidden == 100:
            args.n_hidden = 20
        if args.fast_n_epochs == 100:
            args.fast_n_epochs = 2
        if args.train_n_epochs == 300:
            args.train_n_epochs = 3
        if args.fast_earlystop_rounds == 2:
            args.fast_earlystop_rounds = 1
        if args.train_earlystop_rounds == 20:
            args.train_earlystop_rounds = 1
        if args.treatment_n_estimators == 100:
            args.treatment_n_estimators = 10
        if args.parallel_jobs is None or args.parallel_jobs > 1:
            args.parallel_jobs = 1
        if args.parallel_verbose == 1:
            args.parallel_verbose = 0

    if args.seed is None:
        run_seeds = list(range(args.seed_start, args.seed_end + 1))
    else:
        run_seeds = args.seed

    aggregate_seeds = list(range(args.aggregate_seed_start, args.aggregate_seed_end + 1))
    fast_train_opt, train_opt = build_training_options(args)

    if args.mode in ["all", "simulate"]:
        w, t = load_bhp_data()
        mu_t, sigma2_t = fit_treatment_models(
            w, t, n_estimators=args.treatment_n_estimators
        )
        for seed in run_seeds:
            run_seed(
                seed,
                w,
                mu_t,
                sigma2_t,
                n_sim=args.n_sim,
                plot=not args.no_plot,
                max_designs=args.max_designs,
                n_hidden=args.n_hidden,
                drop_prob=args.drop_prob,
                fast_train_opt=fast_train_opt,
                train_opt=train_opt,
                parallel_jobs=args.parallel_jobs,
                parallel_verbose=args.parallel_verbose,
                torch_num_threads=args.torch_num_threads,
            )

    if args.mode in ["all", "summarize"]:
        write_latex_table(aggregate_seeds)
        if not args.no_plot:
            write_histograms(aggregate_seeds)


if __name__ == "__main__":
    main()
