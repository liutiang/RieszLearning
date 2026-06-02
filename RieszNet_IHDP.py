#!/usr/bin/env python3

import argparse
import gc
import glob
import os
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
from pathlib import Path

from joblib import dump, load
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.ihdp_data import load_and_format_covariates, load_other_stuff
from utils.moments import ate_moment_fn
from utils.riesznet import RieszNet, RieszNetRR


MAE_METHODS = ["dr", "direct", "ips"]
MAE_METHOD_LABELS = ["DR", "Direct", "IPS"]
ABLATION_TABLE_METHODS = ["direct", "ips", "dr"]
SRR_MAP = {"dr": True, "direct": False, "ips": True}
ROBUSTNESS_METHODS = ["dr"]

MAE_BASE_DIR = Path("./results/IHDP/RieszNet/MAE")
MAE_SHARD_DIR = MAE_BASE_DIR / "shards"
COVERAGE_BASE_DIR = Path("./results/IHDP/RieszNet/coverage")
COVERAGE_SHARD_DIR = COVERAGE_BASE_DIR / "shards"
ABLATION_BASE_DIR = Path("./results/IHDP/RieszNet/ablation")
ABLATION_SHARD_DIR = ABLATION_BASE_DIR / "shards"
ROBUSTNESS_BASE_DIR = Path("./results/IHDP/RieszNet/robustness")
ROBUSTNESS_SHARD_DIR = ROBUSTNESS_BASE_DIR / "shards"
MODEL_DIR = Path.cwd() / ".riesznet_models" / "ihdp"

ROBUSTNESS_VARIANTS = [
    ("baseline", "Default"),
    ("ten_x_lr", "10x learning rates"),
    ("one_tenth_lr", "1/10 learning rates"),
    ("ten_x_weights", "10x target_reg + riesz_weight"),
    ("one_tenth_weights", "1/10 target_reg + riesz_weight"),
]


def rmse_fn(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def select_simulation_files(data_base_dir, nsims, base_seed):
    simulation_files = sorted(glob.glob(f"{data_base_dir}/*.csv"))
    if nsims > len(simulation_files):
        raise ValueError(
            f"Requested nsims={nsims}, but only found {len(simulation_files)} files in {data_base_dir}."
        )

    rng = np.random.RandomState(base_seed)
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


def get_device():
    return torch.cuda.current_device() if torch.cuda.is_available() else None


def configure_torch(torch_num_threads):
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_model(model):
    tempdir = getattr(model, "tempdir", None)
    if tempdir is not None:
        tempdir.cleanup()


def _combinations(n_features, degree, interaction_only):
    comb = combinations if interaction_only else combinations_w_r
    return chain.from_iterable(comb(range(n_features), i) for i in range(0, degree + 1))


class Learner(nn.Module):
    def __init__(self, n_t, n_hidden, p, degree, interaction_only=False):
        super().__init__()
        n_common = 200
        self.monomials = list(_combinations(n_t, degree, interaction_only))
        self.common = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_t, n_common),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ELU(),
        )
        self.riesz_nn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))
        self.reg_nn0 = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_common, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, 1),
        )
        self.reg_nn1 = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_common, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, 1),
        )
        self.reg_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))

    def forward(self, x):
        poly = torch.cat(
            [torch.prod(x[:, t], dim=1, keepdim=True) for t in self.monomials], dim=1
        )
        feats = self.common(x)
        riesz = self.riesz_nn(feats) + self.riesz_poly(poly)
        reg = (
            self.reg_nn0(feats) * (1 - x[:, [0]])
            + self.reg_nn1(feats) * x[:, [0]]
            + self.reg_poly(poly)
        )
        return torch.cat([reg, riesz], dim=1)


class SharedFeatureRieszLearner(nn.Module):
    def __init__(self, n_t, p):
        super().__init__()
        n_common = 200
        self.common = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_t, n_common),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ELU(),
        )
        self.riesz_nn0 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_nn1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))

    def forward(self, x):
        feats = self.common(x)
        riesz = self.riesz_nn0(feats) * (1 - x[:, [0]]) + self.riesz_nn1(feats) * x[:, [0]]
        return torch.cat([riesz, feats], dim=1)


class SharedFeatureRegLearner(nn.Module):
    def __init__(self, n_hidden, p):
        super().__init__()
        n_common = 200
        self.reg_nn0 = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_common, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, 1),
        )
        self.reg_nn1 = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_common, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        t = x[:, [0]]
        riesz = x[:, [1]]
        feats = x[:, 2:]
        reg = self.reg_nn0(feats) * (1 - t) + self.reg_nn1(feats) * t
        return torch.cat([reg, riesz], dim=1)


class SeparateRieszLearner(nn.Module):
    def __init__(self, n_t, p):
        super().__init__()
        n_common = 200
        self.common = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_t, n_common),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ELU(),
        )
        self.riesz_nn0 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_nn1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))

    def forward(self, x):
        feats = self.common(x)
        return self.riesz_nn0(feats) * (1 - x[:, [0]]) + self.riesz_nn1(feats) * x[:, [0]]


class SeparateRegLearner(nn.Module):
    def __init__(self, n_t, n_hidden, p):
        super().__init__()
        n_common = 200
        self.common = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_t, n_common),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ELU(),
        )
        self.reg_nn0 = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_common, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, 1),
        )
        self.reg_nn1 = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_common, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        X = x[:, 0:-1]
        riesz = x[:, [-1]]
        feats = self.common(X)
        reg = self.reg_nn0(feats) * (1 - X[:, [0]]) + self.reg_nn1(feats) * X[:, [0]]
        return torch.cat([reg, riesz], dim=1)


def build_train_options(args):
    main_fast = {
        "earlystop_rounds": args.fast_earlystop_rounds,
        "earlystop_delta": args.earlystop_delta,
        "learner_lr": args.fast_learner_lr,
        "learner_l2": args.learner_l2,
        "learner_l1": args.learner_l1,
        "n_epochs": args.fast_n_epochs,
        "bs": args.batch_size,
        "target_reg": args.target_reg,
        "riesz_weight": args.riesz_weight,
        "optimizer": "adam",
    }
    main_train = {
        "earlystop_rounds": args.train_earlystop_rounds,
        "earlystop_delta": args.earlystop_delta,
        "learner_lr": args.learner_lr,
        "learner_l2": args.learner_l2,
        "learner_l1": args.learner_l1,
        "n_epochs": args.train_n_epochs,
        "bs": args.batch_size,
        "target_reg": args.target_reg,
        "riesz_weight": args.riesz_weight,
        "optimizer": "adam",
    }
    rr_fast = {
        "earlystop_rounds": args.fast_earlystop_rounds,
        "earlystop_delta": args.rr_earlystop_delta,
        "learner_lr": args.rr_fast_learner_lr,
        "learner_l2": args.learner_l2,
        "learner_l1": args.learner_l1,
        "n_epochs": args.fast_n_epochs,
        "bs": args.batch_size,
        "optimizer": "adam",
    }
    rr_train = {
        "earlystop_rounds": args.train_earlystop_rounds,
        "earlystop_delta": args.rr_earlystop_delta,
        "learner_lr": args.rr_learner_lr,
        "learner_l2": args.learner_l2,
        "learner_l1": args.learner_l1,
        "n_epochs": args.train_n_epochs,
        "bs": args.batch_size,
        "optimizer": "adam",
    }
    return main_fast, main_train, rr_fast, rr_train


def scale_main_learning_rates(fast_cfg, train_cfg, factor):
    scaled_fast = fast_cfg.copy()
    scaled_train = train_cfg.copy()
    scaled_fast["learner_lr"] *= factor
    scaled_train["learner_lr"] *= factor
    return scaled_fast, scaled_train


def build_robustness_variants(args, main_fast_cfg, main_train_cfg):
    lr_fast_cfg, lr_train_cfg = scale_main_learning_rates(main_fast_cfg, main_train_cfg, 10.0)
    low_lr_fast_cfg, low_lr_train_cfg = scale_main_learning_rates(
        main_fast_cfg, main_train_cfg, 0.1
    )
    return [
        {
            "name": "baseline",
            "label": "Default",
            "n_hidden": args.n_hidden,
            "target_reg": args.target_reg,
            "riesz_weight": args.riesz_weight,
            "fast_cfg": main_fast_cfg.copy(),
            "train_cfg": main_train_cfg.copy(),
        },
        {
            "name": "ten_x_lr",
            "label": "10x learning rates",
            "n_hidden": args.n_hidden,
            "target_reg": args.target_reg,
            "riesz_weight": args.riesz_weight,
            "fast_cfg": lr_fast_cfg,
            "train_cfg": lr_train_cfg,
        },
        {
            "name": "one_tenth_lr",
            "label": "1/10 learning rates",
            "n_hidden": args.n_hidden,
            "target_reg": args.target_reg,
            "riesz_weight": args.riesz_weight,
            "fast_cfg": low_lr_fast_cfg,
            "train_cfg": low_lr_train_cfg,
        },
        {
            "name": "ten_x_weights",
            "label": "10x target_reg + riesz_weight",
            "n_hidden": args.n_hidden,
            "target_reg": args.target_reg * 10.0,
            "riesz_weight": args.riesz_weight * 10.0,
            "fast_cfg": main_fast_cfg.copy(),
            "train_cfg": main_train_cfg.copy(),
        },
        {
            "name": "one_tenth_weights",
            "label": "1/10 target_reg + riesz_weight",
            "n_hidden": args.n_hidden,
            "target_reg": args.target_reg * 0.1,
            "riesz_weight": args.riesz_weight * 0.1,
            "fast_cfg": main_fast_cfg.copy(),
            "train_cfg": main_train_cfg.copy(),
        },
    ]


def fit_riesznet(
    learner,
    moment_fn,
    x_train,
    y_train,
    x_test,
    y_test,
    fast_cfg,
    train_cfg,
    *,
    device,
    target_reg=None,
    riesz_weight=None,
):
    agmm = RieszNet(learner, moment_fn)
    fast_cfg = fast_cfg.copy()
    train_cfg = train_cfg.copy()
    if target_reg is not None:
        fast_cfg["target_reg"] = target_reg
        train_cfg["target_reg"] = target_reg
    if riesz_weight is not None:
        fast_cfg["riesz_weight"] = riesz_weight
        train_cfg["riesz_weight"] = riesz_weight

    agmm.fit(
        x_train,
        y_train,
        Xval=x_test,
        yval=y_test,
        model_dir=str(MODEL_DIR),
        device=device,
        verbose=0,
        **fast_cfg,
    )
    agmm.fit(
        x_train,
        y_train,
        Xval=x_test,
        yval=y_test,
        warm_start=True,
        model_dir=str(MODEL_DIR),
        device=device,
        verbose=0,
        **train_cfg,
    )
    return agmm


def fit_riesz_rr(learner, moment_fn, x_train, x_test, fast_cfg, train_cfg, *, device):
    rrnn = RieszNetRR(learner, moment_fn)
    rrnn.fit(
        x_train,
        Xval=x_test,
        model_dir=str(MODEL_DIR),
        device=device,
        verbose=0,
        **fast_cfg,
    )
    rrnn.fit(
        x_train,
        Xval=x_test,
        warm_start=True,
        model_dir=str(MODEL_DIR),
        device=device,
        verbose=0,
        **train_cfg,
    )
    return rrnn


def load_ihdp_sample(simulation_file):
    x = load_and_format_covariates(simulation_file, delimiter=" ")
    t, y, _, mu_0, mu_1 = load_other_stuff(simulation_file, delimiter=" ")
    X = np.c_[t, x]
    true_ate = float(np.mean(mu_1 - mu_0))
    return X, y, true_ate


def scale_outcome(y):
    y_scaler = StandardScaler(with_mean=True).fit(y)
    y_scaled = y_scaler.transform(y)
    return y_scaled, float(y_scaler.scale_[0])


def run_standard_experiment(
    simulation_file,
    sim_seed,
    args,
    fast_cfg,
    train_cfg,
    *,
    device,
    target_reg,
    riesz_weight,
    prediction_kwargs,
    n_hidden=None,
    methods=None,
):
    set_all_seeds(sim_seed)
    X, y, true_ate = load_ihdp_sample(simulation_file)
    y_scaled, y_scale = scale_outcome(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=sim_seed
    )

    torch.cuda.empty_cache()
    hidden_units = args.n_hidden if n_hidden is None else n_hidden
    methods = MAE_METHODS if methods is None else methods
    learner = Learner(X_train.shape[1], hidden_units, args.drop_prob, 0, interaction_only=True)
    agmm = fit_riesznet(
        learner,
        ate_moment_fn,
        X_train,
        y_train,
        X_test,
        y_test,
        fast_cfg,
        train_cfg,
        device=device,
        target_reg=target_reg,
        riesz_weight=riesz_weight,
    )
    params = (
        tuple(
            value * y_scale
            for method in methods
            for value in agmm.predict_avg_moment(
                X,
                y_scaled,
                model="earlystop",
                method=method,
                srr=prediction_kwargs.get("srr", SRR_MAP).get(method, True),
                postTMLE=prediction_kwargs.get("postTMLE", False),
            )
        )
        + (true_ate,)
    )
    cleanup_model(agmm)
    del agmm, learner
    gc.collect()
    torch.cuda.empty_cache()
    return params


def run_shared_ablation(simulation_file, sim_seed, args, main_fast_cfg, main_train_cfg, rr_fast_cfg, rr_train_cfg, *, device):
    set_all_seeds(sim_seed)
    X, y, true_ate = load_ihdp_sample(simulation_file)
    y_scaled, y_scale = scale_outcome(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=sim_seed
    )

    torch.cuda.empty_cache()
    rrlearner = SharedFeatureRieszLearner(X_train.shape[1], args.drop_prob)
    rrnn = fit_riesz_rr(
        rrlearner, ate_moment_fn, X_train, X_test, rr_fast_cfg, rr_train_cfg, device=device
    )

    reglearner = SharedFeatureRegLearner(args.n_hidden, args.drop_prob)
    regnn = RieszNet(reglearner, ate_moment_fn)

    inputs = np.hstack((X[:, [0]], rrnn.predict(X, model="earlystop")))
    input_train = np.hstack((X_train[:, [0]], rrnn.predict(X_train, model="earlystop")))
    input_test = np.hstack((X_test[:, [0]], rrnn.predict(X_test, model="earlystop")))

    regnn = fit_riesznet(
        reglearner,
        ate_moment_fn,
        input_train,
        y_train,
        input_test,
        y_test,
        main_fast_cfg,
        main_train_cfg,
        device=device,
        target_reg=args.target_reg,
        riesz_weight=0.0,
    )

    params = (
        tuple(
            value * y_scale
            for method in MAE_METHODS
            for value in regnn.predict_avg_moment(
                inputs,
                y_scaled,
                model="earlystop",
                method=method,
                srr=SRR_MAP[method],
            )
        )
        + (true_ate,)
    )
    cleanup_model(rrnn)
    cleanup_model(regnn)
    del rrnn, regnn, rrlearner, reglearner
    gc.collect()
    torch.cuda.empty_cache()
    return params


def run_separate_ablation(simulation_file, sim_seed, args, main_fast_cfg, main_train_cfg, rr_fast_cfg, rr_train_cfg, *, device):
    set_all_seeds(sim_seed)
    X, y, true_ate = load_ihdp_sample(simulation_file)
    y_scaled, y_scale = scale_outcome(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=sim_seed
    )

    torch.cuda.empty_cache()
    rrlearner = SeparateRieszLearner(X_train.shape[1], args.drop_prob)
    rrnn = fit_riesz_rr(
        rrlearner, ate_moment_fn, X_train, X_test, rr_fast_cfg, rr_train_cfg, device=device
    )

    reglearner = SeparateRegLearner(X_train.shape[1], args.n_hidden, args.drop_prob)
    regnn = RieszNet(reglearner, ate_moment_fn)

    inputs = np.hstack((X, rrnn.predict(X, model="earlystop")))
    input_train = np.hstack((X_train, rrnn.predict(X_train, model="earlystop")))
    input_test = np.hstack((X_test, rrnn.predict(X_test, model="earlystop")))

    regnn = fit_riesznet(
        reglearner,
        ate_moment_fn,
        input_train,
        y_train,
        input_test,
        y_test,
        main_fast_cfg,
        main_train_cfg,
        device=device,
        target_reg=args.target_reg,
        riesz_weight=0.0,
    )

    params = (
        tuple(
            value * y_scale
            for method in MAE_METHODS
            for value in regnn.predict_avg_moment(
                inputs,
                y_scaled,
                model="earlystop",
                method=method,
                srr=SRR_MAP[method],
            )
        )
        + (true_ate,)
    )
    cleanup_model(rrnn)
    cleanup_model(regnn)
    del rrnn, regnn, rrlearner, reglearner
    gc.collect()
    torch.cuda.empty_cache()
    return params


def summarize_results(results, methods, metric_mode, *, include_truth=False):
    if not results:
        raise ValueError("No results available to summarize.")

    res = tuple(np.array(x) for x in zip(*results))
    truth = np.asarray(res[-1], dtype=float)
    res_dict = {}

    for it, method in enumerate(methods):
        point, lb, ub = (np.asarray(x, dtype=float) for x in res[it * 3 : (it + 1) * 3])
        method_result = {"point": point, "lb": lb, "ub": ub}
        if include_truth:
            method_result["truth"] = truth
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


def run_shard(simulation_files, *, shard, n_shards, label, output_path, runner):
    shard_members = get_shard_members(len(simulation_files), n_shards, shard)
    shard_files = [simulation_files[i] for i in shard_members]
    print(f"[{label}] shard {shard + 1}/{n_shards} running {len(shard_members)} datasets")

    results = []
    for idx, simulation_file in enumerate(shard_files, start=1):
        print(
            f"[{label}] shard {shard + 1}/{n_shards} dataset {idx}/{len(shard_files)}: "
            f"{Path(simulation_file).name}"
        )
        results.append(runner(simulation_file, shard_members[idx - 1]))

    ensure_dir(Path(output_path).parent)
    dump(
        {
            "label": label,
            "shard": shard,
            "n_shards": n_shards,
            "simulation_files": shard_files,
            "simulation_file_indices": shard_members,
            "results": results,
        },
        output_path,
    )
    print(f"[{label}] wrote shard output to {output_path}")


def simulate_mae(shards, n_shards, args, main_fast_cfg, main_train_cfg, device):
    files = select_simulation_files("./data/IHDP/sim_data", args.mae_nsims, args.base_seed)
    ensure_dir(MAE_SHARD_DIR)

    for shard in shards:
        run_shard(
            files,
            shard=shard,
            n_shards=n_shards,
            label="MAE",
            output_path=MAE_SHARD_DIR / f"mae_shard_{shard}.joblib",
            runner=lambda simulation_file, global_idx: run_standard_experiment(
                simulation_file,
                args.base_seed + global_idx,
                args,
                main_fast_cfg,
                main_train_cfg,
                device=device,
                target_reg=args.target_reg,
                riesz_weight=args.riesz_weight,
                prediction_kwargs={"srr": SRR_MAP},
            ),
        )


def simulate_coverage_and_ablation(shards, n_shards, args, main_fast_cfg, main_train_cfg, rr_fast_cfg, rr_train_cfg, device):
    files = select_simulation_files(
        "./data/IHDP/sim_data_redraw_T", args.redraw_nsims, args.base_seed
    )
    ensure_dir(COVERAGE_SHARD_DIR)
    ensure_dir(ABLATION_SHARD_DIR)

    for shard in shards:
        run_shard(
            files,
            shard=shard,
            n_shards=n_shards,
            label="coverage",
            output_path=COVERAGE_SHARD_DIR / f"coverage_shard_{shard}.joblib",
            runner=lambda simulation_file, global_idx: run_standard_experiment(
                simulation_file,
                args.base_seed + 10_000 + global_idx,
                args,
                main_fast_cfg,
                main_train_cfg,
                device=device,
                target_reg=args.target_reg,
                riesz_weight=args.riesz_weight,
                prediction_kwargs={"srr": SRR_MAP},
            ),
        )
        run_shard(
            files,
            shard=shard,
            n_shards=n_shards,
            label="shared_ablation",
            output_path=ABLATION_SHARD_DIR / f"shared_ablation_shard_{shard}.joblib",
            runner=lambda simulation_file, global_idx: run_shared_ablation(
                simulation_file,
                args.base_seed + 20_000 + global_idx,
                args,
                main_fast_cfg,
                main_train_cfg,
                rr_fast_cfg,
                rr_train_cfg,
                device=device,
            ),
        )
        run_shard(
            files,
            shard=shard,
            n_shards=n_shards,
            label="posttmle_ablation",
            output_path=ABLATION_SHARD_DIR / f"posttmle_ablation_shard_{shard}.joblib",
            runner=lambda simulation_file, global_idx: run_standard_experiment(
                simulation_file,
                args.base_seed + 30_000 + global_idx,
                args,
                main_fast_cfg,
                main_train_cfg,
                device=device,
                target_reg=0.0,
                riesz_weight=args.riesz_weight,
                prediction_kwargs={"srr": {method: False for method in MAE_METHODS}, "postTMLE": True},
            ),
        )
        run_shard(
            files,
            shard=shard,
            n_shards=n_shards,
            label="separate_ablation",
            output_path=ABLATION_SHARD_DIR / f"separate_ablation_shard_{shard}.joblib",
            runner=lambda simulation_file, global_idx: run_separate_ablation(
                simulation_file,
                args.base_seed + 40_000 + global_idx,
                args,
                main_fast_cfg,
                main_train_cfg,
                rr_fast_cfg,
                rr_train_cfg,
                device=device,
            ),
        )


def run_robustness_shard(simulation_files, *, shard, n_shards, args, variants, device):
    shard_members = get_shard_members(len(simulation_files), n_shards, shard)
    shard_files = [simulation_files[i] for i in shard_members]
    print(f"[robustness] shard {shard + 1}/{n_shards} running {len(shard_members)} datasets")

    results = []
    for idx, simulation_file in enumerate(shard_files, start=1):
        global_idx = shard_members[idx - 1]
        sim_seed = args.base_seed + 50_000 + global_idx
        print(
            f"[robustness] shard {shard + 1}/{n_shards} dataset {idx}/{len(shard_files)}: "
            f"{Path(simulation_file).name}"
        )
        dataset_results = {}
        for variant in variants:
            print(
                f"[robustness] shard {shard + 1}/{n_shards} dataset {idx}/{len(shard_files)} "
                f"variant={variant['name']}"
            )
            dataset_results[variant["name"]] = run_standard_experiment(
                simulation_file,
                sim_seed,
                args,
                variant["fast_cfg"],
                variant["train_cfg"],
                device=device,
                target_reg=variant["target_reg"],
                riesz_weight=variant["riesz_weight"],
                prediction_kwargs={"srr": SRR_MAP},
                n_hidden=variant["n_hidden"],
                methods=ROBUSTNESS_METHODS,
            )
        results.append(dataset_results)

    output_path = ROBUSTNESS_SHARD_DIR / f"robustness_shard_{shard}.joblib"
    ensure_dir(output_path.parent)
    dump(
        {
            "label": "robustness",
            "shard": shard,
            "n_shards": n_shards,
            "simulation_files": shard_files,
            "simulation_file_indices": shard_members,
            "methods": ROBUSTNESS_METHODS,
            "variants": [{"name": name, "label": label} for name, label in ROBUSTNESS_VARIANTS],
            "results": results,
        },
        output_path,
    )
    print(f"[robustness] wrote shard output to {output_path}")


def simulate_robustness(shards, n_shards, args, main_fast_cfg, main_train_cfg, device):
    files = select_simulation_files(
        "./data/IHDP/sim_data_redraw_T", args.redraw_nsims, args.base_seed
    )
    ensure_dir(ROBUSTNESS_SHARD_DIR)
    variants = build_robustness_variants(args, main_fast_cfg, main_train_cfg)

    for shard in shards:
        run_robustness_shard(
            files,
            shard=shard,
            n_shards=n_shards,
            args=args,
            variants=variants,
            device=device,
        )


def write_mae_table(res_dict):
    ensure_dir(MAE_BASE_DIR)
    with open(MAE_BASE_DIR / "IHDP_MAE_NN.tex", "w") as f:
        f.write(
            "\\begin{tabular}{lc} \n"
            "\\toprule \n"
            "& MAE $\\pm$ std. err. \\\\ \n"
            "\\midrule \n"
            "\\multicolumn{2}{l}{\\textbf{Auto-DML:}} \\\\ \n"
        )

        for label, method in zip(MAE_METHOD_LABELS, MAE_METHODS):
            f.write(
                " & ".join(
                    [
                        label,
                        "{:.3f} $\\pm$ {:.3f}".format(
                            res_dict[method]["MAE"], res_dict[method]["std. err."]
                        ),
                    ]
                )
                + " \\\\ \n"
            )

        f.write(
            "\\multicolumn{2}{l}{\\textbf{Benchmark:}} \\\\ \n"
            "Dragonnet & 0.146 $\\pm$ 0.010 \\\\ \n"
            "\\bottomrule \n \\end{tabular}"
        )


def write_coverage_plot(res_dict, truth):
    ensure_dir(COVERAGE_BASE_DIR)
    plt.figure()
    plot_methods = ["dr", "direct", "ips"]
    method_strs = [
        "{}. Bias: {:.3f}, RMSE: {:.3f}, Coverage: {:.3f}".format(
            method,
            res_dict[method]["bias"],
            res_dict[method]["rmse"],
            res_dict[method]["cov"],
        )
        for method in plot_methods
    ]
    plt.title("\n".join(method_strs))
    for method in plot_methods:
        d = res_dict[method]
        plt.hist(np.array(d["point"]), alpha=0.5, label=method)
    plt.axvline(x=float(np.mean(truth)), label="true", color="red")
    handles, labels = plt.gca().get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    legend_order = ["true"] + plot_methods
    plt.legend(
        [label_to_handle[label] for label in legend_order],
        legend_order,
    )
    plt.savefig(COVERAGE_BASE_DIR / "IHDP_coverage_NN.pdf", bbox_inches="tight")
    plt.close()


def write_ablation_table():
    ensure_dir(ABLATION_BASE_DIR)
    files = [
        COVERAGE_BASE_DIR / "IHDP_coverage_NN.joblib",
        ABLATION_BASE_DIR / "IHDP_separateNNs_ablation.joblib",
        ABLATION_BASE_DIR / "IHDP_shared_ablation.joblib",
        ABLATION_BASE_DIR / "IHDP_postTMLE_ablation.joblib",
    ]
    names = ["RieszNet", "Separate NNs", "No end-to-end", "TMLE post-proc."]

    with open(ABLATION_BASE_DIR / "ablation.tex", "w") as f:
        f.write(
            "\\begin{tabular}{*{10}{r}} \n"
            "\\toprule \n"
            "& \\multicolumn{3}{c}{Direct} & \\multicolumn{3}{c}{IPS} & \\multicolumn{3}{c}{DR} \\\\ \n"
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \n"
            "&  Bias &  RMSE &  Cov. &  Bias &  RMSE &  Cov. &  Bias &  RMSE &  Cov. \\\\ \n"
            "\\midrule \n"
        )

        for name, path in zip(names, files):
            loaded = load(path)
            f.write(name + " & ")
            f.write(
                " & ".join(
                    [
                        "{:.3f}".format(float(loaded[method][metric]))
                        for method in ABLATION_TABLE_METHODS
                        for metric in ["bias", "rmse", "cov"]
                    ]
                )
                + " \\\\ \n"
            )

        f.write("\\bottomrule \n \\end{tabular}")


def write_robustness_table(summary):
    ensure_dir(ROBUSTNESS_BASE_DIR)
    with open(ROBUSTNESS_BASE_DIR / "IHDP_robustness.tex", "w") as f:
        f.write(
            "\\begin{tabular}{lrrr} \n"
            "\\toprule \n"
            "& \\multicolumn{3}{c}{DR} \\\\ \n"
            "\\cmidrule(lr){2-4} \n"
            "&  Bias &  RMSE &  Cov. \\\\ \n"
            "\\midrule \n"
        )

        for variant_name, variant_label in ROBUSTNESS_VARIANTS:
            f.write(variant_label + " & ")
            f.write(
                " & ".join(
                    [
                        "{:.3f}".format(float(summary[variant_name]["dr"][metric]))
                        for metric in ["bias", "rmse", "cov"]
                    ]
                )
                + " \\\\ \n"
            )

        f.write("\\bottomrule \n \\end{tabular}")


def summarize_mae(shards):
    results = load_shard_results(MAE_SHARD_DIR, "mae", shards)
    res_dict, _ = summarize_results(results, MAE_METHODS, metric_mode="mae")
    ensure_dir(MAE_BASE_DIR)
    dump(res_dict, MAE_BASE_DIR / "IHDP_MAE_NN.joblib")
    write_mae_table(res_dict)
    for method in MAE_METHODS:
        print(
            "{} : MAE = {:.3f} +/- {:.3f}".format(
                method, res_dict[method]["MAE"], res_dict[method]["std. err."]
            )
        )


def summarize_coverage(shards, *, plot):
    results = load_shard_results(COVERAGE_SHARD_DIR, "coverage", shards)
    res_dict, truth = summarize_results(results, MAE_METHODS, metric_mode="coverage")
    ensure_dir(COVERAGE_BASE_DIR)
    dump(res_dict, COVERAGE_BASE_DIR / "IHDP_coverage_NN.joblib")
    if plot:
        write_coverage_plot(res_dict, truth)
    for method in MAE_METHODS:
        print(
            "{} : bias = {:.3f}, rmse = {:.3f}, cov = {:.3f}".format(
                method,
                res_dict[method]["bias"],
                res_dict[method]["rmse"],
                res_dict[method]["cov"],
            )
        )


def summarize_ablation_variant(shards, prefix, output_name):
    results = load_shard_results(ABLATION_SHARD_DIR, prefix, shards)
    res_dict, _ = summarize_results(
        results, MAE_METHODS, metric_mode="coverage", include_truth=True
    )
    ensure_dir(ABLATION_BASE_DIR)
    dump(res_dict, ABLATION_BASE_DIR / output_name)
    for method in MAE_METHODS:
        print(
            "{} [{}] : bias = {:.3f}, rmse = {:.3f}, cov = {:.3f}".format(
                method,
                prefix,
                res_dict[method]["bias"],
                res_dict[method]["rmse"],
                res_dict[method]["cov"],
            )
        )


def summarize_robustness(shards):
    dataset_results = load_shard_results(ROBUSTNESS_SHARD_DIR, "robustness", shards)
    summary = {}

    for variant_name, variant_label in ROBUSTNESS_VARIANTS:
        variant_results = [result[variant_name] for result in dataset_results]
        summary[variant_name], _ = summarize_results(
            variant_results, ROBUSTNESS_METHODS, metric_mode="coverage"
        )
        print(
            "{} [{}] : bias = {:.3f}, rmse = {:.3f}, cov = {:.3f}".format(
                "dr",
                variant_label,
                summary[variant_name]["dr"]["bias"],
                summary[variant_name]["dr"]["rmse"],
                summary[variant_name]["dr"]["cov"],
            )
        )

    ensure_dir(ROBUSTNESS_BASE_DIR)
    dump(
        {
            "methods": ROBUSTNESS_METHODS,
            "variants": [{"name": name, "label": label} for name, label in ROBUSTNESS_VARIANTS],
            "summary": summary,
        },
        ROBUSTNESS_BASE_DIR / "IHDP_robustness.joblib",
    )
    write_robustness_table(summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster-friendly Python script version of RieszNet_IHDP.ipynb."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "simulate", "summarize", "robustness"],
        default="all",
        help=(
            "all: run existing simulations/summaries plus robustness; "
            "simulate: run only the existing shard simulations; "
            "summarize: build the existing aggregate outputs from shard files; "
            "robustness: with --shard run robustness shard simulations, otherwise summarize robustness shard outputs."
        ),
    )
    parser.add_argument(
        "--shard",
        type=int,
        action="append",
        default=None,
        help="Shard to run. Repeat to pass multiple shards. If omitted, uses --shard-start..--shard-end.",
    )
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=9)
    parser.add_argument("--aggregate-shard-start", type=int, default=0)
    parser.add_argument("--aggregate-shard-end", type=int, default=9)
    parser.add_argument("--n-shards", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--mae-nsims", type=int, default=1000)
    parser.add_argument("--redraw-nsims", type=int, default=100)
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--n-hidden", type=int, default=100)
    parser.add_argument("--learner-lr", type=float, default=1e-5)
    parser.add_argument("--fast-learner-lr", type=float, default=1e-4)
    parser.add_argument("--rr-learner-lr", type=float, default=1e-3)
    parser.add_argument("--rr-fast-learner-lr", type=float, default=1e-1)
    parser.add_argument("--learner-l2", type=float, default=1e-3)
    parser.add_argument("--learner-l1", type=float, default=0.0)
    parser.add_argument("--fast-n-epochs", type=int, default=100)
    parser.add_argument("--train-n-epochs", type=int, default=600)
    parser.add_argument("--fast-earlystop-rounds", type=int, default=2)
    parser.add_argument("--train-earlystop-rounds", type=int, default=40)
    parser.add_argument("--earlystop-delta", type=float, default=1e-4)
    parser.add_argument("--rr-earlystop-delta", type=float, default=1e-2)
    parser.add_argument("--target-reg", type=float, default=1.0)
    parser.add_argument("--riesz-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Number of CPU threads used by PyTorch. Defaults to SLURM_CPUS_PER_TASK when running under Slurm.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip histogram generation during summarize/all runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.torch_num_threads is None and "SLURM_CPUS_PER_TASK" in os.environ:
        args.torch_num_threads = int(os.environ["SLURM_CPUS_PER_TASK"])

    configure_torch(args.torch_num_threads)
    ensure_dir(MODEL_DIR)
    device = get_device()
    print("GPU:", torch.cuda.is_available())

    main_fast_cfg, main_train_cfg, rr_fast_cfg, rr_train_cfg = build_train_options(args)

    if args.mode in ["all", "simulate"]:
        simulate_shards = resolve_shards(args.shard, args.shard_start, args.shard_end, args.n_shards)
        simulate_mae(simulate_shards, args.n_shards, args, main_fast_cfg, main_train_cfg, device)
        simulate_coverage_and_ablation(
            simulate_shards,
            args.n_shards,
            args,
            main_fast_cfg,
            main_train_cfg,
            rr_fast_cfg,
            rr_train_cfg,
            device,
        )

    if args.mode in ["all", "summarize"]:
        aggregate_shards = resolve_shards(
            None, args.aggregate_shard_start, args.aggregate_shard_end, args.n_shards
        )
        summarize_mae(aggregate_shards)
        summarize_coverage(aggregate_shards, plot=not args.no_plot)
        summarize_ablation_variant(
            aggregate_shards,
            "shared_ablation",
            "IHDP_shared_ablation.joblib",
        )
        summarize_ablation_variant(
            aggregate_shards,
            "posttmle_ablation",
            "IHDP_postTMLE_ablation.joblib",
        )
        summarize_ablation_variant(
            aggregate_shards,
            "separate_ablation",
            "IHDP_separateNNs_ablation.joblib",
        )
        write_ablation_table()

    if args.mode in ["all", "robustness"]:
        if args.mode == "all" or args.shard is not None:
            robustness_shards = resolve_shards(
                args.shard, args.shard_start, args.shard_end, args.n_shards
            )
            simulate_robustness(
                robustness_shards,
                args.n_shards,
                args,
                main_fast_cfg,
                main_train_cfg,
                device,
            )

        if args.mode == "all" or args.shard is None:
            aggregate_shards = resolve_shards(
                None, args.aggregate_shard_start, args.aggregate_shard_end, args.n_shards
            )
            summarize_robustness(aggregate_shards)


if __name__ == "__main__":
    main()
