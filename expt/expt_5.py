import contextlib
import copy
import itertools
import os
from collections import defaultdict, namedtuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from expt.common import (
    _run_single_instance,
    dataset_name_map,
    enrich_training_data,
    load_models,
    method_map,
    method_name_map,
    synthetic_params,
)
from expt.expt_config import Expt5
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from tqdm import tqdm
from utils import helpers
from utils.funcs import compute_max_distance, find_pareto
from utils.transformer import get_transformer


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


Results = namedtuple("Results", ["l1_cost", "cur_vald", "fut_vald", "feasible"])

param_to_vary = {
    "wachter": ["none"],
    "lime_roar": ["delta_max"],
    "limels_roar": ["delta_max"],
    "mpm_roar": ["delta_max"],
    "rbr": ["epsilon_pe", "delta_plus", "epsilon_op"],
}


def run(ec, wdir, dname, cname, mname, num_proc, seed, logger, start_index=None, num_ins=None, device="cpu"):
    # logger.info("Running dataset: %s, classifier: %s, method: %s...",
    # dname, cname, mname)
    print("Running dataset: %s, classifier: %s, method: %s..." % (dname, cname, mname))

    df, _ = helpers.get_dataset(dname, params=synthetic_params)
    y = df["label"].to_numpy()
    X_df = df.drop("label", axis=1)
    transformer = get_transformer(dname)
    X = transformer.transform(X_df)
    if not type(X) is np.ndarray:
        X = X.toarray()
    cat_indices = transformer.cat_indices

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    enriched_data = enrich_training_data(5000, X_train, cat_indices, seed)

    cur_models = load_models(dname, cname, ec.kfold, wdir)

    kf = KFold(n_splits=ec.kfold)

    ptv_lst = param_to_vary[mname]
    param_grid = {}
    for ptv in ptv_lst:
        min_ptv = ec.params_to_vary[ptv]["min"]
        max_ptv = ec.params_to_vary[ptv]["max"]
        step_size = ec.params_to_vary[ptv]["step"]
        values = np.arange(min_ptv, max_ptv + step_size, step_size)
        param_grid[ptv] = values

    grid = ParameterGrid(param_grid)

    mname_short = mname.replace("_delta", "").replace("_rho", "")
    method = method_map[mname_short]

    res = dict()
    res["params"] = grid
    res["delta_max_df"] = ec.roar_params["delta_max"]
    res["cost"] = []
    res["cur_vald"] = []
    res["fut_vald"] = []
    res["feasible"] = []

    x_train_max_dist = compute_max_distance(X_train)

    for pid, params in enumerate(grid):
        # logger.info("run with params {}".format(params))
        print("run with params {}".format(params))
        # new_config = copy.deepcopy(ec)
        new_config = Expt5(ec.to_dict())
        new_config.max_distance = x_train_max_dist
        for ptv, value in params.items():
            if ptv == "delta_max":
                new_config.roar_params[ptv] = value
            if ptv == "epsilon_pe" or ptv == "delta_plus" or ptv == "perturb_radius":
                new_config.rbr_params[ptv] = value

        train_index, _ = next(kf.split(X_train))
        X_training, y_training = X_train[train_index], y_train[train_index]

        model = cur_models[0]
        shifted_models = load_models(dname + f"_shift_{0}", cname, ec.num_future, wdir)

        X_all = np.vstack([X_test, X_training])
        y_all = np.concatenate([y_test, y_training])
        y_pred = model.predict(X_all)
        uds_X, uds_y = X_all[y_pred == 0], y_all[y_pred == 0]

        if start_index is not None or num_ins is not None:
            num_ins = num_ins or 1
            start_index = start_index or 0
            uds_X = uds_X[start_index : start_index + num_ins]
            uds_y = uds_y[start_index : start_index + num_ins]
        else:
            uds_X, uds_y = uds_X[: ec.max_ins], uds_y[: ec.max_ins]

        params = dict(
            train_data=X_training,
            enriched_data=enriched_data,
            cat_indices=cat_indices,
            config=new_config,
            method_name=mname_short,
            dataset_name=dname,
            device=device,
            perturb_radius=ec.perturb_radius[dname],
        )

        jobs_args = []

        for idx, x0 in enumerate(uds_X):
            jobs_args.append((idx, method, x0, model, shifted_models, seed, logger, params))

        with tqdm_joblib(tqdm(desc="Running recourse method", total=len(jobs_args))) as _:
            rets = joblib.Parallel(n_jobs=num_proc, prefer="threads")(
                joblib.delayed(_run_single_instance)(*jobs_args[i]) for i in range(len(jobs_args))
            )

        # rets = []
        # for idx, x0 in enumerate(uds_X):
        #     t_start = time()
        #     # print("run with params {}".format(grid[pid]))
        #     # print(f'run_single_instance id={idx}')
        #     ret = _run_single_instance(*jobs_args[idx])
        #     rets.append(ret)
        #     print(f'Finished {idx+1} / {len(uds_X)} | Time elapsed: {(time() - t_start):.4f} seconds')

        l1_cost = []
        cur_vald = []
        fut_vald = []
        feasible = []

        for ret in rets:
            l1_cost.append(ret.l1_cost)
            cur_vald.append(ret.cur_vald)
            fut_vald.append(ret.fut_vald)
            feasible.append(ret.feasible)

        res["cost"].append(np.array(l1_cost))
        res["cur_vald"].append(np.array(cur_vald))
        res["fut_vald"].append(np.array(fut_vald))
        res["feasible"].append(np.array(feasible))

    helpers.pdump(res, f"{cname}_{dname}_{mname}.pickle", wdir)

    logger.info("Done dataset: %s, classifier: %s, method: %s!", dname, cname, mname)


label_map = {
    "fut_vald": "Future Validity",
    "cur_vald": "Current Validity",
    "cost": "Cost",
}


def plot_5(ec, wdir, cname, dname, methods):
    def plot(methods, x_label, y_label, data):
        plt.rcParams.update({"font.size": 20})
        fig, ax = plt.subplots()
        marker = reversed(["*", "v", "^", "o", (5, 1), (5, 0), "+", "s"])
        iter_marker = itertools.cycle(marker)

        for mname in methods:
            x, y = find_pareto(data[mname][x_label], data[mname][y_label])
            ax.plot(x, y, marker=next(iter_marker), label=method_name_map[mname], alpha=0.8)

        ax.set_ylabel(label_map[y_label])
        ax.set_xlabel(label_map[x_label])
        # ax.set_yscale('log')
        ax.legend(prop={"size": 14})
        filepath = os.path.join(wdir, f"{cname}_{dname}_{x_label}_{y_label}.png")
        plt.savefig(filepath, dpi=400, bbox_inches="tight")

    data = defaultdict(dict)

    for mname in methods:
        res = helpers.pload(f"{cname}_{dname}_{mname}.pickle", wdir)
        data[mname]["params"] = res["params"]
        data[mname]["delta_max"] = res["delta_max_df"]
        data[mname]["cost"] = []
        data[mname]["fut_vald"] = []
        data[mname]["cur_vald"] = []

        for i in range(len(res["params"])):
            data[mname]["cost"].append(np.mean(res["cost"][i]))
            data[mname]["fut_vald"].append(np.mean(res["fut_vald"][i]))
            data[mname]["cur_vald"].append(np.mean(res["cur_vald"][i]))

    print(data["rbr"])
    plot(methods, "cost", "fut_vald", data)
    plot(methods, "cost", "cur_vald", data)


def plot_5_1(ec, wdir, cname, datasets, methods):
    def __plot(ax, data, dname, x_label, y_label):
        marker = reversed(["+", "v", "^", "o", (5, 0)])
        iter_marker = itertools.cycle(marker)

        for mname, o in data[dname].items():
            if mname == "wachter":
                ax.scatter(
                    data[dname][mname][x_label],
                    data[dname][mname][y_label],
                    marker=(5, 1),
                    label=method_name_map[mname],
                    alpha=0.7,
                    color="black",
                    zorder=10,
                )
            else:
                x, y = find_pareto(data[dname][mname][x_label], data[dname][mname][y_label])
                ax.plot(x, y, marker=next(iter_marker), label=method_name_map[mname], alpha=0.7)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_title(dataset_name_map[dname])

    data = defaultdict(dict)
    for dname in datasets:
        for mname in methods:
            res = helpers.pload(f"{cname}_{dname}_{mname}.pickle", wdir)

            # print(res)
            data[dname][mname] = {}
            data[dname][mname]["params"] = res["params"]
            data[dname][mname]["delta_max"] = res["delta_max_df"]
            data[dname][mname]["cost"] = []
            data[dname][mname]["fut_vald"] = []
            data[dname][mname]["cur_vald"] = []

            for i in range(len(res["params"])):
                data[dname][mname]["cost"].append(np.mean(res["cost"][i]))
                data[dname][mname]["fut_vald"].append(np.mean(res["fut_vald"][i]))
                data[dname][mname]["cur_vald"].append(np.mean(res["cur_vald"][i]))

    plt.style.use("seaborn-deep")
    plt.rcParams.update({"font.size": 10.5})
    num_ds = len(datasets)
    figsize_map = {5: (17, 5.5), 4: (12, 5.5), 3: (10, 5.5), 2: (8, 5.5), 1: (4, 4)}
    fig, axs = plt.subplots(2, num_ds, figsize=figsize_map[num_ds])
    if num_ds == 1:
        axs = axs.reshape(-1, 1)

    metrics = ["cur_vald", "fut_vald"]

    for i in range(num_ds):
        for j in range(len(metrics)):
            __plot(axs[j, i], data, datasets[i], "cost", metrics[j])
            if i == 0:
                axs[j, i].set_ylabel(label_map[metrics[j]])
            if j == len(metrics) - 1:
                axs[j, i].set_xlabel(label_map["cost"])

    marker = reversed(["+", "v", "^", "o", (5, 0)])
    iter_marker = itertools.cycle(marker)
    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    for mname in methods:
        if mname == "wachter":
            ax.scatter([], [], marker=(5, 1), label=method_name_map[mname], alpha=0.7, color="black")
        else:
            ax.plot([], marker=next(iter_marker), label=method_name_map[mname], alpha=0.7)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23 - 0.1 * (len(methods) > 5)),
        ncol=min(len(methods), 5),
        frameon=False,
    )
    plt.tight_layout()
    joint_dname = "".join([e[:2] for e in datasets])
    filepath = os.path.join(wdir, f"{cname}_{joint_dname}.pdf")
    plt.savefig(filepath, dpi=400, bbox_inches="tight")


def run_expt_5(
    ec,
    wdir,
    datasets,
    classifiers,
    methods,
    num_proc=4,
    plot_only=False,
    seed=None,
    logger=None,
    start_index=None,
    num_ins=None,
    rerun=True,
    device="cpu",
):
    logger.info("Running ept 5...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.e5.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.e5.all_clfs

    if methods is None or len(methods) == 0:
        methods = ec.e5.all_methods

    if not plot_only:
        jobs_args = []

        for cname in classifiers:
            cmethods = copy.deepcopy(methods)

            for dname in datasets:
                for mname in cmethods:
                    filepath = os.path.join(wdir, f"{cname}_{dname}_{mname}.pickle")
                    if not os.path.exists(filepath) or rerun:
                        jobs_args.append(
                            (ec.e5, wdir, dname, cname, mname, num_proc, seed, logger, start_index, num_ins, device)
                        )

        # rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(run)(
        #     *jobs_args[i]) for i in range(len(jobs_args)))
        for i in range(len(jobs_args)):
            run(*jobs_args[i])

    for cname in classifiers:
        cmethods = copy.deepcopy(methods)
        for dname in datasets:
            plot_5(ec.e5, wdir, cname, dname, cmethods)
        plot_5_1(ec.e5, wdir, cname, datasets, cmethods)

    logger.info("Done ept 5.")
