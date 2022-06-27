import os
from collections import namedtuple

import numpy as np
import torch
from classifiers import mlp
from libs.roar import lime_roar, limels_roar
from libs.wachter import wachter
from rbr import rbr
from sklearn.utils import check_random_state
from utils import helpers
from utils.funcs import lp_dist


Results = namedtuple("Results", ["l1_cost", "cur_vald", "fut_vald", "feasible"])


def to_numpy_array(lst):
    pad = len(max(lst, key=len))
    return np.array([i + [0] * (pad - len(i)) for i in lst])


def load_models(dname, cname, n, wdir):
    pdir = os.path.dirname(wdir)
    pdir = os.path.join(pdir, "checkpoints")
    models = helpers.pload(f"{cname}_{dname}_{n}.pickle", pdir)
    return models


def calc_future_validity(x, shifted_models):
    preds = []
    for model in shifted_models:
        pred = model.predict(x)
        preds.append(pred)
    preds = np.array(preds)
    return np.mean(preds)


def enrich_training_data(num_samples, train_data, cat_indices, rng):
    rng = check_random_state(rng)
    cur_n, d = train_data.shape
    if cur_n < num_samples:
        min_f_val = np.min(train_data, axis=0)
        max_f_val = np.max(train_data, axis=0)
        new_data = rng.uniform(min_f_val, max_f_val, (num_samples - cur_n, d))

        new_data[:, cat_indices] = new_data[:, cat_indices] >= 0.5

        new_data = np.vstack([train_data, new_data])
    else:
        new_data = train_data[:num_samples]
    return new_data


def to_mean_std(m, s, is_best):
    if is_best:
        return "\\textbf{" + "{:.2f}".format(m) + "}" + " $\pm$ {:.2f}".format(s)
    else:
        return "{:.2f} $\pm$ {:.2f}".format(m, s)


def _run_single_instance(idx, method, x0, model, shifted_models, seed, logger, params=dict()):
    # logger.info("Generating recourse for instance : %d", idx)

    torch.manual_seed(seed + 2)
    np.random.seed(seed + 1)
    random_state = check_random_state(seed)

    x_ar, report = method.generate_recourse(x0, model, random_state, params)

    l1_cost = lp_dist(x0, x_ar, p=1)
    cur_vald = model.predict(x_ar)
    fut_vald = calc_future_validity(x_ar, shifted_models)

    return Results(l1_cost, cur_vald, fut_vald, report["feasible"])


method_name_map = {"lime_roar": "LIME-ROAR", "limels_roar": "LIMELS-ROAR", "wachter": "Wachter", "rbr": "RBR"}


dataset_name_map = {
    "synthesis": "Synthetic data",
    "german": "German",
    "sba": "SBA",
    "gmc": "Give Me Some Credit",
}

metric_order = {"cost": -1, "cur-vald": 1, "fut-vald": 1}


method_map = {
    "wachter": wachter,
    "lime_roar": lime_roar,
    "limels_roar": limels_roar,
    "rbr": rbr,
}


clf_map = {
    "net0": mlp.Net0,
    "mlp": mlp.Net0,
}


train_func_map = {
    "net0": mlp.train,
    "mlp": mlp.train,
}


synthetic_params = dict(
    num_samples=1000,
    x_lim=(-2, 4),
    y_lim=(-2, 7),
    f=lambda x, y: y >= 1 + x + 2 * x**2 + x**3 - x**4,
    random_state=42,
)
