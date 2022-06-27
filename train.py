import argparse
import os

import joblib
import numpy as np
import torch
import yaml
from expt.common import clf_map, synthetic_params, train_func_map
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from utils import helpers
from utils.transformer import get_transformer


arrival_data = {"synthesis": False, "german": False, "sba": False, "gmc": False}


def eval_performance(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=-1)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob[:, 1])

    return accuracy, auc


def train_model(X_train, y_train, X_test, y_test, train_func, clf, d, lr, num_epoch, idx, verbose, random_state):
    print("training classifier ", idx, flush=True)
    torch.manual_seed(random_state)
    np.random.seed(random_state + 1)
    model = clf(d)
    train_func(model, X_train, y_train, lr, num_epoch, verbose)
    acc, auc = eval_performance(model, X_test, y_test)
    return model, acc, auc


def train(
    clf_name,
    data_name,
    wdir,
    lr,
    num_epoch,
    kfold=5,
    num_future=100,
    seed=123,
    verbose=False,
    num_proc=1,
    append_arrival=True,
    arrival_ratio=0.50,
    train_shift_size=0.8,
):

    transformer = get_transformer(data_name)
    df, _ = helpers.get_dataset(data_name, params=synthetic_params)
    y = df["label"].to_numpy()
    X = df.drop("label", axis=1)
    X = transformer.transform(X)
    if not type(X) is np.ndarray:
        X = X.toarray()

    df_shift, _ = helpers.get_dataset(data_name + "shift", params=synthetic_params)
    y_shift = df_shift["label"].to_numpy()
    X_shift = df_shift.drop("label", axis=1)
    X_shift = transformer.transform(X_shift)
    if not type(X_shift) is np.ndarray:
        X_shift = X_shift.toarray()

    d = X.shape[1]
    clf = clf_map[clf_name]
    train_func = train_func_map[clf_name]
    report = {}

    # Train present data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    kf = KFold(n_splits=kfold)
    jobs_args = []

    for i, (train_index, cross_index) in enumerate(kf.split(X_train)):
        X_training = X_train[train_index]
        y_training = y_train[train_index]

        jobs_args.append((X_training, y_training, X_test, y_test, train_func, clf, d, lr, num_epoch, i, verbose, seed))

    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(train_model)(*args) for args in jobs_args)
    # rets = []
    # for i in range(len(jobs_args)):
    #     rets.append(train_model(*jobs_args[i]))

    cur_auc = []
    cur_acc = []
    cur_models = []
    for model, acc, auc in rets:
        cur_acc.append(acc)
        cur_auc.append(auc)
        cur_models.append(model)

    model = cur_models[0]

    report["append_arrival"] = append_arrival
    report["arrival_ratio"] = arrival_ratio
    report["train_shift_size"] = train_shift_size
    report["cur_acc_mean"] = float(np.mean(cur_acc))
    report["cur_acc_std"] = float(np.std(cur_acc))
    report["cur_auc_mean"] = float(np.mean(cur_auc))
    report["cur_auc_std"] = float(np.std(cur_auc))
    name = f"{clf_name}_{data_name}_{kfold}.pickle"
    helpers.pdump(cur_models, name, wdir)
    print(
        "Trained classifier: {} on current dataset: {}, and saved to {}".format(
            clf_name, data_name, os.path.join(wdir, name)
        )
    )

    # Train shift data

    for i, (train_index, cross_index) in enumerate(kf.split(X_train)):
        jobs_args = []
        X_training = X_train[train_index]
        y_training = y_train[train_index]

        for rng in range(num_future):
            if append_arrival:
                if arrival_ratio != 0:
                    arrival_X, _, arrival_y, _ = train_test_split(
                        X_shift, y_shift, train_size=arrival_ratio, random_state=rng, stratify=y_shift
                    )
                    future_X = np.vstack([X_training, arrival_X])
                    future_y = np.concatenate([y_training, arrival_y])
                else:
                    future_X, future_y = X_training, y_training
            else:
                future_X, X_test, future_y, y_test = train_test_split(
                    X_shift, y_shift, train_size=train_shift_size, random_state=rng, stratify=y_shift
                )

            jobs_args.append(
                (future_X, future_y, X_test, y_test, train_func, clf, d, lr, num_epoch, rng, verbose, seed)
            )

        rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(train_model)(*args) for args in jobs_args)

        shift_auc = []
        shift_acc = []
        models = []
        for model, acc, auc in rets:
            shift_acc.append(acc)
            shift_auc.append(auc)
            models.append(model)

        report["shift_acc_mean"] = float(np.mean(shift_acc))
        report["shift_acc_std"] = float(np.std(shift_acc))
        report["shift_auc_mean"] = float(np.mean(shift_auc))
        report["shift_auc_std"] = float(np.std(shift_auc))
        name = f"{clf_name}_{data_name}_shift_{i}_{num_future}.pickle"
        helpers.pdump(models, name, wdir)
        print(
            "Trained classifier: {} on shifted dataset: {}, and saved to {}".format(
                clf_name, data_name, os.path.join(wdir, name)
            )
        )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument("--clf", "-c", dest="clfs", nargs="*")
    parser.add_argument("--data", "-d", dest="datasets", nargs="*")
    parser.add_argument("--lr", "-lr", default=1e-3, type=float)
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--kfold", default=5, type=int)
    parser.add_argument("--num-future", "-nf", default=100, type=int)
    parser.add_argument("--num-proc", default=1, type=int)
    parser.add_argument("--run-id", default=0, type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--seed", "-s", default=123, type=int)

    args = parser.parse_args()

    torch.set_printoptions(sci_mode=False)
    seed = 46
    torch.manual_seed(args.seed + 12)
    np.random.seed(args.seed + 11)
    np.set_printoptions(suppress=False)
    wdir = f"results/run_{args.run_id}/checkpoints"
    os.makedirs(wdir, exist_ok=True)

    report = {}
    for clf in args.clfs:
        clf_report = {}
        for data in args.datasets:
            print("training on dataset: ", data)
            data_report = train(
                clf,
                data,
                wdir,
                args.lr,
                args.epoch,
                args.kfold,
                args.num_future,
                args.seed,
                args.verbose,
                args.num_proc,
            )
            clf_report[data] = data_report
        report[clf] = clf_report

    filepath = f"{wdir}/report.txt"
    with open(filepath, mode="w") as file:
        yaml.dump(report, file)
