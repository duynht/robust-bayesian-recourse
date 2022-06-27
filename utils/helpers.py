import collections
import logging
import os
import pickle

import numpy as np
import pandas as pd
from utils.validation import check_random_state


def gen_synthetic_data(num_samples=1000, mean_0=None, cov_0=None, mean_1=None, cov_1=None, random_state=None):
    random_state = check_random_state(random_state)

    if mean_0 is None or cov_0 is None or mean_1 is None or cov_1 is None:
        mean_0 = np.array([-2, -2])
        cov_0 = np.array([[0.5, 0], [0, 0.5]])
        mean_1 = np.array([2, 2])
        cov_1 = np.array([[0.5, 0], [0, 0.5]])

    num_class0 = random_state.binomial(n=num_samples, p=0.5)
    x_class0 = random_state.multivariate_normal(mean_0, cov_0, num_class0)
    x_class1 = random_state.multivariate_normal(mean_1, cov_1, num_samples - num_class0)
    data0 = np.hstack([x_class0, np.zeros((num_class0, 1))])
    data1 = np.hstack([x_class1, np.ones((num_samples - num_class0, 1))])
    raw_data = np.vstack([data0, data1])
    random_state.shuffle(raw_data)
    column_names = ["f" + str(i) for i in range(len(mean_0))] + ["label"]
    df = pd.DataFrame(raw_data, columns=column_names)
    return df


def gen_synthetic_data_nl(
    num_samples=1000,
    x_lim=(-2, 4),
    y_lim=(-2, 7),
    f=lambda x, y: y >= 1 + x + 2 * x**2 + x**3 - x**4,
    random_state=42,
    add_noise=False,
):
    random_state = check_random_state(random_state)
    std = 1.0
    x = random_state.uniform(x_lim[0], x_lim[1], num_samples)
    y = random_state.uniform(y_lim[0], y_lim[1], num_samples)
    noisy_y = y + random_state.normal(0, std, size=y.shape)
    label = f(x, noisy_y if add_noise else y).astype(np.int32)
    raw_data = {"f0": x, "f1": y, "label": label}
    df = pd.DataFrame(raw_data)
    return df


def get_dataset(dataset="synthesis", params=list()):
    if "synthesis" in dataset:
        if isinstance(params, collections.Sequence):
            params.append("shift" in dataset)
            dataset = gen_synthetic_data_nl(*params)
        else:
            if "shift" in dataset:
                params["add_noise"] = "shift" in dataset
            dataset = gen_synthetic_data_nl(**params)
        numerical = list(dataset.columns)
        numerical.remove("label")
    elif "german" in dataset:
        dataset = pd.read_csv("./data/corrected_german_small.csv" if "shift" in dataset else "./data/german_small.csv")
        # numerical = ['Duration', 'Credit amount', 'Installment rate',
        # 'Present residence', 'Age', 'Existing credits', 'Number people']
        numerical = ["Duration", "Credit amount", "Age"]
    elif "sba" in dataset:
        dataset = pd.read_csv("./data/sba_8905.csv" if "shift" in dataset else "./data/sba_0614.csv")
        categorical = ["LowDoc", "RevLineCr", "NewExist", "MIS_Status", "UrbanRural", "label"]
        numerical = list(dataset.columns.difference(categorical))
    elif "gmc" in dataset:
        dataset = pd.read_csv("./data/gmc_shift.csv" if "shift" in dataset else "./data/gmc.csv").rename(
            columns={"SeriousDlqin2yrs": "label"}
        )
        numerical = [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ]
    else:
        raise ValueError("Unknown dataset")

    return dataset, numerical


def get_full_dataset(dataset="synthesis", params=list()):
    if "synthesis" in dataset:
        joint_dataset = gen_synthetic_data(*params)
        numerical = list(joint_dataset.columns)
        numerical.remove("label")
    elif "german" in dataset:
        dataset = pd.read_csv("./data/german_small.csv")
        shift_dataset = pd.read_csv("./data/corrected_german_small.csv")
        joint_dataset = dataset.append(shift_dataset)
        # numerical = ['Duration', 'Credit amount', 'Installment rate',
        # 'Present residence', 'Age', 'Existing credits', 'Number people']
        numerical = ["Duration", "Credit amount", "Age"]
    elif "sba" in dataset:
        dataset = pd.read_csv("./data/sba_0614.csv")
        shift_dataset = pd.read_csv("./data/sba_8905.csv")
        joint_dataset = dataset.append(shift_dataset)
        categorical = ["LowDoc", "RevLineCr", "NewExist", "MIS_Status", "UrbanRural", "label"]
        numerical = list(dataset.columns.difference(categorical))
    elif "gmc" in dataset:
        dataset = pd.read_csv("./data/gmc.csv").rename(columns={"SeriousDlqin2yrs": "label"})
        shift_dataset = pd.read_csv("./data/gmc_shift.csv").rename(columns={"SeriousDlqin2yrs": "label"})
        joint_dataset = dataset.append(shift_dataset)
        numerical = [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ]
    else:
        raise ValueError("Unknown dataset")

    return joint_dataset, numerical


def make_logger(name, log_dir):
    log_dir = log_dir or "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "debug.log")
    handler = logging.FileHandler(log_file)
    formater = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formater)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formater)
    logger.addHandler(stream_handler)

    return logger


def pdump(x, name, outdir="."):
    with open(os.path.join(outdir, name), mode="wb") as f:
        pickle.dump(x, f)


def pload(name, outdir="."):
    with open(os.path.join(outdir, name), mode="rb") as f:
        return pickle.load(f)
