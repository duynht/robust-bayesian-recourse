import numpy as np
from libs.explainers.lime_wrapper import LimeWrapper
from libs.roar.linear_roar import LinearROAR
from sklearn.utils import check_random_state


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params["train_data"]
    ec = params["config"]
    cat_indices = params["cat_indices"]

    delta_max = ec.roar_params["delta_max"]

    explainer = LimeWrapper(train_data, class_names=["0", "1"], discretize_continuous=False, random_state=rng)

    w, b = explainer.explain_instance(x0, model.predict_proba, num_samples=ec.num_samples)

    arg = LinearROAR(
        train_data, w, b, cat_indices, lambd=0.1, dist_type=1, lr=0.01, delta_max=delta_max, max_iter=1000
    )
    x_ar = arg.fit_instance(x0, verbose=False)
    report = dict(feasible=arg.feasible)

    return x_ar, report


def search_lambda(model, X, y, params, logger):
    lbd_list = np.arange(0.01, 0.1, 0.01)
    logger.info("ROAR: Search best lambda")

    y_pred = model.predict(X)
    uds_X = X[y_pred == 0]
    max_ins = 50
    uds_X = uds_X[:max_ins]

    logger.info("ROAR: cross_validation size: %d", len(uds_X))

    best_lbd = 0
    best_sum_f = 0

    for lbd in lbd_list:
        logger.info("ROAR: try with lambda = %.2f", lbd)
        params["lambda"] = lbd
        sum_f = 0
        for x0 in uds_X:
            x_ar, _ = generate_recourse(x0, model, 1, params)
            sum_f += model.predict(x_ar)

        logger.info("ROAR: number of valid instances = %d", sum_f)

        if sum_f >= best_sum_f:
            best_sum_f, best_lbd = sum_f, lbd

    logger.info("ROAR: best lambda = %f", best_lbd)
    return best_lbd
