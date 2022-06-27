from functools import partial

import numpy as np
from lime import lime_base
from sklearn.utils import check_random_state
from utils.funcs import uniform_ball


class LimeLS:
    def __init__(self, train_data, predict_fn, num_cfs=5, random_state=None):
        self.random_state = check_random_state(random_state)
        self.train_data = train_data
        train_prob = predict_fn(train_data)
        self.train_label = np.argmax(train_prob, axis=1)
        self.predict_fn = predict_fn
        self.num_cfs = num_cfs
        self.num_features = train_data.shape[1]
        kernel_width = np.sqrt(self.num_features) * 0.75
        kernel_width = float(kernel_width)

        def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.base = lime_base.LimeBase(kernel_fn, False, random_state=self.random_state)

    def make_prediction(self, x):
        return np.argmax(self.predict_fn(x), axis=-1)

    def dist(self, x, y):
        return np.linalg.norm(x - y, ord=2, axis=-1)

    def find_counterfactual(self, x, k=None):
        k = k or self.num_cfs
        x_label = self.make_prediction(x)

        d = self.dist(self.train_data, x)
        order = np.argsort(d)
        x_cfs = self.train_data[order[self.train_label[order] == 1 - x_label]][:k]
        best_x_b = None
        best_dist = np.inf

        for x_cf in x_cfs:
            lambd_list = np.linspace(0, 1, 100)
            for lambd in lambd_list:
                x_b = (1 - lambd) * x + lambd * x_cf
                label = self.make_prediction(x_b)
                if label == 1 - x_label:
                    dist = self.dist(x, x_b)
                    if dist < best_dist:
                        best_x_b = x_b
                        best_dist = dist
                    break
        return best_x_b

    def sample_perturbations(self, x, radius=0.3, num_samples=5000, random_state=None):
        return uniform_ball(x, radius, num_samples, random_state)

    def explain_instance(self, x, perturb_radius=0.3, num_samples=5000):
        x_b = self.find_counterfactual(x)
        X_s = self.sample_perturbations(x_b, perturb_radius, num_samples, self.random_state)
        y_s = self.predict_fn(X_s)

        distances = np.ones(X_s.shape[0])

        (intercept, local_exp, _, _) = self.base.explain_instance_with_data(
            X_s, y_s, distances, 1, self.num_features, model_regressor=None, feature_selection="auto"
        )
        b = intercept - 0.5
        exp = local_exp
        exp = sorted(exp, key=lambda x: x[0])
        w = np.zeros(self.num_features)
        for e in exp:
            w[e[0]] = e[1]

        self.data = X_s
        self.data_pred = y_s
        return w, b
