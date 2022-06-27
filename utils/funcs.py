import numpy as np
import scipy as sp
from scipy.linalg import eigh, sqrtm
from sklearn.utils import check_random_state


def gelbrich_dist(mean_0, cov_0, mean_1, cov_1):
    t1 = np.linalg.norm(mean_0 - mean_1)
    t2 = np.trace(cov_0 + cov_1 - 2 * sqrtm(sqrtm(cov_1) @ cov_0 @ sqrtm(cov_1)))
    return np.sqrt(t1**2 + t2)


def check_symmetry(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def sqrtm_psd(A, check_finite=True):
    A = np.asarray(A)
    if len(A.shape) != 2:
        raise ValueError("Non-matrix input to matrix function.")
    w, v = eigh(A, check_finite=check_finite)
    w = np.maximum(w, 0)
    return (v * np.sqrt(w)).dot(v.conj().T)


def lp_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=p)


def l2_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=2)


def l1_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=1)


def uniform_ball(x, r, n, random_state=None):
    # muller method
    random_state = check_random_state(random_state)
    d = len(x)
    V_x = random_state.randn(n, d)
    V_x = V_x / np.linalg.norm(V_x, axis=1).reshape(-1, 1)
    V_x = V_x * (random_state.random(n) ** (1.0 / d)).reshape(-1, 1)
    V_x = V_x * r + x
    return V_x


def normalize_exp(w, b):
    m = np.linalg.norm(np.hstack([w, b]), 2)
    return w / m, b / m


def compute_robustness(exps):
    w0, b0 = exps[0]
    w0, _ = normalize_exp(w0, b0)
    # e0 = np.hstack([w0, b0])
    ret = 0

    for w, b in exps[1:]:
        # e = np.hstack([w, b])
        w, _ = normalize_exp(w, b)
        ret = max(ret, np.linalg.norm(w0 - w, 2))
        # ret = max(ret, sp.spatial.distance.cosine(e, e0))

    return ret


def compute_fidelity_on_samples(exps, X, y):
    w, b = exps
    exp_pred = X @ w.T + b >= 0
    ret = np.sum(y == exp_pred) / len(y)
    return ret


def compute_fidelity(x, e, predict_fn, r_fid, num_samples=1000, random_state=None, return_data=False):
    V_x = uniform_ball(x, r_fid, num_samples, random_state)
    y = np.argmax(predict_fn(V_x), axis=-1)
    w, b = e
    y_ls = V_x @ w.T + b >= 0
    ret = np.mean(y == y_ls)
    if return_data:
        return ret, V_x, y
    else:
        return ret


def compute_max_distance(x):
    max_dist = -np.inf
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d = np.linalg.norm(x[i] - x[j], ord=2)
            max_dist = max(max_dist, d)

    return max_dist


def quadratic_divergence(covsa, cov):
    return np.trace((covsa - cov) @ (covsa - cov))


def bures_divergence(covsa, cov):
    covsa_sqrtm = sqrtm_psd(covsa)
    return np.trace(covsa + cov - 2 * sqrtm_psd(covsa_sqrtm @ cov @ covsa_sqrtm))


def fisher_rao_distance(covsa, cov):
    covsa_sqrtm_inv = np.linalg.inv(sqrtm_psd(covsa))
    return np.linalg.norm(sp.linalg.logm(covsa_sqrtm_inv @ cov @ covsa_sqrtm_inv))


distance_funcs = {
    "l2": l2_dist,
}


def compute_max_shift(covsa, covs, metric="l2"):
    max_dist = -np.inf
    for cov in covs:
        d = distance_funcs[metric](covsa, cov)
        max_dist = max(max_dist, d)
    return max_dist


def is_dominated(a, b):
    dominated = False
    for i in range(len(a)):
        if a[i] > b[i]:
            return False
        elif a[i] < b[i]:
            dominated = True
    return dominated


def find_pareto(x, y):
    a = list(zip(x, y))
    a = sorted(a, key=lambda x: (x[0], -x[1]))
    best = -1
    pareto = []
    for e in a:
        if e[1] >= best:
            pareto.append(e)
            best = e[1]

    return [e[0] for e in pareto], [e[1] for e in pareto]
