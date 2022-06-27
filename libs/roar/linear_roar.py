import numpy as np
import torch
import torch.nn as nn


def reconstruct_encoding_constraints(x, cat_pos):
    x_enc = x.clone()
    for pos in cat_pos:
        x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
    return x_enc


class LinearROAR(object):
    """Class for generate counterfactual samples for framework: ROAR"""

    DECISION_THRESHOLD = 0.5

    def __init__(
        self,
        data,
        coef,
        intercept,
        cat_indices=list(),
        lambd=0.1,
        delta_min=None,
        delta_max=0.1,
        lr=0.5,
        dist_type=1,
        max_iter=20,
        encoding_constraints=False,
    ):
        self.data = data
        self.intercept_ = intercept / np.linalg.norm(coef, 2)
        self.coef_ = coef / np.linalg.norm(coef, 2)
        self.cat_indices = cat_indices
        self.lambd = lambd
        self.lr = lr
        self.dist_type = dist_type
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.max_iter = max_iter
        self.encoding_constraints = encoding_constraints

    def f(self, x, w, b):
        return torch.sigmoid(torch.dot(x, w) + b)

    def logit(self, x, w, b):
        return torch.dot(x, w) + b

    def fit_instance(self, x_0, verbose=False):
        (d,) = x_0.shape
        x_0 = torch.tensor(x_0.copy()).float()
        x_t = x_0.clone().detach().requires_grad_(True)
        x_enc = reconstruct_encoding_constraints(x_t, self.cat_indices)

        w = torch.from_numpy(self.coef_.copy()).float()
        b = torch.tensor(self.intercept_).float()
        y_target = torch.tensor(1).float()
        lambda_ = torch.tensor(self.lambd).float()
        lr = torch.tensor(self.lr).float()
        loss_fn = nn.BCELoss()

        glob_it = 0

        while True:
            w_, b_ = w, b
            it = 0
            while it < self.max_iter:
                if x_t.grad is not None:
                    x_t.grad.data.zero_()

                if self.encoding_constraints:
                    x_enc = reconstruct_encoding_constraints(x_t, self.cat_indices)
                else:
                    x_enc = x_t.clone()

                with torch.no_grad():
                    lar_mul = self.delta_max / torch.sqrt(torch.linalg.norm(x_enc) ** 2 + 1)
                    delta_w = -x_enc * lar_mul
                    delta_b = -lar_mul
                    w_ = w + delta_w
                    b_ = b + delta_b

                f_x = torch.sigmoid(torch.dot(x_enc, w_) + b_).float()
                cost = torch.dist(x_enc, x_0, self.dist_type)
                f_loss = loss_fn(f_x, y_target)

                loss = f_loss + lambda_ * cost
                loss.backward()

                if verbose:
                    with torch.no_grad():
                        print("=" * 10, " Iter ", it, "=" * 10)
                        print("Loss: ", loss.data.item())
                        print("-" * 10)
                        print(
                            "Loss components: f_loss: ",
                            f_loss.data.item(),
                            "; cost: ",
                            lambda_.data.item() * cost.data.item(),
                        )
                        print("w_: ", w_, "; b_: ", b_)
                        print("x_enc: ", x_enc.data)
                        print("f_x:", f_x.data.item())
                        print("f_x w.r.t. (w_, b_): ", self.f(x_enc, w_, b_))
                        print("f_x w.r.t. (w, b): ", self.f(x_enc, w, b))
                        print("x_t.grad:", x_t.grad)

                with torch.no_grad():
                    x_t -= lr * x_t.grad
                it += 1
                if f_x >= LinearROAR.DECISION_THRESHOLD:
                    break

            lambda_ *= 0.5
            glob_it += 1
            if verbose:
                with torch.no_grad():
                    print("F_x = ", f_x.data.item())
                    print("Logit w.r.t. (w, b): ", self.logit(x_enc, w, b).data.item())
                    print("Logit w.r.t. (w_, b_): ", self.logit(x_enc, w_, b_).data.item())

            if glob_it >= 10 or f_x.data.item() > LinearROAR.DECISION_THRESHOLD:
                break

        self.feasible = f_x.data.item() > LinearROAR.DECISION_THRESHOLD

        return x_enc.detach().numpy()
