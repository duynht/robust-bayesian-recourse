import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.utils import check_random_state
from .rbr_loss import RBRLoss


plt.gca().set_aspect("equal", adjustable="box")
obj_fig, obj_ax = plt.subplots()
point_fig, point_ax = plt.subplots()
pe_fig, pe_ax = plt.subplots()
op_fig, op_ax = plt.subplots()
grad0_fig, grad0_ax = plt.subplots()
grad1_fig, grad1_ax = plt.subplots()


def power10floor(x):
    return 10 ** math.floor(math.log10(x))


def reconstruct_encoding_constraints(x, cat_pos):
    x_enc = x.clone()
    for pos in cat_pos:
        x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
    return x_enc


class RBR(object):
    """Class for generate counterfactual samples for framework: Wachter"""

    DECISION_THRESHOLD = 0.5

    def __init__(self, model, train_data, y_target=1, num_cfacts=10, max_iter=1000, random_state=None, device="cuda"):
        self.random_state = check_random_state(random_state)
        self.model = model
        self.max_iter = max_iter
        self.y_target = y_target
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = "cpu"
        self.feasible = True
        self.train_data = torch.tensor(train_data).float()
        self.train_label = self.make_prediction(self.train_data).to(self.device)
        self.train_data = self.train_data.to(self.device)
        self.num_cfacts = num_cfacts

    def make_prediction(self, x):
        return torch.tensor(self.model.predict(x.cpu().detach().numpy()))

    def dist(self, a, b):
        return torch.linalg.norm(a - b, ord=1, axis=-1)

    def find_x_boundary(self, x, k=None):
        k = k or self.num_cfacts
        x_label = self.make_prediction(x)

        d = self.dist(self.train_data, x)
        order = torch.argsort(d)
        x_cfact_list = self.train_data[order[self.train_label[order] == (1 - x_label)]][:k]
        best_x_b = None
        best_dist = torch.tensor(float("inf"))

        for x_cfact in x_cfact_list:
            lambd_list = torch.linspace(0, 1, 100)
            for lambd in lambd_list:
                x_b = (1 - lambd) * x + lambd * x_cfact
                label = self.make_prediction(x_b)
                if label == 1 - x_label:
                    dist = self.dist(x, x_b)
                    if dist < best_dist:
                        best_x_b = x_b
                        best_dist = dist
                    break
        return best_x_b, best_dist

    def uniform_ball(self, x, r, n, random_state=None):
        # muller method
        rng = check_random_state(random_state)
        d = len(x)
        V_x = rng.randn(n, d)
        V_x = V_x / np.linalg.norm(V_x, axis=1).reshape(-1, 1)
        V_x = V_x * (rng.random(n) ** (1.0 / d)).reshape(-1, 1)
        V_x = V_x * r + x.numpy()
        return torch.from_numpy(V_x).to(self.device)

    def feasible_set(self, x, radius=0.3, num_samples=20, random_state=None):
        return self.uniform_ball(x.cpu(), radius, num_samples, random_state)

    def simplex_projection(self, x, delta):
        """
        Euclidean projection on a positive simplex
        """
        (p,) = x.shape
        if torch.linalg.norm(x, ord=1) == delta and torch.all(x >= 0):
            return x
        u, _ = torch.sort(x, descending=True)
        cssv = torch.cumsum(u, 0)
        rho = torch.nonzero(u * torch.arange(1, p + 1).to(self.device) > (cssv - delta))[-1, 0]
        theta = (cssv[rho] - delta) / (rho + 1.0)
        w = torch.clip(x - theta, min=0)
        return w

    def projection(self, x, delta):
        """
        Euclidean projection on an L1-ball
        """
        x_abs = torch.abs(x)
        if x_abs.sum() <= delta:
            return x

        proj = self.simplex_projection(x_abs, delta=delta)
        proj *= torch.sign(x)

        return proj

    def optimize(self, x, delta, loss_fn, theta=0.7, beta=1, verbose=False):
        x_t = x.detach().clone()
        x_t.requires_grad_(True)

        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        F_list = []
        pe_list = []
        op_list = []
        grad0_list = []
        grad1_list = []

        for t in range(self.max_iter):

            if x_t.grad is not None:
                x_t.grad.data.zero_()
            F, pe, op = loss_fn(x_t)

            F.backward()
            if verbose:
                print(f"Iteration {t+1}/{self.max_iter}")
                print(x_t)
                print(x_t.grad)
                print(F)
                print(op)
                print(pe)
                print(self.dist(x_t.detach().cpu(), self.x0.detach().cpu()))
                if x_t.shape[0] == 2:
                    x_t_handle = x_t.detach()
                else:
                    x_t_handle = x_t.detach().unsqueeze(0) @ self.pca
                    x_t_handle = x_t_handle.squeeze()
                point_ax.scatter(x_t_handle[0].cpu().numpy(), x_t_handle[1].cpu().numpy(), color="black")
                point_ax.arrow(
                    x_t_handle[0].cpu().numpy(),
                    x_t_handle[1].cpu().numpy(),
                    1 / np.sqrt(1e3) * x_t.grad[0].detach().cpu().numpy(),
                    1 / np.sqrt(1e3) * x_t.grad[1].detach().cpu().numpy(),
                    head_width=0.05,
                    head_length=0.1,
                )
                point_fig.savefig("points.png")
                F_list.append(F.detach().cpu().numpy())
                pe_list.append(pe.detach().cpu().numpy())
                op_list.append(op.detach().cpu().numpy())
                grad0_list.append(x_t.grad.data[0].item())
                grad1_list.append(x_t.grad.data[1].item())

            if torch.ge(self.dist(x_t.detach(), self.x0), delta):
                break

            with torch.no_grad():
                x_new = x_t - 1 / torch.sqrt(torch.tensor(1e3, device=self.device)) * x_t.grad
                if verbose:
                    if x_new.shape[0] == 2:
                        x_new_handle = x_new.detach()
                    else:
                        x_new_handle = x_new.detach().unsqueeze(0) @ self.pca
                        x_new_handle = x_new_handle.squeeze()
                    point_ax.scatter(x_new_handle[0].cpu().numpy(), x_new_handle[1].cpu().numpy(), color="cyan")
                    point_fig.savefig("points.png")

                x_new = self.projection(x_new - self.x0, delta) + self.x0  # shift to origin before project

            for i, elem in enumerate(x_new.data):
                x_t.data[i] = elem

            if verbose:
                obj_ax.plot(F_list)
                obj_fig.savefig("loss.png")
                pe_ax.plot(pe_list)
                pe_fig.savefig("pessimistic.png")
                op_ax.plot(op_list)
                op_fig.savefig("optimistic.png")
                grad0_ax.plot(grad0_list)
                grad0_fig.savefig("grad0.png")
                grad1_ax.plot(grad1_list)
                grad1_fig.savefig("grad1.png")

            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum

            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0

            min_loss = min(min_loss, loss_sum)

        return F, x_t.detach()

    def fit_instance(
        self, x0, num_samples, perturb_radius, delta_plus, sigma, epsilon_op, epsilon_pe, ec, verbose=False
    ):
        x0 = torch.from_numpy(x0.copy()).float().to(self.device)
        self.x0 = x0
        self.delta_plus = delta_plus
        x = x0.clone()
        x_b, delta_base = self.find_x_boundary(x)

        x_b = x_b.detach().clone()
        delta_base = delta_base.detach().clone()
        delta = delta_base + delta_plus

        X_feas = self.feasible_set(
            x_b, radius=perturb_radius, num_samples=num_samples, random_state=self.random_state
        ).float()

        if verbose:
            _, _, self.pca = torch.pca_lowrank(X_feas)
            self.pca = self.pca[:, :2]

        y_feas = self.make_prediction(X_feas).to(self.device)

        X_feas_pos = X_feas[y_feas == self.y_target].reshape([sum(y_feas == self.y_target), -1])
        X_feas_neg = X_feas[y_feas == (1 - self.y_target)].reshape([sum(y_feas == (1 - self.y_target)), -1])

        if verbose:
            if x0.shape[0] == 2:
                x0_handle = x0.detach()
            else:
                x0_handle = x0.detach().unsqueeze(0) @ self.pca
                x0_handle = x0_handle.squeeze()

            if X_feas_pos.shape[1] == 2:
                X_feas_pos_handle = X_feas_pos
            else:
                X_feas_pos_handle = X_feas_pos.detach() @ self.pca

            if X_feas_neg.shape[1] == 2:
                X_feas_neg_handle = X_feas_neg
            else:
                X_feas_neg_handle = X_feas_neg.detach() @ self.pca

            point_ax.scatter(x0_handle[0].cpu().numpy(), x0_handle[1].cpu().numpy(), color="blue")
            point_ax.scatter(
                [x[0].cpu().numpy() for x in X_feas_neg_handle],
                [x[1].cpu().numpy() for x in X_feas_neg_handle],
                color="red",
            )
            point_ax.scatter(
                [x[0].cpu().numpy() for x in X_feas_pos_handle],
                [x[1].cpu().numpy() for x in X_feas_pos_handle],
                color="green",
            )
            point_fig.savefig("points.png")

        loss_fn = RBRLoss(X_feas, X_feas_pos, X_feas_neg, epsilon_op, epsilon_pe, sigma, device=self.device)

        loss, x = self.optimize(x_b, delta, loss_fn, verbose=verbose)

        self.feasible = self.make_prediction(x) == self.y_target
        return x.cpu().detach().numpy().squeeze()


def generate_recourse(x0, model, random_state, params=dict()):
    train_data = params["train_data"]

    ec = params["config"]
    arg = RBR(model, train_data, num_cfacts=1000, max_iter=500, random_state=random_state, device=params["device"])

    x_ar = arg.fit_instance(
        x0,
        ec.num_samples,
        params["perturb_radius"] * ec.max_distance,
        ec.rbr_params["delta_plus"],
        ec.rbr_params["sigma"],
        ec.rbr_params["epsilon_op"],
        ec.rbr_params["epsilon_pe"],
        ec,
    )
    report = dict(feasible=arg.feasible)

    return x_ar, report
