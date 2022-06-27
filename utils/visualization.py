import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse


def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def visualize_explanations(
    model,
    X=None,
    y=None,
    x_test=None,
    lines=None,
    mean_neg=None,
    cov_neg=None,
    mean_pos=None,
    cov_pos=None,
    line_labels=None,
    xlim=None,
    ylim=None,
    N=1000,
    name=1,
    show=False,
    save=False,
):
    if xlim is None:
        xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    else:
        xmin, xmax = xlim
    if ylim is None:
        ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
    else:
        ymin, ymax = ylim

    fig, ax = plt.subplots()

    xd = np.linspace(xmin, xmax, N)
    yd = np.linspace(ymin, ymax, N)
    X_mesh, Y_mesh = np.meshgrid(xd, yd)

    if X is not None and y is not None:
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=8, alpha=0.5)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=8, alpha=0.5)

    colors = ["red", "green", "blueviolet", "magenta", "darkblue", "olive"]
    line_labels = line_labels or ["LIME"]

    if lines is not None:
        i = 0
        for w, b in lines:
            xdd = np.array([xmin, xmax])
            ydd = -w[0] / w[1] * xdd - b / w[1]
            ax.plot(xdd, ydd, colors[i], lw=2, ls="--", label=line_labels[i], zorder=30)
            i += 1

    plt.rcParams.update({"font.size": 17})
    if mean_neg is not None:
        ax.scatter(mean_neg[0], mean_neg[1], color="olive", marker="^", zorder=20, s=40, label="$\hat{\mu}_{-1}$")
        if cov_neg is not None:
            confidence_ellipse(mean_neg, cov_neg, ax, edgecolor="olive", zorder=20, label="$\hat{\Sigma}_{-1}$")

    if mean_pos is not None:
        ax.scatter(mean_pos[0], mean_pos[1], color="brown", marker="v", zorder=20, s=40, label="$\hat{\mu}_{+1}$")
        if cov_pos is not None:
            confidence_ellipse(mean_pos, cov_pos, ax, edgecolor="brown", zorder=20, label="$\hat{\Sigma}_{+1}$")

    if x_test is not None:
        ax.scatter(x_test[0], x_test[1], s=40, marker="*", zorder=20, color="black")
        ax.annotate("$x_0$", (x_test + np.full_like(x_test, 0.1)))

    ax.set_ylabel(r"$x_2$")
    ax.set_xlabel(r"$x_1$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()

    if save:
        os.makedirs("results/figures", exist_ok=True)
        plt.savefig(f"results/figures/example_{name}.png", dpi=500, bbox_inches="tight")
    if show:
        plt.show()


def plot_linear_model_2d(X, y, w, b):
    w1, w2 = w
    # Calculate the intercept and gradient of the decision boundary.
    c = b / w2
    m = -w1 / w2

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    print(xd)
    print(yd)
    ymin, ymax = min(ymin, np.min(yd)), max(ymax, np.max(yd))

    plt.plot(xd, yd, "k", lw=1, ls="--")
    plt.fill_between(xd, yd, ymin, color="tab:blue", alpha=0.2)
    plt.fill_between(xd, yd, ymax, color="tab:orange", alpha=0.2)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=8, alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=8, alpha=0.5)

    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")

    plt.show()
