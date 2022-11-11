import matplotlib.pyplot as plt
import numpy as np


def _compute_maximum_likelihood_estimate(x: np.ndarray) -> np.ndarray:
    """
    Calculates MLE of images
    :param x: numpy array of shape (N, D)
    :return: MLE estimate
    """
    return np.mean(x, axis=0)


def _compute_maximum_a_priori_estimate(
    x: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """
    Calculates MAP estimate of images
    :param x: numpy array of shape (N, D)
    :param alpha: param of prior distribution
    :param beta: param of prior distribution
    :return: MAP estimate
    """

    n, _ = x.shape
    return (alpha - 1 + np.sum(x, axis=0)) / (n + alpha + beta - 2)


def d(x: np.ndarray, figure_path: str, figure_title: str) -> None:
    """
    Produces answers for question 1d
    :param x: numpy array of shape (N, D)
    :param figure_path: path to store figure
    :param figure_title: figure title
    :return:
    """
    maximum_likelihood = _compute_maximum_likelihood_estimate(x)
    plt.figure()
    plt.imshow(
        np.reshape(maximum_likelihood, (8, 8)),
        interpolation="None",
    )
    plt.colorbar()
    plt.axis("off")
    plt.title(figure_title)
    plt.savefig(figure_path)


def e(
    x: np.ndarray, alpha: float, beta: float, figure_path: str, figure_title: str
) -> None:
    """
    Produces answers for question 1e
    :param x: numpy array of shape (N, D)
    :param alpha: param of prior distribution
    :param beta: param of prior distribution
    :param figure_path: path to store figure
    :param figure_title: figure title
    :return:
    """
    maximum_a_priori = _compute_maximum_a_priori_estimate(x, alpha, beta)
    plt.figure()
    plt.imshow(
        np.reshape(maximum_a_priori, (8, 8)),
        interpolation="None",
    )
    plt.colorbar()
    plt.axis("off")
    plt.title(figure_title)
    plt.savefig(f"{figure_path}.png")

    maximum_likelihood = _compute_maximum_likelihood_estimate(x)
    plt.figure()
    plt.imshow(
        np.reshape(maximum_a_priori - maximum_likelihood, (8, 8)),
        interpolation="None",
    )
    plt.colorbar()
    plt.axis("off")
    plt.title(f"MAP vs MLE")
    plt.savefig(f"{figure_path}-mle-vs-map.png")
