from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from src.constants import DEFAULT_SEED


@dataclass
class Theta:
    """
    log_pi: the logarithm of the mixing proportions (1, k)
    log_p_matrix: the logarithm of the probability where the (i,j)th element is the probability that
                  pixel j takes value 1 under mixture component i (d, k)
    """

    log_pi: np.ndarray
    log_p_matrix: np.ndarray

    @property
    def pi(self):
        return np.exp(self.log_pi)

    @property
    def p_matrix(self):
        d, k = self.log_p_matrix.shape
        image_dimension = int(np.sqrt(d))
        return np.exp(self.log_p_matrix).reshape(image_dimension, image_dimension, -1)

    @property
    def log_one_minus_p_matrix(self) -> np.ndarray:
        """
        Compute log(1-P) where P=exp(log_p_matrix)
        :return: an array of the same shape as log_p_matrix (d, k)
        """
        log_of_one = np.zeros(self.log_p_matrix.shape)
        stacked_sum = np.stack((log_of_one, self.log_p_matrix))
        weights = np.ones(stacked_sum.shape)
        weights[1] = -1  # scale p matrix by -1 for subtraction
        return np.array(logsumexp(stacked_sum, b=weights, axis=0))

    def log_pi_repeated(self, n: int):
        """
        Repeats the log_pi vector n times along axis 0
        :param n: number of repetitions
        :return: an array of shape (n, k)
        """
        return np.repeat(self.log_pi, n, axis=0)


def _init_params(k: int, d: int, seed: int = DEFAULT_SEED) -> Theta:
    """
    Random initialisation of theta parameters (log_pi and log_p_matrix)
    :param k: Number of components
    :param d: Image dimension (number of pixels in a single image)
    :param seed: seed initialisation for random methods
    :return: theta: the parameters of the model
    """
    np.random.seed(seed)
    return Theta(
        log_pi=np.log(np.random.dirichlet(np.ones(k), size=1)),
        log_p_matrix=np.log(np.random.uniform(low=0, high=1, size=(d, k))),
    )


def _compute_log_component_p_x_i_given_theta(x: np.ndarray, theta: Theta) -> np.ndarray:
    """
    Compute the unweighted probability of each mixing component for each image
    :param x: the image data (n, d)
    :param theta: the parameters of the model
    :return: an array of the unweighted probabilities (n, k)
    """
    return x @ theta.log_p_matrix + (1 - x) @ theta.log_one_minus_p_matrix


def _compute_log_p_x_i_given_theta(x: np.ndarray, theta: Theta) -> np.ndarray:
    """
    Computes the log likelihood of each image in the dataset x
    :param x: the image data (n, d)
    :param theta: the parameters of the model
    :return: log_p_x_i_given_theta: a log likelihood array containing the log likelihood of each image (n ,1)
    """
    n, _ = x.shape
    log_component_probabilities = _compute_log_component_p_x_i_given_theta(
        x, theta
    )  # (n, k)
    return np.array(
        logsumexp(
            log_component_probabilities
            + theta.log_pi_repeated(n),  # scale each component by component probability
            axis=1,
        )
    )


def _compute_log_likelihood(x: np.ndarray, theta: Theta) -> float:
    """
    Computes the log likelihood of all images in the dataset x
    :param x: the image data (n, d)
    :param theta: the parameters of the model
    :return: log_p_x_given_theta: the log likelihood array across all images
    """
    return np.sum(_compute_log_p_x_i_given_theta(x, theta)).item()


def _compute_log_e_step(x: np.ndarray, theta: Theta) -> np.ndarray:
    """
    Compute the e step of expectation maximisation
    :param x: the image data (n, d)
    :param theta: the parameters of the model
    :return: an array of the log responsibilities of k mixture components for each image (n, k)
    """
    log_r_unnormalised = _compute_log_component_p_x_i_given_theta(x, theta)
    log_r_normaliser = logsumexp(log_r_unnormalised, axis=1)
    log_responsibility = log_r_unnormalised - log_r_normaliser[:, np.newaxis]
    return log_responsibility


def _compute_log_pi_hat(log_responsibility: np.ndarray) -> np.ndarray:
    """
    Compute the log of the maximised mixing proportions
    :param log_responsibility: an array of the log responsibilities of k mixture components for each image (n, k)
    :return: an array of the maximised log mixing proportions (1, k)
    """
    n, _ = log_responsibility.shape
    summed_responsibilities = logsumexp(log_responsibility, axis=0)

    alpha = 2
    beta = 2
    summed_responsibilities = logsumexp(
        np.stack(
            (
                (alpha - 1) * np.ones(summed_responsibilities.shape),
                summed_responsibilities,
            ),
            axis=0,
        ),
        axis=0,
    )
    return (summed_responsibilities - np.log(n + alpha + beta - 2)).reshape(1, -1)


def _compute_log_p_matrix_hat(
    x: np.ndarray, log_responsibility: np.ndarray
) -> np.ndarray:
    """
    Compute the log of the maximised pixel probabilities
    :param x: the image data (n, d)
    :param log_responsibility: an array of the log responsibilities of k mixture components for each image (n, k)
    :return: an array of the maximised pixel probabilities for each component (d, k)
    """
    n, d = x.shape
    _, k = log_responsibility.shape

    x_repeated = np.repeat(x[:, :, np.newaxis], k, axis=2)  # (n, d, k)
    log_responsibility_repeated = np.repeat(
        log_responsibility[:, np.newaxis, :], d, axis=1
    )  # (n, d, k)

    log_p_matrix_unnormalised = logsumexp(
        log_responsibility_repeated, b=x_repeated, axis=0
    )  # (d, k)
    log_p_matrix_normaliser = np.array(
        logsumexp(log_responsibility_repeated, axis=0)
    )  # (d, k)

    alpha = 2
    beta = 2
    log_p_matrix_unnormalised = logsumexp(
        np.stack(
            (
                (alpha - 1) * np.ones(log_p_matrix_unnormalised.shape),
                log_p_matrix_unnormalised,
            ),
            axis=0,
        ),
        axis=0,
    )
    log_p_matrix_normaliser = logsumexp(
        np.stack(
            (
                (alpha + beta - 2) * np.ones(log_p_matrix_normaliser.shape),
                log_p_matrix_normaliser,
            ),
            axis=0,
        ),
        axis=0,
    )
    log_p_matrix_normalised = log_p_matrix_unnormalised - log_p_matrix_normaliser
    return log_p_matrix_normalised


def _compute_log_m_step(x: np.ndarray, log_responsibility: np.ndarray) -> Theta:
    """
    Compute the m step of expectation maximisation
    :param x: the image data (n, d)
    :param log_responsibility: an array of the log responsibilities of k mixture components for each image (n, k)
    :return: thetas optimised after maximisation step
    """
    return Theta(
        log_pi=_compute_log_pi_hat(log_responsibility),
        log_p_matrix=_compute_log_p_matrix_hat(x, log_responsibility),
    )


def _run_expectation_maximisation(
    x: np.ndarray, theta: Theta, max_number_of_steps: int, epsilon: float
) -> Tuple[Theta, List[float]]:
    """
    Run the expectation maximisation algorithm
    :param x: the image data (n, d)
    :param theta: initial theta parameters
    :param max_number_of_steps: the maximum number of steps to run the algorithm
    :param epsilon: the minimum required change in log likelihood, otherwise the algorithm stops early
    :return: a tuple containing the optimised thetas and the log likelihood at each step of the algorithm
    """
    log_likelihoods = []
    for _ in range(max_number_of_steps):
        log_r = _compute_log_e_step(x, theta)
        theta = _compute_log_m_step(x, log_r)

        log_likelihoods.append(_compute_log_likelihood(x, theta))

        #  check for early stopping
        if len(log_likelihoods) > 1:
            if (log_likelihoods[-1] - log_likelihoods[-2]) < epsilon:
                break
    return theta, log_likelihoods


def _plot_p_matrix(
    thetas: List[Theta], ks: List[int], figure_title: str, figure_path: str
):
    n = len(ks)
    m = np.max(ks)
    fig = plt.figure()
    for i, k in enumerate(ks):
        for j in range(k):
            ax = plt.subplot(n, m, m * i + j + 1)
            ax.imshow(
                thetas[i].p_matrix[:, :, j],
                interpolation="None",
            )
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
            )
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
            )
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if j == 0:
                ax.set_ylabel(f"{k=}")
    fig.suptitle(figure_title)
    plt.savefig(figure_path)


def _plot_log_likelihoods(
    log_likelihoods: List[List[float]],
    ks: List[int],
    figure_title: str,
    figure_path: str,
) -> None:
    fig, ax = plt.subplots(len(ks), 1, constrained_layout=True)
    fig.set_figwidth(10)
    fig.set_figheight(15)
    for i, k in enumerate(ks):
        ax[i].plot(np.arange(1, len(log_likelihoods[i]) + 1), log_likelihoods[i])
        ax[i].set_xlabel("Step")
        ax[i].set_ylabel("Log-Likelihood")
        ax[i].set_title(f"{k=}")
    plt.suptitle(figure_title)

    plt.savefig(figure_path)


def e(
    x: np.ndarray,
    number_of_trials: int,
    ks: List[int],
    epsilon: float,
    max_number_of_steps: int,
    figure_path: str,
    figure_title: str,
) -> None:
    n, d = x.shape
    seeds = np.random.randint(
        low=number_of_trials * len(ks), size=(number_of_trials, len(ks))
    )
    for i in range(number_of_trials):
        init_thetas = []
        em_thetas = []
        log_likelihoods = []
        for j, k in enumerate(ks):
            init_theta = _init_params(k, d, seed=seeds[i, j])
            em_theta, log_likelihood = _run_expectation_maximisation(
                x,
                theta=init_theta,
                epsilon=epsilon,
                max_number_of_steps=max_number_of_steps,
            )
            init_thetas.append(init_theta)
            em_thetas.append(em_theta)
            log_likelihoods.append(log_likelihood)

        _plot_p_matrix(
            init_thetas,
            ks,
            figure_title=f"{figure_title} Trial {i}: Initialised P",
            figure_path=f"{figure_path}-{i}-initialised-p.png",
        )
        _plot_p_matrix(
            em_thetas,
            ks,
            figure_title=f"{figure_title} Trial {i}: EM Optimised P",
            figure_path=f"{figure_path}-{i}-optimised-p.png",
        )
        _plot_log_likelihoods(
            log_likelihoods,
            ks,
            figure_title=f"{figure_title} Trial {i}: Negative Log-Likelihood",
            figure_path=f"{figure_path}-{i}-log-like.png",
        )
