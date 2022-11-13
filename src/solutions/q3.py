from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln, logsumexp
from sklearn.manifold import TSNE

from src.constants import DEFAULT_SEED


@dataclass
class Theta:
    """
    Data class containing the model parameters
    log_pi: the logarithm of the mixing proportions (1, k)
    log_p_matrix: the logarithm of the probability where the (i,j)th element is the probability that
                  pixel j takes value 1 under mixture component i (d, k)
    """

    log_pi: np.ndarray
    log_p_matrix: np.ndarray

    @property
    def pi(self) -> np.ndarray:
        """
        Calculates the mixing proportions
        :return: vector of mixing proportions (1, k)
        """
        return np.exp(self.log_pi)

    @property
    def p_matrix(self) -> np.ndarray:
        """
        Calculates the Bernoulli parameters
        :return: matrix Bernoulli parameters (d, k)
        """
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

    def log_pi_repeated(self, n: int) -> np.ndarray:
        """
        Repeats the log_pi vector n times along axis 0
        :param n: number of repetitions
        :return: an array of shape (n, k)
        """
        return np.repeat(self.log_pi, n, axis=0)


def _init_params(k: int, d: int) -> Theta:
    """
    Random initialisation of theta parameters (log_pi and log_p_matrix)
    :param k: Number of components
    :param d: Image dimension (number of pixels in a single image)
    :return: theta: the parameters of the model
    """
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


def _compute_log_prior(
    theta: Theta, alpha_parameter: float, beta_parameter: float
) -> float:
    """
    Compute the prior log probability of the P matrix under a Beta prior
    :param theta: the parameters of the model
    :param alpha_parameter: alpha parameter of the beta prior
    :param beta_parameter: beta parameter of the beta prior
    :return: log_p_of_p_matrix
    """
    return np.sum(
        -betaln(alpha_parameter, beta_parameter)
        + (alpha_parameter - 1) * theta.log_p_matrix
        + (beta_parameter - 1) * theta.log_one_minus_p_matrix
    ).item()


def _compute_unnormalised_log_posterior_likelihood(
    x: np.ndarray, theta: Theta, alpha_parameter: float, beta_parameter: float
) -> float:
    """
    Compute the unnormalised posterior log probability of the P matrix
    :param x: the image data (n, d)
    :param theta: the parameters of the model
    :param alpha_parameter: alpha parameter of the beta prior
    :param beta_parameter: beta parameter of the beta prior
    :return: log_p_of_p_matrix
    """
    log_likelihood = _compute_log_likelihood(x, theta)
    log_prior = _compute_log_prior(theta, alpha_parameter, beta_parameter)
    return log_likelihood + log_prior


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
    return (logsumexp(log_responsibility, axis=0) - np.log(n)).reshape(1, -1)


def _compute_log_p_matrix_hat(
    x: np.ndarray,
    log_responsibility: np.ndarray,
    alpha_parameter: float,
    beta_parameter: float,
) -> np.ndarray:
    """
    Compute the log of the maximised pixel probabilities
    :param x: the image data (n, d)
    :param log_responsibility: an array of the log responsibilities of k mixture components for each image (n, k)
    :param alpha_parameter: alpha parameter of the beta prior
    :param beta_parameter: beta parameter of the beta prior
    :return: an array of the maximised pixel probabilities for each component (d, k)
    """
    n, d = x.shape
    _, k = log_responsibility.shape

    x_repeated = np.repeat(x[:, :, np.newaxis], k, axis=2)  # (n, d, k)
    log_responsibility_repeated = np.repeat(
        log_responsibility[:, np.newaxis, :], d, axis=1
    )  # (n, d, k)

    log_p_matrix_unnormalised_posterior = logsumexp(
        log_responsibility_repeated, b=(x_repeated + alpha_parameter - 1), axis=0
    )  # (d, k)

    log_p_matrix_normaliser_posterior = logsumexp(
        log_responsibility_repeated, b=(alpha_parameter + beta_parameter - 1), axis=0
    )  # (d, k)

    log_p_matrix_normalised_posterior = (
        log_p_matrix_unnormalised_posterior - log_p_matrix_normaliser_posterior
    )  # (d, k)
    return log_p_matrix_normalised_posterior


def _compute_log_m_step(
    x: np.ndarray,
    log_responsibility: np.ndarray,
    alpha_parameter: float,
    beta_parameter: float,
) -> Theta:
    """
    Compute the m step of expectation maximisation
    :param x: the image data (n, d)
    :param log_responsibility: an array of the log responsibilities of k mixture components for each image (n, k)
    :param alpha_parameter: alpha parameter of the beta prior
    :param beta_parameter: beta parameter of the beta prior
    :return: thetas optimised after maximisation step
    """
    return Theta(
        log_pi=_compute_log_pi_hat(log_responsibility),
        log_p_matrix=_compute_log_p_matrix_hat(
            x, log_responsibility, alpha_parameter, beta_parameter
        ),
    )


def _run_expectation_maximisation(
    x: np.ndarray,
    theta: Theta,
    alpha_parameter: float,
    beta_parameter: float,
    max_number_of_steps: int,
    epsilon: float,
) -> Tuple[Theta, np.ndarray, List[float]]:
    """
    Run the expectation maximisation algorithm
    :param x: the image data (n, d)
    :param theta: initial theta parameters
    :param alpha_parameter: alpha parameter of the beta prior
    :param beta_parameter: beta parameter of the beta prior
    :param max_number_of_steps: the maximum number of steps to run the algorithm
    :param epsilon: the minimum required change in log posterior, otherwise the algorithm stops early
    :return: a tuple containing the optimised thetas, the log responsibilities,
             and the log log_posteriors at each step of the algorithm
    """
    log_responsibility = None
    log_posteriors = []
    for _ in range(max_number_of_steps):
        log_responsibility = _compute_log_e_step(x, theta)
        theta = _compute_log_m_step(
            x, log_responsibility, alpha_parameter, beta_parameter
        )

        log_posteriors.append(
            _compute_unnormalised_log_posterior_likelihood(
                x, theta, alpha_parameter, beta_parameter
            )
        )

        #  check for early stopping
        if len(log_posteriors) > 1:
            if (log_posteriors[-1] - log_posteriors[-2]) < epsilon:
                break
    return theta, log_responsibility, log_posteriors


def _visualise_p_matrix(
    thetas: List[Theta], ks: List[int], figure_title: str, figure_path: str
) -> None:
    """
    Visualises the P matrix for different thetas and ks
    :param thetas: list of Theta instances
    :param ks: list of k values used for each Theta
    :param figure_title: name of figure
    :param figure_path: path to store figure
    :return:
    """
    n = len(ks)
    m = np.max(ks)
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(10)
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
            ax.set_title(f"pi_{j}: {np.round(thetas[i].pi[0, j], 2)}")
            if j == 0:
                ax.set_ylabel(f"{k=}")
    fig.suptitle(figure_title)
    plt.savefig(figure_path)


def _visualise_responsibility_clusters(
    log_responsibilities: List[np.ndarray],
    ks: List[int],
    figure_title: str,
    figure_path: str,
) -> None:
    """
    Visualise responsibility vectors of images using TSNE for different k values
    :param log_responsibilities: list of log responsibilities for different ks
    :param ks: list of k values used for each Theta
    :param figure_title: name of figure
    :param figure_path: path to store figure
    :return:
    """
    n = len(ks)
    fig = plt.figure()
    fig.set_figwidth(5 * n)
    fig.set_figheight(5)
    for i, k in enumerate(ks):
        if k > 2:
            # use TSNE when we have more than 2 dimensions
            embedding = TSNE(
                n_components=2,
                learning_rate="auto",
                init="random",
                perplexity=10,
                random_state=DEFAULT_SEED,
            ).fit_transform(log_responsibilities[i])
        else:
            # otherwise we can visualise responsibility vectors without dimensionality reduction
            embedding = np.exp(log_responsibilities[i])
        ax = plt.subplot(1, n, i + 1)
        ax.scatter(embedding[:, 0], embedding[:, 1])
        ax.set_title(f"{k=}")
    fig.suptitle(figure_title)
    plt.savefig(figure_path, bbox_inches="tight")


def _plot_log_posteriors(
    log_posteriors: List[List[float]],
    ks: List[int],
    epsilon: float,
    figure_title: str,
    figure_path: str,
) -> None:
    """
    Plot log posteriors as a function of EM iteration for different ks
    :param log_posteriors: list of vectors, each representing the log posterior during EM for a specific k
    :param ks: list of k values used for each Theta
    :param epsilon: value used for early stopping of EM
    :param figure_title: name of figure
    :param figure_path: path to store figure
    :return:
    """
    fig, ax = plt.subplots(len(ks), 1, constrained_layout=True)
    fig.set_figwidth(10)
    fig.set_figheight(10)
    for i, k in enumerate(ks):
        ax[i].plot(np.arange(1, len(log_posteriors[i]) + 1), log_posteriors[i])
        ax[i].set_xlabel("Step")
        ax[i].set_ylabel(f"Unnorm'd Log-Posterior")
        ax[i].set_title(f"{k=} {epsilon=}")
    plt.suptitle(figure_title)

    plt.savefig(figure_path)


def e(
    x: np.ndarray,
    alpha_parameter: float,
    beta_parameter: float,
    number_of_trials: int,
    ks: List[int],
    epsilon: float,
    max_number_of_steps: int,
    figure_path: str,
    figure_title: str,
) -> None:
    """
    Produces answers for question 3e
    :param x: numpy array of shape (N, D)
    :param alpha_parameter: alpha parameter of the beta prior
    :param beta_parameter: beta parameter of the beta prior
    :param number_of_trials: number of trails to run EM
    :param ks: k values to use for each trial
    :param epsilon: value used for early stopping of EM
    :param max_number_of_steps: maximum number of steps during EM
    :param figure_title: base name of figures
    :param figure_path: base paths to store figure
    :return:
    """
    n, d = x.shape
    np.random.seed(DEFAULT_SEED)
    for i in range(number_of_trials):
        init_thetas: List[Theta] = []
        em_thetas: List[Theta] = []
        log_posteriors: List[List[float]] = []
        log_responsibilities: List[np.ndarray] = []
        for j, k in enumerate(ks):
            init_theta = _init_params(k, d)
            em_theta, log_responsibility, log_posterior = _run_expectation_maximisation(
                x,
                theta=init_theta,
                alpha_parameter=alpha_parameter,
                beta_parameter=beta_parameter,
                epsilon=epsilon,
                max_number_of_steps=max_number_of_steps,
            )
            init_thetas.append(init_theta)
            em_thetas.append(em_theta)
            log_responsibilities.append(log_responsibility)
            log_posteriors.append(log_posterior)

        _visualise_p_matrix(
            init_thetas,
            ks,
            figure_title=f"{figure_title} Trial {i}: Initialised P",
            figure_path=f"{figure_path}-{i}-initialised-p.png",
        )
        _visualise_p_matrix(
            em_thetas,
            ks,
            figure_title=f"{figure_title} Trial {i}: EM Optimised P",
            figure_path=f"{figure_path}-{i}-optimised-p.png",
        )
        _visualise_responsibility_clusters(
            log_responsibilities,
            ks,
            figure_title=f"{figure_title} Trial {i}: TSNE Responsibility Visualisation",
            figure_path=f"{figure_path}-{i}-tsne.png",
        )
        _plot_log_posteriors(
            log_posteriors,
            ks,
            epsilon,
            figure_title=f"{figure_title} Trial {i}: Unnormalised Log-Posterior",
            figure_path=f"{figure_path}-{i}-log-pos.png",
        )
