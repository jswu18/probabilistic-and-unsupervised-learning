import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp


def _log_p_d_given_m1(x: np.ndarray) -> float:
    """
    Calculates log likelihood of model 1
    :param x: numpy array of shape (N, D)
    :return: log likelihood
    """
    n, d = x.shape
    return n * d * np.log(0.5)


def _log_p_d_given_m2(x: np.ndarray):
    """
    Calculates log likelihood of model 2
    :param x: numpy array of shape (N, D)
    :return: log likelihood
    """
    n, d = x.shape
    k = np.sum(x).astype(int)
    return betaln(k + 1, n * d - k + 1)


def _log_p_d_given_m3(x: np.ndarray):
    """
    Calculates log likelihood of model 3
    :param x: numpy array of shape (N, D)
    :return: log likelihood
    """
    n, _ = x.shape
    k_d = np.sum(x, axis=0).astype(int)
    return logsumexp(betaln(k_d + 1, n - k_d + 1))


def _log_p_model_given_data(x) -> np.ndarray:
    """
    Calculates posterior log likelihood of models given image data
    :param x: numpy array of shape (N, D)
    :return: posterior log likelihood
    """
    log_p_d_given_m = np.array(
        [
            _log_p_d_given_m1(x),
            _log_p_d_given_m2(x),
            _log_p_d_given_m3(x),
        ]
    )
    log_p_m_given_data = log_p_d_given_m - logsumexp(log_p_d_given_m)
    return log_p_m_given_data


def c(x: np.ndarray, table_path: str) -> None:
    """
    Produces answers for question 2c
    :param x: numpy array of shape (N, D)
    :param table_path: path to store table posterior likelihoods
    :return:
    """
    log_p_m_given_data = _log_p_model_given_data(x)
    df = pd.DataFrame(
        data=np.array(
            [
                np.arange(len(log_p_m_given_data)).astype(int) + 1,
                [f"1E{int(x/np.log(10))}" for x in log_p_m_given_data[:-1]]
                + [
                    f"1-{'-'.join([f'(1E{int(x/np.log(10))})' for x in log_p_m_given_data[:-1]])}"
                ],
            ]
        ).T,
        columns=["Model", "P(M_i|D)"],
    )
    df.set_index("Model", inplace=True)
    df.to_csv(table_path)
