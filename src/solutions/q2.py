import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp


def _log_p_d_given_m1(x):
    n, d = x.shape
    return n * d * np.log(0.5)


def _log_p_d_given_m2(x):
    n, d = x.shape
    k = np.sum(x, axis=0).astype(int)
    return betaln(np.sum(k) + 1, n * d - np.sum(k) + 1)


def _log_p_d_given_m3(x):
    n, _ = x.shape
    k = np.sum(x, axis=0).astype(int)
    return logsumexp(betaln(k + 1, n - k + 1))


def c(x, table_path):
    log_p_d_given_m = np.array(
        [
            _log_p_d_given_m1(x),
            _log_p_d_given_m2(x),
            _log_p_d_given_m3(x),
        ]
    )
    log_p_m_given_d = log_p_d_given_m - logsumexp(log_p_d_given_m)
    df = pd.DataFrame(
        data=np.array(
            [
                np.arange(len(log_p_m_given_d)).astype(int) + 1,
                [f"1E{int(x/np.log(10))}" for x in log_p_m_given_d[:-1]]
                + [
                    f"1-{'-'.join([f'(1E{int(x/np.log(10))})' for x in log_p_m_given_d[:-1]])}"
                ],
            ]
        ).T,
        columns=["Model", "P(M_i|D)"],
    )
    df.set_index("Model", inplace=True)
    df.to_csv(table_path)
