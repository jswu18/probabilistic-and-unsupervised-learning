from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src.constants import DEFAULT_SEED


class Decrypter:
    def __init__(self, decryption_dict):
        self.decryption_dict = decryption_dict

    def decrypt(self, encrypted_message):
        return "".join([self.decryption_dict[x] for x in encrypted_message])


class Statistics:
    def __init__(
        self,
        training_text: str,
        symbols: List[str],
        invariant_stopping_epsilon: float = 5e-20,
    ):
        self.training_text = training_text
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.symbols_dict = {k: v for v, k in enumerate(symbols)}
        self.text_numbers = [
            self.symbols_dict[symbol]
            for symbol in list(training_text)
            if symbol in self.symbols_dict
        ]
        self.transition_matrix = self._construct_transition_matrix(
            training_text, self.symbols_dict
        )
        self.invariant_distribution = self._approximate_invariant_distribution(
            invariant_stopping_epsilon
        )
        self.log_transition_matrix = np.log(self.transition_matrix)
        self.log_invariant_distribution = np.log(self.invariant_distribution)

    def _construct_transition_matrix(
        self, training_text: str, symbols_dict: Dict[str, int]
    ) -> np.ndarray:

        # initialise with ones to ensure ergodicity
        transition_matrix = np.ones((self.num_symbols, self.num_symbols))
        for i in range(1, len(training_text)):
            # check symbols are valid
            if (
                training_text[i] in symbols_dict
                and training_text[i - 1] in symbols_dict
            ):
                transition_matrix[
                    symbols_dict[training_text[i - 1]], symbols_dict[training_text[i]]
                ] += 1
        # normalise to get transition probabilities
        transition_matrix = normalize(transition_matrix, axis=0, norm="l1")
        return transition_matrix

    def _approximate_invariant_distribution(
        self, invariant_stopping_epsilon: float
    ) -> np.ndarray:
        invariant_distribution = np.zeros((self.num_symbols, 1))
        previous_invariant_distribution = invariant_distribution.copy()
        invariant_distribution[0] = 1

        while (
            np.linalg.norm(invariant_distribution - previous_invariant_distribution)
            > invariant_stopping_epsilon
        ):
            previous_invariant_distribution = invariant_distribution.copy()
            invariant_distribution = self.transition_matrix @ invariant_distribution
        return invariant_distribution

    def log_transition_probability(self, alpha: str, beta: str) -> float:
        return self.log_transition_matrix[
            self.symbols_dict[beta], self.symbols_dict[alpha]
        ]

    def log_invariant_probability(self, gamma: str) -> float:
        return self.log_invariant_distribution[self.symbols_dict[gamma]].item()

    def compute_log_probability(self, message: str) -> float:
        log_probability = self.log_invariant_probability(message[0])
        for i in range(1, len(message)):
            s_i = message[i]
            s_i_minus_1 = message[i - 1]
            log_probability += self.log_transition_probability(s_i, s_i_minus_1)
        return log_probability


class MetropolisHastingsDecryption:
    def __init__(self, symbols):
        self.symbols = symbols
        self._random_generator = np.random.default_rng()

    def generate_random_decrypter(self) -> Decrypter:
        return Decrypter(
            {
                self.symbols[i]: self.symbols[x]
                for i, x in enumerate(
                    np.random.permutation(np.arange(len(self.symbols)))
                )
            }
        )

    @staticmethod
    def generate_proposal_decryption(decrypter: Decrypter) -> Decrypter:
        x1 = np.random.choice(list(decrypter.decryption_dict.keys()))
        x2 = np.random.choice(list(decrypter.decryption_dict.keys()))
        proposal_decryption = decrypter.decryption_dict.copy()
        proposal_decryption[x2], proposal_decryption[x1] = (
            decrypter.decryption_dict[x1],
            decrypter.decryption_dict[x2],
        )
        return Decrypter(proposal_decryption)

    def _choose_decrypter(
        self,
        statistics,
        encrypted_message,
        current_decrypter: Decrypter,
        proposal_decrypter: Decrypter,
    ) -> Decrypter:
        current_log_probability = statistics.compute_log_probability(
            message=current_decrypter.decrypt(encrypted_message),
        )
        proposal_log_probability = statistics.compute_log_probability(
            message=proposal_decrypter.decrypt(encrypted_message),
        )
        acceptance_probability = np.min(
            [1, np.exp(proposal_log_probability - current_log_probability)]
        )
        return self._random_generator.choice(
            [current_decrypter, proposal_decrypter],
            p=[1 - acceptance_probability, acceptance_probability],
        )

    def _find_good_starting_decrypter(
        self,
        statistics: Statistics,
        encrypted_message,
        number_start_attempts,
    ) -> Decrypter:
        best_log_likelihood = -np.float("inf")
        best_decrypter = None
        for _ in range(number_start_attempts):
            decrypter = self.generate_random_decrypter()
            if (
                statistics.compute_log_probability(
                    message=decrypter.decrypt(encrypted_message)
                )
                > best_log_likelihood
            ):
                best_decrypter = decrypter
        return best_decrypter

    def run(
        self,
        encrypted_message: str,
        statistics: Statistics,
        number_of_mh_loops: int,
        number_start_attempts: int,
        check_decryption_interval: int,
        check_decryption_size: int,
    ) -> Tuple[Decrypter, List[str]]:
        decrypter = self._find_good_starting_decrypter(
            statistics, encrypted_message, number_start_attempts
        )
        logged_decryption_message = [
            decrypter.decrypt(encrypted_message)[:check_decryption_size]
        ]
        for i in range(1, number_of_mh_loops + 1):
            if (i + 1) % check_decryption_interval == 0:
                logged_decryption_message.append(
                    decrypter.decrypt(encrypted_message)[:check_decryption_size]
                )
            proposal_decrypter = self.generate_proposal_decryption(decrypter)
            decrypter = self._choose_decrypter(
                statistics, encrypted_message, decrypter, proposal_decrypter
            )
        return decrypter, logged_decryption_message


def _convert_to_scientific_notation(x: float) -> str:
    return "{:.1e}".format(float(x))


def a(
    symbols: List[str],
    training_text: str,
    transition_matrix_path: str,
    invariant_distribution_path: str,
):
    statistics = Statistics(
        training_text,
        symbols,
    )
    symbols_for_df = statistics.symbols.copy()
    symbols_for_df[symbols_for_df.index(" ")] = "space"
    symbols_for_df[symbols_for_df.index('"')] = "double quotes"
    df = pd.DataFrame(
        data=statistics.transition_matrix,
        columns=symbols_for_df,
    )
    df.index = symbols_for_df
    df.applymap(_convert_to_scientific_notation).to_csv(transition_matrix_path)

    df = (
        pd.DataFrame(
            data=statistics.invariant_distribution.reshape(1, -1),
            columns=symbols_for_df,
        )
        .applymap(_convert_to_scientific_notation)
        .transpose()
        .reset_index()
    )
    df.columns = ["Symbol", "Probability"]
    df.set_index("Symbol").to_csv(invariant_distribution_path, sep="|")


def d(
    encrypted_message: str,
    symbols: List[str],
    training_text: str,
    number_trials: int,
    number_of_mh_loops: int,
    number_start_attempts: int,
    check_decryption_interval: int,
    check_decryption_size: int,
    decryptor_table_path: str,
    decrypted_message_iterations_table_path: str,
):
    statistics = Statistics(
        training_text,
        symbols,
    )
    np.random.seed(DEFAULT_SEED)
    metropolis_hastings_decryption = MetropolisHastingsDecryption(symbols)
    decrypters = []
    log_likelihoods = []
    logged_decryption_messages = []
    decryption_messages = []
    for i in range(number_trials):
        (decrypter, logged_decryption_message,) = metropolis_hastings_decryption.run(
            encrypted_message,
            statistics,
            number_of_mh_loops,
            number_start_attempts,
            check_decryption_interval,
            check_decryption_size,
        )
        decrypters.append(decrypter)
        log_likelihoods.append(
            statistics.compute_log_probability(
                decrypter.decrypt(encrypted_message)
            )
        )
        logged_decryption_messages.append(logged_decryption_message)
        decryption_messages.append(
            decrypter.decrypt(encrypted_message)[:check_decryption_size]
        )

    # sort trials by log likelihood
    best_trial = np.argmax(log_likelihoods)

    decrpyter_table = pd.DataFrame(
        decrypters[best_trial].decryption_dict.items(), columns=["s", "sigma(s)"]
    )
    decrpyter_table[decrpyter_table == " "] = "space"
    decrpyter_table[decrpyter_table == '"'] = "double quotes"
    decrpyter_table.set_index("s").to_csv(decryptor_table_path, sep="|")

    decrypted_message_iterations_table = pd.DataFrame(
        [
            np.arange(0, len(logged_decryption_messages[best_trial]))
            * check_decryption_interval,
            logged_decryption_messages[best_trial],
        ]
    ).transpose()
    decrypted_message_iterations_table.columns = ["MH Iteration", "Current Decryption"]
    decrypted_message_iterations_table.set_index("MH Iteration").to_csv(
        decrypted_message_iterations_table_path, sep="|"
    )
