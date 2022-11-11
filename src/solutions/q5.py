from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src.constants import DEFAULT_SEED


def _convert_to_scientific_notation(x: float) -> str:
    """
    Convert value to string in scientific notation
    :param x: value to convert
    :return: string of x in scientific notation
    """
    return "{:.1e}".format(float(x))


class Decrypter:
    def __init__(self, decryption_dict: Dict[str, str]) -> None:
        """
        Decrypter containing the mapping a symbol to its encrypted symbol
        :param decryption_dict:
        """
        self.decryption_dict = decryption_dict

    def decrypt(self, encrypted_message: str) -> str:
        """
        Decrypts an encrypted message using the decryption dictionary
        :param encrypted_message: the encrypted message to decrypt
        :return: decrypted message
        """
        return "".join([self.decryption_dict[x] for x in encrypted_message])

    @property
    def table(self) -> pd.DataFrame:
        """
        Generate table containing symbol decryptions
        :return: pandas table of decryptions
        """
        decrpyter_table = pd.DataFrame(
            self.decryption_dict.items(), columns=["s", "sigma(s)"]
        )
        decrpyter_table[decrpyter_table == " "] = "space"
        decrpyter_table[decrpyter_table == '"'] = "double quotes"
        return decrpyter_table.set_index("s")


class Statistics:
    def __init__(
        self,
        training_text: str,
        symbols: List[str],
        invariant_stopping_epsilon: float = 5e-20,
    ) -> None:
        """
        Statistics for text
        :param training_text: training text for calculating transition and invariant probability
        :param symbols: symbols in the training text
        :param invariant_stopping_epsilon: stopping condition for constructing the invariant distribution
        """
        self.training_text = training_text
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.symbols_dict = self._construct_symbols_dictionary(symbols)
        self.transition_matrix = self._construct_transition_matrix(
            training_text, self.symbols_dict
        )
        self.invariant_distribution = self._approximate_invariant_distribution(
            invariant_stopping_epsilon
        )
        self.log_transition_matrix = np.log(self.transition_matrix)
        self.log_invariant_distribution = np.log(self.invariant_distribution)

    @property
    def list_of_symbols_for_df(self) -> List[str]:
        """
        Replace certain symbols to prepare for dataframe
        :return: list of symbols with some replacements
        """
        x = self.symbols.copy()
        x[x.index(" ")] = "space"
        x[x.index('"')] = "double quotes"
        return x

    @property
    def transition_table(self) -> pd.DataFrame:
        """
        Generate a table containing transition probabilities
        :return: transition probabilities
        """
        df_transitions = pd.DataFrame(
            data=self.transition_matrix,
            columns=self.list_of_symbols_for_df,
        )
        df_transitions.index = self.list_of_symbols_for_df
        return df_transitions.applymap(_convert_to_scientific_notation)

    @property
    def invariant_distribution_table(self) -> pd.DataFrame:
        """
        Generate a table containing invariant distribution probabilities
        :return: invariant distribution probabilities
        """
        df = (
            pd.DataFrame(
                data=self.invariant_distribution.reshape(1, -1),
                columns=self.list_of_symbols_for_df,
            )
            .applymap(_convert_to_scientific_notation)
            .transpose()
            .reset_index()
        )
        df.columns = ["Symbol", "Probability"]
        return df.set_index("Symbol")

    @staticmethod
    def _construct_symbols_dictionary(symbols: List[str]) -> Dict[str, int]:
        """
        Construct a dictionary mapping each symbol to an integer
        :param symbols: list of symbols to map
        :return: symbol to integer mapping
        """
        return {k: v for v, k in enumerate(symbols)}

    def _construct_transition_matrix(
        self, text: str, symbols_dict: Dict[str, int]
    ) -> np.ndarray:
        """
        Constructs the transition matrix for a given text
        :param text: string to calculate transition matrix with
        :param symbols_dict: dictionary mapping symbol to a dictionary
        :return:
        """
        # initialise with ones to ensure ergodicity
        transition_matrix = np.ones((self.num_symbols, self.num_symbols))
        for i in range(1, len(text)):
            # check symbols are valid
            if text[i] in symbols_dict and text[i - 1] in symbols_dict:
                transition_matrix[symbols_dict[text[i - 1]], symbols_dict[text[i]]] += 1
        # normalise to get transition probabilities
        transition_matrix = normalize(transition_matrix, axis=0, norm="l1")
        return transition_matrix

    def _approximate_invariant_distribution(
        self, invariant_stopping_epsilon: float
    ) -> np.ndarray:
        """
        Approximate the invariant distribution with the power method
        :param invariant_stopping_epsilon: stopping condition for constructing the invariant distribution
        :return: the invariant distribution as a vector (number of symbols, 1)
        """
        invariant_distribution = np.zeros((self.num_symbols, 1))
        previous_invariant_distribution = invariant_distribution.copy()

        # make sure it's a proper distribution that sums to one
        invariant_distribution[0] = 1

        while (
            np.linalg.norm(invariant_distribution - previous_invariant_distribution)
            > invariant_stopping_epsilon
        ):
            previous_invariant_distribution = invariant_distribution.copy()
            invariant_distribution = self.transition_matrix @ invariant_distribution
        return invariant_distribution

    def log_transition_probability(self, alpha: str, beta: str) -> float:
        """
        Look up the log probability of the transition from symbol alpha to beta
        :param alpha: symbol that is being transitioned from
        :param beta: symbol that is being transitioned to
        :return: probability of transition
        """
        return self.log_transition_matrix[
            self.symbols_dict[beta], self.symbols_dict[alpha]
        ]

    def log_invariant_probability(self, gamma: str) -> float:
        """
        Look up the log probability of a symbol with respect to the invariant distribution
        :param gamma: symbol to query
        :return: log probability of the symbol
        """
        return self.log_invariant_distribution[self.symbols_dict[gamma]].item()

    def compute_log_probability(self, text: str) -> float:
        """
        Compute the log probability of a given text containing symbols
        :param text: text to compute log probability for
        :return: log probability of the text
        """
        log_probability = self.log_invariant_probability(text[0])
        for i in range(1, len(text)):
            log_probability += self.log_transition_probability(text[i], text[i - 1])
        return log_probability


class MetropolisHastingsDecryption:
    def __init__(self, symbols: List[str]):
        """
        Metropolis Hastings MCMC for Decryption
        :param symbols: set of symbols to decrypt
        """
        self.symbols = symbols

    def generate_random_decrypter(self) -> Decrypter:
        """
        Generates a random decrypter
        :return: a Decrypter instantiation
        """
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
        """
        Generate a proposal decrypter by randomly swapping two of the decryption mappings
        :param decrypter: the decrypter used to generate the proposal
        :return: a proposal decrypter
        """
        x1 = np.random.choice(list(decrypter.decryption_dict.keys()))
        x2 = np.random.choice(list(decrypter.decryption_dict.keys()))
        proposal_decryption = decrypter.decryption_dict.copy()
        proposal_decryption[x2], proposal_decryption[x1] = (
            decrypter.decryption_dict[x1],
            decrypter.decryption_dict[x2],
        )
        return Decrypter(proposal_decryption)

    @staticmethod
    def _choose_decrypter(
        statistics: Statistics,
        encrypted_message: str,
        current_decrypter: Decrypter,
        proposal_decrypter: Decrypter,
    ) -> Decrypter:
        """
        Choose between the current and proposal decrypter
        :param statistics: Statistics instantiation for calculating log probabilities
        :param encrypted_message: the encrypted message
        :param current_decrypter: the current decrypter
        :param proposal_decrypter: the proposal decrypter
        :return:
        """
        # calculate log probabilities
        current_log_probability = statistics.compute_log_probability(
            text=current_decrypter.decrypt(encrypted_message),
        )
        proposal_log_probability = statistics.compute_log_probability(
            text=proposal_decrypter.decrypt(encrypted_message),
        )

        # calculate acceptance probability
        acceptance_probability = np.min(
            [1, np.exp(proposal_log_probability - current_log_probability)]
        )
        # choose decrypter using the acceptance probability
        return np.random.choice(
            [current_decrypter, proposal_decrypter],
            p=[1 - acceptance_probability, acceptance_probability],
        )

    def _find_good_starting_decrypter(
        self,
        statistics: Statistics,
        encrypted_message,
        number_start_attempts,
    ) -> Decrypter:
        """
        Find a good starting decrypter for the sampler by choosing the one with the best log likelihood
        :param statistics: Statistics instantiation for calculating log probabilities
        :param encrypted_message: the encrypted message
        :param number_start_attempts: number of possible starting decrypters to check
        :return: the best starting decrypter for the sampler
        """
        best_log_likelihood = -np.float("inf")
        best_decrypter = None
        for _ in range(number_start_attempts):
            decrypter = self.generate_random_decrypter()
            if (
                statistics.compute_log_probability(
                    text=decrypter.decrypt(encrypted_message)
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
        log_decryption_interval: int,
        log_decryption_size: int,
    ) -> Tuple[Decrypter, List[str]]:
        """
        Run the sampler with two steps:
            1. find a good starting decrypter for the sampler
            2. run the sampler
        :param encrypted_message: the encrypted message
        :param statistics: Statistics instantiation for calculating log probabilities
        :param number_of_mh_loops: number of loops to run the metropolis hastings sampler
        :param number_start_attempts: number of possible starting decrypters to check
        :param log_decryption_interval: number of samples between logging the decrypted message
        :param log_decryption_size: number of symbols to decrypt when logging the decrypted message
        :return: a tuple containing the decrypter found from the sampler and the logged decryption message
        """
        decrypter = self._find_good_starting_decrypter(
            statistics, encrypted_message, number_start_attempts
        )
        logged_decryption_message = [
            decrypter.decrypt(encrypted_message)[:log_decryption_size]
        ]
        for i in range(1, number_of_mh_loops + 1):
            if (i + 1) % log_decryption_interval == 0:
                logged_decryption_message.append(
                    decrypter.decrypt(encrypted_message)[:log_decryption_size]
                )
            proposal_decrypter = self.generate_proposal_decryption(decrypter)
            decrypter = self._choose_decrypter(
                statistics, encrypted_message, decrypter, proposal_decrypter
            )
        return decrypter, logged_decryption_message


def _construct_logged_decryptions_table(
    logged_decryption_message, log_decryption_interval
) -> pd.DataFrame:
    decrypted_message_iterations_table = pd.DataFrame(
        [
            np.arange(0, len(logged_decryption_message)) * log_decryption_interval,
            logged_decryption_message,
        ]
    ).transpose()
    decrypted_message_iterations_table.columns = ["MH Iteration", "Current Decryption"]
    return decrypted_message_iterations_table.set_index("MH Iteration")


def a(
    symbols: List[str],
    training_text: str,
    transition_matrix_path: str,
    invariant_distribution_path: str,
) -> None:
    """
    Produces answers for question 5a
    :param symbols: symbols in the training text
    :param training_text: training text for calculating transition and invariant probability
    :param transition_matrix_path: path to store transition matrix
    :param invariant_distribution_path: path to store invariant distribution
    :return:
    """
    statistics = Statistics(
        training_text,
        symbols,
    )
    statistics.transition_table.to_csv(transition_matrix_path)
    statistics.invariant_distribution_table.to_csv(invariant_distribution_path, sep="|")


def d(
    encrypted_message: str,
    symbols: List[str],
    training_text: str,
    number_trials: int,
    number_of_mh_loops: int,
    number_start_attempts: int,
    log_decryption_interval: int,
    log_decryption_size: int,
    decryptor_table_path: str,
    decrypted_message_iterations_table_path: str,
) -> None:
    """
    Produces answers for question 5d
    :param encrypted_message: the encrypted message
    :param symbols: symbols in the training text
    :param training_text: training text for calculating transition and invariant probability
    :param number_trials: number of times to restart and run the sampler
    :param number_of_mh_loops: number of loops to run the metropolis hastings sampler
    :param number_start_attempts: number of possible starting decrypters to check
    :param log_decryption_interval: number of samples between logging the decrypted message
    :param log_decryption_size: number of symbols to decrypt when logging the decrypted message
    :param decryptor_table_path: path to store decrypter mapping table
    :param decrypted_message_iterations_table_path: path to store logged decryption messages
    :return:
    """
    statistics = Statistics(
        training_text,
        symbols,
    )
    np.random.seed(DEFAULT_SEED)
    metropolis_hastings_decryption = MetropolisHastingsDecryption(symbols)
    decrypters: List[Decrypter] = []
    log_likelihoods: List[float] = []
    logged_decryption_messages: List[List[str]] = []
    decryption_messages = []
    for i in range(number_trials):
        (decrypter, logged_decryption_message,) = metropolis_hastings_decryption.run(
            encrypted_message,
            statistics,
            number_of_mh_loops,
            number_start_attempts,
            log_decryption_interval,
            log_decryption_size,
        )
        decrypters.append(decrypter)
        log_likelihoods.append(
            statistics.compute_log_probability(decrypter.decrypt(encrypted_message))
        )
        logged_decryption_messages.append(logged_decryption_message)
        decryption_messages.append(
            decrypter.decrypt(encrypted_message)[:log_decryption_size]
        )

    # sort trials by log likelihood
    best_trial = np.argmax(log_likelihoods)
    decrypters[best_trial].table.to_csv(decryptor_table_path, sep="|")
    df_logged_decryptions = _construct_logged_decryptions_table(
        logged_decryption_messages[best_trial], log_decryption_interval
    )
    df_logged_decryptions.to_csv(decrypted_message_iterations_table_path, sep="|")
