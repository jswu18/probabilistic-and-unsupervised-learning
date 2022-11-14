import os

import numpy as np

from src.constants import (
    BINARY_DIGITS_FILE_PATH,
    MESSAGE_FILE_PATH,
    OUTPUTS_FOLDER,
    SYMBOLS_FILE_PATH,
    TRAINING_TEXT_FILE_PATH,
)
from src.solutions import q1, q2, q3, q5

if __name__ == "__main__":
    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)
    x = np.loadtxt(BINARY_DIGITS_FILE_PATH)

    # Question 1
    Q1_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q1")
    if not os.path.exists(Q1_OUTPUT_FOLDER):
        os.makedirs(Q1_OUTPUT_FOLDER)
    q1.d(
        x,
        figure_path=os.path.join(Q1_OUTPUT_FOLDER, "q1d.png"),
        figure_title="Q1d: Maximum Likelihood Estimate",
    )
    q1.e(
        x,
        alpha=3,
        beta=3,
        figure_path=os.path.join(Q1_OUTPUT_FOLDER, "q1e"),
        figure_title="Q1e: Maximum A Prior",
    )

    # Question 2
    Q2_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q2")
    if not os.path.exists(Q2_OUTPUT_FOLDER):
        os.makedirs(Q2_OUTPUT_FOLDER)
    q2.c(x, table_path=os.path.join(Q2_OUTPUT_FOLDER, "q2c.csv"))

    # Question 3
    Q3_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q3")
    if not os.path.exists(Q3_OUTPUT_FOLDER):
        os.makedirs(Q3_OUTPUT_FOLDER)
    q3.e(
        x,
        alpha_parameter=1 + 1e-5,
        beta_parameter=1 + 1e-5,
        number_of_trials=4,
        ks=[2, 3, 4, 7, 10],
        epsilon=1e-5,
        max_number_of_steps=int(1e2),
        figure_path=os.path.join(Q3_OUTPUT_FOLDER, "q3e"),
        figure_title="Q3e",
        compression_csv_path=os.path.join(Q3_OUTPUT_FOLDER, "q3e-compression"),
    )

    # Question 5
    Q5_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q5")
    if not os.path.exists(Q5_OUTPUT_FOLDER):
        os.makedirs(Q5_OUTPUT_FOLDER)
    with open(TRAINING_TEXT_FILE_PATH) as fp:
        training_text = fp.read().replace("\n", "").lower()
    with open(SYMBOLS_FILE_PATH) as fp:
        symbols = fp.read().split("\n")
    with open(MESSAGE_FILE_PATH) as fp:
        encrypted_message = fp.read()
    q5.a(
        symbols,
        training_text,
        transition_matrix_path=os.path.join(Q5_OUTPUT_FOLDER, "q5a-transition.csv"),
        invariant_distribution_path=os.path.join(Q5_OUTPUT_FOLDER, "q5a-invariant.csv"),
    )
    q5.d(
        encrypted_message,
        symbols,
        training_text,
        number_trials=10,
        number_of_mh_loops=int(1e4),
        number_start_attempts=int(1e4),
        log_decryption_interval=100,
        log_decryption_size=60,
        trial_decryptions_table_path=os.path.join(Q5_OUTPUT_FOLDER, "q5d-trials.csv"),
        decryptor_table_path=os.path.join(Q5_OUTPUT_FOLDER, "q5d-decrypter.csv"),
        decrypted_message_iterations_table_path=os.path.join(
            Q5_OUTPUT_FOLDER, "q5d-iterations.csv"
        ),
    )
