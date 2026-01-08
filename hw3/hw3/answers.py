r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=128, #100
        seq_len=200, # 100
        h_dim=256,
        n_layers=2, #2
        dropout=0.4, #0.4
        learn_rate=0.01, #0.005
        lr_sched_factor=0.5, #0.5
        lr_sched_patience=3, #5
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = """ACT I."""
    temperature = 0.45
    # ========================
    return start_seq, temperature


part1_q1 = r"""
First, training on the entire corpus as a single sequence is computationally infeasible, as it would require backpropagation through an extremely long sequence, leading to excessive memory usage and severe vanishing or exploding gradient issues.

Second, treating the whole corpus as one sequence would yield only a single training sample and thus very few gradient updates per epoch. Splitting the data into many sequences produces multiple training samples, enabling frequent gradient updates and more effective optimization.
"""

part1_q2 = r"""
Although the RNN is trained on fixed-length sequences, its hidden state is carried forward across time steps and summarizes past information in a compressed form. The hidden state has a fixed dimension that does not depend on the sequence length, and its parameters are shared across all time steps.

As a result, during text generation the model can propagate information through the hidden state over arbitrarily many steps, allowing it to exhibit memory that exceeds the training sequence length.
"""

part1_q3 = r"""
We do not shuffle the order of batches because the hidden state of the RNN is typically carried over between consecutive batches. Shuffling would break the temporal continuity of the data and make the propagated hidden state inconsistent with the input sequence.

Therefore, batches must remain in their original order to preserve the sequential structure of the text and allow the model to learn long-range dependencies.
"""

part1_q4 = r"""
1. The temperature parameter controls the sharpness of the sampling distribution. Lowering the temperature increases the distinctiveness between probabilities, making high-probability characters much more likely to be selected, which leads to more coherent and model-consistent text generation.

2. When the temperature is very high, the softmax distribution approaches a uniform distribution. This follows from
$$
\text{softmax}_T(y_i) = \frac{e^{y_i/T}}{\sum_k e^{y_k/T}},
$$
since as $T \to \infty$, we have $y_i/T \to 0$ and thus $e^{y_i/T} \to 1$, resulting in nearly random sampling.

3. When the temperature is very low, the distribution becomes extremely peaked. In the limit $T \to 0$, the softmax converges to an argmax operation, meaning the most likely character is selected almost deterministically.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
