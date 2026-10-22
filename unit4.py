"""Unit 4: Markov models, discrete HMMs, message passing, and an LDS."""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT = "results/unit4_sequence.png"


def forward_scaled(observations, initial, transition, emission):
    """Forward sum-product messages; scaling factors prevent underflow."""
    length = len(observations)
    states = len(initial)
    alpha = np.zeros((length, states))
    scales = np.zeros(length)
    alpha[0] = initial * emission[:, observations[0]]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]
    for t in range(1, length):
        alpha[t] = (alpha[t - 1] @ transition) * emission[:, observations[t]]
        scales[t] = alpha[t].sum()
        alpha[t] /= scales[t]
    return alpha, scales, np.log(scales).sum()


def backward_scaled(observations, scales, initial, transition, emission):
    """Backward sum-product messages, scaled using the forward factors."""
    beta = np.ones((len(observations), len(initial)))
    for t in range(len(observations) - 2, -1, -1):
        beta[t] = transition @ (emission[:, observations[t + 1]] * beta[t + 1])
        beta[t] /= scales[t + 1]
    first = np.sum(initial * emission[:, observations[0]] * beta[0])
    log_likelihood = np.log(first) + np.log(scales[1:]).sum()
    return beta, log_likelihood


def fit_hmm_em(observations, states=2, symbols=3, max_iter=300, tolerance=1e-4):
    """Baum-Welch maximum likelihood: update parameters from soft state counts."""
    initial = np.full(states, 1 / states)
    transition = np.full((states, states), 0.10 / (states - 1))
    np.fill_diagonal(transition, 0.90)
    if states == 2 and symbols == 3:
        emission = np.array([[0.20, 0.60, 0.20], [0.40, 0.20, 0.40]])
    else:
        emission = np.full((states, symbols), 1 / symbols)

    previous = -np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        alpha, scales, log_likelihood = forward_scaled(observations, initial, transition, emission)
        beta, _ = backward_scaled(observations, scales, initial, transition, emission)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi_sum = np.zeros((states, states))
        for t in range(len(observations) - 1):
            next_message = emission[:, observations[t + 1]] * beta[t + 1]
            xi = alpha[t, :, None] * transition * next_message[None, :] / scales[t + 1]
            xi_sum += xi

        initial = np.maximum(gamma[0], 1e-8)
        initial /= initial.sum()
        transition = xi_sum + 1e-8
        transition /= transition.sum(axis=1, keepdims=True)
        for symbol in range(symbols):
            emission[:, symbol] = gamma[observations == symbol].sum(axis=0) + 1e-8
        emission /= emission.sum(axis=1, keepdims=True)

        if iteration > 1 and abs(log_likelihood - previous) < tolerance:
            converged = True
            break
        previous = log_likelihood

    alpha, scales, log_likelihood = forward_scaled(observations, initial, transition, emission)
    beta, backward_likelihood = backward_scaled(observations, scales, initial, transition, emission)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return initial, transition, emission, gamma, iteration, converged, log_likelihood, backward_likelihood


def viterbi(observations, initial, transition, emission):
    """Max-product dynamic program; returns the single most likely state path."""
    log_a = np.log(np.maximum(transition, 1e-12))
    log_b = np.log(np.maximum(emission, 1e-12))
    score = np.zeros((len(observations), len(initial)))
    previous_state = np.zeros((len(observations), len(initial)), dtype=int)
    score[0] = np.log(np.maximum(initial, 1e-12)) + log_b[:, observations[0]]
    for t in range(1, len(observations)):
        candidates = score[t - 1][:, None] + log_a
        previous_state[t] = candidates.argmax(axis=0)
        score[t] = candidates.max(axis=0) + log_b[:, observations[t]]
    path = np.zeros(len(observations), dtype=int)
    path[-1] = score[-1].argmax()
    for t in range(len(observations) - 2, -1, -1):
        path[t] = previous_state[t + 1, path[t + 1]]
    return path


def observed_markov_model(observations, split):
    """First-order Markov baseline over observed return categories."""
    symbols = int(observations.max()) + 1
    counts = np.ones((symbols, symbols))  # add-one smoothing
    for current, following in zip(observations[:split - 1], observations[1:split]):
        counts[current, following] += 1
    transition = counts / counts.sum(axis=1, keepdims=True)
    predicted = transition[observations[split - 1:-1]].argmax(axis=1)
    accuracy = np.mean(predicted == observations[split:])
    return transition, accuracy


def kalman_filter(observations, split):
    """Scalar linear dynamical system: latent state evolves linearly, returns are noisy observations."""
    variance = np.var(observations[:split])
    dynamics, observation_scale = 0.98, 0.95
    process_noise = (1 - observation_scale) * variance
    observation_noise = observation_scale * variance
    state_mean, state_variance = 0.0, variance
    predictions, filtered = [], []
    for value in observations:
        prior_mean = dynamics * state_mean
        prior_variance = dynamics ** 2 * state_variance + process_noise
        predictions.append(prior_mean)
        gain = prior_variance / (prior_variance + observation_noise)
        state_mean = prior_mean + gain * (value - prior_mean)
        state_variance = (1 - gain) * prior_variance
        filtered.append(state_mean)
    return np.asarray(predictions), np.asarray(filtered)


def run(data):
    print("\nUNIT 4 - HIDDEN MARKOV MODELS AND SEQUENTIAL DATA")
    stock = data[data["Ticker"] == "AAPL"].sort_values("Date").dropna(subset=["Return_1D"])
    values = stock["Return_1D"].to_numpy()
    dates = stock["Date"].to_numpy()
    split = int(0.70 * len(values))
    cut_points = np.quantile(values[:split], [1 / 3, 2 / 3])
    observations = np.digitize(values, cut_points)
    symbols = len(np.unique(observations))

    print(f"AAPL daily-return sequence: {len(values)} observations; train={split}, test={len(values)-split}.")
    print("The observation symbols are return terciles: low, middle, high.")
    print("A first-order Markov model assumes the next state depends on the current state only.")
    markov_transition, markov_accuracy = observed_markov_model(observations, split)
    print("Observed-state Markov transition probabilities:")
    print(np.round(markov_transition, 3))
    print(f"Next-symbol test accuracy={markov_accuracy:.3f}")

    train_sequence, test_sequence = observations[:split], observations[split:]
    initial, transition, emission, gamma_train, iterations, converged, train_ll, _ = fit_hmm_em(
        train_sequence, states=2, symbols=symbols)
    alpha_test, scales_test, forward_ll = forward_scaled(test_sequence, initial, transition, emission)
    beta_test, backward_ll = backward_scaled(test_sequence, scales_test, initial, transition, emission)
    posterior_test = alpha_test * beta_test
    posterior_test /= posterior_test.sum(axis=1, keepdims=True)
    states_test = viterbi(test_sequence, initial, transition, emission)

    print("\nTwo-state discrete HMM, trained by Baum-Welch / EM maximum likelihood:")
    status = "converged" if converged else "iteration limit reached"
    print(f"Iterations={iterations} ({status}); training log-likelihood={train_ll:.2f}; "
          f"test log-likelihood per step={forward_ll/len(test_sequence):.4f}")
    print("Learned hidden-state transition matrix:")
    print(np.round(transition, 3))
    print("Learned hidden-state emission probabilities (columns: low/middle/high return):")
    print(np.round(emission, 3))
    print(f"Forward/backward log-likelihood difference={abs(forward_ll-backward_ll):.3e}")
    print(f"Forward scaling-factor range on test={scales_test.min():.3f} to {scales_test.max():.3f}")
    print(f"Average posterior certainty (largest state probability)={posterior_test.max(axis=1).mean():.3f}")
    print(f"Viterbi most-likely test-state counts={np.bincount(states_test, minlength=2).tolist()}")

    # Kalman filtering is inference in a linear Gaussian state-space model (LDS).
    predictions, filtered = kalman_filter(values, split)
    lds_rmse = np.sqrt(np.mean((values[split:] - predictions[split:]) ** 2))
    print("\nLinear dynamical system / Kalman filter:")
    print("Latent state: z[t] = 0.98*z[t-1] + process noise; observation: return[t] = z[t] + measurement noise.")
    print(f"One-step test RMSE={lds_rmse:.5f}; latent state estimates update as each return arrives.")

    # Visualize the test sequence, decoded hidden regimes, and continuous LDS estimate.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    shown = min(400, len(test_sequence))
    axes[0].scatter(dates[split:split + shown], values[split:split + shown] * 100,
                    c=states_test[:shown], cmap="coolwarm", s=12)
    axes[0].set(title="HMM decoded hidden regimes over AAPL returns", ylabel="daily return (%)")
    axes[1].plot(dates[split:split + shown], values[split:split + shown] * 100,
                 alpha=0.45, label="observed return")
    axes[1].plot(dates[split:split + shown], filtered[split:split + shown] * 100,
                 label="Kalman latent-state estimate")
    axes[1].set(title="Linear dynamical system (Kalman filter)", ylabel="daily return (%)", xlabel="date")
    axes[1].legend()
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"Unit 4 complete. Plot saved: {OUT}")
