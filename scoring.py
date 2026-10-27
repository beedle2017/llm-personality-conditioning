import numpy as np


def bfi_error(bfi_score, target):
    """
    Absolute error between measured BFI score and the requested target.

    bfi_score : float or array, shape (N,)  measured BFI score for the varied trait (1-5)
    target : float or array, shape (N,)  requested target score t (1-5)

    Output : (N, ) error, 0 = perfect, larger = worse.
    """
    return np.abs(np.asarray(bfi_score, dtype=float) - np.asarray(target, dtype=float))


def pis(rate, target, baseline_rate):
    """
    Personality Influence Score, relative to the unconditioned baseline.

    rate : float or array, shape (N,)  endorsement rate r (percent, 0-100) for the conditioned run(s)
    target : float or array, shape (N,)  requested target score t (1-5)
    baseline_rate : float  unconditioned endorsement rate for the same trait (e.g. mean over baseline runs)

    Output : same shape as `rate`. Positive = behavior moved in the requested direction (up if t > 3, down if t < 3). NaN where t == 3.
    """
    rate = np.asarray(rate, dtype=float)
    target = np.asarray(target, dtype=float)
    direction = np.sign(target - 3)
    out = direction * (rate - baseline_rate)
    return out