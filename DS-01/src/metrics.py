import numpy as np

def qini_curve(y, t, score):
    """Cumulative incremental outcomes when targeting customers in order of score."""
    y, t, score = np.asarray(y), np.asarray(t), np.asarray(score)
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    ratio = np.divide(n_t, n_c, out=np.zeros(len(y)), where=n_c > 0)
    return y_t - y_c * ratio

def qini_coefficient(y, t, score):
    """Area between the model's Qini curve and the random-targeting line (higher = better)."""
    q = qini_curve(y, t, score)
    n = len(q)
    random_line = q[-1] * np.arange(1, n + 1) / n
    return float(np.mean(q - random_line))