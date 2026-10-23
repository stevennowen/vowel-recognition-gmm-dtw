import numpy as np

from config import COV_REG


def fit_gaussian(sequences):
    frames = np.vstack(sequences)
    mean = frames.mean(axis=0)
    cov = np.cov(frames, rowvar=False) + COV_REG * np.eye(frames.shape[1])
    return mean, cov


def negative_log_likelihood(x, mean, cov):
    dim = x.shape[1]
    _, logdet = np.linalg.slogdet(cov)
    diff = x - mean
    mahalanobis = np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)
    nll = 0.5 * (dim * np.log(2 * np.pi) + logdet) + 0.5 * mahalanobis
    return nll.mean()
