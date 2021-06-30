import numpy as np
from scipy.stats import norm

from code.parameters import WINDOW_SIZE, MINIMAL_WINDOW_KERNEL_VALUE


def step_forward(signals: np.ndarray, threshold=0.1) -> np.ndarray:
    """
    Encodes a time series based on the step-forward algorithm provided in:
        Petro et al. (2020)
    Args:
        signals: (U x T) time series set
        threshold: threshold
    Returns:
        (2U x T) spikes
    """
    s = signals.shape[0]
    T = signals.shape[1]
    signals = normalize_signals(signals).transpose()
    spikes = np.zeros((T, 2 * s))
    base = signals[0]
    for t in range(T):
        base_pos = base + threshold
        base_neg = base - threshold
        spikes[t, :s] = np.where(signals[t] > base_pos, 1, 0)
        spikes[t, s:] = np.where(signals[t] < base_neg, 1, 0)
        base += np.where(signals[t] > base_pos, threshold, 0)
        base += np.where(signals[t] < base_neg, -threshold, 0)
    return spikes.transpose().astype('int8')


def rate_encoding(signals: np.ndarray, threshold=1.78) -> np.ndarray:
    """
    Uses neurons with a threshold to generate a rate encoding
    Args:
        threshold: threshold of the spiking neuron
        signals: (U x T) time series set
    Returns:
        (U x T) spikes
    """
    signals = normalize_signals(signals).transpose()
    spikes = np.zeros_like(signals)
    potential = np.zeros_like(signals[0])
    for t in range(signals.shape[0]):
        potential += signals[t]
        spikes[t] = np.where(potential > threshold, 1, 0)
        potential = np.where(potential > threshold, 0, potential)
    return spikes.transpose().astype('int8')


def decode(spikes: np.ndarray, denoise=False,
           window_size=WINDOW_SIZE,
           minimal_window_value=MINIMAL_WINDOW_KERNEL_VALUE) -> np.ndarray:
    """
    Uses np.convolve to apply the sliding window over the spikes very fast.
    Args:
        spikes: (U x T) spikes
        window_size: sliding window size (tau)
        minimal_window_value: weight of the oldest spike, or gamma^(tau-1)
        denoise: use a denoise kernel to smooth the output
    Returns:
        (U x T) time series signal
    """
    window_kernel = get_window_kernel(window_size, minimal_window_value)
    signals = np.empty_like(spikes, dtype=np.float32)
    T = signals.shape[1]
    for i in range(spikes.shape[0]):
        # not using 'same' convolution here, because time step t must correspond with the first value of the kernel
        signals[i] = np.convolve(spikes[i], window_kernel, 'full')[:T]
    if denoise:
        kernel = get_denoise_kernel()
        for i in range(spikes.shape[0]):
            signals[i] = np.convolve(signals[i], kernel, 'same')
    return signals


def normalize_signals(signals) -> np.ndarray:
    """
    Normalizes the data based on the variance
    Args:
        signals: (U x T) data streams
    Returns:
        Normalized (U x T) data streams
    """
    signals = signals - np.min(signals, axis=1).reshape(signals.shape[0], 1)
    return signals / (3 * np.std(signals, axis=1).reshape(signals.shape[0], 1))


def get_window_kernel(window_size, minimal_window_value):
    """
    Gamma is calculated from: gamma^(tau-1)=minimal_window_value
    Returns:
        kernel [gamma^0, gamma^1,..,gamma^(tau-1)]
    """
    gamma = minimal_window_value ** (1 / (window_size - 1))
    return gamma ** np.arange(window_size)


def get_denoise_kernel():
    """
    Create a bell curve with a certain std.
    Returns:
        A kernel (length=7) that can be used to denoise the data
    """
    kernel = norm.pdf(np.arange(-3, 4), 0.0, 1.0)
    return kernel / np.sum(kernel)


def combine_dataset(*args) -> np.ndarray:
    """
    Combines any amount of timeseries sets and timeseries to a single timeseries set
    Args:
        *args: Any amount of timeseries sets and timeseries
    Returns:
        A single timeseries set
    """
    data = list(args)
    for i in range(len(data)):
        if data[i].ndim == 1:
            data[i] = data[i].reshape(1, data[i].shape[0])
        elif data[i].ndim > 2:
            raise Exception("Only 1 and 2 dimensions are allowed!")
    return np.concatenate(data, axis=0)
