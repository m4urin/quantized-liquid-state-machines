import numpy as np


class TimeSeries:
    def generate(self, length: int) -> np.ndarray:
        """
        Generate the time series
        Args:
            length: length of the time series

        Returns:
            A time series of a certain length
        """
        raise Exception("Not implemented")

    def get_prediction_set(self, length, k, sum_predictions=False):
        """
        Generates a time series x and the shifted time series y that should be predicted
        Args:
            length: length of both time series
            k: amount of steps that should be predicted
            sum_predictions: should the prediction values be summed?
        Returns:
            The time series x and the shifted time series y
        """
        data = self.generate(length=length + abs(k))
        if k > 0:
            x, y = data[:, :-k], data[:, k:]
        elif k < 0:
            x, y = data[:, -k:], data[:, :k]
        else:
            x, y = data, data

        if sum_predictions:
            y = np.sum(y, axis=0).reshape((1, y.shape[1]))

        return x, y


class MackeyGlass(TimeSeries):
    def __init__(self, tau: int = 17, a: float = 0.2, b: float = 0.1, n: int = 10,
                 x0: float = 1.2, h: float = 1.0, seed: int = 1234567):
        """
        Code altered from https://github.com/reservoirpy/reservoirpy/blob/master/reservoirpy/datasets/_chaos.py
        """
        self.tau = tau
        self.a = a
        self.b = b
        self.n = n
        self.x0 = x0
        self.h = h
        self.seed = seed

    def mg_rk4(self, xt, xtau):
        """
        Runge-Kuta method (RK4) for Mackey-Glass data discretization.
        """
        g = (self.a * xtau) / (1 + xtau ** self.n)  # first part of equation can be pre computed
        k1 = self.h * (g - (self.b * xt))
        k2 = self.h * (g - (self.b * (xt + k1 / 2)))
        k3 = self.h * (g - (self.b * (xt + k2 / 2)))
        k4 = self.h * (g - (self.b * (xt + k3)))
        return xt + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def generate(self, length: int) -> np.ndarray:
        """
        Generates the Mackey-Glass time series
        Args:
            length: length of the time series
        Returns:
            Mackey-Glass time series
        """
        warmup = int(50 / self.h)

        hist = int(self.tau / self.h)
        x = np.zeros(length + hist + warmup, dtype=np.float)

        # fill with random values
        np.random.seed(self.seed)
        x[:hist] = self.x0 * np.ones(hist) + 0.2 * (np.random.rand(hist) - 0.5)

        # initial value
        x[hist] = self.x0

        # calculate the rest of the values
        for i in range(hist, len(x) - 1):
            x[i + 1] = self.mg_rk4(xt=x[i], xtau=x[i - hist])

        x = x[hist + warmup:]

        # reshape to 1 channel
        return x.reshape(1, x.shape[0])


class SineWaves(TimeSeries):
    def __init__(self, time_steps: [67, 88, 117]):
        """
        Creates a set of sine wave time series
        Args:
            time_steps: Per sine wave: Amount of time steps needed for one cycle
        """
        self.widths = np.array(time_steps)

    def generate(self, length: int) -> np.ndarray:
        """
        Generates multiple sine waves
        Args:
            length: length of the time series
        Returns:
            A set of sine wave time series
        """
        step_sizes = 2 * np.pi / self.widths
        x = np.array([np.arange(100, length + 100), ] * len(step_sizes))
        x = x * step_sizes.reshape(step_sizes.shape[0], 1)
        return np.sin(x).astype('float32')
