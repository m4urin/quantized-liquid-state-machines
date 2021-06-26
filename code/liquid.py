import numpy as np

from code.parameters import *


class Liquid:
    def __init__(self, nbr_neurons: int, bits: int, nbr_inputs: int, seed: int):
        """
        Creates a quantized liquid that converts input spikes to output spikes

        Args:
            nbr_neurons: liquid size
            bits: lower-bit precision used in the reservoir
            nbr_inputs: amount of input channels
            **_: ignored parameters
        """

        # calculate amount of excitatory neurons
        self.nbr_ex_neurons = int(PERCENTAGE_EX_NEURONS * nbr_neurons)
        self.nbr_neurons = nbr_neurons

        # set neuron potentials and weight matrices
        self.v_t = np.zeros(nbr_neurons, dtype=np.int32)
        self.s_t = np.zeros(nbr_neurons, dtype=np.int8)
        self.threshold = (2 ** bits) - 1
        self.b = bits
        self.W = np.zeros((nbr_neurons, nbr_neurons), dtype=np.int32)
        self.W_in = np.zeros((nbr_inputs, nbr_neurons), dtype=np.int32)

        # sets of indices corresponding to the sets used in the paper
        E = np.arange(self.nbr_ex_neurons)  # 0, 1, .., e
        I = np.arange(self.nbr_ex_neurons, nbr_neurons)  # e+1, e+2, .., n
        U = np.arange(nbr_inputs)  # 0, 1, .., u

        # set seed to generate a new liquid or repeat the experiment
        np.random.seed(seed)
        # create internal connections (neurons cannot connect to themselves)
        for i in E:
            self.W[i, np.random.choice(E[E != i], size=2, replace=False)] = 1
            self.W[i, np.random.choice(I, size=2, replace=False)] = 1
        for i in I:
            self.W[i, np.random.choice(E, size=1, replace=False)] = -1
            self.W[i, np.random.choice(I[I != i], size=1, replace=False)] = -1

        # create input connections (to excitatory neurons!) based on neurons_per_input
        for i in U:
            neurons_per_input = int(PERCENTAGE_NEURONS_PER_INPUT * self.nbr_ex_neurons)
            self.W_in[i, np.random.choice(E, size=neurons_per_input, replace=False)] = 1

        # sample from the set of discrete weights w_b (without 0)
        w_b = np.arange(1, 2 ** bits)  # 1, .., (2^b)-1
        self.W *= np.random.choice(w_b, size=self.W.shape)
        self.W_in *= np.random.choice(w_b, size=self.W_in.shape)

        self.v_t_hist = []

    def tick(self, u_t: np.ndarray) -> (np.ndarray, np.float32):
        """
        Takes input spikes at time step t and creates a new liquid state

        Args:
            u_t: (U x 1) input vector at time step t

        Returns:
            The excitatory spikes (E x 1) generated along with the sum of all the spikes in the reservoir
        """
        # check if threshold is reached and generate new (binary) spikes
        s_t = np.where(self.v_t > self.threshold, 1, 0).astype('int8')

        self.v_t_hist.append(self.v_t[:10])

        # neurons that have spiked are reset to 0,
        v_t = np.where(self.v_t > self.threshold, 0, self.v_t).astype('int32')

        # apply a leak_factor
        v_t = v_t - get_leak(v_t)
        v_t = v_t.clip(min=0)

        # calculate new potential
        v_t = (u_t @ self.W_in) + (s_t @ self.W) + v_t

        # negative potentials not allowed
        self.v_t = v_t.clip(min=0)

        # read the spikes of the excitatory neurons and return the general activity
        return s_t[:self.nbr_ex_neurons], np.sum(s_t), np.sum(self.v_t) / self.threshold

    def get_v_hist(self):
        return np.array(self.v_t_hist).transpose()

    def run_complete(self, u: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Runs an entire input by calling tick() for every timestep

        Args:
            u: (U x T) input spikes

        Returns:
            All the excitatory output spikes (E x T) along with the activity series of the liquid
        """
        # transpose u to get an input vector at each timestep
        u = u.astype('int8').transpose()
        T = u.shape[0]

        # output
        output_spikes = np.zeros((u.shape[0], self.nbr_ex_neurons), dtype=np.int8)
        activity = np.zeros(u.shape[0], dtype=np.int32)
        energy = np.zeros(u.shape[0], dtype=np.float32)

        # run through data
        for t in range(T):
            output_spikes[t], activity[t], energy[t] = self.tick(u[t])

        # transpose the output to get a spike streak for each channel
        output_spikes = output_spikes.transpose()

        return output_spikes.astype('int8'), activity, energy


def get_leak(v) -> np.ndarray:
    """
    Fast bitwise operations to calculate the leak_factor

    Args:
        v: current membrane potential

    Returns:
        2^(floor(log2(x))-leak_factor)
    """
    v = v >> LEAK
    v |= (v >> 1)
    v |= (v >> 2)
    v |= (v >> 4)
    v |= (v >> 8)
    v |= (v >> 16)
    v = v - (v >> 1)
    return v.clip(min=1)

