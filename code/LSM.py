import numpy as np

from code.spikecoding import decode
from code.liquid import Liquid
from code.parameters import *
from code.readout import ReadoutLayer


class LSM:
    def __init__(self, liquid_size, nbr_inputs, bits=4):
        self.seed = 12345
        self.liquid_size = liquid_size
        self.nbr_inputs = nbr_inputs
        self.bits = bits

        self.readout = ReadoutLayer(in_features=int(PERCENTAGE_EX_NEURONS * self.liquid_size))
        self.readout.reset_weights()
        self.liquid = Liquid(self.liquid_size, self.bits, self.nbr_inputs, self.seed)

    def set_liquid_size(self, liquid_size):
        self.seed += 1
        self.liquid_size = liquid_size
        self.readout = ReadoutLayer(in_features=int(PERCENTAGE_EX_NEURONS * self.liquid_size))
        self.readout.reset_weights()
        self.liquid = Liquid(self.liquid_size, self.bits, self.nbr_inputs, self.seed)

    def set_nbr_inputs(self, nbr_inputs):
        self.seed += 1
        self.nbr_inputs = nbr_inputs
        self.liquid = Liquid(self.liquid_size, self.bits, self.nbr_inputs, self.seed)

    def set_bits(self, bits: int):
        self.seed += 1
        self.bits = bits
        self.liquid = Liquid(self.liquid_size, self.bits, self.nbr_inputs, self.seed)

    def train_readout(self, input_spikes: np.ndarray, expected_y: np.ndarray):
        output_spikes, activity, energy = self.liquid.run_complete(input_spikes)
        output_signals = decode(output_spikes, denoise=True)
        train_loss, val_loss = self.readout.train_readout(output_signals, expected_y)
        prediction = self.readout.predict(output_signals)
        return activity, energy, output_spikes, output_signals, train_loss, val_loss, prediction
