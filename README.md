# A simplified Liquid State Machine

[[PDF]](https://raw.githubusercontent.com/m4urin/SimpleLSM/main/paper.pdf)

This project implements a Liquid State Machine using a liquid consisting of quantized neurons that are operating on lower-bit representations and fixed point computations. It provides a next step towards the implementation of efficient accelerators that can be used in the field of neuromorphic computing. A minimal implementation of the liquid is realized by only using a neuron potential vector, weight matrix, a threshold value and a leak function. These components all make use of the lower-bit representation of the neurons. The liquid dynamics are not event-driven, but simulated using a clock function. The Liquid State Machine is tasked to predict a chaotic Mackey-Glass time series as to compare various parameters in terms of accuracy and efficiency. Parameters include the amount of bits used to represent a neuron in the liquid, the liquid’s size and the influence of different encodings to represent the time series. The accuracy is measured by the minimal validation loss of the readout layer and the efficiency is measured by the total amount of generated spikes by the liquid.

A brief overview on the use of this code can be found in [this Jupyter Notebook](experiment.ipynb).
