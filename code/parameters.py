import torch

# STATIC PARAMETERS
PERCENTAGE_EX_NEURONS = 0.8  # percentage of inhibitory(Inh) neurons (over nbr_neurons)
PERCENTAGE_NEURONS_PER_INPUT = 0.2  # percentage of excitatory neurons 1 input channel should input to
LEAK = 2  # leak_factor is around 2^(-leak_factor)
WINDOW_SIZE = 50
MINIMAL_WINDOW_KERNEL_VALUE = 0.02
EPOCHS = 500
LEARNING_RATE = 0.0004
BATCH_SIZE = 50
TEST_SIZE = 150

# OTHER PARAMETERS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
