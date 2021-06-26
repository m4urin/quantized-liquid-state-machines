import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim, tensor
from torch.utils.data import Dataset, DataLoader

from code.parameters import DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE


class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Creates a tensor dataset of the numpy arrays stored on the selected device
        Args:
            x: input data
            y: expected output
        """
        self.x = tensor(x).to(DEVICE).float()
        self.y = tensor(y).to(DEVICE).float()
        self.size = len(self.x)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LSMDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 liquid_warmup=100, validate_percentage=0.2, test_size=25):
        """
        Creates a Dataset from the LSM output signal and expected predictions

        Args:
            x: LSM output signal
            y: prediction
            liquid_warmup: first time steps that should not be used for
                            training/evaluating
            validate_percentage: standard 20% validation size
            test_size: final time steps that can be used for testing the prediction quality
                        (the readout is not trained on this data correlation of data)

        """

        # transpose (U x T) to (T x U) so we have inout states and output states
        x = np.transpose(x)
        y = np.transpose(y)

        # test set is the last n states that we are interested in (the actual prediction)
        x_test, y_test = x[-test_size:], y[-test_size:]

        # the rest of the data (starting after warm up) is randomly split into 80% train data and 20% validation data
        x_train, x_eval, y_train, y_eval = train_test_split(x[liquid_warmup:-test_size], y[liquid_warmup:-test_size],
                                                            test_size=validate_percentage)

        # create the various data sets that we can loop over
        self.train = DataLoader(
            dataset=NumpyDataset(x_train, y_train),
            batch_size=BATCH_SIZE, pin_memory=False, shuffle=True)
        self.eval = DataLoader(
            dataset=NumpyDataset(x_eval, y_eval),
            batch_size=BATCH_SIZE, pin_memory=False)
        self.test = DataLoader(
            dataset=NumpyDataset(x_test, y_test),
            batch_size=BATCH_SIZE, pin_memory=False)


class ReadoutLayer(nn.Module):
    def __init__(self, in_features):
        """
        Creates a single layer using pytorch to do regression

        Args:
            in_features: number of excitatory neurons to read from
        """
        super().__init__()

        # fully connected layer
        self.fc = nn.Linear(in_features, 1)

        # optimize and loss function
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        # use gpu if possible
        self.to(DEVICE)

    def forward(self, x: tensor):
        return self.fc(x)

    def predict(self, x: np.ndarray):
        x = tensor(x).t().float().to(DEVICE)
        y = self(x).detach().cpu().t().numpy()
        return y

    def eval_step(self, x: tensor, y: tensor):
        return self.loss(self(x), y).item()

    def train_step(self, x: tensor, y: tensor):
        loss = self.loss(self(x), y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def run_epoch(self, data, step_function):
        total_loss = 0
        for i, (x, y) in enumerate(data):
            total_loss += step_function(x, y)
        return total_loss / len(data)

    def train_readout(self, x: np.ndarray, y: np.ndarray):
        dataset = LSMDataset(x, y)
        all_train_loss, all_val_loss = [], []

        for epoch in range(EPOCHS):
            self.train()
            train_loss = self.run_epoch(dataset.train, self.train_step)
            all_train_loss.append(train_loss)

            self.eval()
            val_loss = self.run_epoch(dataset.eval, self.eval_step)
            all_val_loss.append(val_loss)

        return np.array(all_train_loss), np.array(all_val_loss)

    def reset_weights(self):
        self.fc.reset_parameters()
