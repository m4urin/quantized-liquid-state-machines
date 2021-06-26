import matplotlib.pyplot as plt
import numpy as np


def plot_spikes(ax: plt.axes, spikes: np.ndarray, start=None, end=None, **kwargs):
    """
    Plots the binary spikes by converting them to time steps
    Args:
        ax: plot
        spikes: binary spikes
        start: start time
        end: end time
        **kwargs: other parameters used by plt.eventplot()
    """
    if spikes.ndim == 1:
        spikes = spikes.reshape((1, spikes.shape[0]))
    timestamps = [[t - start for t in range(start, end) if channel[t] == 1] for channel in spikes]
    ax.eventplot(timestamps, **kwargs)


def plot_signals(ax: plt.axes, signals: np.ndarray, start=None, end=None, **kwargs):
    """
    Plots the signals
    Args:
        ax: plot
        signals: signals
        start: start time
        end: end time
        **kwargs: other parameters used by plt.plot()
    """
    if signals.ndim == 1:
        signals = signals.reshape((1, signals.shape[0]))
    for channel in signals:
        ax.plot(channel[start: end], **kwargs)


def plot_experiments(ax: plt.axes, results: np.ndarray, labels: [str], x_ticks: np.ndarray):
    """
    Make an 25/50/75 percentile error plot comparing samples of 2 parameters
    Args:
        ax: plot
        results: (parameter_1 x parameter_2 x samples) numpy list
        labels: the labels of parameter_1
        x_ticks: the values of parameter_2
    """
    for i in results.shape[0]:
        percentiles = np.percentile(results[i], [50, 25, 75], axis=1)
        ax.errorbar(x_ticks, percentiles[0],
                    yerr=np.abs(percentiles[0] - percentiles[1:]),
                    marker=['s', 'o', 'x', 'D', 'P', '*', 'p', '^'][i % 8],
                    capsize=2, label=labels[i], elinewidth=1)


def plot_x_lines(ax: plt.axes, x_values, labels):
    for i in range(len(x_values)):
        ax.axvline(x=x_values[i], linestyle=['solid', 'dashdot', 'dashed', 'dotted'][i % 4],
                   color='black', label=labels[i])


def show_dataset(start, end, time_series, encoded_spikes):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set(xlabel='Time steps', ylabel='Mackey-Glass')
    plot_signals(ax, time_series, start, end, color='red')
    plot_spikes(ax, encoded_spikes, start, end)
    plt.show()


def show_states(start, end, nbr_parameters, parameter_labels, activity,
                energy, output_spikes, output_decoded, actual, prediction):
    fig, axs = plt.subplots(4, nbr_parameters, sharex='col', sharey='row', figsize=(15, 15))

    for i in range(nbr_parameters):
        axs[0, i].set_title(parameter_labels[i])
        axs[0, i].set(xlabel='Time steps', ylabel='Percentage')
        plot_signals(axs[0, i], activity[i], start, end)
        plot_signals(axs[0, i], energy[i], start, end)
        axs[1, i].set(xlabel='Time steps', ylabel='Excitatory Neurons')
        plot_spikes(axs[1, i], output_spikes[i], start, end)
        axs[2, i].set(xlabel='Time steps', ylabel='Decoded Spikes (50 neurons)')
        plot_signals(axs[2, i], np.array(output_decoded)[i, :50], start, end)
        axs[3, i].set(xlabel='Time steps', ylabel='Prediction')
        plot_signals(axs[3, i], actual, start, end, linestyle='dashed')
        plot_signals(axs[3, i], prediction[i], start, end)

    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.show()


def show_learning(nbr_parameters, parameter_labels, val_loss, train_loss, x_markers, min_y, max_y):
    fig, axs = plt.subplots(1, nbr_parameters, sharey='row', figsize=(15, 4))
    for i in range(nbr_parameters):
        axs[i].set_title(f'{parameter_labels[i]}, loss={round(min(val_loss[i]), 4)})')
        axs[i].set(xlabel='Epochs', ylabel='Validation Loss')
        plot_signals(axs[i], val_loss[i], label='val')
        plot_signals(axs[i], train_loss[i], label='train')
        axs[i].set_ylim([min_y, max_y])
        current = 0
        for t in range(len(val_loss[i])):
            if val_loss[i][t] < x_markers[current]:
                axs[i].axvline(x=t, linestyle=['solid', 'dashdot', 'dashed', 'dotted'][current % 4], color='black',
                               label=f'<{x_markers[current]}')
                current += 1
                if current >= len(x_markers):
                    break
        axs[i].legend(loc=1)
    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()
    plt.show()
