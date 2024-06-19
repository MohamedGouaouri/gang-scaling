
import matplotlib.pyplot as plt
import numpy as np


def save_performance_plots(losses, rewards, experiment_name):
    # Plot losses and rewards side by side
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(losses, color='tab:blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')
    plt.title('Loss per Episode')

    # Create figure for rewards
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward', color='tab:red')
    ax2.plot(rewards, color='tab:red', label='Reward')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper left')
    plt.title('Reward per Episode')

    # Adjust layout and display both figures
    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig(f'models/{experiment_name}_loss.png')
    fig2.savefig(f'models/{experiment_name}_reward.png')


def exponential_moving_average(arr, window_size):
    alpha = 2 / (window_size + 1)
    ema = np.zeros_like(arr, dtype=np.float64)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema
