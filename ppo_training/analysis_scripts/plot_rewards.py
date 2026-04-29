from matplotlib import use
use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    rewards = np.load(r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\training_rewards.npy")

    plt.plot(rewards)
    plt.show()
