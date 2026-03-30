import sys
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) > 1:
        np_file = sys.argv[1]
    else:
        np_file = "./logs/sac/evaluations.npz"

    data = np.load(np_file)
    x = data["timesteps"]
    y = data["results"].mean(axis=1)
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("SAC on CarRacing-v3")
    plt.grid()
    plt.show()
