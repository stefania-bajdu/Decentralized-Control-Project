from model import Model
import matplotlib.pyplot as plt
from utils import *


def plot_results(time, states):
    """
    Plot the simulation results (x, y, psi).

    :param time: Time vector from the simulation.
    :param states: State history of the UAV (x, y, psi).
    """
    figsize = (12, 6)
    padding = 4

    plt.figure(figsize=figsize)

    # Plot x position
    plt.subplot(3, 1, 1)
    plt.plot(time, states[0, :], label='x')
    plt.legend()
    plt.ylabel('x (m)')
    plt.grid(True)

    # Plot y position
    plt.subplot(3, 1, 2)
    plt.plot(time, states[1, :], label='y')
    plt.legend()
    plt.ylabel('y (m)')
    plt.grid(True)

    # Plot yaw (psi)
    plt.subplot(3, 1, 3)
    plt.plot(time, states[2, :], label=r'$\psi$')
    plt.legend()
    plt.ylabel('Yaw (rad)')
    plt.xlabel('Time (s)')
    plt.grid(True)

    # Adjust layout and display
    plt.tight_layout(pad=padding)
    plt.suptitle("UAV States", fontsize=16)
    plt.show()


time, states = read_from_mat("states.mat")
plot_results(time, states)
