import matplotlib.pyplot as plt
from utils import *


def plot_UAV_states(time, states):
    """
    Plot the simulation results (x, y, psi).

    :param time: Time vector from the simulation.
    :param states: State history of the UAV (x, y, psi).
    """
    figsize = (12, 6)
    padding = 4

    plt.figure(figsize=figsize)
    for i, (label, unit) in enumerate(zip(["x", "y", r"$\psi$"], ["(m)", "(m)", "(rad)"])):
        plt.subplot(3, 1, i + 1)
        plt.plot(time, states[i, :], label=label)
        plt.legend()
        plt.ylabel(f"{label} {unit}")
        plt.grid(True)

    plt.tight_layout(pad=padding)
    plt.suptitle("UAV States", fontsize=16)
    plt.show()


def plot_states_with_references(time, xsim, usim, xref, uref, Wp):
    figsize = (12, 6)
    padding = 2.5

    # Plot 2d
    plt.figure(figsize=figsize)
    plt.plot(xsim[0, :], xsim[1, :], label="sim")
    plt.plot(xref[0, :], xref[1, :], "--r", label=f"ref")
    plt.legend()
    plt.ylabel(f"y")
    plt.xlabel(f"x")
    plt.grid(True)
    for j in range(len(Wp) - 1):
        plt.stem([Wp[j][1]], [Wp[j][2]], linefmt='r-', markerfmt='ro', basefmt=' ')
    plt.tight_layout(pad=padding)
    plt.suptitle("States", fontsize=16)

    # Plot states
    plt.figure(figsize=figsize)
    for i, (label, unit) in enumerate(zip(["x", "y", r"$\psi$"], ["(m)", "(m)", "(rad)"])):
        plt.subplot(3, 1, i + 1)
        plt.plot(time, xsim[i, :], label=label)
        plt.plot(time, xref[i, :], "--r", label=f"{label}_ref")
        plt.legend()
        plt.ylabel(f"{label} {unit}")
        plt.xlabel(f"Time (s)")
        plt.grid(True)

    plt.tight_layout(pad=padding)
    plt.suptitle("States", fontsize=16)

    # Plot control inputs
    plt.figure(figsize=figsize)
    for i, (label, unit) in enumerate(zip([r"$v_a$", r"$\phi$"], ["(m/s)", "(rad)"])):
        plt.subplot(2, 1, i + 1)
        plt.plot(time, usim[i, :], label=label)
        plt.plot(time, uref[i, :], "--r", label=f"{label}_ref")
        plt.legend()
        plt.ylabel(f"{label} {unit}")
        plt.xlabel(f"Time (s)")
        plt.grid(True)

    plt.tight_layout(pad=padding)
    plt.suptitle("Commands", fontsize=16)
    plt.show()

if __name__ == "__main__":
    time, states = read_from_mat("states.mat", "states")
    plot_UAV_states(time, states)
