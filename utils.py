from scipy.io import loadmat, savemat
import numpy as np


def read_from_mat(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read simulation data from a MAT file.

    :param filename: Name of the MAT file to read (e.g., "data.mat").

    :return: A tuple containing:
        - time: Time vector (NumPy array).
        - states: State history (NumPy array where rows are states and columns are time steps).
    """
    data = loadmat(filename)
    time = data['time'].squeeze()
    states = data['states']
    print(f"Data succesfully read from {filename}!")
    return time, states


def save_to_mat(filename: str, time: np.array, states: np.array) -> None:
    """
    Save simulation data to a MAT file.

    :param filename: Name of the output MAT file (e.g., "data.mat").
    :param time: Time vector.
    :param states: State history (array where rows are states and columns are time steps).
    """
    data = {'time': time, 'states': states}
    savemat(filename, data)
    print(f"Data saved to {filename}!")
