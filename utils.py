from scipy.io import loadmat, savemat
import numpy as np

def read_from_mat(filename: str, signal_name: str) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(filename)
    time = data['time'].squeeze()
    signal = data[signal_name] 
    print(f"Data successfully read from {filename}!")
    return time, signal

def save_to_mat(filename: str, time: np.ndarray, signals: np.ndarray, signal_name: str) -> None:
    data = {'time': time, signal_name: signals}
    savemat(filename, data)
    print(f"Data saved to {filename}!")
