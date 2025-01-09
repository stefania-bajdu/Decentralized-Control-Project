from utils import *


class Model:
    def __init__(self, Ts: float = 0.1) -> None:
        """
        Initialize the model's parameters.

        :param Ts: Sampling time (seconds).
        :param g: Gravitational acceleration (m/s^2).
        """
        self.g = 9.81
        self.Ts = Ts

    def dynamics(self, state: np.ndarray, va: float, roll: float) -> np.ndarray:
        """
        Compute the system's dynamics (state derivatives).

        :param state: Current state [x, y, psi (yaw)].
        :param va: Airspeed of the UAV (m/s).
        :param roll: Roll angle of the UAV (radians).

        :return: State derivatives [x_dot, y_dot, psi_dot].
        """
        psi = state[2]  
        return np.array([va * np.cos(psi), va * np.sin(psi), self.g * np.tan(roll) / va])

    def simulate(self,
                 initial_state: np.ndarray,
                 va: float,
                 roll: float,
                 time_span: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the UAV dynamics using Euler integration over a given time span.

        :param initial_state: Initial state [x, y, psi (yaw)].
        :param va: Airspeed of the UAV (m/s).
        :param roll: Roll angle of the UAV (radians).
        :param time_span: Tuple defining the simulation start and end times (seconds).

        :return: Tuple containing the time vector and state evolution history.
        """
        time_points = np.arange(time_span[0], time_span[1] + self.Ts, self.Ts)

        # Initialize state history (each column represents a state at a time point)
        states = np.zeros((len(initial_state), len(time_points)))
        states[:, 0] = initial_state

        for k in range(1, len(time_points)):
            xi_dot = self.dynamics(states[:, k-1], va, roll)
            states[:, k] = states[:, k-1] + self.Ts * xi_dot

        return time_points, states


initial_state = np.array([0.0, 0.0, 0.0])
time_span = (0, 100)
Ts = 0.01
va = 22
roll = 0.4

uav = Model(Ts=Ts)
time, states = uav.simulate(initial_state, va, roll, time_span)

save_to_mat("states.mat", time, states)
