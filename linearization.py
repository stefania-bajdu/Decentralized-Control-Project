import numpy as np
import casadi as ca
# from flatness import GenerateTrajectory
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


WPListState = [
    np.array([0, 50, 100, np.pi/6]),
    np.array([10, 250, 200, np.pi/4]),
    np.array([20, 450, 300, np.pi/3]),
    np.array([30, 650, 400, np.pi/3]),
    np.array([40, 850, 500, np.pi/6]),
    np.array([50, 1050, 700, np.pi/6]),
]

WPCommands = [
    np.array([20, 0.2618]),
    np.array([20, 0.1745]),
    np.array([20, -0.2618]),
    np.array([20, -0.2618]),
    np.array([20, 0.1745]),
    np.array([20, 0.1745]),
]


def create_dynamics_function():
    # State Variables
    x_b = ca.MX.sym('x_b')    # x-position
    y_b = ca.MX.sym('y_b')    # y-position
    psi = ca.MX.sym('psi')    # yaw orientation

    # Control Inputs
    v = ca.MX.sym('v')        # linear velocity
    roll = ca.MX.sym('roll')  # angular velocity

    # Parameters
    g = 9.81  # gravitational acceleration (m/s^2)

    # State Vector and Control Vector
    x = ca.vertcat(x_b, y_b, psi)
    u = ca.vertcat(v, roll)

    # Kinematic Equations
    x_b_dot = v * ca.cos(psi)
    y_b_dot = v * ca.sin(psi)
    psi_dot = (g * np.tan(roll)) / v

    # State Derivatives
    xdot = ca.vertcat(x_b_dot, y_b_dot, psi_dot)

    # CasADi Function
    f = ca.Function('f', [x, u], [xdot])
    return f


def linearize_system(f, x0, u0):
    x = ca.MX.sym('x', f.size1_in(0))
    u = ca.MX.sym('u', f.size1_in(1))

    A = ca.jacobian(f(x, u), x)
    B = ca.jacobian(f(x, u), u)

    A_func = ca.Function('A', [x, u], [A])
    B_func = ca.Function('B', [x, u], [B])

    A_at_x0_u0 = np.array(A_func(x0, u0)).astype(float)
    B_at_x0_u0 = np.array(B_func(x0, u0)).astype(float)

    return A_at_x0_u0, B_at_x0_u0


def simulate_linearized(linearized_systems, Ts):
    time_points = np.arange(linearized_systems[0]['time'], linearized_systems[-1]['time'], Ts)

    states = np.zeros((len(linearized_systems[0]['x0']), len(time_points)))
    states[:, 0] = linearized_systems[0]['x0']

    for k in range(1, len(time_points)):
        for i in range(1, len(linearized_systems)):
            if linearized_systems[i-1]['time'] <= time_points[k] < linearized_systems[i]['time']:
                system = linearized_systems[i-1]
                break

        xi_dot = system['A'] @ states[:, k-1] + system['B'] @ system['u0']
        states[:, k] = states[:, k-1] + Ts * xi_dot

    return states

# def simulate_linearized(linearized_systems, Ts):
#     # Define time points for simulation
#     time_points = np.arange(linearized_systems[0]['time'], linearized_systems[-1]['time'], Ts)

#     # Initialize state storage
#     states = []

#     # Start with the initial state
#     x0 = linearized_systems[0]["x0"]
#     states.append(x0)

#     # Iterate over time points
#     for k in range(1, len(time_points)):
#         t_start = time_points[k-1]
#         t_end = time_points[k]
#         t_span = [t_start, t_end]

#         # Find the appropriate linearized system for the current time
#         for i in range(1, len(linearized_systems)):
#             if linearized_systems[i-1]['time'] <= t_start < linearized_systems[i]['time']:
#                 system = linearized_systems[i-1]
#                 break
#         else:
#             system = linearized_systems[-1]  # Use the last system if no match is found

#         # Extract system matrices and input
#         A = system["A"]
#         B = system["B"]
#         u = system["u0"]

#         # Define linearized dynamics
#         def linear_dynamics(t, x):
#             return A @ x + B @ u

#         # Solve the linearized dynamics for the current time step
#         sol = solve_ivp(linear_dynamics, t_span, x0, t_eval=[t_end])

#         # Update the state
#         x_next = sol.y[:, -1]  # Get the state at t_end
#         states.append(x_next)

#         # Update the initial state for the next iteration
#         x0 = x_next

#     return np.array(states).T


def compare(time, xsim, usim):
    Ts = 0.1
    f = create_dynamics_function()

    indices_to_linearize = np.arange(0, len(xsim[0]), 1)
    linearized_systems = []

    for idx in indices_to_linearize:
        x0 = xsim[:, idx]
        u0 = usim[:, idx]

        A, B = linearize_system(f, x0, u0)

        linearized_systems.append({'time': time[idx], 'index': idx, 'A': A, 'B': B, 'x0': x0, 'u0': u0})

    fig, axs = plt.subplots(3, 1, figsize=(12, 6))
    state_labels = ["x (Position)", "y (Position)", "ψ (Yaw)"]

    lin_response = simulate_linearized(linearized_systems, Ts)

    for i, label in enumerate(state_labels):
        axs[i].plot(time, lin_response[i, :], label=f"Linearized", color="red")
        axs[i].plot(time, xsim[i, :], label="Reference Trajectory", color="blue")
        axs[i].set_title(f"{label} Over Time")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel(label)
        axs[i].grid()

    plt.tight_layout(pad=2.5)
    plt.show()
    
    return lin_response
