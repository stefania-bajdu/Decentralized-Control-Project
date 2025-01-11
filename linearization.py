import numpy as np
import casadi as ca
from flatness import GenerateTrajectory
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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


    A = ca.jacobian(f(x,u), x)
    B = ca.jacobian(f(x,u), u)

    A_func = ca.Function('A', [x, u], [A])
    B_func = ca.Function('B', [x, u], [B])

    A_at_x0_u0 = np.array(A_func(x0, u0)).astype(float)
    B_at_x0_u0 = np.array(B_func(x0, u0)).astype(float)

    return A_at_x0_u0, B_at_x0_u0

def simulate_linearized(A, B, x0, u, t_span):

    def linear_dynamics(t, x):
        return A @ x + B @ u

    # Solve the linear system using solve_ivp
    t_eval = np.linspace(t_span[0], t_span[1], 100)  # Time points for evaluation
    sol = solve_ivp(linear_dynamics, t_span, x0, t_eval=t_eval)

    return sol.t, sol.y

if __name__ == '__main__':
    Ts = 0.1  
    Nsim = 50 

    f = create_dynamics_function()

    traj_generator = GenerateTrajectory(Ts, Nsim)
    x_ref, u_ref = traj_generator.generate_trajectory()

    indices_to_linearize = np.arange(0, len(x_ref[0]), 10)
    linearized_points = []

    for idx in indices_to_linearize:
        x0 = x_ref[:, idx] 
        u0 = u_ref[:, idx]  

        A, B = linearize_system(f, x0, u0)
        linearized_points.append({'time': traj_generator.t[idx], 'A': A, 'B': B, 'x0': x0, 'u0': u0})

    
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    state_labels = ["x (Position)", "y (Position)", "ψ (Yaw)"]

    for i, label in enumerate(state_labels):
        for point in linearized_points:
            t_start = point['time']
            t_end = t_start + 2 
            t_span = (t_start, t_end)

            t, x_lin = simulate_linearized(point['A'], point['B'], point['x0'], point['u0'], t_span)

            axs[i].plot(t, x_lin[i, :], linestyle="--", label=f"Linearized @ t={t_start:.2f}s")

        axs[i].plot(traj_generator.t, x_ref[i, :], label="Reference Trajectory", color="blue")

        axs[i].set_title(f"{label} Over Time")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel(label)
        axs[i].grid()

    plt.tight_layout()
    plt.show()

        

