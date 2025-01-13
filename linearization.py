import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def create_dynamics_function():
    """Create the UAV dynamics function using CasADi."""
    # State Variables
    x_b = ca.MX.sym('x_b')    # x-position
    y_b = ca.MX.sym('y_b')    # y-position
    psi = ca.MX.sym('psi')    # yaw orientation

    # Control Inputs
    v = ca.MX.sym('v')        # linear velocity
    roll = ca.MX.sym('roll')  # roll angle

    g = 9.81

    x = ca.vertcat(x_b, y_b, psi)
    u = ca.vertcat(v, roll)

    # Kinematic Equations
    x_b_dot = v * ca.cos(psi)
    y_b_dot = v * ca.sin(psi)
    psi_dot = (g * ca.tan(roll)) / v

    xdot = ca.vertcat(x_b_dot, y_b_dot, psi_dot)

    f = ca.Function('f', [x, u], [xdot])
    return f


def linearize_with_remainder(f, x0, u0):
    """Linearize the system and compute Taylor remainder terms."""
    x = ca.MX.sym('x', f.size1_in(0))
    u = ca.MX.sym('u', f.size1_in(1))

    # First-order derivatives (A and B matrices)
    A = ca.jacobian(f(x, u), x)
    B = ca.jacobian(f(x, u), u)

    # Second-order derivatives (Hessians)
    H_xx = ca.hessian(f(x, u)[0], x)[0]  # Hessian of the first state equation w.r.t. x
    H_xu = ca.jacobian(ca.jacobian(f(x, u)[0], x), u)  # Mixed partial derivatives
    H_uu = ca.hessian(f(x, u)[0], u)[0]  # Hessian of the first state equation w.r.t. u

    # Convert CasADi expressions to functions
    A_func = ca.Function('A', [x, u], [A])
    B_func = ca.Function('B', [x, u], [B])
    H_xx_func = ca.Function('H_xx', [x, u], [H_xx])
    H_xu_func = ca.Function('H_xu', [x, u], [H_xu])
    H_uu_func = ca.Function('H_uu', [x, u], [H_uu])

    # Evaluate at the operating point (x0, u0)
    A_at_x0_u0 = np.array(A_func(x0, u0)).astype(float)
    B_at_x0_u0 = np.array(B_func(x0, u0)).astype(float)
    H_xx_at_x0_u0 = np.array(H_xx_func(x0, u0)).astype(float)
    H_xu_at_x0_u0 = np.array(H_xu_func(x0, u0)).astype(float)
    H_uu_at_x0_u0 = np.array(H_uu_func(x0, u0)).astype(float)

    # Taylor Remainder (Second-order terms approximation)
    def taylor_remainder(dx, du):
        second_order_term = (
            0.5 * dx.T @ H_xx_at_x0_u0 @ dx +
            dx.T @ H_xu_at_x0_u0 @ du +
            0.5 * du.T @ H_uu_at_x0_u0 @ du
        )
        return second_order_term
    
    f_at_x0_u0 = np.array(f(x0, u0)).flatten()

    return A_at_x0_u0, B_at_x0_u0, taylor_remainder, f_at_x0_u0


def simulate_linearized_with_remainder(linearized_systems, Ts):
    """Simulate the linearized system including the Taylor remainder."""
    time_points = np.arange(linearized_systems[0]['time'], linearized_systems[-1]['time'], Ts)

    states = np.zeros((len(linearized_systems[0]['x0']), len(time_points)))
    states[:, 0] = linearized_systems[0]['x0']

    for k in range(1, len(time_points)):
        if time_points[k] >= linearized_systems[-1]['time']:
                system = linearized_systems[-1]
        else:
            for i in range(1, len(linearized_systems)):
                if linearized_systems[i-1]['time'] <= time_points[k] < linearized_systems[i]['time']:
                    system = linearized_systems[i-1]
                    break

        dx = states[:, k-1] - system['x0']
        du = system['u0']
        
        xi_dot = (
            system['A'] @ dx +
            system['B'] @ du +
            system['remainder'](dx, du) 
        )
        states[:, k] = states[:, k-1] + Ts * xi_dot

    return states


def compare(time, xsim, usim):
    """Compare the linearized system response with the original system."""
    Ts = 0.1
    f = create_dynamics_function()
    
    # print(time)
    
    sparse_rate = 9

    indices_to_linearize = np.arange(0, len(xsim[0]), sparse_rate)
    linearized_systems = []

    for idx in indices_to_linearize:
        x0 = xsim[:, idx]
        u0 = usim[:, idx]

        A, B, remainder, f_at_x0_u0 = linearize_with_remainder(f, x0, u0)

        linearized_systems.append({
            'time': time[idx],
            'index': idx,
            'A': A,
            'B': B,
            'x0': x0,
            'u0': u0,
            'remainder': remainder,
            'f_at_x0_u0': f_at_x0_u0
        })

    fig, axs = plt.subplots(3, 1, figsize=(12, 6))
    state_labels = ["x (Position)", "y (Position)", "ψ (Yaw)"]

    lin_response = simulate_linearized_with_remainder(linearized_systems, Ts)

    for i, label in enumerate(state_labels):
        axs[i].plot(time[0:-sparse_rate], lin_response[i, :], label=f"Linearized (with remainder)", color="red")
        axs[i].plot(time, xsim[i, :], label="Reference Trajectory", color="blue")
        axs[i].set_title(f"{label} Over Time")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel(label)
        axs[i].grid()
        axs[i].legend()

    plt.tight_layout(pad=2.5)
    plt.show()

    return lin_response
