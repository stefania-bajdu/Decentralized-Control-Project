import numpy as np
import casadi as cas
import matplotlib.pyplot as plt


class MPC_Controller:
    def __init__(self):
        self._initialize_parameters()
        self.solver = cas.Opti()
        self._setup_solver()

    def _initialize_parameters(self):
        self.g = 9.81

        self.dx = 3     # x, y, yaw angle (heading)
        self.du = 2     # air speed v_a, roll angle

        self.Npred = 10
        self.Nsim = 1000
        self.Ts = 0.1

        self.umin = np.array([1, -1.04])
        self.umax = np.array([25, 1.04])
        self.delta_umin = np.array([-0.1, -0.5]) / self.Ts
        self.delta_umax = np.array([0.2, 1.1]) / self.Ts

        self.Q = np.diag([1, 1, 1])
        self.R = np.diag([1, 1])
        self.P = self.Q

        self.x0 = np.array([150, 100, 0])
        self.u0 = np.array([22, 0])

        t = np.arange(0, self.Nsim) * self.Ts

        r = 50
        self.xref = np.vstack([100 + r * np.cos(0.1*t), 100 + r * np.sin(0.1*t), np.zeros(t.shape)])

    def _setup_solver(self):
        """Set up MPC solver."""
        x = self.solver.variable(self.dx, self.Npred + 1)
        u = self.solver.variable(self.du, self.Npred)
        xinit = self.solver.parameter(self.dx)
        uinit = self.solver.parameter(self.du, 1)
        xref = self.solver.parameter(self.dx, self.Npred)
        uref = self.solver.parameter(self.du, self.Npred)

        f_dynamics = self._create_dynamics_function()

        self.solver.subject_to(x[:, 0] == xinit)

        for k in range(self.Npred):
            self.solver.subject_to(x[:, k + 1] == (x[:, k] + self.Ts * f_dynamics(x[:, k], u[:, k])))

            # if k == 0:
            #     self.solver.subject_to(self.delta_umin <= u[:, k] - uinit)
            #     self.solver.subject_to(u[:, k] - uinit <= self.delta_umax)
            # else:
            #     self.solver.subject_to(self.delta_umin <= u[:, k] - u[:, k-1])
            #     self.solver.subject_to(u[:, k] - u[:, k-1] <= self.delta_umax)

            self.solver.subject_to(self.umin <= u[:, k])
            self.solver.subject_to(u[:, k] <= self.umax)

        # objective = cas.mtimes([cas.transpose(x[:, self.Npred] - self.xref[:]), self.P, x[:, self.Npred] - self.xref[:]])
        objective = 0
        
        for k in range(self.Npred):
            if k != 0:
                objective += cas.mtimes([(x[:, k] - xref[:, k]).T, self.Q, (x[:, k] - xref[:, k])]) + \
                            cas.mtimes([(u[:, k] - u[:, k - 1]).T, self.R, (u[:, k] - u[:, k - 1])])
            else:
                objective += cas.mtimes([(x[:, k] - xref[:, k]).T, self.Q, (x[:, k] - xref[:, k])]) + \
                            cas.mtimes([(u[:, k] - uinit).T, self.R, (u[:, k] - uinit)])

        self.solver.minimize(objective)
        self.solver.solver("ipopt", {"print_time": False, "ipopt.print_level": 0})

        self.x = x
        self.u = u
        self.xinit = xinit
        self.uinit = uinit
        self.solver_xref = xref
        self.solver_uref = uref

        self.f_dynamics = f_dynamics

    def _create_dynamics_function(self):
        """Defines the model's dynamics."""
        x = cas.MX.sym("x", self.dx)
        u = cas.MX.sym("u", self.du)
        va, roll, yaw = u[0], u[1], x[2]
        dynamics = cas.vertcat(va * cas.cos(yaw), va * cas.sin(yaw), self.g * cas.tan(roll) / va)
        return cas.Function("dynamics", [x, u], [dynamics])

    def simulate(self):
        """Simulate the quadcopter dynamics."""
        xsim = np.zeros((self.dx, self.Nsim + 1))
        usim = np.zeros((self.du, self.Nsim))

        xsim[:, 0] = self.x0
        usim_init = self.u0
        
        u_init = np.tile(self.u0, (self.Npred, 1)).T

        for i in range(self.Nsim):
            self.solver.set_value(self.xinit, xsim[:, i])
            self.solver.set_value(self.uinit, usim_init)

            sol = self.solver.solve()
            usim[:, i] = sol.value(self.u)[:, 0]
            
            usim_init = usim[:, i]

            xsim[:, i + 1] = xsim[:, i] + self.Ts * self.f_dynamics(xsim[:, i], usim[:, i]).full().flatten()

        return xsim, usim

    def plot_results(self, xsim, usim):
        """Plots the simulation results."""
        positions = xsim[0:3, :]

        figsize = (12, 6)
        padding = 2.5

        # Positions
        plt.figure(figsize=figsize)
        for i, (label, ref) in enumerate(zip(["x", "y", r"$\psi$"], self.xref)):
            plt.subplot(3, 1, i + 1)
            plt.plot(positions[i, :], label=label)
            plt.plot(self.xref[i, :], "--r", label=f"{label}_ref")  # Fixing the reference plotting
            plt.legend()
            plt.ylabel(f"{label} (m)")
            plt.grid(True)

        plt.tight_layout(pad=padding)
        plt.suptitle("Quadcopter Positions", fontsize=16)

        # Control Inputs
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(usim[0, :], label="va")
        plt.legend()
        plt.ylabel("Air Speed (va)")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(usim[1, :], label="roll")
        plt.legend()
        plt.ylabel("Roll Angle")
        plt.grid(True)

        plt.tight_layout(pad=3.0)
        plt.suptitle("Control Inputs (Air Speed and Roll)", fontsize=16)

        plt.show()

# Run simulation
quadcopter = MPC_Controller()
xsim, usim = quadcopter.simulate()
quadcopter.plot_results(xsim, usim)
