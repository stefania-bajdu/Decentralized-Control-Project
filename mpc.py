from utils import *
import casadi as cas
import matplotlib.pyplot as plt
import time


class MPC_Controller:
    def __init__(self, Ts, Nsim, Npred, xref, uref):
        self._initialize_parameters(Ts, Nsim, Npred, xref, uref)
        self.solver = cas.Opti()
        self._setup_solver()

    def _initialize_parameters(self, Ts, Nsim, Npred, xref, uref):
        self.g = 9.81

        self.dx = 3     # x, y, yaw angle (heading)
        self.du = 2     # air speed v_a, roll angle

        self.Npred = Npred
        self.Nsim = Nsim
        self.Ts = Ts

        self.umin = np.array([15, -0.8])
        self.umax = np.array([30, 0.8])
        self.delta_umin = np.array([-0.1, -0.5]) / self.Ts
        self.delta_umax = np.array([0.2, 1.1]) / self.Ts

        self.Q = np.diag([10, 10, 5])
        self.Rdelta = np.diag([1, 1])
        self.R = np.diag([1, 5])
        self.P = self.Q

        self.u_ref = uref
        self.x_ref = xref

        self.x0 = xref[:, 0]
        self.u0 = uref[:, 0]

        self.t = np.arange(0, self.Nsim + Ts, Ts)

    def _create_dynamics_function(self, x, u):
        """Defines the model's dynamics."""
        x = cas.MX.sym("x", self.dx)
        u = cas.MX.sym("u", self.du)
        va, roll, yaw = u[0], u[1], x[2]
        dynamics = cas.vertcat(va * cas.cos(yaw), va * cas.sin(yaw), self.g * cas.tan(roll) / va)
        return cas.Function("dynamics", [x, u], [dynamics])

    def _setup_solver(self):
        """Set up MPC solver."""
        x = self.solver.variable(self.dx, self.Npred + 1)
        u = self.solver.variable(self.du, self.Npred)
        xinit = self.solver.parameter(self.dx)
        uinit = self.solver.parameter(self.du, 1)
        xref = self.solver.parameter(self.dx, self.Npred)
        uref = self.solver.parameter(self.du, self.Npred)

        f_dynamics = self._create_dynamics_function(x, u)

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

        objective = 0

        for k in range(self.Npred):
            if k != 0:
                objective += cas.mtimes([(x[:, k] - xref[:, k]).T, self.Q, (x[:, k] - xref[:, k])]) + \
                    cas.mtimes([(u[:, k] - u[:, k - 1]).T, self.Rdelta, (u[:, k] - u[:, k - 1])]) + \
                    cas.mtimes([(u[:, k] - uref[:, k]).T, self.R, (u[:, k] - uref[:, k])])
            else:
                objective += cas.mtimes([(x[:, k] - xref[:, k]).T, self.Q, (x[:, k] - xref[:, k])]) + \
                    cas.mtimes([(u[:, k] - uinit).T, self.Rdelta, (u[:, k] - uinit)]) + \
                    cas.mtimes([(u[:, k] - uref[:, k]).T, self.R, (u[:, k] - uref[:, k])])

        self.solver.minimize(objective)
        self.solver.solver("ipopt", {"print_time": False, "ipopt.print_level": 0})

        self.x = x
        self.u = u
        self.xinit = xinit
        self.uinit = uinit
        self.solver_xref = xref
        self.solver_uref = uref

        self.f_dynamics = f_dynamics

    def simulate(self):
        """Simulate the response."""
        xsim = np.zeros((self.dx, self.Nsim + 1))
        usim = np.zeros((self.du, self.Nsim))

        xsim[:, 0] = self.x0
        usim_init = self.u0

        self.solver.set_initial(self.u, np.tile(self.u0, (self.Npred, 1)).T)

        start_time = time.time()
        for i in range(self.Nsim - self.Npred):
            self.solver.set_value(self.xinit, xsim[:, i])
            self.solver.set_value(self.uinit, usim_init)

            if i + self.Npred <= self.Nsim:
                self.solver.set_value(self.solver_xref, self.x_ref[:, i: i + self.Npred])
                self.solver.set_value(self.solver_uref, self.u_ref[:, i: i + self.Npred])
            # else:
            #     remaining_refs = self.x_ref[:, i:]
            #     padding_refs = np.tile(self.x_ref[:, -1:], (1, self.Npred - (self.Nsim - i)))
            #     full_refs = np.hstack([remaining_refs, padding_refs])
            #     self.solver.set_value(self.solver_xref, full_refs)

            #     remaining_refs = self.u_ref[:, i:]
            #     padding_refs = np.tile(self.u_ref[:, -1:], (1, self.Npred - (self.Nsim - i)))
            #     full_refs = np.hstack([remaining_refs, padding_refs])
            #     self.solver.set_value(self.solver_uref, full_refs)

            sol = self.solver.solve()
            usim[:, i] = sol.value(self.u)[:, 0]

            usim_init = usim[:, i]

            xsim[:, i + 1] = xsim[:, i] + self.Ts * self.f_dynamics(xsim[:, i], usim[:, i]).full().flatten()

        execution_time = time.time() - start_time
        execution_time_per_iteration = execution_time / (self.Nsim - self.Npred)
        
        print(f"Total execution time for the main loop: {execution_time:.4f} seconds")
        print(f"Mean execution time per iteration: {execution_time_per_iteration:.4f} seconds")
        return xsim, usim
