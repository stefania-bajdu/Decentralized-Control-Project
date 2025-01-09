import numpy as np
import matplotlib.pyplot as plt


class GenerateTrajectory():
    def __init__(self, Ts, Nsim):
        self.g = 9.81
        self.Ts = Ts

        self.dx = 3   # x, y, yaw
        self.du = 2   # air speed v, roll

        self.umin = np.array([18, -0.43])
        self.umax = np.array([25, 0.43])

        self.Nsim = Nsim  # final time in seconds
        self.t = np.arange(0, self.Nsim + Ts, Ts)
        
        self.coeffs = []

    def _set_waypoints(self):
        self.WPNum = 5
        self.WPListState = [
            np.array([0, 50, 100, np.pi / 6]),
            np.array([10, 250, 200, np.pi / 6]),
            np.array([20, 450, 300, np.pi / 6]),
            np.array([30, 650, 400, np.pi / 6]),
            np.array([40, 850, 500, np.pi / 6]),
        ]
        
        self.WPCommands = [
            np.array([20, 0]),
            np.array([20, 0]),
            np.array([20, 0]),
            np.array([20, 0]),
            np.array([20, 0]),
        ]


    def _compute_coeffs(self):
        for i in range(self.WPNum - 1):
            Ti = self.WPListState[i][0]
            Tf = self.WPListState[i + 1][0]
            xi = self.WPListState[i][1:4]
            xf = self.WPListState[i + 1][1:4]
            ui = self.WPCommands[i][:]
            uf = self.WPCommands[i + 1][:]

            M = np.zeros((12, 12))
            pol = np.array([1, Ti, Ti**2, Ti**3, Ti**4, Ti**5])
            dpol = np.array([0, 1, 2*Ti, 3*Ti**2, 4*Ti**3, 5*Ti**4])
            ddpol = np.array([0, 0, 2, 6*Ti, 12*Ti**2, 20*Ti**3])

            M[:3, :] = np.hstack((np.vstack((pol, dpol, ddpol)), np.zeros((3, 6))))
            M[3:6, :] = np.hstack((np.zeros((3, 6)), np.vstack((pol, dpol, ddpol))))

            pol = np.array([1, Tf, Tf**2, Tf**3, Tf**4, Tf**5])
            dpol = np.array([0, 1, 2*Tf, 3*Tf**2, 4*Tf**3, 5*Tf**4])
            ddpol = np.array([0, 0, 2, 6*Tf, 12*Tf**2, 20*Tf**3])

            M[6:9, :] = np.hstack((np.vstack((pol, dpol, ddpol)), np.zeros((3, 6))))
            M[9:12, :] = np.hstack((np.zeros((3, 6)), np.vstack((pol, dpol, ddpol))))

            b = np.concatenate([
                [xi[0], ui[0] * np.cos(xi[2]), -np.sin(xi[2]) * self.g * np.tan(ui[1])],
                [xi[1], ui[0] * np.sin(xi[2]), np.cos(xi[2]) * self.g * np.tan(ui[1])],
                [xf[0], uf[0] * np.cos(xf[2]), -np.sin(xf[2]) * self.g * np.tan(uf[1])],
                [xf[1], uf[0] * np.sin(xf[2]), np.cos(xf[2]) * self.g * np.tan(uf[1])]
            ])

            self.coeffs.append(np.linalg.solve(M, b))

    def generate_trajectory(self):
        self._set_waypoints()
        self._compute_coeffs()        

        z1 = lambda t, a: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
        z2 = lambda t, a: a[6] + a[7]*t + a[8]*t**2 + a[9]*t**3 + a[10]*t**4 + a[11]*t**5

        z1_dot = lambda t, a: a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
        z2_dot = lambda t, a: a[7] + 2*a[8]*t + 3*a[9]*t**2 + 4*a[10]*t**3 + 5*a[11]*t**4

        z1_ddot = lambda t, a: 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3
        z2_ddot = lambda t, a: 2*a[8] + 6*a[9]*t + 12*a[10]*t**2 + 20*a[11]*t**3

        X = lambda t, a: np.array([
            z1(t, a), z2(t, a), np.arctan2(z2_dot(t, a), z1_dot(t, a))
        ])

        U = lambda t, a: np.array([
            np.sqrt(z1_dot(t, a)**2 + z2_dot(t, a)**2),
            np.arctan2(
                z2_ddot(t, a)*z1_dot(t, a) - z2_dot(t, a)*z1_ddot(t, a),
                self.g * np.sqrt(z1_dot(t, a)**2 + z2_dot(t, a)**2)
            )
        ])

        # Reference trajectories
        x_ref = np.zeros((self.dx, len(self.t)))
        u_ref = np.zeros((self.du, len(self.t)))

        for k in range(len(self.t)):
            for j in range(self.WPNum - 1):
                if self.t[k] <= self.WPListState[j + 1][0]:
                    x_ref[:, k] = X(self.t[k], self.coeffs[j])
                    u_ref[:, k] = U(self.t[k], self.coeffs[j])
                    break
                
        return x_ref, u_ref
        
    def plot_trajectories(self, x_ref, u_ref):
        figsize = (12, 6)
        padding = 2.5
        
        # Plot state trajectory
        plt.figure(figsize=figsize)
        for i, (label, axis_label) in enumerate(zip(["x (m)", "y (m)", r"$\psi$ (rad)"], ["Position x", "Position y", "Yaw Angle"])):
            plt.subplot(3, 1, i + 1)
            plt.grid()
            plt.plot(self.t, x_ref[i, :], 'b', label=f"{label}")
            for j in range(self.WPNum):
                plt.stem([self.WPListState[j][0]], [self.WPListState[j][i + 1]], linefmt='r-', markerfmt='ro', basefmt=' ')
            plt.xlabel("Time (t)")
            plt.ylabel(f"{label}")
            plt.title(axis_label)
            
        plt.tight_layout(pad=padding)

        # Plot command trajectory
        plt.figure(figsize=figsize)
        for i, (label, axis_label, unit) in enumerate(zip([r"$v_a$", r"$\phi$"], ["Air Speed", "Roll Angle"], ["(m/s)", "(rad)"])):
            plt.subplot(2, 1, i + 1)
            plt.grid()
            plt.plot(self.t, u_ref[i, :], 'b', label=label)
            plt.axhline(self.umin[i], color='r', linestyle='--')
            plt.axhline(self.umax[i], color='r', linestyle='--')
            plt.xlabel("Time (t)")
            plt.ylabel(f"{label} {unit}")
            plt.title(axis_label)

        plt.tight_layout(pad=padding)
        plt.show()

