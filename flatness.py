import numpy as np
import matplotlib.pyplot as plt


class GenerateTrajectory():
    def __init__(self, Ts):
        self.g = 9.81
        self.Ts = Ts

        self.dx = 3   # x, y, yaw
        self.du = 2   # air speed v, roll

        self.umin = np.array([18, -0.43])
        self.umax = np.array([25, 0.43])

        self.Nsim = 40  # final time in seconds
        self.t = np.arange(0, self.Nsim + Ts, Ts)

        # Waypoints
        self.WPNum = 5
        self.WPListState = [
            np.array([0, 50, 100, np.pi / 6]),
            np.array([10, 250, 200, np.pi / 6]),
            np.array([20, 450, 300, np.pi / 6]),
            np.array([30, 650, 400, np.pi / 6]),
            np.array([40, 850, 500, np.pi / 6]),
        ]

        self.coeffs = []
        self.ui = np.array([20, 0])
        self.uf = np.array([20, 0])

        for i in range(self.WPNum - 1):
            Ti = self.WPListState[i][0]
            Tf = self.WPListState[i + 1][0]
            xi = self.WPListState[i][1:4]
            xf = self.WPListState[i + 1][1:4]

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
                [xi[0], self.ui[0] * np.cos(xi[2]), -np.sin(xi[2]) * self.g * np.tan(self.ui[1])],
                [xi[1], self.ui[0] * np.sin(xi[2]), np.cos(xi[2]) * self.g * np.tan(self.ui[1])],
                [xf[0], self.uf[0] * np.cos(xf[2]), -np.sin(xf[2]) * self.g * np.tan(self.uf[1])],
                [xf[1], self.uf[0] * np.sin(xf[2]), np.cos(xf[2]) * self.g * np.tan(self.uf[1])]
            ])

            self.coeffs.append(np.linalg.solve(M, b))

        # Dynamics and trajectory functions
        f_dyn = lambda x, u: np.array([
            u[0] * np.cos(x[2]),
            u[0] * np.sin(x[2]),
            self.g * np.tan(u[1]) / u[0]
        ])

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

        # Plot results
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.grid()
        plt.plot(self.t, x_ref[0, :], 'b')
        for i in range(self.WPNum):
            plt.stem([self.WPListState[i][0]], [self.WPListState[i][1]], linefmt='r-', markerfmt='ro', basefmt=' ')
        plt.xlabel('Time (t)')
        plt.ylabel('x(t)')
        plt.title('Position x')

        plt.subplot(3, 1, 2)
        plt.grid()
        plt.plot(self.t, x_ref[1, :], 'b')
        for i in range(self.WPNum):
            plt.stem([self.WPListState[i][0]], [self.WPListState[i][2]], linefmt='r-', markerfmt='ro', basefmt=' ')
        plt.xlabel('Time (t)')
        plt.ylabel('y(t)')
        plt.title('Position y')

        plt.subplot(3, 1, 3)
        plt.grid()
        plt.plot(self.t, x_ref[2, :], 'b')
        for i in range(self.WPNum):
            plt.stem([self.WPListState[i][0]], [self.WPListState[i][3]], linefmt='r-', markerfmt='ro', basefmt=' ')
        plt.xlabel('Time (t)')
        plt.ylabel('Yaw Angle')
        plt.title('Yaw Angle')

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.grid()
        plt.plot(self.t, u_ref[0, :], 'b')
        plt.axhline(self.umin[0], color='r', linestyle='--')
        plt.axhline(self.umax[0], color='r', linestyle='--')
        plt.xlabel('Time (t)')
        plt.ylabel('v_a(t)')
        plt.title('Air Speed')

        plt.subplot(2, 1, 2)
        plt.grid()
        plt.plot(self.t, u_ref[1, :], 'b')
        plt.axhline(self.umin[1], color='r', linestyle='--')
        plt.axhline(self.umax[1], color='r', linestyle='--')
        plt.xlabel('Time (t)')
        plt.ylabel('Roll Angle')
        plt.title('Roll Angle')

        plt.show()


x = GenerateTrajectory(0.1)
# Save results
# np.savez("trajectory.npz", x_ref=x_ref, u_ref=u_ref)
