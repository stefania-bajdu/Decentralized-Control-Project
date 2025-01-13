from model import *
from ploting import *
from flatness import *
from mpc import *
from linearization import compare

Ts = 0.1
Npred = 10

testIdx = 2

if testIdx == 1:
    WPListState = [
        np.array([0, 50, 100, np.pi / 6]),
        np.array([10, 250, 200, np.pi / 6]),
        np.array([20, 450, 300, np.pi / 6]),
        np.array([30, 650, 400, np.pi / 6]),
        np.array([40, 850, 500, np.pi / 6]),
        np.array([50, 1050, 600, np.pi / 6]),
    ]

    WPCommands = [
        np.array([20, 0]),
        np.array([20, 0]),
        np.array([20, 0]),
        np.array([20, 0]),
        np.array([20, 0]),
        np.array([20, 0]),
    ]
elif testIdx == 2:
    WPListState = [
        np.array([0, 50, 100, np.pi/6]),
        np.array([10, 250, 200, np.pi/4]),
        np.array([20, 450, 300, np.pi/3]),
        np.array([30, 650, 400, np.pi/3]),
        np.array([40, 850, 500, np.pi/6]),
        np.array([50, 950, 700, np.pi/6]),
    ]

    WPCommands = [
        np.array([20, 0.2618]),
        np.array([20, 0.1745]),
        np.array([20, -0.2618]),
        np.array([20, -0.2618]),
        np.array([20, 0.1745]),
        np.array([20, 0.1745]),
    ]
elif testIdx == 3:
    WPListState = [
        np.array([0, 50, 100, np.pi/6]),
        np.array([10, 240, 185, np.pi/4]),
        np.array([20, 435, 325, -np.pi/20]),
        np.array([30, 630, 420, -np.pi/10]),
        np.array([40, 850, 500, np.pi/12]),
        np.array([50, 1050, 550, -np.pi/6]),
    ]

    WPCommands = [
        np.array([20, 0.2618]),
        np.array([20, 0.1745]),
        np.array([20, -0.2618]),
        np.array([20, -0.2618]),
        np.array([20, 0.1745]),
        np.array([20, 0.1745]),
    ]
elif testIdx == 4:
    WPListState = [
        np.array([0, 50, 100, np.pi/2.3]),
        np.array([10, 180, 230, np.pi/1.6]),
        np.array([20, 290, 380, np.pi/2.3]),
        np.array([30, 440, 450, np.pi/2.2]),
        np.array([40, 490, 640, np.pi/1.12]),
        np.array([50, 520, 830, np.pi/2.5]),
    ]

    WPCommands = [
        np.array([20, 0.2618]),
        np.array([20, 0.2618]),
        np.array([20, 0.1745]),
        np.array([20, -0.2618]),
        np.array([20, 0.1745]),
        np.array([20, 0.1745]),
    ]

SimTime = WPListState[-1][0] - Ts
# t = np.arange(0, SimTime - 10 + Ts, Ts)
t = np.arange(0, SimTime + Ts, Ts)

traj = GenerateTrajectory(Ts, SimTime)
xref, uref = traj.generate_trajectory(WPListState, WPCommands)
# traj.plot_trajectories(xref, uref)

Nsim = len(xref[1, :])

uav = MPC_Controller(Ts, Nsim, Npred, xref, uref)
xsim, usim = uav.simulate()

# plot_states_with_references(t[:len(t)-Npred], xsim[:, :len(t)-Npred], usim[:, :len(t)-Npred], xref[:, :len(t)-Npred], uref[:, :len(t)-Npred], WPListState)
lin_response = compare(t[:len(t)-Npred], xsim[:, :len(t)-Npred], usim[:, :len(t)-Npred])

save_to_mat(f"../project/xref_{testIdx}.mat", t[:len(t)-Npred], xref[:, :len(t)-Npred], "x_ref")
save_to_mat(f"../project/uref_{testIdx}.mat", t[:len(t)-Npred], uref[:, :len(t)-Npred], "u_ref")
save_to_mat(f"../project/xsim_{testIdx}.mat", t[:len(t)-Npred], xsim[:, :len(t)-Npred], "xsim")
save_to_mat(f"../project/usim_{testIdx}.mat", t[:len(t)-Npred], usim[:, :len(t)-Npred], "usim")
save_to_mat(f"../project/lin_x_{testIdx}.mat", t[:len(t)-Npred], lin_response, "lin_x")
