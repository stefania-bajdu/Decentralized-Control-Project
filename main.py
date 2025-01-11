from model import *
from ploting import *
from flatness import *
from mpc import *

Ts = 0.1
SimTime = 50 - Ts
t = np.arange(0, SimTime - 10 + Ts, Ts)

traj = GenerateTrajectory(Ts, SimTime)
xref, uref = traj.generate_trajectory()
traj.plot_trajectories(xref, uref)

save_to_mat("../project/xref.mat", traj.t, xref, "x_ref")
save_to_mat("../project/uref.mat", traj.t, uref, "u_ref")

Nsim = len(xref[1, :])

uav = MPC_Controller(Ts, Nsim, xref, uref)
xsim, usim = uav.simulate()


plot_states_with_references(t, xsim[:, :len(t)], usim[:, :len(t)], xref[:, :len(t)], uref[:, :len(t)])
