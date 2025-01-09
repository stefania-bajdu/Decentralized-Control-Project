from model import *
from ploting import *
from flatness import *
from mpc import *

Ts = 0.1
SimTime = 40 - Ts 
t = np.arange(0, SimTime + Ts, Ts)

traj = GenerateTrajectory(Ts, SimTime)
xref, uref = traj.generate_trajectory()
# traj.plot_trajectories(xref, uref)

Nsim = len(xref[1, :])

uav = MPC_Controller(Ts, Nsim, xref, uref)
xsim, usim = uav.simulate()

plot_states_with_references(t, xsim, usim, xref, uref)

# save_to_mat("states.mat", traj.t, xref)