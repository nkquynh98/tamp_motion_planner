from pyrieef.motion.trajectory import *
import numpy as np
q_init = np.array([0, 0])
x = np.array([1,1,2,2,3,3])
traj = Trajectory(q_init=q_init, x=x)
print(traj)
print(traj.configuration(3))