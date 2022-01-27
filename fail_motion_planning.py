from pybewego.numerical_optimization.ipopt_mo import NavigationOptimization
from pybewego.motion_optimization import CostFunctionParameters
from pybewego.workspace_viewer_server import WorkspaceViewerServer, WorkspaceViewerServerPlanar
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import time

VERBOSE = False
BOXES = True
DRAW_MODE = "pyglet2d"  # None, pyglet2d, pyglet3d or matplotlib
NB_POINTS = 40          # points for the grid on which to perform graph search.
NB_PROBLEMS = 100       # problems to evaluate
TRAJ_LENGTH = 40
VIEWER_ENABLE = False
box = EnvBox(dim=np.array([10, 10]))
workspace = Workspace(box)

workspace.obstacles.append(Box(dim=np.array([1,6])))
#workspace.obstacles.append(Circle(radius=0.1))
q_init = np.array([-2,3])
q_final = np.array([3,0])
trajectory = linear_interpolation_trajectory(q_init=q_init,q_goal=q_final, T=TRAJ_LENGTH)
viewer = WorkspaceViewerServerPlanar(workspace, trajectory, scale = 50)


problem = NavigationOptimization(
    workspace,
    resample(trajectory, TRAJ_LENGTH),
    # trajectory,
    dt=0.3 / float(TRAJ_LENGTH),
    q_goal=trajectory.final_configuration() + .05 * np.random.rand(2),
    bounds=workspace.box.extent_data())
problem.verbose = False
problem.publish_current_solution = VIEWER_ENABLE
problem.publish_current_solution_slow_down = False

p = CostFunctionParameters()
p.s_velocity_norm = 0
p.s_acceleration_norm = 10
p.s_obstacles = 10
p.s_obstacle_alpha = 7
p.s_obstacle_gamma = 40
p.s_obstacle_margin = 0.0
p.s_obstacle_constraint = 1
p.s_terminal_potential = 1e+2
p.s_waypoint_constraint = 0
problem.initialize_objective(p)

# Initialize the viewer with objective function etc.
viewer.initialize_viewer(problem, problem.trajectory)

options = {}
options["tol"] = 1e-2
options["acceptable_tol"] = 5e-3
options["acceptable_constr_viol_tol"] = 5e-1
options["constr_viol_tol"] = 5e-2
options["max_iter"] = 200
options["bound_relax_factor"] = 0
options["obj_scaling_factor"] = 1e+2

if not VIEWER_ENABLE:
    t0 = time.time()
    result, traj = problem.optimize(p, options)
    print("time : ", time.time() - t0)
    print("Result is ", result)
    print("Trajectory ",traj)
    viewer.initialize_viewer(problem, traj)
else:

    p = Process(target=problem.optimize, args=(p, options))
    p.start()
    print("run viewer...")
    t0 = time.time()
    viewer.run()
    p.join()
    print("time : ", time.time() - t0)
    #break
try:
    while(1):
        pass
except:
    viewer.viewer.gl.close()