#import demos_common_imports

# Pybewego
from pybewego.numerical_optimization.ipopt_mo import NavigationOptimization
from pybewego.motion_optimization import CostFunctionParameters
from pybewego.workspace_viewer_server import  WorkspaceViewerServerPlanar

# Pyrieef
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *

# External
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, SimpleQueue
import time

VERBOSE = False
BOXES = True
DRAW_MODE = "pyglet2d"  # None, pyglet2d, pyglet3d or matplotlib
NB_PROBLEMS = 100       # problems to evaluate
TRAJ_LENGTH = 100       # Here you can set it 100
VIEWER_ENABLE = True
box = EnvBox(dim=np.array([10, 10]))
workspace = Workspace(box)

workspace.obstacles.append(Box(dim=np.array([1, 6])))
# workspace.obstacles.append(Circle(radius=0.1))
q_init = np.array([-2, 0])
q_final = np.array([3, 0])
# trajectory = no_motion_trajectory( 
#      q_init=q_init, T=TRAJ_LENGTH)
trajectory = linear_interpolation_trajectory(
    q_init=q_init, q_goal=q_final, T=TRAJ_LENGTH)
viewer = WorkspaceViewerServerPlanar(workspace, trajectory, scale=50)
viewer.points_radii = .07

problem = NavigationOptimization(
    workspace,
    trajectory,
    dt=0.3 / float(TRAJ_LENGTH),
    q_goal=q_final + .05 * np.random.rand(2),
    bounds=workspace.box.extent_data())
problem.verbose = False
problem.publish_current_solution = VIEWER_ENABLE
problem.publish_current_solution_slow_down = True
problem.with_goal_constraint = True
problem.attractor_type = "euclidean"

p = CostFunctionParameters()
p.s_velocity_norm = 0
p.s_acceleration_norm = 10
p.s_obstacles = 50 # increase weight
p.s_obstacle_alpha = 20 # increase alpha
p.s_obstacle_gamma = 40
p.s_obstacle_margin = 0.0
p.s_obstacle_constraint = 1
p.s_terminal_potential = 1
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

optimal_trajectory = None

if not VIEWER_ENABLE:
    t0 = time.time()
    result, optimal_trajectory = problem.optimize(p, options)
    print("time : ", time.time() - t0)
    print("Result is ", result)
    print("Trajectory ", optimal_trajectory)
    viewer.initialize_viewer(problem, optimal_trajectory)
else:
    queue = SimpleQueue()
    p = Process(target=problem.optimize, args=(p, options, queue))
    p.start()
    print("run viewer...")
    t0 = time.time()
    viewer.run()
    p.join()
    print("time : ", time.time() - t0)
    optimal_trajectory = queue.get()[1]
    configurations = optimal_trajectory.list_configurations()
    continuous_traj = optimal_trajectory.continuous_trajectory()
    
    if collision_check_trajectory(workspace, optimal_trajectory):
        print("trajectory in collision !!! ")
    else:
        print("trajectory collision free !!! ")

    # while True:
    #     for t in np.linspace(0, 1, num=200):
    #         print("drawing : ", t)
    #         q_t = continuous_traj.configuration_at_parameter(t)
    #         viewer.viewer.draw_ws_circle(
    #             viewer.points_radii * 2., q_t, color=(1, 1, 1))
    #         viewer.update_viewer()
try:
    while(1):
        pass
except:
    viewer.viewer.gl.close()
