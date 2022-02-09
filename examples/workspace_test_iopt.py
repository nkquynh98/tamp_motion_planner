import numpy as np
from tqdm import tqdm
from multiprocessing import Process

from pybewego.motion_optimization import NavigationOptimization
from pybewego.motion_optimization import CostFunctionParameters
from pybewego.workspace_viewer_server import WorkspaceViewerServer
from pybewego.motion_optimization import MotionOptimization
from pyrieef.optimization import algorithms
from pybewego.motion_optimization import CostFunctionParameters
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *
import pyrieef.learning.demonstrations as demonstrations
from pyrieef.graph.shortest_path import *
from trajectory import *
import time
TRAJ_LENGTH = 200
DRAW_MODE = "pyglet2d" 
NB_POINTS = 40
DEBUG = False

q_init=np.array([-1,-1])
q_final=np.array([1,1])
trajectory = linear_interpolation_trajectory(q_init=q_init,q_goal=q_final, T=TRAJ_LENGTH)
print("Linear Trajectory",trajectory.active_segment())

box_dim = np.array([10., 10.])
box = EnvBox(np.array([0., 0.]),dim=box_dim)
workspace = Workspace(box=box)
viewer = WorkspaceViewerServer(workspace, scale=70)
#workspace = sample_circle_workspaces(5)
rotate_mat = rotation_matrix_2d(90)


#workspace.obstacles.append(OrientedBox(np.array([0, -5]), np.array([0.1, 10]),rotate_mat))
#workspace.obstacles.append(OrientedBox(np.array([0, 5]), np.array([0.1, 10]),rotate_mat))
workspace.obstacles.append(Circle(np.array([0.4, .0]), 0.1))
workspace.obstacles.append(Circle(np.array([2, .0]), 1))
workspace.obstacles.append(Box(np.array([-5, 0]), np.array([1, 10])))
workspace.obstacles.append(Box(np.array([5, 0]), np.array([1, 10])))
workspace.obstacles.append(Box(np.array([0, -5]), np.array([10, 1])))
workspace.obstacles.append(Box(np.array([0, 5]), np.array([10, 1])))
#viewer = WorkspaceViewerServer(workspace)
grid = np.ones((NB_POINTS, NB_POINTS))
graph = CostmapToSparseGraph(grid, average_cost=False)
graph.convert()
#trajectory = demonstrations.graph_search_path(graph, workspace, NB_POINTS)

#viewer = WorkspaceOpenGl(workspace,wait_for_keyboard=True, scale=50)
#sdf = SignedDistanceWorkspaceMap(workspace)
#viewer = WorkspaceViewerServer(Workspace())

problem = NavigationOptimization(
    workspace,
    resample(trajectory, TRAJ_LENGTH),
    # trajectory,
    dt=0.3 / float(TRAJ_LENGTH),
    q_goal=trajectory.final_configuration()+ .05 * np.random.rand(2),
    bounds=workspace.box.extent_data(),
    goal_radius=0.01)
problem.verbose = False
print(workspace.box.extent_data())
p = CostFunctionParameters()
p.s_velocity_norm = 0
p.s_acceleration_norm = 10
p.s_obstacles = 1e+3
p.s_obstacle_alpha = 7
p.s_obstacle_gamma = 60
p.s_obstacle_margin = 0
p.s_obstacle_constraint = 1
p.s_terminal_potential = 1e+4
p.s_waypoint_constraint = 0
problem.initialize_objective(p)
#viewer.draw_configuration(q=np.array([-0.4,-0.4]))
# Initialize the viewer with objective function etc.
viewer.initialize_viewer(problem, problem.trajectory)
if DEBUG:
    viewer.run(draw_goal_manifold=True)
    while(1):
        pass
options = {}
options["tol"] = 9e-3#1e-2
options["acceptable_tol"] = 1e-2
options["acceptable_constr_viol_tol"] = 5e-1
options["constr_viol_tol"] = 5e-2
options["max_iter"] = 200
# options["bound_relax_factor"] = 0
options["obj_scaling_factor"] = 1e+2

p = Process(target=problem.optimize, args=(p, options))
p.start()
print("run viewer...")
t0 = time.time()
viewer.run(draw_goal_manifold=True)
p.join()
print("time : ", time.time() - t0)
viewer.viewer.gl.close()

# viewer.background_matrix_eval = True
# viewer.draw_ws_background(sdf, nb_points=200)
# viewer.draw_ws_obstacles()
# viewer.show_once()



def fetching_urdf(urdf_link):
    pass
