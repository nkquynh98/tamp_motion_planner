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
from pyrieef.rendering.workspace_planar import WorkspaceDrawer, WorkspaceOpenGl
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *
import pyrieef.learning.demonstrations as demonstrations
from pyrieef.graph.shortest_path import *
from trajectory import *
import time
TRAJ_LENGTH = 20
DRAW_MODE = "pyglet2d" 

q_init=np.array([-4,-4])
q_final=np.array([4,4])
trajectory = linear_interpolation_trajectory(q_init=q_init,q_goal=q_final, T=TRAJ_LENGTH)
print("Traj",trajectory.active_segment())
print("")
box_dim = np.array([10, 10])
box = EnvBox(dim=box_dim)
workspace = Workspace(box=box)
#workspace = sample_circle_workspaces(5)
rotate_mat = rotation_matrix_2d(90)
workspace.obstacles.append(Box(np.array([-5, 0]), np.array([1, 10])))
workspace.obstacles.append(Box(np.array([5, 0]), np.array([1, 10])))
workspace.obstacles.append(Box(np.array([0, -5]), np.array([10, 1])))
workspace.obstacles.append(Box(np.array([0, 5]), np.array([10, 1])))
#workspace.obstacles.append(OrientedBox(np.array([0, -5]), np.array([0.1, 10]),rotate_mat))
#workspace.obstacles.append(OrientedBox(np.array([0, 5]), np.array([0.1, 10]),rotate_mat))
workspace.obstacles.append(Circle(np.array([0.2, .0]), 1))
workspace.obstacles.append(Circle(np.array([2, .0]), 1))
workspace.obstacles.append(Circle(np.array([2, 2]), 1))
#viewer = WorkspaceViewerServer(workspace)

print(rotate_mat)
#viewer = WorkspaceOpenGl(workspace,wait_for_keyboard=True, scale=50)
#sdf = SignedDistanceWorkspaceMap(workspace)
#viewer = WorkspaceViewerServer(Workspace())

problem = MotionOptimization(
    workspace,
    trajectory,
    dt=0.01,
    q_goal=trajectory.final_configuration())

p = CostFunctionParameters()
p.s_velocity_norm = 1
p.s_acceleration_norm = 5
p.s_obstacles = 100
p.s_obstacle_alpha = 10
p.s_obstacle_scaling = 1
p.s_terminal_potential = 1e+6
problem.initialize_objective(p)

objective = TrajectoryOptimizationViewer(
    problem,
    draw=DRAW_MODE is not None,
    draw_gradient=True,
    use_3d=DRAW_MODE == "pyglet3d",
    use_gl=DRAW_MODE == "pyglet2d",
    scale=50.)
if DRAW_MODE is not None:
    print("Set to false")
    objective.viewer.background_matrix_eval = False
    objective.viewer.save_images = True
    objective.viewer.workspace_id += 1
    objective.viewer.image_id = 0
    objective.reset_objective()
    #objective.viewer.draw_ws_obstacles()

algorithms.newton_optimize_trajectory(
    objective, trajectory, verbose=True, maxiter=100)

if DRAW_MODE is not None:
    objective.viewer.gl.close()


# viewer.background_matrix_eval = True
# viewer.draw_ws_background(sdf, nb_points=200)
# viewer.draw_ws_obstacles()
# viewer.show_once()



def fetching_urdf(urdf_link):
    pass
