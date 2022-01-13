import numpy as np
from tqdm import tqdm
from multiprocessing import Process

from pybewego.numerical_optimization.ipopt_mo import NavigationOptimization
from pybewego.motion_optimization import CostFunctionParameters
from pybewego.workspace_viewer_server import WorkspaceViewerServer, WorkspaceViewerServerPlanar
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
from motion_planning.core.workspace import WorkspaceFromEnv
from toy_gym.envs.toy_tasks.toy_pickplace_fiveobject import ToyPickPlaceFiveObject
from motion_planning.core.objective import TrajectoryConstraintObjective
import time
TRAJ_LENGTH = 50
DRAW_MODE = "pyglet2d" 
NB_POINTS = 40
DEBUG = False

objective_TrajConstraint=TrajectoryConstraintObjective(T=TRAJ_LENGTH)



env = ToyPickPlaceFiveObject(render=True, map_name="maze_world_with_obs")
workspace_objects = env.get_workspace_objects()
print("Robot dim" ,workspace_objects.robot.robot_dim)
print("Obs dim", workspace_objects.movable_obstacles["object_0"].dimension)

workspace = WorkspaceFromEnv(workspace_objects)

#workspace = sample_circle_workspaces(5)
rotate_mat = rotation_matrix_2d(90)

q_init=workspace.workspace_objects.robot.position
q_final=workspace.workspace_objects.targets["target_0"].position
q_init = np.array([-2,3])
q_final = np.array([3,0])
trajectory = linear_interpolation_trajectory(q_init=q_init,q_goal=q_final, T=TRAJ_LENGTH)
viewer = WorkspaceViewerServerPlanar(workspace, trajectory, scale = 50)
print("Linear Trajectory",trajectory.active_segment())
#workspace.obstacles.append(OrientedBox(np.array([0, -5]), np.array([0.1, 10]),rotate_mat))
#workspace.obstacles.append(OrientedBox(np.array([0, 5]), np.array([0.1, 10]),rotate_mat))
# workspace.obstacles.append(Circle(np.array([0.4, .0]), 0.1))
# workspace.obstacles.append(Circle(np.array([2, .0]), 1))
# workspace.obstacles.append(Box(np.array([-5, 0]), np.array([1, 10])))
# workspace.obstacles.append(Box(np.array([5, 0]), np.array([1, 10])))
# workspace.obstacles.append(Box(np.array([0, -5]), np.array([10, 1])))
# workspace.obstacles.append(Box(np.array([0, 5]), np.array([10, 1])))
#viewer = WorkspaceViewerServer(workspace)
grid = np.ones((NB_POINTS, NB_POINTS))
graph = CostmapToSparseGraph(grid, average_cost=False)
graph.convert()
#trajectory = demonstrations.graph_search_path(graph, workspace, NB_POINTS)
#objective_TrajConstraint=TrajectoryConstraintObjective( enable_viewer=True)
#objective_TrajConstraint.set_problem(workspace=workspace, trajectory=trajectory, goal_manifold=Circle(origin=q_final,radius=0.1))
#viewer = WorkspaceOpenGl(workspace,wait_for_keyboard=True, scale=50)
#sdf = SignedDistanceWorkspaceMap(workspace)
#viewer = WorkspaceViewerServer(Workspace())

problem = NavigationOptimization(
    workspace,
    resample(trajectory, TRAJ_LENGTH),
    # trajectory,
    dt=5 / float(TRAJ_LENGTH),
    q_goal=trajectory.final_configuration()+ .05 * np.random.rand(2),
    bounds=workspace.box.extent_data())
problem.verbose = False
problem.publish_current_solution = True
problem.publish_current_solution_slow_down = False
print(workspace.box.extent_data())
p = CostFunctionParameters()
p.s_velocity_norm = 0
p.s_acceleration_norm = 10
p.s_obstacles = 1000
p.s_obstacle_alpha = 10
p.s_obstacle_gamma = 40
p.s_obstacle_margin = 0.3
p.s_obstacle_constraint = 1
p.s_terminal_potential = 1e+2
p.s_waypoint_constraint = 0
problem.initialize_objective(p)
#viewer.draw_configuration(q=np.array([-0.4,-0.4]))
viewer.viewer.draw_ws_circle(workspace.workspace_objects.robot.robot_dim[1],workspace.workspace_objects.robot.position)
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

#result, traj = objective_TrajConstraint.optimize() 
#print("traj", traj)
#print("resuĺt", result)
p = Process(target=problem.optimize, args=(p, options))
#p = Process(target=objective_TrajConstraint.optimize)
p.start()
print("run viewer...")
t0 = time.time()

viewer.run()
p.join()
print("time : ", time.time() - t0)

time.sleep(100)
viewer.viewer.gl.close()

# viewer.background_matrix_eval = True
# viewer.draw_ws_background(sdf, nb_points=200)
# viewer.draw_ws_obstacles()
# viewer.show_once()



def fetching_urdf(urdf_link):
    pass
