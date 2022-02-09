import numpy as np
#from pyglet.libs.win32.constants import ESB_ENABLE_BOTH
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
from motion_planning.core.action import *
from motion_planning.core.motion_optimization import MotionOptimizer
from toy_gym.policy.ActionExecutor import ActionExecutor
import matplotlib.pyplot as plt
TRAJ_LENGTH = 50
DRAW_MODE = "pyglet2d" 
NB_POINTS = 40
DEBUG = False
NUM_EPS = 10
MAX_STEPS = 10000
VIEWER_ENABLE = True
action_list = []
for i in range(5):
    pick = MoveToPick(parameters=["object_{}".format(i)],duration=10)
    action_list.append(pick)
    place = MoveToPlace(parameters=["object_{}".format(i), "target_{}".format(i)], duration=10)
    action_list.append(place)

#action_list = [MoveToPick(parameters=["object_0"],duration=50), MoveToPlace(parameters=["object_0", "target_0"], duration=20)]
objective_TrajConstraint=TrajectoryConstraintObjective(T=TRAJ_LENGTH)
#env = ToyPickPlaceFiveObject(render=VIEWER_ENABLE, map_name="maze_world_narrow", is_object_random=True, is_target_random=True)
env = ToyPickPlaceFiveObject(render=VIEWER_ENABLE, map_name="maze_world", is_object_random=True, is_target_random=False)
#env = ToyPickPlaceFiveObject(render=VIEWER_ENABLE, map_name="maze_world_with_obs", is_object_random=True, is_target_random=False)
#workspace_objects = env.get_workspace_objects()
# print("Robot dim" ,workspace_objects.robot.robot_dim)
# print("Obs dim", workspace_objects.movable_obstacles["object_0"].dimension)

# workspace = WorkspaceFromEnv(workspace_objects)
# planner = MotionOptimizer(workspace, action_list, enable_viewer=True)
# planner.execute_plan()
# planner.visualize_final_trajectory()
# print(action_list[0].trajectory.active_segment())
# print(action_list[1].trajectory.active_segment())


# policy = ActionExecutor(env)
# policy.set_action_list(action_list)
costs = []
dones = []
success_episode = 0
for i in range(NUM_EPS):
    workspace_objects = env.get_workspace_objects()
    workspace = WorkspaceFromEnv(workspace_objects)
    planner = MotionOptimizer(workspace, action_list, enable_global_planner=False, enable_viewer=VIEWER_ENABLE, flexible_traj_ratio=4)
    planner.execute_plan()
    planner.visualize_final_trajectory()
    print("Total cost", planner.get_total_cost())
    costs.append(planner.get_total_cost())
    policy=ActionExecutor(env)
    policy.set_action_list(action_list)
    for _ in range(MAX_STEPS):
        action = policy.get_action()
        obs, reward, done, info = env.step(action)
        if done:
            dones.append(i)
            success_episode +=1
            break
    env.reset()
    del planner
    del policy


print("Success rate {}/{}".format(success_episode, NUM_EPS))
print("Episode dones ", dones)
print(costs)
plt.plot(costs)
plt.show()

