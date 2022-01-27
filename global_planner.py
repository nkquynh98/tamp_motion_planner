# pybewego
from pybewego import AStarGrid
from pybewego import ValueIteration

# pybewego
from pyrieef.graph.shortest_path import *
from pyrieef.geometry.workspace import *
from pyrieef.motion.cost_terms import *
from pyrieef.motion.trajectory import Trajectory, no_motion_trajectory, linear_interpolation_trajectory
import pyrieef.rendering.workspace_planar as render
from toy_gym.envs.toy_tasks.toy_pickplace_fiveobject import ToyPickPlaceFiveObject
from motion_planning.core.workspace import WorkspaceFromEnv
# external
import numpy as np
from numpy.testing import assert_allclose
import os
import sys
import time
show_result = True
radius = .1
nb_points = 50
average_cost = False
VIEWER_ENABLE = True
env = ToyPickPlaceFiveObject(render=VIEWER_ENABLE, map_name="maze_world_with_obs", is_object_random=True, is_target_random=True)
workspace_objects = env.get_workspace_objects()
workspace = WorkspaceFromEnv(workspace_objects)

def trajectory_from_global_planner(workspace, q_init, q_goal, grid_points=40):
    phi = CostGridPotential2D(SignedDistanceWorkspaceMap(workspace), 10., .1, 10.)
    costmap = phi(workspace.box.stacked_meshgrid(nb_points))
    converter = CostmapToSparseGraph(costmap, average_cost)
    converter.integral_cost = True
    graph = converter.convert()
    if average_cost:
        assert check_symmetric(graph)
    pixel_map = workspace.pixel_map(nb_points)
    q_init_grid = pixel_map.world_to_grid(q_init)
    q_goal_grid = pixel_map.world_to_grid(q_goal)
    try:
        time_0 = time.time()
        print("planning (1)...")
        global_path = converter.dijkstra_on_map(costmap, q_init_grid[0], q_init_grid[1], q_goal_grid[0], q_goal_grid[1])
    except:
        print("Failed to generate the global planner")
        return None
    print("Global planner took t : {} sec.".format(time.time() - time_0))    
    traj_configurations = []
    for i, p in enumerate(global_path):
        traj_configurations.append(pixel_map.grid_to_world(np.array(p)))    
    traj_configurations.append(q_goal)
    trajectory = Trajectory(q_init=q_init, x=traj_configurations)
    return trajectory
# predecessors = shortest_paths(graph)
pixel_map = workspace.pixel_map(nb_points)
def interpolate_global_planner(pixel_map: PixelMap, global_path=None, trajectory_length=10):
    init_traj = []
    for i, p in enumerate(global_path):
        init_traj.append(pixel_map.grid_to_world(np.array(p)))
    print(init_traj)
    traj = no_motion_trajectory(q_init=[0,0], T=len(init_traj)-1)
    print(len(init_traj))
    print(traj.T())
    traj.set_from_configurations(init_traj)
    print(traj)
    pass
def trajectory(pixel_map, path):
    trajectory = [None] * len(path)
    for i, p in enumerate(path):
        trajectory[i] = pixel_map.grid_to_world(np.array(p))
    return trajectory

#workspace = Workspace()
#workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
#workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))
phi = CostGridPotential2D(SignedDistanceWorkspaceMap(workspace), 10., .1, 10.)
costmap = phi(workspace.box.stacked_meshgrid(nb_points))

converter = CostmapToSparseGraph(costmap, average_cost)
converter.integral_cost = True
graph = converter.convert()
if average_cost:
    assert check_symmetric(graph)
# predecessors = shortest_paths(graph)
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)
for i in range(100):
    s_w = env.get_workspace_objects().robot.position
    print("S_w", s_w)
    t_w = env.get_workspace_objects().movable_obstacles["object_{}".format(i%5)].position
    print("T_w", t_w)
    s = pixel_map.world_to_grid(s_w)
    t = pixel_map.world_to_grid(t_w)

    if s[0] == 0 or s[1] == 0:
        continue
    if t[0] == 0 or t[1] == 0:
        continue

    try:
        time_0 = time.time()
        print("planning (1)...")
        path1 = converter.dijkstra_on_map(costmap, s[0], s[1], t[0], t[1])

    except:
        continue
    print("1) took t : {} sec.".format(time.time() - time_0))
    # try:
    #traj = no_motion_trajectory(q_init=np.array([0,0]),T=10)
    #traj.set_from_configurations(path1)
    #print(traj)
    interpolate_global_planner(pixel_map=pixel_map, global_path=path1)
    time_0 = time.time()
    print("planning (2)...")
    print(costmap.shape)
    astar = AStarGrid()
    astar.init_grid(1. / nb_points, [0, 1, 0, 1])
    astar.set_costs(costmap)
    assert astar.solve(s, t)
    path2 = astar.path()
    print("Path 2", path2)
    print("2) took t : {} sec.".format(time.time() - time_0))

    # time_0 = time.time()
    # print("planning (3)...")
    # print(costmap.shape)
    # dstar = Dstar2D()
    # assert dstar.solve(s, t, costmap)
    # path3 = dstar.path().T
    # print("3) took t : {} sec.".format(time.time() - time_0))

    # time_0 = time.time()
    # print("planning (4)...")
    # print(costmap.shape)
    # print("s : ", s)
    # print("t : ", t)
    # viter = ValueIteration()
    # path4 = viter.solve(s, t, costmap)
    # print("4) took t : {} sec.".format(time.time() - time_0))

    if show_result:

        viewer = render.WorkspaceDrawer(
            rows=1, cols=3, workspace=workspace, wait_for_keyboard=True)

        viewer.set_drawing_axis(0)
        # viewer.draw_ws_background(phi, nb_points, interpolate="none")
        viewer.draw_ws_img(costmap.T, interpolate="none")
        #viewer.draw_ws_obstacles()
        viewer.draw_ws_line(trajectory(pixel_map, path1))
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)

        viewer.set_drawing_axis(1)
        viewer.draw_ws_background(phi, nb_points, interpolate="none")
        #viewer.draw_ws_obstacles()
        viewer.draw_ws_line(trajectory(pixel_map, path2), "b")
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)

        # viewer.set_drawing_axis(2)
        # viewer.draw_ws_background(phi, nb_points, interpolate="none")
        # viewer.draw_ws_obstacles()
        # viewer.draw_ws_line(trajectory(pixel_map, path3), "g")
        # viewer.draw_ws_point(s_w)
        # viewer.draw_ws_point(t_w)

        # viewer.set_drawing_axis(3)
        # viewer.draw_ws_background(phi, nb_points, interpolate="none")
        # viewer.draw_ws_obstacles()
        # viewer.draw_ws_line(trajectory(pixel_map, path4), "r")
        # viewer.draw_ws_point(s_w)
        # viewer.draw_ws_point(t_w)

        viewer.show_once()