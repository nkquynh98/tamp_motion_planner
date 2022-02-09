#from motion_planning.core.action import Action
#Common Object
from tkinter import CURRENT
from queue_server.common_object.action import Action, DurativeAction

#Motion planning
from motion_planning.core.workspace import WorkspaceFromEnv
from motion_planning.core.objective import TrajectoryConstraintObjective
from motion_planning.core.optimizer import NavigationOptimizationMultiprocessing, FreeflyerOptimizationMultiprocessing

#Pybewego
from pybewego.numerical_optimization.ipopt_mo import NavigationOptimization
from pybewego.motion_optimization import CostFunctionParameters
from pybewego.motion_optimization import MotionOptimization
from pybewego.workspace_viewer_server import WorkspaceViewerServerPlanar, WorkspaceViewerServerFreeflyer
from pybewego.numerical_optimization.ipopt_mo import FreeflyerOptimization
from pybewego.workspace_viewer_server import WorkspaceViewerServerFreeflyer
from pybewego.kinematics import create_bewego_freeflyer_from_file
from pybewego.collision_checking import *
from pybewego.assets import *

#Pyrieef
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.motion.trajectory import linear_interpolation_trajectory
from pyrieef.kinematics.robot import create_freeflyer_from_segments
from pyrieef.geometry.rotations import rotation_matrix_2d_radian
from pyrieef.motion.cost_terms import *
from pyrieef.graph.shortest_path import *

#Ultis
from multiprocessing.sharedctypes import Value, Array
import time
from multiprocessing import Process
from ctypes import c_bool, c_double
import numpy as np
from typing import List
import os

FREEFLYER_PATH = os.path.dirname(os.path.realpath(__file__))+"/../../configuration/Toy_freeflyer.json"

def calculate_distance(current_pos, target_pos):
    return np.linalg.norm(target_pos-current_pos)
def calculate_angle(vector):
    angle = np.arctan2(vector[1], vector[0])
    return angle

def get_grasp_pose(target, gripper_local_position, grasp_angle):
    position = target - rotation_matrix_2d_radian(grasp_angle)@gripper_local_position
    return np.concatenate([position, [grasp_angle]])

class TAMPMotionOptimizerFreeFlyer:
    def __init__(self, workspace: WorkspaceFromEnv, skeleton: List[Action], 
                initial_traj_gen = linear_interpolation_trajectory, optimizer:MotionOptimization = FreeflyerOptimizationMultiprocessing, 
                enable_viewer=False, flexible_traj_ratio=None, enable_global_planner=False):
        self.workspace = workspace
        self.skeleton = skeleton.copy()
        self.current_action = skeleton[0]
        self.initial_traj_gen = initial_traj_gen
        self.current_robot_position = np.concatenate([self.workspace.robot.position, [self.workspace.robot.yaw]])
        print(self.current_robot_position)
        self.enable_viewer = enable_viewer
        self.optimizer = optimizer
        self.optimizer_options = {}
        self.optimizer_parameters = CostFunctionParameters()
        self.set_optimizer_options()
        self.set_optimizer_parameters()
        self.robot_freeflyer = create_bewego_freeflyer_from_file(filename=FREEFLYER_PATH)
        self.total_cost = 0
        #Ratio between the real distance (float) with the Trajectory length (int): 1 unit in real distance = "flex_traj_ratio" Trajectory length
        self.flexible_traj_ratio = flexible_traj_ratio 
        print("Init robot", self.current_robot_position)
        self.final_trajectory = Trajectory(q_init=self.current_robot_position, x=np.array([self.current_robot_position]))
        if self.enable_viewer:
            self.viewer = WorkspaceViewerServerFreeflyer(workspace, self.final_trajectory, scale = 50)
        
        self.enable_global_planner = enable_global_planner
        if self.enable_global_planner:
            pass
    def set_optimizer_options(self, **kwargs):
        # self.optimizer_options["tol"] = 1e-2
        # self.optimizer_options["acceptable_tol"] = 5e-3
        # self.optimizer_options["acceptable_constr_viol_tol"] = 5e-1
        # self.optimizer_options["constr_viol_tol"] = 5e-2
        # self.optimizer_options["max_iter"] = 200
        # self.optimizer_options["bound_relax_factor"] = 0
        # self.optimizer_options["obj_scaling_factor"] = 1e+2

        self.optimizer_options["max_iter"] = 200
        # options["mu_init"] = 1e-6
        self.optimizer_options["tol"] = 1e-2
        self.optimizer_options["acceptable_tol"] = 5e-3
        self.optimizer_options["acceptable_constr_viol_tol"] = 5e-1
        self.optimizer_options["constr_viol_tol"] = 5e-2
        self.optimizer_options["bound_relax_factor"] = 0
        self.optimizer_options["obj_scaling_factor"] = 1e+2

    def set_optimizer_parameters(self, **kwargs):
        # self.optimizer_parameters.s_velocity_norm = 0
        # self.optimizer_parameters.s_acceleration_norm = 10
        # self.optimizer_parameters.s_obstacles = 10
        # self.optimizer_parameters.s_obstacle_alpha = 7
        # self.optimizer_parameters.s_obstacle_gamma = 40
        # self.optimizer_parameters.s_obstacle_margin = 0.5
        # self.optimizer_parameters.s_obstacle_constraint = 1
        # self.optimizer_parameters.s_terminal_potential = 1e+3
        # self.optimizer_parameters.s_waypoint_constraint = 0

        # self.optimizer_parameters.s_velocity_norm = 1
        # self.optimizer_parameters.s_acceleration_norm = 5
        # self.optimizer_parameters.s_obstacles = 200
        # self.optimizer_parameters.s_obstacle_alpha = 10
        # self.optimizer_parameters.s_obstacle_scaling = 1
        # self.optimizer_parameters.s_terminal_potential = 1e+6        

        self.optimizer_parameters.s_velocity_norm = 10
        self.optimizer_parameters.s_acceleration_norm = 10
        self.optimizer_parameters.s_obstacles = 50
        self.optimizer_parameters.s_obstacle_constraint = 1
        self.optimizer_parameters.s_terminal_potential = 1e+2
        self.optimizer_parameters.s_terminal_endeffector_potential = 0

    def update_final_trajectory(self, new_trajectory: Trajectory):
        T = self.final_trajectory.T() + new_trajectory.T()
        #print(T)
        #print("init", len(self.final_trajectory.x()))
        #print("new", len(new_trajectory.x()))
        x = np.concatenate([self.final_trajectory.x(), new_trajectory.x()])
        #print("X", x)
        self.final_trajectory = Trajectory(q_init = x[0:3], x=x[3:])
        #print("Final Traj", self.final_trajectory.x())
    def execute_plan(self):
        for i in range(len(self.skeleton)):
            self.current_action = self.skeleton[i]
            if self.current_action.name == "movetopick":
                object_to_grasp = self.current_action.parameters[0]
                q_init = self.current_robot_position
                target_position = self.workspace.workspace_objects.movable_obstacles[object_to_grasp].position
                self.workspace.update_ws_obstacles(object_to_grasp)
                q_goal = get_grasp_pose(target_position, self.workspace.workspace_objects.robot._gripper_pose, calculate_angle(target_position-q_init[0:2]))
                if self.flexible_traj_ratio is not None:
                    self.current_action.duration=int(np.max([int(calculate_distance(q_init[0:2], q_goal[0:2])*self.flexible_traj_ratio), 5]))

                    print("New traj length", self.current_action.duration)
                #init_traj = self.initial_traj_gen(q_init=q_init,q_goal=q_goal, T=self.current_action.duration)
                result, found_traj = self.find_trajectory(self.workspace, q_init,q_goal, self.current_action.duration)
                if result:
                    self.current_action.set_trajectoy(found_traj)
                    self.current_robot_position = found_traj.final_configuration()
                    print("Q_goal", q_goal)
                    print("Q_real", self.current_robot_position)
                    self.update_final_trajectory(found_traj)
                else:
                    print("solution not found", self.current_action.name, self.current_action.parameters)
            elif self.current_action.name == "movetoplace":
                holding_object = self.current_action.parameters[0]
                target = self.current_action.parameters[1]
                q_init = self.current_robot_position
                target_position = self.workspace.workspace_objects.targets[target].position
                self.workspace.update_ws_obstacles(holding_object)
                q_goal = get_grasp_pose(target_position, self.workspace.workspace_objects.robot._gripper_pose, np.pi/2) #calculate_angle(target_position-q_init[0:2]))
                if self.flexible_traj_ratio is not None:
                    self.current_action.duration=int(np.max([int(calculate_distance(q_init[0:2], q_goal[0:2])*self.flexible_traj_ratio), 5]))
                    print("New traj length", self.current_action.duration)
                #init_traj = self.initial_traj_gen(q_init=q_init,q_goal=q_goal, T=self.current_action.duration)
                result, found_traj = self.find_trajectory(self.workspace, q_init, q_goal, self.current_action.duration)
                print("Final pose to target ", found_traj.final_configuration())
                if result:
                    #Update the position of the grasped objet and robot
                    self.current_action.set_trajectoy(found_traj)
                    self.workspace.workspace_objects.movable_obstacles[holding_object].position = target_position
                    self.current_robot_position = found_traj.final_configuration()
                    print("Q_goal", q_goal)
                    print("Q_real", self.current_robot_position)
                    self.update_final_trajectory(found_traj)
                else:
                    print("solution not found", self.current_action.name, self.current_action.parameters)
                pass

    def visualize_final_trajectory(self):
        if self.enable_viewer:
            print("Final Trajectory", self.final_trajectory)
            self.workspace.update_ws_obstacles()
            problem = self.optimizer(
                self.robot_freeflyer,
                self.workspace,
                self.final_trajectory,
                # trajectory,
                dt=5 / float(self.final_trajectory.T()),
                x_goal=self.robot_freeflyer.keypoint_map(0)(self.final_trajectory.final_configuration()),
                q_goal=self.final_trajectory.final_configuration(),# + .05 * np.random.rand(2),
                bounds=self.workspace.box.extent_data())
            problem.verbose = False
            problem.publish_current_solution = self.enable_viewer
            problem.publish_current_solution_slow_down = False
            problem.robot = create_freeflyer_from_segments(FREEFLYER_PATH)
            problem.with_goal_constraint = True

            problem.attractor_type = "euclidean"
            problem.initialize_objective(self.optimizer_parameters)
            self.viewer.initialize_viewer(problem, problem.trajectory)
        else:
            print("Viewer is not Enabled")
    def find_trajectory(self, workspace, q_init, q_goal, traj_length):
        trajectory = None
        if self.enable_global_planner:
            trajectory = self.trajectory_from_global_planner(workspace, q_init, q_goal, traj_length=traj_length)
        if trajectory is None:
            trajectory = self.initial_traj_gen(q_init=q_init,q_goal=q_goal, T=traj_length)
        problem = self.optimizer(
            self.robot_freeflyer,
            workspace,
            trajectory,
            # trajectory,
            dt=5 / float(trajectory.T()),
            x_goal=self.robot_freeflyer.keypoint_map(0)(trajectory.final_configuration()),
            q_goal=trajectory.final_configuration(),# + .05 * np.random.rand(2),
            bounds=workspace.box.extent_data())
        problem.verbose = False
        problem.publish_current_solution = self.enable_viewer
        problem.publish_current_solution_slow_down = False
        problem.robot = create_freeflyer_from_segments(FREEFLYER_PATH)
        problem.with_goal_constraint = True

        problem.attractor_type = "euclidean"
        problem.initialize_objective(self.optimizer_parameters)
        status = Value(c_bool, False)
        traj_array = Array(c_double, trajectory.n() * (trajectory.T() + 2))
        init_traj = trajectory.x().copy()
        if self.enable_viewer:
            self.viewer.initialize_viewer(problem, problem.trajectory)
            p = Process(target=problem.optimize, args=(self.optimizer_parameters, self.optimizer_options, status, traj_array))
            p.start()
            print("run viewer...")
            t0 = time.time()
            self.viewer.run()
            p.join()
            print("time : ", time.time() - t0)
            result = status.value
            trajectory = Trajectory(q_init=np.array(traj_array[:3]), x=np.array(traj_array[3:]))
            
        else:
            result, trajectory = problem.optimize(self.optimizer_parameters, self.optimizer_options)
        cost = problem.objective.forward(trajectory.x())[0]
        print("Cost", cost)    
        print("status ",result)
        print("trajectory ", trajectory)
        final_traj = trajectory.x().copy()
        print("Diff traj", init_traj-final_traj)
        if result:
            self.total_cost+=cost
        return [result, trajectory]

    def __del__(self):
        if self.enable_viewer:
            self.viewer.viewer.gl.close()
    
    def get_total_cost(self):
        return self.total_cost

    def trajectory_from_global_planner(self, workspace: Workspace, q_init, q_goal, grid_points=60, average_cost=False, traj_length=None):
        phi = CostGridPotential2D(SignedDistanceWorkspaceMap(workspace), 1., 100., 100.)
        costmap = phi(workspace.box.stacked_meshgrid(grid_points))
        converter = CostmapToSparseGraph(costmap, average_cost)
        converter.integral_cost = True
        graph = converter.convert()
        if average_cost:
            assert check_symmetric(graph)
        pixel_map = workspace.pixel_map(grid_points)
        q_init_grid = pixel_map.world_to_grid(q_init)

        q_goal_grid = pixel_map.world_to_grid(q_goal)

        try:
            time_0 = time.time()
            print("planning (1)...")
            global_path = converter.dijkstra_on_map(costmap, q_goal_grid[0], q_goal_grid[1], q_init_grid[0], q_init_grid[1])
        except:
            print("Failed to generate the global planner")
            return None
        print("Global planner took t : {} sec.".format(time.time() - time_0))    
        traj_configurations = []
        for i, p in enumerate(global_path):
            print(p)
            traj_configurations.append(pixel_map.grid_to_world(np.array(p)))    
        traj_configurations.append(q_goal)
        traj_configurations = np.array(traj_configurations).reshape((-1,))    
        trajectory = Trajectory(q_init=q_init, x=traj_configurations)
        trajectory = resample(trajectory, traj_length)
        return trajectory