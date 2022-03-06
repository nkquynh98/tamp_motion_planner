import socket
import pickle
from time import time, sleep
from logic_planning.planner import LogicPlanner
from policy_learning.core.data_process import process_task_data
from queue_server.common_object.parser import PDDLParser
from queue_server.common_object.message import Message
from queue_server.core.queue_object import TaskQueueObject, MotionQueueObject

# Motion planning
from motion_planning.core.workspace import WorkspaceFromEnv
from motion_planning.core.action import *
from motion_planning.core.TAMP_motion_planner import TAMPMotionOptimizer
from motion_planning.core.GIL_motion_planner import GIL_motion_planner


# Working Environment
from toy_gym.envs.toy_tasks.toy_pickplace_tamp import ToyPickPlaceTAMP
from toy_gym.policy.TAMPActionExecutor import TAMPActionExecutor

class MotionPlanningNode:
    def __init__(self, node_name = "MotionNode_0", host = "127.0.0.1", port = 64563, viewer_enable = False, env_type =  ToyPickPlaceTAMP, planner_type = TAMPMotionOptimizer, action_executor = TAMPActionExecutor, max_steps = 10000):
        self.node_name = node_name
        self.host = host
        self.port = port
        self.message = Message(self.node_name)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.motion_object = MotionQueueObject(is_refined=True)
        self.env_type = env_type
        self.planner_type = planner_type
        self.action_executor = action_executor
        self.viewer_enable = viewer_enable
        self.max_steps = max_steps
    def get_motion_problem(self):
        
        is_problem_received = False
        while not is_problem_received:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print("Trying to get motion problem from Server")
                self.message.set_command("GetMotionProblem", [])
                send_value = pickle.dumps(self.message)
                self.socket.connect((self.host,self.port))
                self.socket.sendall(send_value)
                received_data = self.socket.recv(100000)
                self.socket.close()
                message = pickle.loads(received_data)
                if not isinstance(message, Message):
                    raise Exception("Invalid Message")
                if message.command == "Error":
                    raise Exception("Error: {}".format(message.data))
                motion_object = pickle.loads(message.data)
                assert isinstance(motion_object, MotionQueueObject)
                self.motion_object = motion_object
                is_problem_received = True

            except Exception as e: 
                print(e)
                sleep(5)

    def plan(self):
        assert self.motion_object.is_refined == False
        init_config = self.motion_object.geometric_state
        goal = self.motion_object.problem.get_dict()["positive_goal"]
        skeleton = self.motion_object.skeleton
        env = self.env_type(render=self.viewer_enable, json_data = init_config, goal=goal, enable_physic=False)
        workspace_objects = env.get_workspace_objects()
        workspace = WorkspaceFromEnv(workspace_objects)
        planner = self.planner_type(workspace, skeleton=skeleton, enable_global_planner=False, enable_viewer=self.viewer_enable, flexible_traj_ratio=4)
        planner.execute_plan()
        planner.visualize_final_trajectory()
        policy = self.action_executor(env)
        policy.set_action_list(skeleton)
        for _ in range(self.max_steps):
            action = policy.get_action()
            obs, reward, done, info = env.step(action)
            #print("logic_state", env.get_logic_state())
            if done:
                #dones.append(i)
                #success_episode +=1
                break
        env.reset()
        del planner
        del policy      
        del env  
        self.motion_object.is_refined = True
        return True

class GILMotionPlanningNode(MotionPlanningNode):
    def __init__(self, node_name = "MotionNode_0", host = "127.0.0.1", port = 64563, viewer_enable = False, max_steps = 100):
        super().__init__(node_name,host,port,viewer_enable,env_type = ToyPickPlaceTAMP, planner_type=GIL_motion_planner,max_steps=max_steps)
    def plan(self):
        time_start = time()
        assert self.motion_object.is_refined == False
        init_config = self.motion_object.geometric_state
        skeleton = self.motion_object.skeleton
        env = self.env_type(render=self.viewer_enable, json_data = init_config, problem=self.motion_object.problem, domain=self.motion_object.domain, enable_physic=False)
        workspace_objects = env.get_workspace_objects()
        workspace = WorkspaceFromEnv(workspace_objects)
        planner = self.planner_type(env, workspace, skeleton=skeleton, enable_global_planner=False, enable_viewer=self.viewer_enable, flexible_traj_ratio=6)
        task_data = planner.execute_plan()
        process_task_data(task_data)
        planner.visualize_final_trajectory()
        print("Total cost", planner.get_total_cost())
        print("planning time: {} s".format(time()-time_start))
        env.reset()
        del planner 
        del env  
        self.motion_object.is_refined = True
        return True