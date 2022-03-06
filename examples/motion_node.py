from motion_planning.core.motion_planning_node import MotionPlanningNode, GILMotionPlanningNode
from motion_planning.core.TAMP_motion_planner_freeflyer import TAMPMotionOptimizerFreeFlyer
from toy_gym.policy.TAMPActionExecutor import TAMPActionExecutorFreeFlyer
import argparse

from pyrieef import motion
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python motion_node.py ')

parser.add_argument('-n', '--node-name', help="The task node should have the name MotionNode_x, x is the number of the node", dest="node_name", type=str, default="MotionNode_0")
parser.add_argument("--enable-viewer", help="Enable the viewer", action="store_true")
parser.add_argument("--disable-freeflyer", help="Disable Freeflyer mode: Not consider yaw in motion plannig", action="store_true")
args = parser.parse_args()
print(args)

assert "MotionNode" in args.node_name
print(args.node_name)
while(1):
    motion_node = GILMotionPlanningNode(args.node_name, viewer_enable=args.enable_viewer)
    # if args.disable_freeflyer:
    #     motion_node = MotionPlanningNode(args.node_name, viewer_enable=args.enable_viewer)
    # else:
    #     motion_node = MotionPlanningNode(args.node_name, max_steps=100, viewer_enable=args.enable_viewer,planner_type=TAMPMotionOptimizerFreeFlyer,action_executor=TAMPActionExecutorFreeFlyer)
    motion_node.get_motion_problem()
    motion_node.plan()
