import numpy as np
from typing import List

from pyrieef.motion.trajectory import Trajectory
class Action(object):
    def __init__(self, name=None, parameters=None, order=None, duration=None, traj: Trajectory=None):
        self.name=name
        self.parameters=parameters
        self.order=order
        self.duration=duration
        self.trajectory = traj

    def set_trajectoy(self, traj: Trajectory):
        self.trajectory = traj


class MoveToPick(Action):
    def __init__(self, parameters, order=None, duration=None):
        self.object_to_grasp = parameters[0]
        super().__init__(name="MoveToPick", parameters=parameters, order=order, duration=duration)

class MoveToPlace(Action):
    def __init__(self, parameters, order=None, duration=None):
        self.holding_object=parameters[0]
        self.target=parameters[1]
        super().__init__(name="MoveToPlace",parameters=parameters, order=order, duration=duration)

class PlaceAndRetreat(Action):
    def __init__(self, parameters, order, duration):
        self.holding_object=parameters[0]
        self.target=parameters[1]
        super().__init__(name="PlaceAndRetreat",parameters=parameters, order=order, duration=duration)


if __name__ =="__main__":
    skeleton = []

    skeleton.append(MoveToPick(["object_0"],order=0, duration=10))
    skeleton.append(MoveToPlace(["object_0","target_0"], order=1, duration=10))
    print(skeleton[0].name)
    pass