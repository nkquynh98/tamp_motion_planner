import numpy as np


from pybewego.workspace_viewer_server import WorkspaceViewerServer
from toy_gym.envs.core.workspace_objects import Workspace_objects

from pyrieef.geometry.workspace import *
from pyrieef.rendering.workspace_planar import *

class WorkspaceFromEnv(Workspace):
    def __init__(self, workspace_objects: Workspace_objects):
        self._workspace_objects = workspace_objects
        self.robot = self.workspace_objects.robot
        print(self.workspace_objects.map.map_dim)
        self.env_box = EnvBox(dim=self.workspace_objects.map.map_dim)
        super().__init__(self.env_box)
        self.update_ws_obstacles()
    def add_object(self, object_shape, object_dim, object_position):
        if object_shape=="Box":
            self.obstacles.append(Box(origin=object_position, dim = object_dim + np.array([0.5, 0.5])))
        elif object_shape == "Cylinder":
            self.obstacles.append(Circle(origin=object_position, radius=object_dim[1]+0.2))
        else:
            print("Object shape {} is not supported".format(object_shape))
    def update_ws_obstacles(self, target_object: str = None):
        self.obstacles = []
        #Do not consider the target object as an obstacle
        for obs in self.workspace_objects.fixed_obstacles.values():
            self.add_object(object_shape= obs.shape, object_position=obs.position, object_dim=obs.dimension)
        
        for obs in self.workspace_objects.movable_obstacles.values():
            if target_object is not None and obs.name == target_object:
                    continue
            else:
                self.add_object(object_shape= obs.shape, object_position=obs.position, object_dim=obs.dimension)

    
    @property
    def workspace_objects(self):
        return self._workspace_objects
    @workspace_objects.setter
    def workspace_objects(self, workspace_object: Workspace_objects):
        self._workspace_objects=workspace_object
