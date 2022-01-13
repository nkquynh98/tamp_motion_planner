import numpy as np
from pyrieef.geometry.differentiable_geometry import DifferentiableMap
from pyrieef.geometry.rotations import rotation_matrix_2d_radian

def calculate_angle(vector):
    angle = np.arctan2(vector[1], vector[0])
    return angle
class LinearTranslation(DifferentiableMap):
    """
    Simple linear translation
    """

    def __init__(self, p0=np.zeros(2)):
        assert isinstance(p0, np.ndarray)
        self._p = p0

    def forward(self, q):
        assert q.shape[0] == self.input_dimension
        return self._p + q

    def backward(self, p):  # p is coordinate in parent frame
        assert p.shape[0] == self.output_dimension
        return p - self._p

    def jacobian(self, q):
        assert q.shape[0] == self.input_dimension
        return np.eye(self.input_dimension)

    @property
    def input_dimension(self):
        return self._p.shape[0]

    @property
    def output_dimension(self):
        return self.input_dimension

def LinearRotate2D(LinearTranslation):
    def __init__(self, p0=np.zeros(2)):
        assert isinstance(p0, np.ndarray)
        self._p = p0

    def forward(self, q, yaw):
        assert q.shape[0] == self.input_dimension
        return self._p + rotation_matrix_2d_radian(yaw)@q

    def backward(self, p, yaw):
        assert p.shape[0] == self.output_dimension
        return p - self._p