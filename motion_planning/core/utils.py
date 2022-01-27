import numpy as np
from pyrieef.motion.trajectory import Trajectory
class abcd:
    def __init__(self, a):
        self.a=a 

    def get_a(self):
        return self.a
A = abcd
x = A(5)
print(x.get_a())

A = np.array([0, 0])
B = np.array([1, 1])
C = np.concatenate([A,B])
print(A)
print(C)
a = np.min([0, 1])
print(a)
A = Trajectory(q_init=np.array([0, 0]), x=np.array([]))
print(A.x())
B = Trajectory(q_init=np.array([1, 1]), x=np.array([]))
print(B.x())
x = np.concatenate([A.x(), B.x()])
C = Trajectory(q_init=x[0:2], x=x[2:])
print(C)

print(int(10/3))