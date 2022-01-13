from pybewego.numerical_optimization.ipopt_mo import NavigationOptimization
from multiprocessing.sharedctypes import Value, Array
# This class allow to take the result 
class NavigationOptimizationMultiprocessing(NavigationOptimization):
    def optimize(self, scalars, ipopt_options=...,return_status: Value = None, return_traj=None,  queue=None):
        result = super().optimize(scalars, ipopt_options=ipopt_options, queue=queue)
        if return_status is not None:  # get out status from multiprocessing
            return_status.value = result[0]
            print("return_status", return_status)
        if return_traj is not None:
            return_traj[:] = result[1].x().tolist()
            print("return traj", return_traj)
        return result