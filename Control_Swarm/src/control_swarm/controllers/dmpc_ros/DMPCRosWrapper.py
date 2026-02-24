import numpy as np
from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl


class DMPCRosWrapper:
    """
    Thin ROS wrapper around existing DMPCControl.
    Keeps original implementation untouched.
    """

    def __init__(self,
                 drone_id,
                 num_drones,
                 dt,
                 target,
                 horizon=50,
                 max_acc=5.0,
                 max_vel=2.0,
                 d_safe=0.5):

        self.controller = DMPCControl(
            drone_id=drone_id,
            Np=horizon,
            dt=dt,
            target_pos=np.array(target),
            max_acc=max_acc,
            max_vel=max_vel,
            d_safe=d_safe,
            num_drones=num_drones
        )

    def compute(self, state, neighbor_predictions):
        return self.controller.compute_control(state, neighbor_predictions)

    def get_prediction(self):
        return self.controller.predicted_trajectory
