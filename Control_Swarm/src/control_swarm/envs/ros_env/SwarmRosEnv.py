import numpy as np
from control_swarm.controllers.dmpc_ros.DMPCRosWrapper import DMPCRosWrapper


class SwarmRosEnv:
    """
    ROS-based environment replacing Gym environment.
    """

    def __init__(self,
                 num_drones,
                 dt,
                 targets):

        self.NUM_DRONES = num_drones
        self.DT = dt

        self.states = [np.zeros(6) for _ in range(num_drones)]

        self.controllers = [
            DMPCRosWrapper(
                drone_id=i,
                num_drones=num_drones,
                dt=dt,
                target=targets[i]
            )
            for i in range(num_drones)
        ]

        self.prev_predictions = None

    def update_state(self, drone_id, state):
        self.states[drone_id] = state

    def step(self):

        commands = []

        for i in range(self.NUM_DRONES):

            neighbors = []

            if self.prev_predictions is not None:
                neighbors = np.hstack([
                    self.prev_predictions[j]
                    for j in range(self.NUM_DRONES)
                    if j != i
                ])

            u = self.controllers[i].compute(
                self.states[i],
                neighbors
            )

            pred = self.controllers[i].get_prediction()

            commands.append(pred[1, 3:6])

        # store predictions
        self.prev_predictions = [
            self.controllers[i].get_prediction().flatten()
            for i in range(self.NUM_DRONES)
        ]

        return commands
