#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64MultiArray

# from control_swarm_ros.Scripts_MARL_DMPC.policy_loader import PolicyLoader
# from control_swarm_ros.Scripts_MARL_DMPC.marl_utils import (
#     ACTIONS,
#     build_observation,
# )

from policy_loader import PolicyLoader
from marl_utils import ACTIONS, build_observation

# =========================================================
# CENTERED WORLD <-> GRID TRANSFORMS
# =========================================================
def world_to_grid(x, y, scale, grid_size):
    offset = (grid_size * scale) / 2.0

    gx = int((x + offset) / scale)
    gy = int((y + offset) / scale)

    gx = np.clip(gx, 0, grid_size - 1)
    gy = np.clip(gy, 0, grid_size - 1)

    return gx, gy


def grid_to_world(gx, gy, scale, grid_size):
    offset = (grid_size * scale) / 2.0

    wx = gx * scale - offset + scale / 2.0
    wy = gy * scale - offset + scale / 2.0

    return wx, wy


# =========================================================
# CENTRALIZED NODE
# =========================================================
class CentralizedMARLNode(Node):

    def __init__(self):
        super().__init__("centralized_marl_agent")

        # -----------------------------------------
        # PARAMETERS
        # -----------------------------------------
        self.grid_size = 6
        self.scale = 0.5
        self.num_drones = 2

        self.drone_names = ["cf231", "cf232"]

        # -----------------------------------------
        # SHARED GRID
        # -----------------------------------------
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # positions in grid coordinates
        self.positions = {
            0: (0.0, 0.0),
            1: (0.0, 0.0)
        }

        self.odom_received = {
            0: False,
            1: False
        }

        self.positions_grid = {
            0: (0, 0),
            1: (0, 0)
        }

        # -----------------------------------------
        # POLICY
        # -----------------------------------------
        checkpoint_path = (
            "/home/shuvam/6_Drone_Swarm_DMPC_MARL/"
            "Control_Swarm/experiments/exp_marl/"
            "checkpoints_rllib4_6X6_2Agents"
        )

        self.policy = PolicyLoader(
            checkpoint_path=checkpoint_path,
            grid_size=self.grid_size,
            n_agents=self.num_drones
        )

        # -----------------------------------------
        # SUBSCRIBERS
        # -----------------------------------------

        self.create_subscription(
            TFMessage,
            "/tf",
            self.tf_callback,
            50
        )


        # -----------------------------------------
        # PUBLISHERS
        # -----------------------------------------
        self.target_publishers = {}

        for drone_id, drone_name in enumerate(self.drone_names):

            self.target_publishers[drone_id] = self.create_publisher(
                Float64MultiArray,
                f"/{drone_name}/marl_target",
                10
            )

        # -----------------------------------------
        # TIMER
        # -----------------------------------------
        self.timer = self.create_timer(1.0, self.run_policy)

        self.get_logger().info("Centralized MARL node initialized")

    # =====================================================
    # TF CALLBACK
    # =====================================================
   
    def tf_callback(self, msg):

        for transform in msg.transforms:

            drone_name = transform.child_frame_id

            if drone_name not in self.drone_names:
                continue

            drone_id = self.drone_names.index(drone_name)

            x = transform.transform.translation.x
            y = transform.transform.translation.y

            gx, gy = world_to_grid(
                x,
                y,
                self.scale,
                self.grid_size
            )

            self.positions[drone_id] = (x, y)
            self.positions_grid[drone_id] = (gx, gy)

            self.odom_received[drone_id] = True

    # =====================================================
    # UPDATE SHARED GRID
    # =====================================================
    def update_grid(self):

        # reset dynamic agent markers
        self.grid[self.grid == 9] = 1
        self.grid[self.grid == 10] = 1

        # mark explored + agents
        for drone_id, (gx, gy) in self.positions_grid.items():

            self.grid[gx, gy] = 1

            if drone_id == 0:
                self.grid[gx, gy] = 9
            elif drone_id == 1:
                self.grid[gx, gy] = 10

    # =====================================================
    # MAIN POLICY LOOP
    # =====================================================
    def run_policy(self):

        # wait until all drones available
        if not all(self.odom_received.values()):
            self.get_logger().info("Waiting for all TF...")
            return

        # update shared environment
        self.update_grid()

        self.get_logger().info(
            f"Shared Grid:\n{self.grid}\nPositions={self.positions_grid}"
        )

        # -------------------------------------
        # RUN POLICY FOR EACH AGENT
        # -------------------------------------
        for drone_id in range(self.num_drones):

            obs = build_observation(
                self.grid,
                self.positions_grid,
                drone_id,
                self.grid_size
            )

            action = self.policy.predict(obs)

            dx, dy = ACTIONS[action]

            gx, gy = self.positions_grid[drone_id]

            target_gx = np.clip(gx + dx, 0, self.grid_size - 1)
            target_gy = np.clip(gy + dy, 0, self.grid_size - 1)

            tx, ty = grid_to_world(
                target_gx,
                target_gy,
                self.scale,
                self.grid_size
            )

            target_msg = Float64MultiArray()
            target_msg.data = [float(tx), float(ty), 1.0]

            self.target_publishers[drone_id].publish(target_msg)

            self.get_logger().info(
                f"[Agent {drone_id}] "
                f"Action={action} "
                f"GridTarget=({target_gx},{target_gy}) "
                f"WorldTarget=({tx:.2f},{ty:.2f},1.0)"
            )


# =========================================================
# MAIN
# =========================================================
def main(args=None):

    rclpy.init(args=args)

    node = CentralizedMARLNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()