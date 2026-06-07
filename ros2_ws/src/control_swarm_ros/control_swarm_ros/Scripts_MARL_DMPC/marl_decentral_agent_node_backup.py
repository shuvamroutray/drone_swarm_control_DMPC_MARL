#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# from control_swarm_ros.Scripts_MARL_DMPC.policy_loader import PolicyLoader
# from control_swarm_ros.Scripts_MARL_DMPC.marl_utils import (
#     ACTIONS,
#     world_to_grid,
#     grid_to_world,
#     build_observation
# )

from policy_loader import PolicyLoader
from marl_utils import (
    ACTIONS,
    world_to_grid,
    grid_to_world,
    build_observation
)


class MARLAgent(Node):

    def __init__(self):
        super().__init__("marl_agent")

        # =============================
        # PARAMETERS
        # =============================
        self.declare_parameter("drone_id", 0)
        self.declare_parameter("num_drones", 2)
        self.declare_parameter("drone_name", "cf231")
        self.declare_parameter("drone_names", ["cf231", "cf232"])

        self.drone_id = self.get_parameter("drone_id").value
        self.num_drones = self.get_parameter("num_drones").value
        self.drone_name = self.get_parameter("drone_name").value
        self.drone_names = self.get_parameter("drone_names").value
        self.dt = 1

        # =============================
        # GRID
        # =============================
        self.grid_size = 6
        self.scale = 0.5

        # =============================
        # STATE
        # =============================
        self.positions_world = {
            i: np.zeros(2)
            for i in range(self.num_drones)
        }

        self.positions_grid = {
            i: (0, 0)
            for i in range(self.num_drones)
        }

        self.grid = np.zeros((self.grid_size, self.grid_size))

        # =============================
        # POLICY
        # =============================
        checkpoint = "/home/shuvam/6_Drone_Swarm_DMPC_MARL/Control_Swarm/experiments/exp_marl/checkpoints_rllib4_6X6_2Agents"
        self.policy = PolicyLoader(
            checkpoint,
            self.grid_size,
            self.num_drones
        )

        # =============================
        # SUBSCRIBERS
        # =============================
        for i, name in enumerate(self.drone_names):

            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, i=i: self.odom_callback(msg, i),
                10
            )

        # =============================
        # PUBLISHER
        # =============================
        self.target_pub = self.create_publisher(
            Float64MultiArray,
            f"/{self.drone_name}/marl_target",
            10
        )

        self.grid_pub = self.create_publisher(
            Marker,
            "/grid_marker",
            10
        )

        # =============================
        # TIMER
        # =============================
        self.timer = self.create_timer(self.dt, self.run_policy)

    # =============================
    # CALLBACKS
    # =============================
    def odom_callback(self, msg, drone_id):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.positions_world[drone_id] = np.array([x, y])

        gx, gy = world_to_grid(x, y, self.scale, self.grid_size)
        self.positions_grid[drone_id] = (gx, gy)

    # =============================
    # GRID UPDATE
    # =============================
    def update_grid(self):

        for i in range(self.num_drones):

            gx, gy = self.positions_grid[i]

            if self.grid[gx, gy] == 0:
                self.grid[gx, gy] = 1

    # =============================
    # MAIN LOOP
    # =============================
    def run_policy(self):

        self.update_grid()

        obs = build_observation(
            self.grid,
            self.positions_grid,
            self.drone_id,
            self.grid_size
        )

        action = self.policy.predict(obs)

        gx, gy = self.positions_grid[self.drone_id]
        dx, dy = ACTIONS[action]

        new_gx = np.clip(gx + dx, 0, self.grid_size - 1)
        new_gy = np.clip(gy + dy, 0, self.grid_size - 1)

        wx, wy = grid_to_world(new_gx, new_gy, self.scale)

        target = [wx, wy, 1.0]

        msg = Float64MultiArray()
        msg.data = target

        self.target_pub.publish(msg)

        self.get_logger().info(
            f"[Agent {self.drone_id}] Action={action} Target={target}"
        )

        self.publish_grid()

    # =============================
    # VISUALIZATION
    # =============================
    def publish_grid(self):

        marker = Marker()

        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.type = Marker.CUBE_LIST
        marker.scale.x = self.scale
        marker.scale.y = self.scale
        marker.scale.z = 0.05

        marker.color.a = 0.5
        marker.color.g = 1.0

        for i in range(self.grid_size):
            for j in range(self.grid_size):

                if self.grid[i, j] > 0:
                    p = Point()
                    p.x = i * self.scale
                    p.y = j * self.scale
                    p.z = 0.0
                    marker.points.append(p)

        self.grid_pub.publish(marker)


def main(args=None):

    rclpy.init(args=args)

    node = MARLAgent()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()