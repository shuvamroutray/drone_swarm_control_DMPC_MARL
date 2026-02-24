#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

from control_swarm.envs.ros_env.SwarmRosEnv import SwarmRosEnv


class SwarmDMPCNode(Node):

    def __init__(self):

        super().__init__('swarm_dmpc_node')

        self.NUM_DRONES = 2
        self.DT = 1.0 / 30.0

        self.targets = np.array([
            [0.0,  1.5, 1.0],
            [0.0, -1.5, 1.0]
        ])

        self.env = SwarmRosEnv(
            num_drones=self.NUM_DRONES,
            dt=self.DT,
            targets=self.targets
        )

        self.subscribers = []
        for i in range(self.NUM_DRONES):
            self.create_subscription(
                Odometry,
                f'/cf{i+1}/odom',
                lambda msg, i=i: self.odom_callback(msg, i),
                10
            )

        self.publishers = [
            self.create_publisher(Twist, f'/cf{i+1}/cmd_vel', 10)
            for i in range(self.NUM_DRONES)
        ]

        self.timer = self.create_timer(self.DT, self.control_loop)

    def odom_callback(self, msg, drone_id):

        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        state = np.array([px, py, pz, vx, vy, vz])
        self.env.update_state(drone_id, state)

    def control_loop(self):

        commands = self.env.step()

        for i in range(self.NUM_DRONES):

            cmd = Twist()
            cmd.linear.x = float(commands[i][0])
            cmd.linear.y = float(commands[i][1])
            cmd.linear.z = float(commands[i][2])

            self.publishers[i].publish(cmd)


def main(args=None):

    rclpy.init(args=args)
    node = SwarmDMPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
