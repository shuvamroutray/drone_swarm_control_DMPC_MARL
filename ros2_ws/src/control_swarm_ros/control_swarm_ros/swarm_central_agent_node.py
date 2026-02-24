# This code creates a single node that handles control for all 
# drones. Single node for all the drones, which is kind of centralized
# control of the swarm

# Date created: 20 Feb 2026
# Creator: Shuvam Routray

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

# Import YOUR existing DMPC controller
from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl


class SwarmDMPCNode(Node):

    def __init__(self):
        super().__init__('swarm_dmpc_node')

        # =============================
        # PARAMETERS
        # =============================
        self.NUM_DRONES = 2
        self.NP = 50
        self.DT = 1.0 / 30.0
        self.MAX_ACC = 5.0
        self.MAX_VEL = 2.0
        self.D_SAFE = 0.5

        self.targets = np.array([
            [0.0,  1.5, 1.0],
            [0.0, -1.5, 1.0]
        ])

        # =============================
        # STATE STORAGE
        # =============================
        self.states = [np.zeros(6) for _ in range(self.NUM_DRONES)]
        self.prev_predictions = np.zeros(
            (self.NUM_DRONES, 6*(self.NP+1))
        )

        # =============================
        # DMPC CONTROLLERS
        # =============================
        self.controllers = [
            DMPCControl(
                drone_id=i,
                Np=self.NP,
                dt=self.DT,
                target_pos=self.targets[i],
                max_acc=self.MAX_ACC,
                max_vel=self.MAX_VEL,
                d_safe=self.D_SAFE,
                num_drones=self.NUM_DRONES
            )
            for i in range(self.NUM_DRONES)
        ]

        # =============================
        # SUBSCRIBERS
        # =============================
        for i in range(self.NUM_DRONES):
            self.create_subscription(
                Odometry,
                f'/cf{i+1}/odom',
                lambda msg, i=i: self.odom_callback(msg, i),
                10
            )

        # =============================
        # PUBLISHERS
        # =============================
        self.cmd_publishers = [
            self.create_publisher(Twist, f'/cf{i+1}/cmd_vel', 10)
            for i in range(self.NUM_DRONES)
        ]

        # =============================
        # TIMER
        # =============================
        self.timer = self.create_timer(self.DT, self.control_loop)

        self.get_logger().info("Swarm DMPC Node Started")

    def odom_callback(self, msg, drone_id):

        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        self.states[drone_id] = np.array([px, py, pz, vx, vy, vz])

    def control_loop(self):

        for i in range(self.NUM_DRONES):

            neighbors = np.hstack([
                self.prev_predictions[j]
                for j in range(self.NUM_DRONES)
                if j != i
            ]) if self.NUM_DRONES > 1 else np.array([])

            u = self.controllers[i].compute_control(
                self.states[i],
                neighbors
            )

            pred = self.controllers[i].predicted_trajectory
            target_vel = pred[1, 3:6]

            msg = Twist()
            msg.linear.x = float(target_vel[0])
            msg.linear.y = float(target_vel[1])
            msg.linear.z = float(target_vel[2])

            self.cmd_publishers[i].publish(msg)


            self.prev_predictions[i] = pred.flatten()


def main(args=None):
    rclpy.init(args=args)
    node = SwarmDMPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

