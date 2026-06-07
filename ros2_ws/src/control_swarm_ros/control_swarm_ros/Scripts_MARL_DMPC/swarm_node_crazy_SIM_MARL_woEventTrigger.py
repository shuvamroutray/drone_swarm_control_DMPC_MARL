#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import time

from std_msgs.msg import Float64MultiArray, Empty
from crazyflie_interfaces.msg import FullState
from crazyflie_interfaces.srv import Takeoff

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from tf2_ros import Buffer, TransformListener

from control_swarm.controllers.dmpc_gym.DMPCControlRuntime import DMPCControl


class DMPCAgent(Node):

    def __init__(self):
        super().__init__('dmpc_agent')

        # ==============================
        # PARAMETERS
        # ==============================
        self.declare_parameter("drone_name", "cf231")
        self.declare_parameter("neighbor_names", [])
        self.declare_parameter("num_drones", 1)
        self.declare_parameter("Np", 10)
        self.declare_parameter("dt", 0.2)

        self.drone_name = self.get_parameter("drone_name").value
        self.neighbor_names = self.get_parameter("neighbor_names").value
        self.num_drones = self.get_parameter("num_drones").value
        self.Np = self.get_parameter("Np").value
        self.dt = self.get_parameter("dt").value

        # ==============================
        # CONTROL LIMITS (TUNED)
        # ==============================
        self.MAX_ACC = 1.2
        self.MAX_VEL = 0.8
        self.D_SAFE = 0.5

        # ==============================
        # STATE
        # ==============================
        self.state = np.zeros(6)
        self.last_pos = None

        self.target = np.array([0.0, 0.0, 1.0])
        self.target_locked = False

        # ==============================
        # NEIGHBORS
        # ==============================
        self.neighbor_predictions = {
            name: None for name in self.neighbor_names
        }

        # ==============================
        # TF
        # ==============================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ==============================
        # CONTROLLER
        # ==============================
        self.controller = DMPCControl(
            drone_id=0,
            Np=self.Np,
            dt=self.dt,
            target_pos=self.target,
            max_acc=self.MAX_ACC,
            max_vel=self.MAX_VEL,
            d_safe=self.D_SAFE,
            num_drones=self.num_drones
        )

        # ==============================
        # PUBLISHERS
        # ==============================
        self.cmd_pub = self.create_publisher(
            FullState,
            f'/{self.drone_name}/cmd_full_state',
            10
        )

        self.pred_pub = self.create_publisher(
            Float64MultiArray,
            f'/{self.drone_name}/prediction',
            10
        )

        # ==============================
        # SUBSCRIBERS
        # ==============================
        for name in self.neighbor_names:
            self.create_subscription(
                Float64MultiArray,
                f'/{name}/prediction',
                lambda msg, name=name: self.prediction_callback(msg, name),
                10
            )

        self.create_subscription(
            Float64MultiArray,
            f'/{self.drone_name}/marl_target',
            self.target_callback,
            10
        )

        # ==============================
        # TAKEOFF
        # ==============================
        self.takeoff_client = self.create_client(Takeoff, 'takeoff')

        self.get_logger().info("Taking off...")
        req = Takeoff.Request()
        req.height = 1.0
        req.duration.sec = 2
        self.takeoff_client.call_async(req)
        time.sleep(3)

        # ==============================
        # TIMER
        # ==============================
        self.timer = self.create_timer(self.dt, self.control_loop)

    # ==========================================
    # STATE UPDATE
    # ==========================================
    def update_state(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'world',
                self.drone_name,
                rclpy.time.Time()
            )

            px = trans.transform.translation.x
            py = trans.transform.translation.y
            pz = trans.transform.translation.z

            if self.last_pos is not None:
                vx = (px - self.last_pos[0]) / self.dt
                vy = (py - self.last_pos[1]) / self.dt
                vz = (pz - self.last_pos[2]) / self.dt

                # LOW PASS FILTER
                alpha = 0.6
                vx = alpha * vx + (1 - alpha) * self.state[3]
                vy = alpha * vy + (1 - alpha) * self.state[4]
                vz = alpha * vz + (1 - alpha) * self.state[5]

            else:
                vx, vy, vz = 0.0, 0.0, 0.0

            self.last_pos = np.array([px, py, pz])
            self.state = np.array([px, py, pz, vx, vy, vz])

        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")

    # ==========================================
    # TARGET CALLBACK (SMART LOCKING)
    # ==========================================
    def target_callback(self, msg):

        new_target = np.array(msg.data)

        # Ignore small changes
        if np.linalg.norm(new_target - self.target) < 0.2:
            return

        # Only accept new target if current reached
        if not self.target_locked:
            self.target = new_target
            self.controller.target_pos = self.target
            self.target_locked = True

            self.get_logger().info(f"New Target: {np.round(self.target,2)}")

    # ==========================================
    # NEIGHBOR CALLBACK
    # ==========================================
    def prediction_callback(self, msg, name):
        self.neighbor_predictions[name] = np.array(msg.data)

    # ==========================================
    # CONTROL LOOP
    # ==========================================
    def control_loop(self):

        self.update_state()

        # Check if target reached
        dist = np.linalg.norm(self.state[:3] - self.target)

        if dist < 0.15:
            self.target_locked = False
            return

        # Wait for neighbors
        if self.neighbor_names:
            if any(self.neighbor_predictions[n] is None for n in self.neighbor_names):
                return

            neighbors = np.hstack([
                self.neighbor_predictions[n]
                for n in sorted(self.neighbor_names)
            ])
        else:
            neighbors = np.array([])

        # Solve MPC
        self.controller.target_pos = self.target
        self.controller.target_state = np.hstack([self.target, np.zeros(3)])

        self.controller.compute_control(self.state, neighbors)
        pred = self.controller.predicted_trajectory

        target_pos = pred[1, 0:3]

        # Publish command
        msg = FullState()

        msg.pose.position.x = float(target_pos[0])
        msg.pose.position.y = float(target_pos[1])
        msg.pose.position.z = float(target_pos[2])

        msg.twist.linear.x = float(pred[1, 3])
        msg.twist.linear.y = float(pred[1, 4])
        msg.twist.linear.z = float(pred[1, 5])

        msg.pose.orientation.w = 1.0

        self.cmd_pub.publish(msg)

        # Publish prediction
        pred_msg = Float64MultiArray()
        pred_msg.data = pred.flatten().tolist()
        self.pred_pub.publish(pred_msg)

        self.get_logger().info(
            f"[{self.drone_name}] → {np.round(target_pos,2)} | dist={dist:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DMPCAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()