#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Empty
from crazyflie_interfaces.msg import FullState

import numpy as np
import time

from tf2_ros import Buffer, TransformListener

from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl


class DMPCAgent(Node):

    def __init__(self):
        super().__init__('dmpc_agent')

        # ============================================================
        # PARAMETERS
        # ============================================================
        self.declare_parameter("drone_name", "cf231")
        self.declare_parameter("neighbor_names", [""])
        self.declare_parameter("num_drones", 1)
        self.declare_parameter("Np", 10)
        self.declare_parameter("dt", 1.0/30.0)
        self.declare_parameter("target", [0.0, 1.0, 1.0])

        self.drone_name = self.get_parameter("drone_name").value
        self.neighbor_names = self.get_parameter("neighbor_names").value
        self.num_drones = self.get_parameter("num_drones").value
        self.Np = self.get_parameter("Np").value
        self.dt = self.get_parameter("dt").value
        self.target = np.array(
            self.get_parameter("target").value,
            dtype=float
        )

        # Controller limits
        self.MAX_ACC = 3.0
        self.MAX_VEL = 1.5
        self.D_SAFE = 0.4

        # ============================================================
        # STATE STORAGE
        # ============================================================
        self.state = np.zeros(6)
        self.last_pos = None

        # Neighbor predicted trajectories (name-based)
        self.neighbor_predictions = {
            name: np.zeros(6 * (self.Np + 1))
            for name in self.neighbor_names
        }

        # ============================================================
        # TF LISTENER (Crazyswarm2 uses TF)
        # ============================================================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ============================================================
        # DMPC CONTROLLER
        # ============================================================
        self.controller = DMPCControl(
            drone_id=0,  # only used internally/logging
            Np=self.Np,
            dt=self.dt,
            target_pos=self.target,
            max_acc=self.MAX_ACC,
            max_vel=self.MAX_VEL,
            d_safe=self.D_SAFE,
            num_drones=self.num_drones
        )

        # ============================================================
        # PUBLISHERS
        # ============================================================

        # # Position setpoint (relative to namespace)
        # self.cmd_pub = self.create_publisher(
        #     Twist,
        #     'cmd_position',
        #     10
        # )

        self.cmd_pub = self.create_publisher(
            FullState,
            'cmd_full_state',
            10
        )



        # Takeoff / Land
        self.takeoff_pub = self.create_publisher(Empty, 'takeoff', 10)
        self.land_pub = self.create_publisher(Empty, 'land', 10)

        # Prediction sharing
        self.pred_pub = self.create_publisher(
            Float64MultiArray,
            'prediction',
            10
        )

        # ============================================================
        # NEIGHBOR SUBSCRIPTIONS
        # ============================================================
        for name in self.neighbor_names:
            topic = f'/{name}/prediction'
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, name=name: self.prediction_callback(msg, name),
                10
            )

        # ============================================================
        # WAIT FOR TF + TAKEOFF
        # ============================================================
        self.get_logger().info(f"[{self.drone_name}] Waiting for TF...")
        time.sleep(2.0)

        self.get_logger().info(f"[{self.drone_name}] Taking off...")
        self.takeoff_pub.publish(Empty())
        time.sleep(3.0)

        # ============================================================
        # CONTROL TIMER
        # ============================================================
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"DMPC Agent for {self.drone_name} started. "
            f"Neighbors: {self.neighbor_names}"
        )

    # ============================================================
    # TF STATE UPDATE
    # ============================================================
    def update_state_from_tf(self):

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
            else:
                vx, vy, vz = 0.0, 0.0, 0.0

            self.last_pos = np.array([px, py, pz])
            self.state = np.array([px, py, pz, vx, vy, vz])

        except Exception as e:
            self.get_logger().warn(
                f"[{self.drone_name}] TF lookup failed: {e}"
            )

    # ============================================================
    # NEIGHBOR CALLBACK
    # ============================================================
    def prediction_callback(self, msg, drone_name):
        self.neighbor_predictions[drone_name] = np.array(msg.data)

    # ============================================================
    # CONTROL LOOP
    # ============================================================
    def control_loop(self):

        # Update own state
        self.update_state_from_tf()

        # Stack neighbor trajectories
        if self.neighbor_names:
            neighbors = np.hstack([
                self.neighbor_predictions[name]
                for name in sorted(self.neighbor_names)
            ])
        else:
            neighbors = np.array([])

        # Solve DMPC
        u = self.controller.compute_control(self.state, neighbors)
        pred = self.controller.predicted_trajectory

        # Use next predicted position
        target_pos = pred[1, 0:3]

        #***********************************************
        # Send position command
        #************************************************
        # msg = Twist()
        # msg.linear.x = float(target_pos[0])
        # msg.linear.y = float(target_pos[1])
        # msg.linear.z = float(target_pos[2])
        # self.cmd_pub.publish(msg)

        msg = FullState()

        # Position
        msg.pose.position.x = float(target_pos[0])
        msg.pose.position.y = float(target_pos[1])
        msg.pose.position.z = float(target_pos[2])

        # Velocity (use predicted next velocity if available)
        msg.twist.linear.x = float(pred[1, 3])
        msg.twist.linear.y = float(pred[1, 4])
        msg.twist.linear.z = float(pred[1, 5])

        # Zero angular stuff for now
        msg.pose.orientation.w = 1.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        self.cmd_pub.publish(msg)



        

        # Publish predicted trajectory
        pred_msg = Float64MultiArray()
        pred_msg.data = pred.flatten().tolist()
        self.pred_pub.publish(pred_msg)

        self.get_logger().info(
            f"[{self.drone_name}] Pos: {np.round(self.state[:3],2)} "
            f"→ Cmd: {np.round(target_pos,2)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DMPCAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()