#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Empty
from crazyflie_interfaces.msg import FullState
from crazyflie_interfaces.srv import Takeoff
from visualization_msgs.msg import Marker

import numpy as np
import time
import os

from tf2_ros import Buffer, TransformListener
from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl


class DMPCAgent(Node):

    def __init__(self):
        super().__init__('dmpc_agent')

        # ================= PARAMETERS =================
        self.declare_parameter("drone_name", "cf231")
        self.declare_parameter("neighbor_names", [""])
        self.declare_parameter("num_drones", 1)
        self.declare_parameter("Np", 10)
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("target", [0.0, 0.0, 1.0])

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

        # ================= STATE =================
        self.state = np.zeros(6)
        self.last_pos = None

        self.neighbor_predictions = {
            name: np.zeros(6 * (self.Np + 1))
            for name in self.neighbor_names
        }

        # ================= LOGGING =================
        self.time_log = []
        self.track_log = []
        self.min_dist_log = []

        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # ================= TF =================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ================= CONTROLLER =================
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

        # ================= PUBLISHERS =================
        self.cmd_pub = self.create_publisher(
            FullState,
            'cmd_full_state',
            10
        )

        self.pred_pub = self.create_publisher(
            Float64MultiArray,
            'prediction',
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            "target_marker",
            10
        )

        self.takeoff_client = self.create_client(
            Takeoff,
            'takeoff'
        )

        # ================= NEIGHBOR SUBS =================
        for name in self.neighbor_names:
            topic = f'/{name}/prediction'
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, name=name: self.pred_callback(msg, name),
                10
            )

        # ================= TAKEOFF =================
        time.sleep(2.0)
        req = Takeoff.Request()
        req.height = 1.0
        req.duration.sec = 2
        self.takeoff_client.call_async(req)
        time.sleep(3.0)

        # ================= TIMERS =================
        self.create_timer(self.dt, self.control_loop)
        self.create_timer(0.5, self.publish_marker)

        self.get_logger().info(f"{self.drone_name} started.")

    # ====================================================
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
            else:
                vx, vy, vz = 0.0, 0.0, 0.0

            self.last_pos = np.array([px, py, pz])
            self.state = np.array([px, py, pz, vx, vy, vz])

        except:
            pass

    # ====================================================
    def pred_callback(self, msg, name):
        self.neighbor_predictions[name] = np.array(msg.data)

    # ====================================================
    def control_loop(self):

        self.update_state()

        if self.neighbor_names:
            neighbors = np.hstack([
                self.neighbor_predictions[n]
                for n in sorted(self.neighbor_names)
            ])
        else:
            neighbors = np.array([])

        self.controller.compute_control(self.state, neighbors)
        pred = self.controller.predicted_trajectory
        next_pos = pred[1, 0:3]

        # Send command
        msg = FullState()
        msg.pose.position.x = float(next_pos[0])
        msg.pose.position.y = float(next_pos[1])
        msg.pose.position.z = float(next_pos[2])
        msg.twist.linear.x = float(pred[1, 3])
        msg.twist.linear.y = float(pred[1, 4])
        msg.twist.linear.z = float(pred[1, 5])
        msg.pose.orientation.w = 1.0
        self.cmd_pub.publish(msg)

        # Share prediction
        pmsg = Float64MultiArray()
        pmsg.data = pred.flatten().tolist()
        self.pred_pub.publish(pmsg)

        # ===== METRICS =====
        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        tracking_error = np.linalg.norm(self.state[:3] - self.target)

        min_dist = np.inf
        for n in self.neighbor_names:
            try:
                trans = self.tf_buffer.lookup_transform(
                    'world', n, rclpy.time.Time()
                )
                np_pos = np.array([
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z
                ])
                d = np.linalg.norm(self.state[:3] - np_pos)
                min_dist = min(min_dist, d)
            except:
                pass

        if min_dist == np.inf:
            min_dist = np.nan

        self.time_log.append(t)
        self.track_log.append(tracking_error)
        self.min_dist_log.append(min_dist)

    # ====================================================
    def publish_marker(self):

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "targets"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(self.target[0])
        marker.pose.position.y = float(self.target[1])
        marker.pose.position.z = float(self.target[2])
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25

        marker.color.a = 1.0
        marker.color.r = np.random.rand()
        marker.color.g = np.random.rand()
        marker.color.b = np.random.rand()

        self.marker_pub.publish(marker)

    # ====================================================
    def save_logs(self):
        data = np.vstack([
            self.time_log,
            self.track_log,
            self.min_dist_log
        ]).T
        np.save(f"{self.drone_name}_metrics.npy", data)
        self.get_logger().info(f"{self.drone_name} logs saved.")


def main(args=None):
    rclpy.init(args=args)
    node = DMPCAgent()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.save_logs()
    node.destroy_node()
    rclpy.shutdown()