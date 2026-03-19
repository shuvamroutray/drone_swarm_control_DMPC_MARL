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
        self.declare_parameter("num_drones", 2)
        self.declare_parameter("Np", 10)
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("switch_time", 10.0)

        self.drone_name = self.get_parameter("drone_name").value
        self.neighbor_names = self.get_parameter("neighbor_names").value
        self.num_drones = self.get_parameter("num_drones").value
        self.Np = self.get_parameter("Np").value
        self.dt = self.get_parameter("dt").value
        self.switch_time = self.get_parameter("switch_time").value

        # ================= INITIAL POSITIONS =================
        if self.drone_name == "cf231":
            self.target = np.array([-1.0, 0.0, 1.0])
            self.target_2 = np.array([1.0, 0.0, 1.0])
        else:
            self.target = np.array([1.0, 0.0, 1.0])
            self.target_2 = np.array([-1.0, 0.0, 1.0])

        self.target_switched = False
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # ================= CONTROLLER LIMITS =================
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
        self.tracking_log = []
        self.inter_log = []

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
        self.cmd_pub = self.create_publisher(FullState, 'cmd_full_state', 10)
        self.pred_pub = self.create_publisher(Float64MultiArray, 'prediction', 10)
        self.marker_pub = self.create_publisher(Marker, 'target_marker', 10)

        self.takeoff_client = self.create_client(Takeoff, 'takeoff')

        for name in self.neighbor_names:
            self.create_subscription(
                Float64MultiArray,
                f'/{name}/prediction',
                lambda msg, name=name: self.pred_callback(msg, name),
                10
            )

        # ================= TAKEOFF =================
        req = Takeoff.Request()
        req.height = 1.0
        req.duration.sec = 2
        self.takeoff_client.call_async(req)
        time.sleep(3.0)

        # ================= TIMERS =================
        self.create_timer(self.dt, self.control_loop)
        self.create_timer(0.5, self.publish_marker)

    # ============================================================
    # STATE UPDATE
    # ============================================================
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

    # ============================================================
    # NEIGHBOR
    # ============================================================
    def pred_callback(self, msg, name):
        self.neighbor_predictions[name] = np.array(msg.data)

    # ============================================================
    # CONTROL LOOP
    # ============================================================
    def control_loop(self):

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self.start_time

        if not self.target_switched and elapsed > self.switch_time:
            self.target = self.target_2.copy()
            self.controller.target_pos = self.target
            self.target_switched = True
            self.get_logger().info(f"{self.drone_name} SWITCHED TARGET")

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

        cmd = FullState()
        cmd.pose.position.x = float(pred[1, 0])
        cmd.pose.position.y = float(pred[1, 1])
        cmd.pose.position.z = float(pred[1, 2])
        cmd.twist.linear.x = float(pred[1, 3])
        cmd.twist.linear.y = float(pred[1, 4])
        cmd.twist.linear.z = float(pred[1, 5])
        cmd.pose.orientation.w = 1.0
        self.cmd_pub.publish(cmd)

        pred_msg = Float64MultiArray()
        pred_msg.data = pred.flatten().tolist()
        self.pred_pub.publish(pred_msg)

        # ========== METRICS ==========
        tracking_error = np.linalg.norm(self.state[0:3] - self.target)

        try:
            trans_n = self.tf_buffer.lookup_transform(
                'world',
                self.neighbor_names[0],
                rclpy.time.Time()
            )
            neighbor_pos = np.array([
                trans_n.transform.translation.x,
                trans_n.transform.translation.y,
                trans_n.transform.translation.z
            ])
            inter_dist = np.linalg.norm(self.state[0:3] - neighbor_pos)
        except:
            inter_dist = np.nan

        self.time_log.append(elapsed)
        self.tracking_log.append(tracking_error)
        self.inter_log.append(inter_dist)

    # ============================================================
    # MARKER
    # ============================================================
    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.target[0])
        marker.pose.position.y = float(self.target[1])
        marker.pose.position.z = float(self.target[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.25
        marker.color.a = 1.0

        if self.drone_name == "cf231":
            marker.color.r = 1.0
        else:
            marker.color.g = 1.0

        self.marker_pub.publish(marker)

    # ============================================================
    # SAVE
    # ============================================================
    def save_logs(self):
        data = np.vstack([
            self.time_log,
            self.tracking_log,
            self.inter_log
        ]).T
        np.save(f"{self.drone_name}_metrics.npy", data)


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