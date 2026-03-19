#!/usr/bin/env python3

#*********With Plots**************

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

        # ============================================================
        # PARAMETERS
        # ============================================================
        self.declare_parameter("drone_name", "cf231")
        self.declare_parameter("neighbor_names", [""])
        self.declare_parameter("num_drones", 1)
        self.declare_parameter("Np", 10)
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("target", [0.0, 1.0, 1.0])

        # Mission switch
        self.declare_parameter("target_2", [1.0, 0.0, 1.0])
        self.declare_parameter("switch_time", 20.0)

        # ============================================================
        # LOAD PARAMETERS
        # ============================================================
        self.drone_name = self.get_parameter("drone_name").value
        self.neighbor_names = self.get_parameter("neighbor_names").value
        self.num_drones = self.get_parameter("num_drones").value
        self.Np = self.get_parameter("Np").value
        self.dt = self.get_parameter("dt").value

        self.target = np.array(
            self.get_parameter("target").value,
            dtype=float
        )

        self.target_2 = np.array(
            self.get_parameter("target_2").value,
            dtype=float
        )

        self.switch_time = self.get_parameter("switch_time").value

        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        self.target_switched = False

        # ============================================================
        # CONTROLLER LIMITS
        # ============================================================
        self.MAX_ACC = 3.0
        self.MAX_VEL = 1.5
        self.D_SAFE = 0.4

        # ============================================================
        # STATE STORAGE
        # ============================================================
        self.state = np.zeros(6)
        self.last_pos = None

        self.neighbor_predictions = {
            name: np.zeros(6 * (self.Np + 1))
            for name in self.neighbor_names
        }

        # ============================================================
        # LOGGING STORAGE
        # ============================================================
        self.time_log = []
        self.tracking_error_log = []
        self.inter_drone_distance_log = []

        self.sim_start_time = self.start_time

        # ============================================================
        # TF LISTENER
        # ============================================================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ============================================================
        # DMPC CONTROLLER
        # ============================================================
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

        # ============================================================
        # PUBLISHERS
        # ============================================================
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

        self.target_pub = self.create_publisher(
            Marker,
            "target_marker",
            10
        )

        self.takeoff_client = self.create_client(
            Takeoff,
            'takeoff'
        )

        self.land_pub = self.create_publisher(Empty, 'land', 10)

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
        # TAKEOFF
        # ============================================================
        self.get_logger().info(f"[{self.drone_name}] Taking off...")
        req = Takeoff.Request()
        req.height = 1.0
        req.duration.sec = 2
        self.takeoff_client.call_async(req)
        time.sleep(3.0)

        # ============================================================
        # TIMERS
        # ============================================================
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.create_timer(0.5, self.publish_target_marker)
    # ============================================================
    # STATE UPDATE FROM TF
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

        except:
            pass

    # ============================================================
    # NEIGHBOR CALLBACK
    # ============================================================
    def prediction_callback(self, msg, drone_name):
        self.neighbor_predictions[drone_name] = np.array(msg.data)



        # ============================================================
    # TARGET MARKER
    # ============================================================
    def publish_target_marker(self):

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

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0

        if "231" in self.drone_name:
            marker.color.r = 1.0
        elif "232" in self.drone_name:
            marker.color.g = 1.0
        else:
            marker.color.b = 1.0

        self.target_pub.publish(marker)



    # ============================================================
    # CONTROL LOOP
    # ============================================================
    def control_loop(self):

        # Mission switch
        current_time = self.get_clock().now().nanoseconds * 1e-9
        elapsed = current_time - self.start_time

        if (not self.target_switched) and (elapsed > self.switch_time):
            self.get_logger().info(
                f"[{self.drone_name}] Switching mission objective!"
            )
            self.target = self.target_2.copy()
            self.controller.target_pos = self.target
            self.target_switched = True

        # Update state
        self.update_state_from_tf()

        # Stack neighbors
        if self.neighbor_names:
            neighbors = np.hstack([
                self.neighbor_predictions[name]
                for name in sorted(self.neighbor_names)
            ])
        else:
            neighbors = np.array([])

        # Compute control
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

        # ============================================================
        # METRICS LOGGING
        # ============================================================
        sim_time = current_time - self.sim_start_time

        tracking_error = np.linalg.norm(self.state[0:3] - self.target)

        if self.neighbor_names:
            try:
                trans_neighbor = self.tf_buffer.lookup_transform(
                    'world',
                    self.neighbor_names[0],
                    rclpy.time.Time()
                )
                neighbor_pos = np.array([
                    trans_neighbor.transform.translation.x,
                    trans_neighbor.transform.translation.y,
                    trans_neighbor.transform.translation.z
                ])
                inter_dist = np.linalg.norm(self.state[0:3] - neighbor_pos)
            except:
                inter_dist = np.nan
        else:
            inter_dist = np.nan

        self.time_log.append(sim_time)
        self.tracking_error_log.append(tracking_error)
        self.inter_drone_distance_log.append(inter_dist)

    # ============================================================
    # SAVE LOGS
    # ============================================================
    def save_logs(self):

        log_data = np.vstack([
            self.time_log,
            self.tracking_error_log,
            self.inter_drone_distance_log
        ]).T

        filename = f"{self.drone_name}_metrics.npy"
        np.save(filename, log_data)

        self.get_logger().info(f"[{self.drone_name}] Metrics saved.")


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























#***********Without Plots************************


# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float64MultiArray, Empty
# from crazyflie_interfaces.msg import FullState
# from crazyflie_interfaces.srv import Takeoff
# from visualization_msgs.msg import Marker

# import numpy as np
# import time

# from tf2_ros import Buffer, TransformListener
# from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl


# class DMPCAgent(Node):

#     def __init__(self):
#         super().__init__('dmpc_agent')

#         # ============================================================
#         # PARAMETERS
#         # ============================================================
#         self.declare_parameter("drone_name", "cf231")
#         self.declare_parameter("neighbor_names", [""])
#         self.declare_parameter("num_drones", 1)
#         self.declare_parameter("Np", 10)
#         self.declare_parameter("dt", 0.1)
#         self.declare_parameter("target", [0.0, 1.0, 1.0])

#         # Step change parameters
#         self.declare_parameter("target_2", [1.0, 0.0, 1.0])
#         self.declare_parameter("switch_time", 10.0)

#         # ============================================================
#         # GET PARAMETERS
#         # ============================================================
#         self.drone_name = self.get_parameter("drone_name").value
#         self.neighbor_names = self.get_parameter("neighbor_names").value
#         self.num_drones = self.get_parameter("num_drones").value
#         self.Np = self.get_parameter("Np").value
#         self.dt = self.get_parameter("dt").value

#         self.target = np.array(
#             self.get_parameter("target").value,
#             dtype=float
#         )

#         self.target_2 = np.array(
#             self.get_parameter("target_2").value,
#             dtype=float
#         )

#         self.switch_time = self.get_parameter("switch_time").value

#         # Store initial target
#         self.target_1 = self.target.copy()

#         self.start_time = self.get_clock().now().nanoseconds * 1e-9
#         self.target_switched = False

#         # ============================================================
#         # CONTROLLER LIMITS
#         # ============================================================
#         self.MAX_ACC = 3.0
#         self.MAX_VEL = 1.5
#         self.D_SAFE = 0.4

#         # ============================================================
#         # STATE STORAGE
#         # ============================================================
#         self.state = np.zeros(6)
#         self.last_pos = None

#         self.neighbor_predictions = {
#             name: np.zeros(6 * (self.Np + 1))
#             for name in self.neighbor_names
#         }

#         # ============================================================
#         # TF LISTENER
#         # ============================================================
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # ============================================================
#         # DMPC CONTROLLER
#         # ============================================================
#         self.controller = DMPCControl(
#             drone_id=0,
#             Np=self.Np,
#             dt=self.dt,
#             target_pos=self.target,
#             max_acc=self.MAX_ACC,
#             max_vel=self.MAX_VEL,
#             d_safe=self.D_SAFE,
#             num_drones=self.num_drones
#         )

#         # ============================================================
#         # PUBLISHERS
#         # ============================================================
#         self.cmd_pub = self.create_publisher(
#             FullState,
#             'cmd_full_state',
#             10
#         )

#         self.target_pub = self.create_publisher(
#             Marker,
#             "target_marker",
#             10
#         )

#         self.takeoff_client = self.create_client(
#             Takeoff,
#             'takeoff'
#         )

#         self.land_pub = self.create_publisher(Empty, 'land', 10)

#         self.pred_pub = self.create_publisher(
#             Float64MultiArray,
#             'prediction',
#             10
#         )

#         # ============================================================
#         # NEIGHBOR SUBSCRIPTIONS
#         # ============================================================
#         for name in self.neighbor_names:
#             topic = f'/{name}/prediction'
#             self.create_subscription(
#                 Float64MultiArray,
#                 topic,
#                 lambda msg, name=name: self.prediction_callback(msg, name),
#                 10
#             )

#         # ============================================================
#         # TAKEOFF
#         # ============================================================
#         self.get_logger().info(f"[{self.drone_name}] Waiting for TF...")
#         time.sleep(2.0)

#         self.get_logger().info(f"[{self.drone_name}] Taking off...")
#         req = Takeoff.Request()
#         req.height = 1.0
#         req.duration.sec = 2
#         self.takeoff_client.call_async(req)
#         time.sleep(3.0)

#         # ============================================================
#         # TIMERS
#         # ============================================================
#         self.timer = self.create_timer(self.dt, self.control_loop)
#         self.create_timer(0.5, self.publish_target_marker)

#         self.get_logger().info(
#             f"DMPC Agent for {self.drone_name} started."
#         )

#     # ============================================================
#     # TF STATE UPDATE
#     # ============================================================
#     def update_state_from_tf(self):
#         try:
#             trans = self.tf_buffer.lookup_transform(
#                 'world',
#                 self.drone_name,
#                 rclpy.time.Time()
#             )

#             px = trans.transform.translation.x
#             py = trans.transform.translation.y
#             pz = trans.transform.translation.z

#             if self.last_pos is not None:
#                 vx = (px - self.last_pos[0]) / self.dt
#                 vy = (py - self.last_pos[1]) / self.dt
#                 vz = (pz - self.last_pos[2]) / self.dt
#             else:
#                 vx, vy, vz = 0.0, 0.0, 0.0

#             self.last_pos = np.array([px, py, pz])
#             self.state = np.array([px, py, pz, vx, vy, vz])

#         except Exception as e:
#             self.get_logger().warn(f"[{self.drone_name}] TF lookup failed: {e}")

#     # ============================================================
#     # NEIGHBOR CALLBACK
#     # ============================================================
#     def prediction_callback(self, msg, drone_name):
#         self.neighbor_predictions[drone_name] = np.array(msg.data)

#     # ============================================================
#     # CONTROL LOOP
#     # ============================================================
#     def control_loop(self):

#         # Mission objective switch
#         current_time = self.get_clock().now().nanoseconds * 1e-9
#         elapsed = current_time - self.start_time

#         if (not self.target_switched) and (elapsed > self.switch_time):
#             self.get_logger().info(
#                 f"[{self.drone_name}] Switching mission objective!"
#             )
#             self.target = self.target_2.copy()
#             self.controller.target_pos = self.target
#             self.target_switched = True

#         # Update state
#         self.update_state_from_tf()

#         # Stack neighbors
#         if self.neighbor_names:
#             neighbors = np.hstack([
#                 self.neighbor_predictions[name]
#                 for name in sorted(self.neighbor_names)
#             ])
#         else:
#             neighbors = np.array([])

#         # Solve DMPC
#         self.controller.compute_control(self.state, neighbors)
#         pred = self.controller.predicted_trajectory
#         target_pos = pred[1, 0:3]

#         # Publish command
#         msg = FullState()
#         msg.pose.position.x = float(target_pos[0])
#         msg.pose.position.y = float(target_pos[1])
#         msg.pose.position.z = float(target_pos[2])
#         msg.twist.linear.x = float(pred[1, 3])
#         msg.twist.linear.y = float(pred[1, 4])
#         msg.twist.linear.z = float(pred[1, 5])
#         msg.pose.orientation.w = 1.0

#         self.cmd_pub.publish(msg)

#         # Publish prediction
#         pred_msg = Float64MultiArray()
#         pred_msg.data = pred.flatten().tolist()
#         self.pred_pub.publish(pred_msg)

#     # ============================================================
#     # TARGET MARKER
#     # ============================================================
#     def publish_target_marker(self):
#         marker = Marker()
#         marker.header.frame_id = "world"
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "target"
#         marker.id = 0
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD

#         marker.pose.position.x = float(self.target[0])
#         marker.pose.position.y = float(self.target[1])
#         marker.pose.position.z = float(self.target[2])
#         marker.pose.orientation.w = 1.0

#         marker.scale.x = 0.2
#         marker.scale.y = 0.2
#         marker.scale.z = 0.2

#         marker.color.a = 1.0
#         marker.color.r = 1.0 if "231" in self.drone_name else 0.0
#         marker.color.g = 1.0 if "232" in self.drone_name else 0.0
#         marker.color.b = 1.0 if "233" in self.drone_name else 0.0

#         self.target_pub.publish(marker)


# def main(args=None):
#     rclpy.init(args=args)
#     node = DMPCAgent()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()