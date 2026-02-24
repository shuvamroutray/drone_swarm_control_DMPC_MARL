# Here we are creating one drone one node architecture. Every drone 
# has its own control node and other relavant subcriptions and publisher.
# Its decentralized execution where every drone has its independent controller.


# Date created: 20 Feb 2026
# Creator: Shuvam Routray



#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

import numpy as np

from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl


class DMPCAgent(Node):

    def __init__(self):
        super().__init__('dmpc_agent')

        # =============================
        # PARAMETERS
        # =============================
        self.declare_parameter("drone_id", 0)
        self.declare_parameter("num_drones", 2)
        self.declare_parameter("Np", 50)
        self.declare_parameter("dt", 1.0/30.0)
        self.declare_parameter("target",[0.0, 1.5, 1.0])

        self.drone_id = self.get_parameter("drone_id").value
        self.num_drones = self.get_parameter("num_drones").value
        self.Np = self.get_parameter("Np").value
        self.dt = self.get_parameter("dt").value
        self.target = np.array(
            self.get_parameter("target").value,
            dtype=float
        )

        self.get_logger().info(f"Target: {self.target}")

        self.MAX_ACC = 5.0
        self.MAX_VEL = 2.0
        self.D_SAFE = 0.5

        #self.target = np.array([0.0, 1.5, 1.0])  # Can parametrize later

        # =============================
        # STATE
        # =============================
        self.state = np.zeros(6)
        self.neighbor_predictions = {}

        for i in range(self.num_drones):
            if i != self.drone_id:
                self.neighbor_predictions[i] = np.zeros(6*(self.Np+1))

        # =============================
        # DMPC CONTROLLER
        # =============================
        self.controller = DMPCControl(
            drone_id=self.drone_id,
            Np=self.Np,
            dt=self.dt,
            target_pos=self.target,
            max_acc=self.MAX_ACC,
            max_vel=self.MAX_VEL,
            d_safe=self.D_SAFE,
            num_drones=self.num_drones
        )

        # =============================
        # SUBSCRIBE TO OWN ODOM
        # =============================
        self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # =============================
        # PUBLISH VELOCITY
        # =============================
        self.cmd_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # =============================
        # PUBLISH PREDICTION
        # =============================
        self.pred_pub = self.create_publisher(
            Float64MultiArray,
            'prediction',
            10
        )

        # =============================
        # SUBSCRIBE TO NEIGHBOR PREDICTIONS
        # =============================
        for i in range(self.num_drones):
            if i != self.drone_id:
                topic = f'/cf{i+1}/prediction'
                self.create_subscription(
                    Float64MultiArray,
                    topic,
                    lambda msg, i=i: self.prediction_callback(msg, i),
                    10
                )

        # =============================
        # TIMER
        # =============================
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"DMPC Agent {self.drone_id} started in namespace {self.get_namespace()}"
        )

    # =============================
    # CALLBACKS
    # =============================

    def odom_callback(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        self.state = np.array([px, py, pz, vx, vy, vz])

    def prediction_callback(self, msg, drone_id):
        self.neighbor_predictions[drone_id] = np.array(msg.data)

    # =============================
    # CONTROL LOOP
    # =============================

    def control_loop(self):

        #self.get_logger().info("Control loop running")
        #self.get_logger().info(f"State: {self.state}")

        if len(self.neighbor_predictions) < self.num_drones - 1:
            return  # wait for neighbors

        neighbors = np.hstack([
            self.neighbor_predictions[i]
            for i in sorted(self.neighbor_predictions.keys())
        ]) if self.num_drones > 1 else np.array([])

        u = self.controller.compute_control(self.state, neighbors)

        pred = self.controller.predicted_trajectory

        # Publish velocity
        target_vel = pred[1, 3:6]
        msg = Twist()
        msg.linear.x = float(target_vel[0])
        msg.linear.y = float(target_vel[1])
        msg.linear.z = float(target_vel[2])
        self.cmd_pub.publish(msg)

        # Publish predicted trajectory
        pred_msg = Float64MultiArray()
        pred_msg.data = pred.flatten().tolist()


        
        
        self.get_logger().info(f"Publishing vel: {target_vel}")
        self.get_logger().info(f"Computed control u: {u}")
        #self.get_logger().info(f"Predicted vel: {target_vel}")
        
        
        
        self.pred_pub.publish(pred_msg)


    # def control_loop(self):
    #     msg = Twist()
    #     msg.linear.x = 0.5
    #     msg.linear.y = 0.0
    #     self.cmd_pub.publish(msg)
        



def main(args=None):
    rclpy.init(args=args)
    node = DMPCAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()