#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time

from std_msgs.msg import Float64MultiArray
from crazyflie_interfaces.msg import FullState
from crazyflie_interfaces.srv import Takeoff
from tf2_ros import Buffer, TransformListener
from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl

class DMPCAgent(Node):

    def __init__(self):
        super().__init__('dmpc_agent')

        # ==============================
        # PARAMETERS
        # ==============================
        self.declare_parameter("drone_name", "cf231")
        self.declare_parameter("neighbor_names", [""])
        self.declare_parameter("num_drones", 1)
        self.declare_parameter("Np", 10)
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("drone_id", 0)

        self.drone_name = self.get_parameter("drone_name").value
        self.neighbor_names = [n for n in self.get_parameter("neighbor_names").value if n.strip() != ""]
        self.num_drones = self.get_parameter("num_drones").value
        self.Np = self.get_parameter("Np").value
        self.dt = self.get_parameter("dt").value
        self.drone_id = self.get_parameter("drone_id").value

        # ==============================
        # TUNED COUPLING PARAMETERS
        # ==============================
        self.last_solve_time = 0.0
        self.solve_interval = 0.1  # Fast re-evaluation to keep tracking stable
        self.last_solution = None

        self.deviation_threshold = 0.15  
        self.trigger_mpc = True  
        self.traj_step = 1
        
        self.MAX_ACC = 1.2
        self.MAX_VEL = 0.8
        self.D_SAFE = 0.5

        self.state = np.zeros(6)
        self.last_pos = None
        self.target = np.array([999.0, 999.0, 999.0])
        self.target_locked = False

        self.neighbor_predictions = {
            name: np.tile(np.zeros(6), self.Np + 1) for name in self.neighbor_names
        }

        # ==============================
        # ROS INFRASTRUCTURE
        # ==============================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

        self.cmd_pub = self.create_publisher(FullState, f'/{self.drone_name}/cmd_full_state', 10)
        self.pred_pub = self.create_publisher(Float64MultiArray, f'/{self.drone_name}/prediction', 10)

        for name in self.neighbor_names:
            self.create_subscription(
                Float64MultiArray,
                f'/{name}/prediction',
                lambda msg, n=name: self.prediction_callback(msg, n),
                10
            )

        self.create_subscription(Float64MultiArray, f'/{self.drone_name}/marl_target', self.target_callback, 10)

        # Takeoff handling sequence
        self.takeoff_client = self.create_client(Takeoff, f'/{self.drone_name}/takeoff')
        while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for takeoff service... /{self.drone_name}")
        
        req = Takeoff.Request()
        req.height = 1.0
        req.duration.sec = 2
        self.takeoff_client.call_async(req)
        time.sleep(3.0)

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("DMPC Node Setup Complete.")

    def update_state(self):
        try:
            trans = self.tf_buffer.lookup_transform('world', self.drone_name, rclpy.time.Time())
            px = trans.transform.translation.x
            py = trans.transform.translation.y
            pz = trans.transform.translation.z

            if self.last_pos is not None:
                vx = (px - self.last_pos[0]) / self.dt
                vy = (py - self.last_pos[1]) / self.dt
                vz = (pz - self.last_pos[2]) / self.dt

                # Low-pass filter for raw velocities
                alpha = 0.6
                vx = alpha * vx + (1 - alpha) * self.state[3]
                vy = alpha * vy + (1 - alpha) * self.state[4]
                vz = alpha * vz + (1 - alpha) * self.state[5]
            else:
                vx, vy, vz = 0.0, 0.0, 0.0

            self.last_pos = np.array([px, py, pz])
            self.state = np.array([px, py, pz, vx, vy, vz])
        except Exception as e:
            self.get_logger().warn(f"Transform processing fault: {e}")

    def target_callback(self, msg):
        new_target = np.array(msg.data)
        if not np.allclose(new_target, self.target, atol=0.01):
            self.target = new_target
            self.controller.target_pos = self.target
            self.trigger_mpc = True  
            self.target_locked = True 

    def should_solve(self):
        if self.trigger_mpc:
            self.trigger_mpc = False
            return True

        if self.last_solution is not None:
            idx = min(self.traj_step, self.Np)
            if np.linalg.norm(self.state[:3] - self.last_solution[idx, :3]) > self.deviation_threshold:
                return True

        if (time.time() - self.last_solve_time) > self.solve_interval:
            return True
        
        if self.traj_step > self.Np:
            return True

        # Inter-agent safety proximity trigger
        for name in self.neighbor_names:
            traj = self.neighbor_predictions[name]
            if traj is not None:
                idx = min(self.traj_step, self.Np)
                other_pos = traj.reshape(self.Np + 1, 6)[idx, 0:3]
                if np.linalg.norm(self.state[:3] - other_pos) < (self.D_SAFE + 0.15):
                    return True
        return False

    def prediction_callback(self, msg, name):
        self.neighbor_predictions[name] = np.array(msg.data)

    def control_loop(self):
        self.update_state()

        if not self.target_locked:
            return

        dist = np.linalg.norm(self.state[:3] - self.target)

        # =====================================================
        # TARGET ENVELOPE SETTLEMENT
        # =====================================================
        if dist < 0.15: 
            self.target_locked = False
            self.last_solution = None
            
            # Publish an active zero-velocity hold command to prevent drift 
            stop_msg = FullState()
            stop_msg.pose.position.x = self.state[0]
            stop_msg.pose.position.y = self.state[1]
            stop_msg.pose.position.z = 1.0  # Safe hold altitude
            stop_msg.pose.orientation.w = 1.0
            self.cmd_pub.publish(stop_msg)
            return

        # Stack predictions for the DMPC controller
        neighbors = np.hstack([self.neighbor_predictions[n] for n in sorted(self.neighbor_names)]) if self.neighbor_names else np.array([])

        if self.should_solve():
            self.controller.target_pos = self.target
            self.controller.target_state = np.hstack([self.target, np.zeros(3)])
            self.controller.compute_control(self.state, neighbors)

            pred = self.controller.predicted_trajectory
            if pred is not None and pred.shape[0] > 0:
                self.last_solution = pred
                self.last_solve_time = time.time()
                self.traj_step = 1
            else:
                self.get_logger().warn("DMPC failed optimization. Reusing fallback trajectory profile.")

        if self.last_solution is None:
            return

        idx = min(self.traj_step, self.Np)
        self.traj_step += 1   
        if self.traj_step > self.Np:
            self.trigger_mpc = True     

        # Build command message
        msg = FullState()
        msg.pose.position.x = float(self.last_solution[idx, 0])
        msg.pose.position.y = float(self.last_solution[idx, 1])
        msg.pose.position.z = float(self.last_solution[idx, 2])
        msg.twist.linear.x = float(self.last_solution[idx, 3])
        msg.twist.linear.y = float(self.last_solution[idx, 4])
        msg.twist.linear.z = float(self.last_solution[idx, 5])
        msg.pose.orientation.w = 1.0
        self.cmd_pub.publish(msg)

        # Publish local predictive trajectory step mapping
        pred_msg = Float64MultiArray()
        pred_msg.data = self.last_solution.flatten().tolist()
        self.pred_pub.publish(pred_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DMPCAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()