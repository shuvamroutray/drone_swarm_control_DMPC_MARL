#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import time

from std_msgs.msg import Float64MultiArray, Empty
from crazyflie_interfaces.msg import FullState
from crazyflie_interfaces.srv import Takeoff

from std_msgs.msg import Empty

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from tf2_ros import Buffer, TransformListener

import sys
sys.path.append(
    "/home/shuvam/6_Drone_Swarm_DMPC_MARL/Control_Swarm/src"
)

from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl

"""
Extension of swarm_node_crazy_SIM_MARL_EventTrigger_2.py with MARL integrated. Target comes from MARL node.
"""


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


        self.declare_parameter(
            "scenario_seed",
            0
        )


        self.drone_name = self.get_parameter("drone_name").value
        self.neighbor_names = [
            n for n in self.get_parameter("neighbor_names").value
            if n.strip() != ""
        ]
        self.num_drones = self.get_parameter("num_drones").value
        self.Np = self.get_parameter("Np").value
        self.dt = self.get_parameter("dt").value
        self.drone_id = self.get_parameter("drone_id").value

        self.get_logger().info(
            f"Drone={self.drone_name}, neighbors={self.neighbor_names}, drone_id={self.drone_id}"
        )

        self.scenario_seed = self.get_parameter(
            "scenario_seed"
        ).value

        

        # ==============================
        # EVENT TRIGGER RELATED PARAMETERS
        # ==============================


        self.solve_count = 0
        self.solve_times = []

        self.min_separation = 999.0

        self.path_length = 0.0

        self.last_position_for_path = None

        # ==========================================
        # BENCHMARK METRICS
        # ==========================================

        self.total_optimization_time = 0.0

        self.mission_start_time = time.time()
        self.mission_end_time = None

        self.target_reached = False







        self.last_solve_time = 0.0
        self.solve_interval = 0.5   # fallback
       
        self.last_solution = None

        self.trigger_threshold = 0.15
        self.deviation_threshold = 0.2  
        

        self.trigger_mpc = True  
        self.traj_step = 1
        
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

        # self.target = np.array([0.0, 0.0, 1.0])
        # goal = np.array(
        #     self.get_parameter(
        #         "goal_position"
        #     ).value
        # )

        # self.target = goal
        # self.controller.target_pos = goal
        # self.target_locked = True

        # scenario = self.generate_scenario()

        # my_data = scenario[
        #     self.drone_name
        # ]

        # self.target = (
        #     my_data["goal"]
        # )


        self.target = self.generate_goal()

        self.get_logger().info(
            f"GOAL={np.round(self.target,2)}"
        )



        self.target_locked = True

        # ==============================
        # NEIGHBORS
        # ==============================
        # self.neighbor_predictions = {
        #     name: None for name in self.neighbor_names
        # }

        self.neighbor_predictions = {
            name: np.tile(np.zeros(6), self.Np + 1)
            for name in self.neighbor_names
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
            #drone_id=0,
            drone_id=self.drone_id,
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

        from std_msgs.msg import Empty

        # self.mission_done_pub = self.create_publisher(
        #     Empty,
        #     f"/{self.drone_name}/mission_done",
        #     10
        # )

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

        # self.create_subscription(
        #     Float64MultiArray,
        #     f'/{self.drone_name}/marl_target',
        #     self.target_callback,
        #     10
        # )

        # ==============================
        # TAKEOFF
        # ==============================
        self.takeoff_client = self.create_client(
            Takeoff,
            f'/{self.drone_name}/takeoff'
        )
        self.get_logger().info("Taking off...")
        req = Takeoff.Request()
        req.height = 1.0
        req.duration.sec = 2
        # self.takeoff_client.call_async(req)
        # time.sleep(3)

        while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for /{self.drone_name}/takeoff...")

        future = self.takeoff_client.call_async(req)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=10.0
        )

        if future.result() is None:
            self.get_logger().error(
                "Takeoff failed!"
            )
            return

        self.get_logger().info(
            "Takeoff successful"
        )

        time.sleep(3)

        # ==============================
        # TIMER
        # ==============================
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("DMPC node initialized successfully")

    # ==========================================
    # STATE UPDATE
    # ==========================================

    def generate_goal(self):

        rng = np.random.default_rng(
            self.scenario_seed
        )

        goal1 = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            1.0
        ])

        goal2 = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            1.0
        ])

        while np.linalg.norm(
            goal1[:2] - goal2[:2]
        ) < 1.0:

            goal2 = np.array([
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                1.0
            ])

        goals = {
            "cf231": goal1,
            "cf232": goal2
        }

        return goals[self.drone_name]

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

            if self.last_position_for_path is not None:

                self.path_length += np.linalg.norm(
                    self.state[:3]
                    -
                    self.last_position_for_path
                )

            self.last_position_for_path = (
                self.state[:3].copy()
            )


        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")

   


    # ==========================================
    # TRIGGER FUNCTION FOR DMPC
    # ==========================================
    def should_solve(self):

        # 1. Target changed
        if self.trigger_mpc:
            self.trigger_mpc = False
            return True

        # 2. State deviation
        if self.last_solution is not None:
            
            idx = min(self.traj_step, self.Np)
            pred_state = self.last_solution[idx, :3]
            actual = self.state[:3]

            if np.linalg.norm(actual - pred_state) > self.deviation_threshold:
                return True

        # 3. Time fallback
        if (time.time() - self.last_solve_time) > self.solve_interval:
            return True
        
        if self.traj_step > self.Np:
            return True
        

        # 4. Safety trigger
        if self.neighbor_names:
            for name in self.neighbor_names:
                traj = self.neighbor_predictions[name]
                if traj is not None:

                    idx = min(self.traj_step, self.Np)
                    traj_reshaped = traj.reshape(self.Np+1, 6)
                    other_pos = traj_reshaped[idx, 0:3]
                    dist = np.linalg.norm(self.state[:3] - other_pos)

                    if dist < self.D_SAFE + 0.2:
                        return True

        return False

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

        if not self.target_locked:
            return

        dist = np.linalg.norm(self.state[:3] - self.target)

        # TARGET REACHED
        if dist < 0.25 and np.linalg.norm(self.state[3:6]) < 0.10:

            if not self.target_reached:

                self.target_reached = True

                self.mission_end_time = time.time()

                self.save_stats()
                # done_msg = Empty()
                # self.mission_done_pub.publish(done_msg)

            self.target_locked = False
            self.last_solution = None

            return

        # NEIGHBORS
        if self.neighbor_names:
            if any(self.neighbor_predictions[n] is None for n in self.neighbor_names):
                self.get_logger().warn(
                    f"Waiting for neighbors: {self.neighbor_predictions}"
                )

                return

            neighbors = np.hstack([
                self.neighbor_predictions[n]
                for n in sorted(self.neighbor_names)
            ])
        else:
            neighbors = np.array([])

        # EVENT TRIGGER
        solve_now = True

        if solve_now:

            self.controller.target_pos = self.target
            self.controller.target_state = np.hstack([self.target, np.zeros(3)])

            start = time.perf_counter()

            self.controller.compute_control(
                self.state,
                neighbors
            )

            solve_time = (
                time.perf_counter() - start
            )

            # self.solve_times.append(
            #     solve_time
            # )

            # self.solve_count += 1



            pred = self.controller.predicted_trajectory

            if pred is not None and pred.shape[0] > 0:

                self.last_solution = pred

                print(
                    f"\n[{self.drone_name}] CURRENT = "
                    f"{np.round(self.state[:3],3)}"
                )

                print(
                    f"[{self.drone_name}] STEP1 = "
                    f"{np.round(pred[1,0:3],3)}"
                )

                print(
                    f"[{self.drone_name}] STEP2 = "
                    f"{np.round(pred[2,0:3],3)}"
                )

                print(
                    f"[{self.drone_name}] STEP5 = "
                    f"{np.round(pred[5,0:3],3)}"
                )

                print(
                    f"[{self.drone_name}] GOAL  = "
                    f"{np.round(self.target,3)}\n"
                )

                print(
                    "Pred shape:",
                    pred.shape
                )

                self.last_solve_time = time.time()
                self.traj_step = 1

                self.solve_times.append(
                    solve_time
                )

                self.solve_count += 1

                self.total_optimization_time += solve_time


            else:
                self.get_logger().warn("MPC solve failed, reusing last trajectory")

        # REUSE TRAJECTORY
        if self.last_solution is None:
            return

        #pred = self.last_solution

        idx = 1
        target_pos = self.last_solution[idx, 0:3]      


     

        for name in self.neighbor_names:

            traj = self.neighbor_predictions[name]

            if traj is None:
                continue

            other = traj.reshape(
                self.Np+1,
                6
            )[0,:3]

            d = np.linalg.norm(
                self.state[:3]
                -
                other
            )
            self.min_separation = min(
                self.min_separation,
                d
            )  

        msg = FullState()

        msg.pose.position.x = float(self.last_solution[idx, 0])
        msg.pose.position.y = float(self.last_solution[idx, 1])
        msg.pose.position.z = float(self.last_solution[idx, 2])

        msg.twist.linear.x = float(self.last_solution[idx, 3])
        msg.twist.linear.y = float(self.last_solution[idx, 4])
        msg.twist.linear.z = float(self.last_solution[idx, 5])

        msg.pose.orientation.w = 1.0

        self.cmd_pub.publish(msg)

        # publish prediction
        pred_msg = Float64MultiArray()
        pred_msg.data = self.last_solution.flatten().tolist()
        self.pred_pub.publish(pred_msg)

        self.get_logger().info(
            f"[{self.drone_name}] → {np.round(target_pos,2)} | dist={dist:.2f}"
        )


        if (
            time.time()
            - self.mission_start_time
        ) > 120:

            self.get_logger().error(
                "MISSION TIMEOUT"
            )

            with open(
                f"/tmp/{self.drone_name}_failed",
                "w"
            ) as f:
                f.write("timeout")

            self.mission_end_time = time.time()

            self.save_stats()

            self.target_reached = True

            return





    def save_stats(self):

        avg_solve = (
            np.mean(self.solve_times)
            if len(self.solve_times) > 0
            else 0.0
        )

        max_solve = (
            np.max(self.solve_times)
            if len(self.solve_times) > 0
            else 0.0
        )

        min_solve = (
            np.min(self.solve_times)
            if len(self.solve_times) > 0
            else 0.0
        )

        mission_time = (
            self.mission_end_time
            -
            self.mission_start_time
            if self.mission_end_time is not None
            else 0.0
        )

        filename = (
            f"/tmp/{self.drone_name}_benchmark.txt"
        )

        with open(filename, "w") as f:

            f.write("THESIS_VERSION_2:PeriodicDMPC\n")

            f.write(f"scenario_seed={self.scenario_seed}\n")

            f.write(
                f"drone={self.drone_name}\n"
            )

            f.write(
                f"solve_count={self.solve_count}\n"
            )

            f.write(
                f"avg_solve_ms="
                f"{avg_solve*1000:.4f}\n"
            )

            f.write(
                f"max_solve_ms="
                f"{max_solve*1000:.4f}\n"
            )

            f.write(
                f"min_solve_ms="
                f"{min_solve*1000:.4f}\n"
            )

            f.write(
                f"total_optimization_time_ms="
                f"{self.total_optimization_time*1000:.4f}\n"
            )

            f.write(
                f"path_length="
                f"{self.path_length:.4f}\n"
            )

            f.write(
                f"min_separation="
                f"{self.min_separation:.4f}\n"
            )

            f.write(
                f"mission_time="
                f"{mission_time:.4f}\n"
            )

            f.write(
                f"replans_per_second="
                f"{self.solve_count/max(mission_time,1e-6):.4f}\n"
            )

            f.write(
                f"goal_x={self.target[0]:.4f}\n"
            )

            f.write(
                f"goal_y={self.target[1]:.4f}\n"
            )

        self.get_logger().info(
            f"Saved benchmark stats to {filename}"
        )

        with open(
            f"/tmp/{self.drone_name}_done",
            "w"
        ) as f:
            f.write("done")


def main(args=None):
    rclpy.init(args=args)
    node = DMPCAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()