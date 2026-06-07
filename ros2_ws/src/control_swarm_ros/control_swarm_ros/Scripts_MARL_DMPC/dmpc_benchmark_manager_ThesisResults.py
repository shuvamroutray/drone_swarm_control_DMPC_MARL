#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import csv
import os
import time

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Empty


class DMPCBenchmarkManager(Node):

    def __init__(self):

        super().__init__("dmpc_benchmark_manager")

        self.DRONES = [
            "cf231",
            "cf232"
            # "cf233",
            # "cf234"
        ]

        self.NUM_SCENARIOS = 1

        self.GOAL_RANGE = 2.0

        self.declare_parameter("scenario_seed", 0)

        self.seed = self.get_parameter(
            "scenario_seed"
        ).value



        self.done_flags = {
            d: False
            for d in self.DRONES
        }

        self.goal_publishers = {}

        for drone in self.DRONES:

            self.goal_publishers[drone] = (
                self.create_publisher(
                    Float64MultiArray,
                    f"/{drone}/marl_target",
                    10
                )
            )

            self.create_subscription(
                Empty,
                f"/{drone}/mission_done",
                lambda msg,
                drone=drone:
                self.done_callback(drone),
                10
            )

        self.csv_file = open(
            "benchmark_results.csv",
            "w",
            newline=""
        )

        self.writer = csv.writer(
            self.csv_file
        )

        self.writer.writerow([
            "scenario",
            "drone",
            "solve_count",
            "avg_solve_ms",
            "max_solve_ms",
            "min_solve_ms",
            "total_optimization_time_ms",
            "path_length",
            "min_separation",
            "mission_time"
        ])

        # self.run_benchmark()

    # =====================================================
    # DONE CALLBACK
    # =====================================================

    def done_callback(self, drone):

        self.done_flags[drone] = True

        self.get_logger().info(
            f"{drone} completed mission"
        )

    # =====================================================
    # RANDOM GOAL GENERATOR
    # =====================================================

    def generate_goals(self, seed):

        np.random.seed(seed)

        goals = {}

        accepted = []

        for drone in self.DRONES:

            while True:

                goal = np.array([
                    np.random.uniform(
                        -self.GOAL_RANGE,
                        self.GOAL_RANGE
                    ),
                    np.random.uniform(
                        -self.GOAL_RANGE,
                        self.GOAL_RANGE
                    ),
                    1.0
                ])

                valid = True

                for g in accepted:

                    if np.linalg.norm(
                        goal[:2] - g[:2]
                    ) < 0.75:

                        valid = False
                        break

                if valid:

                    accepted.append(goal)

                    goals[drone] = goal

                    break

        return goals

    # =====================================================
    # PUBLISH GOALS
    # =====================================================

    def publish_goals(self, goals):

        for drone, goal in goals.items():

            msg = Float64MultiArray()

            msg.data = goal.tolist()

            self.goal_publishers[
                drone
            ].publish(msg)

    # =====================================================
    # WAIT FOR COMPLETION
    # =====================================================

    def wait_until_complete(self):

        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            if all(
                self.done_flags.values()
            ):
                return

    # =====================================================
    # READ METRICS
    # =====================================================

    def read_metrics(self, scenario):

        for drone in self.DRONES:

            filename = (
                f"/tmp/"
                f"{drone}_benchmark.txt"
            )

            if not os.path.exists(
                filename
            ):
                continue

            metrics = {}

            with open(
                filename,
                "r"
            ) as f:

                for line in f:

                    key, value = (
                        line.strip()
                        .split("=")
                    )

                    metrics[key] = value

            self.writer.writerow([
                scenario,
                drone,
                metrics.get(
                    "solve_count", 0
                ),
                metrics.get(
                    "avg_solve_ms", 0
                ),
                metrics.get(
                    "max_solve_ms", 0
                ),
                metrics.get(
                    "min_solve_ms", 0
                ),
                metrics.get(
                    "total_optimization_time_ms",
                    0
                ),
                metrics.get(
                    "path_length", 0
                ),
                metrics.get(
                    "min_separation", 0
                ),
                metrics.get(
                    "mission_time", 0
                )
            ])

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run_benchmark(self):


        self.get_logger().info(
            "Starting benchmark..."
        )

        self.get_logger().error(
            f"RUNNING BENCHMARK VERSION - NUM_SCENARIOS={self.NUM_SCENARIOS}"
        )

        self.get_logger().fatal(
            "THESIS_BENCHMARK_V7_SINGLE_SCENARIO"
        )       

        time.sleep(5)

        # for scenario in range(
        #     self.NUM_SCENARIOS
        # ):

        #     self.get_logger().info(
        #         f"Scenario {scenario+1}"
        #     )

        self.done_flags = {
            d: False
            for d in self.DRONES
        }

        goals = self.generate_goals(
            self.seed
        )

        self.publish_goals(
            goals
        )

        self.wait_until_complete()

        self.read_metrics(
            self.seed
        )

        self.csv_file.flush()

        self.csv_file.close()

        self.get_logger().info(
            "Scenario complete"
        )

        # rclpy.shutdown()

        # import signal
        # import os

        # self.get_logger().info(
        #     "Scenario complete - terminating launch"
        # )

        # os.kill(
        #     os.getppid(),
        #     signal.SIGINT
        # )

        self.get_logger().info(
            "Scenario complete"
        )

        with open("/tmp/benchmark_done", "w") as f:
            f.write("done")


def main(args=None):

    rclpy.init(args=args)

    node = DMPCBenchmarkManager()

    node.run_benchmark()

    node.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()