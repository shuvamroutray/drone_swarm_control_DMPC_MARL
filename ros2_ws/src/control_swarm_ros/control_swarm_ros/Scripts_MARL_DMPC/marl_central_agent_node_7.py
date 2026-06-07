#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64MultiArray

from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose

# from control_swarm_ros.Scripts_MARL_DMPC.policy_loader import PolicyLoader
# from control_swarm_ros.Scripts_MARL_DMPC.marl_utils import (
#     ACTIONS,
#     build_observation,
# )

from policy_loader import PolicyLoader
from marl_utils import ACTIONS, build_observation

# =========================================================
# CENTERED WORLD <-> GRID TRANSFORMS
# =========================================================
# def world_to_grid(x, y, scale, grid_size):
#     offset = (grid_size * scale) / 2.0

#     gx = int((x + offset) / scale)
#     gy = int((y + offset) / scale)

#     gx = np.clip(gx, 0, grid_size - 1)
#     gy = np.clip(gy, 0, grid_size - 1)

#     return gx, gy

def world_to_grid(x, y, scale, grid_size):

    offset = (grid_size * scale) / 2.0

    gx = int((x + offset) / scale)
    gy = int((y + offset) / scale)

    gx = np.clip(gx, 0, grid_size - 1)
    gy = np.clip(gy, 0, grid_size - 1)

    return gx, gy


def grid_to_world(gx, gy, scale, grid_size):

    offset = (grid_size * scale) / 2.0

    
    wx = gx * scale - offset + scale / 2.0
    wy = gy * scale - offset + scale / 2.0
    

    return wx, wy


# =========================================================
# CENTRALIZED NODE
# =========================================================
class CentralizedMARLNode(Node):

    def __init__(self):
        super().__init__("centralized_marl_agent")

        # -----------------------------------------
        # PARAMETERS
        # -----------------------------------------
        self.grid_size = 6
        self.scale = 1.0
        self.num_drones = 2

        self.drone_names = ["cf231", "cf232"]

        # -----------------------------------------
        # SHARED GRID
        # -----------------------------------------
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Heatmap visit count
        self.visit_count = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.int32
        )
        # positions in grid coordinates
        self.positions = {
            0: (0.0, 0.0),
            1: (0.0, 0.0)
        }

        self.recent_positions = {
            0: [],
            1: []
        }

        self.odom_received = {
            0: False,
            1: False
        }

        self.positions_grid = {
            0: (0, 0),
            1: (0, 0)
        }

        self.current_targets = {
            0: None,
            1: None
        }

        self.target_reached = {
            0: True,
            1: True
        }

        self.last_targets = {
            0: None,
            1: None
        }

        self.anomaly_discoverer = None
        self.anomaly_resolver = None


        # Internal true world state (for debug only)
        self.true_grid = np.zeros((self.grid_size, self.grid_size))

        
        self.detected = False      


        # Random hidden anomaly
        while True:
            ax = np.random.randint(0, self.grid_size)
            ay = np.random.randint(0, self.grid_size)

            # Avoid spawning on drone start cells
            # if (ay, ax) not in self.positions_grid.values():
            if (ax, ay) not in [(3,5), (3,0)]:
                break

        
        ax = 0
        ay = 2
        self.anomaly_pos = (ax, ay)

        # Hidden anomaly ONLY in true grid
        self.true_grid[ax, ay] = 2

        self.get_logger().info(
            f"Hidden anomaly generated at {self.anomaly_pos}"
        )


        # -----------------------------------------
        # POLICY
        # -----------------------------------------
        checkpoint_path = (
            "/home/shuvam/6_Drone_Swarm_DMPC_MARL/"
            "Control_Swarm/experiments/exp_marl/"
            "checkpoints_rllib4_6X6_2Agents"
        )

        # checkpoint_path = (
        #     "/home/shuvam/6_Drone_Swarm_DMPC_MARL/"
        #     "Control_Swarm/experiments/exp_marl/"
        #     "checkpoints"
        # )

       

        self.policy = PolicyLoader(
            checkpoint_path=checkpoint_path,
            grid_size=self.grid_size,
            n_agents=self.num_drones
        )

        # -----------------------------------------
        # SUBSCRIBERS
        # -----------------------------------------

        self.create_subscription(
            TFMessage,
            "/tf",
            self.tf_callback,
            50
        )


        # -----------------------------------------
        # PUBLISHERS
        # -----------------------------------------
        self.target_publishers = {}

        for drone_id, drone_name in enumerate(self.drone_names):

            self.target_publishers[drone_id] = self.create_publisher(
                Float64MultiArray,
                f"/{drone_name}/marl_target",
                10
            )

        # -------------------------------------------------
        # RVIZ VISUALIZATION PUBLISHERS
        # -------------------------------------------------

        # Anomaly sphere marker
        self.anomaly_marker_pub = self.create_publisher(
            Marker,
            "/anomaly_marker",
            10
        )

        # Exploration heatmap
        self.heatmap_pub = self.create_publisher(
            OccupancyGrid,
            "/marl_heatmap",
            10
        )

        # Scan area boundary
        self.scan_area_pub = self.create_publisher(
            Marker,
            "/scan_area",
            10
        )

        # -----------------------------------------
        # TIMER
        # -----------------------------------------
        self.timer = self.create_timer(3.0, self.run_policy)

        self.get_logger().info("Centralized MARL node initialized")

    # =====================================================
    # TF CALLBACK
    # =====================================================
   
    def tf_callback(self, msg):

        for transform in msg.transforms:

            drone_name = transform.child_frame_id

            if drone_name not in self.drone_names:
                continue

            drone_id = self.drone_names.index(drone_name)

            x = transform.transform.translation.x
            y = transform.transform.translation.y

            gx, gy = world_to_grid(
                x,
                y,
                self.scale,
                self.grid_size
            )

            self.positions[drone_id] = (x, y)
            self.positions_grid[drone_id] = (gx, gy)

            self.odom_received[drone_id] = True


    def has_reached_target(self, drone_id):

        if self.current_targets[drone_id] is None:
            return True

        tx, ty = self.current_targets[drone_id]

        gx, gy = self.positions_grid[drone_id]

        dist = abs(gx - tx) + abs(gy - ty)

        return dist <= 1

    # =====================================================
    # UPDATE SHARED GRID
    # =====================================================
    def update_grid(self):

        # reset dynamic agent markers
        self.grid[self.grid == 9] = 1
        self.grid[self.grid == 10] = 1
        
        self.true_grid[self.true_grid == 9] = 1
        self.true_grid[self.true_grid == 10] = 1


        # ax, ay = self.anomaly_pos

        # If anomaly cell not yet explored
        # if self.grid[ax, ay] == 0:
        #     self.grid[ax, ay] = 2

        # mark explored + agents
        for drone_id, (gx, gy) in self.positions_grid.items():

            # Update heatmap visitation
            self.visit_count[gx, gy] += 1

            if self.grid[gx, gy] not in [3,4]:
                self.grid[gx, gy] = 1

            if self.true_grid[gx, gy] not in [2,3,4]:
                self.true_grid[gx, gy] = 1

            if (gx, gy) == self.anomaly_pos:

                # First discovery
                if not self.detected:
                    self.detected = True

                    self.anomaly_discoverer = drone_id

                    # Assign other drone as resolver
                    self.anomaly_resolver = next(
                        i for i in range(self.num_drones)
                        if i != drone_id
                    )

                    # Visible only after detection
                    self.grid[gx, gy] = 3
                    self.true_grid[gx, gy] = 3

                    self.get_logger().info(
                        f"Agent {drone_id} DETECTED anomaly!"
                    )

                # Resolution
                # -------------------------------------------------
                # ONLY RESOLVER CAN RESOLVE ANOMALY
                # -------------------------------------------------
                elif (
                    self.grid[gx, gy] == 3
                    and drone_id == self.anomaly_resolver
                ):

                    self.grid[gx, gy] = 4
                    self.true_grid[gx, gy] = 4

                    self.get_logger().info(
                        f"Agent {drone_id} RESOLVED anomaly!"
                    )

            self.recent_positions[drone_id].append((gx, gy))

            if len(self.recent_positions[drone_id]) > 6:
                self.recent_positions[drone_id].pop(0)

            # -------------------------------------------------
            # DO NOT OVERWRITE ACTIVE ANOMALY CELLS
            # -------------------------------------------------

            if self.grid[gx, gy] not in [3, 4]:

                if drone_id == 0:
                    self.grid[gx, gy] = 9
                    self.true_grid[gx, gy] = 9

                elif drone_id == 1:
                    self.grid[gx, gy] = 10
                    self.true_grid[gx, gy] = 10

            ax, ay = self.anomaly_pos
            if self.true_grid[ax, ay] == 4:
                self.anomaly_discoverer = None
                self.anomaly_resolver = None


    # =====================================================
    # ANOMALY MARKER
    # =====================================================
    def publish_anomaly_marker(self):

        ax, ay = self.anomaly_pos

        wx, wy = grid_to_world(
            ax,
            ay,
            self.scale,
            self.grid_size
        )

        marker = Marker()

        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "anomaly"
        marker.id = 0

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = wx
        marker.pose.position.y = wy
        marker.pose.position.z = 0.2

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.45
        marker.scale.y = 0.45
        marker.scale.z = 0.45

        # BEFORE DETECTION
        if not self.detected:

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.15

        # AFTER RESOLUTION
        elif self.true_grid[ax, ay] == 4:

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

        # AFTER DETECTION
        else:

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

        self.anomaly_marker_pub.publish(marker)


    # =====================================================
    # HEATMAP
    # =====================================================
    def publish_heatmap(self):

        msg = OccupancyGrid()

        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.info.resolution = self.scale
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size

        offset = (self.grid_size * self.scale) / 2.0

        msg.info.origin.position.x = -offset
        msg.info.origin.position.y = -offset
        msg.info.origin.position.z = 0.0

        max_visit = np.max(self.visit_count)

        if max_visit == 0:
            max_visit = 1

        data = []

        for y in range(self.grid_size):

            for x in range(self.grid_size):

                val = self.visit_count[x, y]

                # Normalize to 0-100
                # heat = int((val / max_visit) * 100)

                # heat = min(100, int((val ** 0.6) * 35))
                heat = min(100, 25 + int((val ** 0.7) * 30))

                data.append(heat)

        msg.data = data

        self.heatmap_pub.publish(msg)

    # =====================================================
    # SCAN AREA BOUNDARY
    # =====================================================
    def publish_scan_area(self):

        marker = Marker()

        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "scan_area"
        marker.id = 0

        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        size = self.grid_size * self.scale

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = -0.01

        marker.pose.orientation.w = 1.0

        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = 0.02

        # Transparent blue
        marker.color.r = 0.0
        marker.color.g = 0.2
        marker.color.b = 1.0
        marker.color.a = 0.15

        self.scan_area_pub.publish(marker)

    # =====================================================
    # MAIN POLICY LOOP
    # =====================================================
    def run_policy(self):

        # wait until all drones available
        if not all(self.odom_received.values()):
            self.get_logger().info("Waiting for all TF...")
            return

        # update shared environment
        self.update_grid()

        # -------------------------------------------------
        # RVIZ VISUALIZATION
        # -------------------------------------------------

        self.publish_anomaly_marker()

        self.publish_heatmap()

        self.publish_scan_area()

        self.get_logger().info(
            f"DEBUG True grid:\n{self.true_grid}\nPositions={self.positions_grid}"
        )

        if (
            not np.any(self.grid == 0)
            and not np.any(self.grid == 3)
            and not np.any(self.true_grid == 2)
        ):
            self.get_logger().info(
                "MISSION COMPLETE: FULL REGION SCANNED + ALL ANOMALIES RESOLVED"
            )

            self.timer.cancel()
            return
        

        

        # -------------------------------------
        # RUN POLICY FOR EACH AGENT
        # -------------------------------------

        reserved_targets = set()
        for drone_id in range(self.num_drones):


            if not self.target_reached[drone_id]:

                if self.has_reached_target(drone_id):

                    self.target_reached[drone_id] = True

                    self.get_logger().info(
                        f"[Agent {drone_id}] Target reached."
                    )

                else:
                    continue

            obs = build_observation(
                self.grid,
                self.positions_grid,
                drone_id,
                self.grid_size
            )

            action = self.policy.predict(obs)

            dx, dy = ACTIONS[action]

            gx, gy = self.positions_grid[drone_id]

                        # Other drone occupied cells
            other_positions = {
                pos for other_id, pos in self.positions_grid.items()
                if other_id != drone_id
            }


            
            if self.detected and np.any(self.grid == 3):

                ax, ay = self.anomaly_pos

                # =====================================================
                # DISCOVERER MOVES AWAY
                # =====================================================
                if drone_id == self.anomaly_discoverer:

                    valid_moves = []

                    for alt_action, (adx, ady) in ACTIONS.items():

                        nx = gx + adx
                        ny = gy + ady

                        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                            continue

                        if (nx, ny) in other_positions:
                            continue

                        # maximize distance from anomaly
                        dist = abs(nx - ax) + abs(ny - ay)

                        explore_bonus = -5 if self.grid[nx, ny] == 0 else 0

                        score = -dist + explore_bonus

                        valid_moves.append((score, nx, ny))

                    if len(valid_moves) > 0:

                        valid_moves.sort()

                        _, target_gx, target_gy = valid_moves[0]

                        tx, ty = grid_to_world(
                            target_gx,
                            target_gy,
                            self.scale,
                            self.grid_size
                        )

                        msg = Float64MultiArray()
                        msg.data = [tx, ty, 1.0]

                        self.target_publishers[drone_id].publish(msg)

                        self.current_targets[drone_id] = (target_gx, target_gy)

                        self.get_logger().info(
                            f"[DISCOVERER] Agent {drone_id} moving away"
                        )

                        continue

                # =====================================================
                # RESOLVER GOES DIRECTLY TO ANOMALY
                # =====================================================
                elif drone_id == self.anomaly_resolver:

                    # -------------------------------------------------
                    # WAIT UNTIL DISCOVERER LEAVES ANOMALY CELL
                    # -------------------------------------------------

                    discoverer_pos = self.positions_grid[
                        self.anomaly_discoverer
                    ]

                    # If discoverer still on anomaly:
                    # HOLD POSITION
                    if discoverer_pos == (ax, ay):

                        self.get_logger().info(
                            f"[RESOLVER] Agent {drone_id} waiting for cell clearance"
                        )

                        continue

                    # -------------------------------------------------
                    # SAFE TO INVESTIGATE
                    # -------------------------------------------------

                    tx, ty = grid_to_world(
                        ax,
                        ay,
                        self.scale,
                        self.grid_size
                    )

                    msg = Float64MultiArray()
                    msg.data = [tx, ty, 1.0]

                    self.target_publishers[drone_id].publish(msg)

                    self.current_targets[drone_id] = (ax, ay)

                    self.get_logger().info(
                        f"[RESOLVER] Agent {drone_id} investigating anomaly"
                    )

                    continue






            # Candidate from MARL
            candidate_gx = np.clip(gx + dx, 0, self.grid_size - 1)
            candidate_gy = np.clip(gy + dy, 0, self.grid_size - 1)

            # If MARL chooses explored/non-optimal cell while unexplored exists nearby,
            # override with frontier-first heuristic
            

            valid_moves = []

            other_last_targets = {
                self.last_targets[other_id]
                for other_id in range(self.num_drones)
                if other_id != drone_id
            }

            for alt_action, (adx, ady) in ACTIONS.items():

                nx = gx + adx
                ny = gy + ady

                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue

                # Avoid teammate collision
                if (nx, ny) in other_positions or (nx, ny) in reserved_targets:
                    continue

                if (nx, ny) in other_last_targets and (gx, gy) in other_positions:
                    continue

                # Penalize staying unless no choice
                stay_penalty = 2 if (nx, ny) == (gx, gy) else 0

                # Priority
                if self.grid[nx, ny] == 3:
                    score = -20
                elif self.grid[nx, ny] == 0:
                    score = -10
                elif self.grid[nx, ny] == 1:
                    score = 5
                else:
                    score = 20

                recent_penalty = 15 if (nx, ny) in self.recent_positions[drone_id] else 0
                score += recent_penalty

                score += stay_penalty

                valid_moves.append((score, alt_action, nx, ny))

            # =====================================================
            # GLOBAL FRONTIER RECOVERY
            # =====================================================

            unexplored = np.argwhere(self.grid == 0)

            # If no nearby good unexplored move,
            # actively move toward closest frontier
            best_score = valid_moves[0][0] if valid_moves else 999

            if len(unexplored) > 0 and best_score >= 5:

                frontier_candidates = []

                for ux, uy in unexplored:

                    # Manhattan distance
                    dist = abs(gx - ux) + abs(gy - uy)

                    frontier_candidates.append((dist, ux, uy))

                frontier_candidates.sort()

                _, ux, uy = frontier_candidates[0]

                # One-step greedy motion toward frontier
                step_x = np.sign(ux - gx)
                step_y = np.sign(uy - gy)

                candidate_steps = []

                # prioritize x/y movement options
                candidate_steps.append((gx + step_x, gy))
                candidate_steps.append((gx, gy + step_y))

                moved = False

                for nx, ny in candidate_steps:

                    if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                        continue

                    if (nx, ny) in other_positions:
                        continue

                    if (nx, ny) in reserved_targets:
                        continue

                    target_gx, target_gy = nx, ny

                    action_source = "GLOBAL_FRONTIER"

                    moved = True

                    break

                # fallback
                if not moved:

                    target_gx, target_gy = gx, gy

                    action_source = "FRONTIER_WAIT"
            
            
            valid_moves.sort()




            # MARL candidate score
            candidate_score = next(
                (
                    score for score, act, nx, ny in valid_moves
                    if (nx, ny) == (candidate_gx, candidate_gy)
                ),
                999
            )

            # Best available
            best_score, best_action, best_gx, best_gy = valid_moves[0]

            # Override only if MARL worse
            if best_score < candidate_score:
                action_source = "ALT"
                action = best_action
                target_gx, target_gy = best_gx, best_gy
            else:
                action_source = "MARL"
                target_gx, target_gy = candidate_gx, candidate_gy   

            reserved_targets.add((target_gx, target_gy))    


            tx, ty = grid_to_world(
                target_gx,
                target_gy,
                self.scale,
                self.grid_size
            )


            # new_target = (tx, ty)
            new_target = (target_gx, target_gy)

            # Only assign new target if different from previous
            if self.current_targets[drone_id] != new_target:
                self.current_targets[drone_id] = new_target
                self.target_reached[drone_id] = False

            msg = Float64MultiArray()
            msg.data = [tx, ty, 1.0]

            self.target_publishers[drone_id].publish(msg)

            self.last_targets[drone_id] = new_target                    


            self.get_logger().info(
                f"[Agent {drone_id}] Source={action_source} "
                f"MARL_Action={self.policy.predict(obs)} "
                f"FinalAction={action} "
                f"Current=({gx},{gy}) "
                f"Target=({target_gx},{target_gy})"
            )
            self.get_logger().info(
                f"[Agent {drone_id}] CurrentWorld={self.positions[drone_id]} -> Published={msg.data}"
            )

            self.get_logger().info(
                f"[Agent {drone_id}] valid_moves={valid_moves[:3]}"
            )

            self.get_logger().info(
                f"[ANOMALY] Discoverer={self.anomaly_discoverer} Resolver={self.anomaly_resolver}"
            )


# =========================================================
# MAIN
# =========================================================
def main(args=None):

    rclpy.init(args=args)

    node = CentralizedMARLNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()