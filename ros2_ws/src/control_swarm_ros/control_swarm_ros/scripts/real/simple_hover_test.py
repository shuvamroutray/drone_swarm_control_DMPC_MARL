# #!/usr/bin/env python3

#**************Code uisng high level Crazyswarm APIs*****************

#!/usr/bin/env python3


from crazyflie_py import Crazyswarm


def main():

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    print("Taking off...")
    cf.takeoff(targetHeight=0.6, duration=3.0)
    timeHelper.sleep(5.0)

    print("Hovering...")
    #timeHelper.sleep(5.0)

    print("Landing...")
    cf.land(targetHeight=0.03, duration=1.5)
    timeHelper.sleep(3.0)

    print("Done.")


if __name__ == "__main__":
    main()






#**************Code for Streaming position to using cmd_position*****************


# import rclpy
# from rclpy.node import Node
# from crazyflie_interfaces.msg import FullState
# from crazyflie_interfaces.srv import Takeoff
# from tf2_ros import Buffer, TransformListener
# import numpy as np
# import time
# from crazyflie_interfaces.msg import Position


# class SimpleHover(Node):

#     def __init__(self):
#         super().__init__('simple_hover')

#         # ==============================
#         # PARAMETERS
#         # ==============================
#         self.declare_parameter("drone_name", "cf231")
#         self.declare_parameter("hover_height", 0.5)
#         self.declare_parameter("rate", 20.0)

#         self.drone_name = self.get_parameter("drone_name").value
#         self.hover_height = self.get_parameter("hover_height").value
#         self.rate = self.get_parameter("rate").value
#         self.dt = 1.0 / self.rate

#         # ==============================
#         # TF Listener
#         # ==============================
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # ==============================
#         # Publisher (IMPORTANT: absolute topic)
#         # ==============================
#         self.cmd_pub = self.create_publisher(
#             Position,
#             f'/{self.drone_name}/cmd_position',
#             10
#         )

#         # ==============================
#         # Takeoff Client
#         # ==============================
#         self.takeoff_client = self.create_client(
#             Takeoff,
#             f'/{self.drone_name}/takeoff'
#         )

#         while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info("Waiting for takeoff service...")

#         # ==============================
#         # Wait for TF to become available
#         # ==============================
#         self.get_logger().info("Waiting for TF data...")
#         while rclpy.ok():
#             try:
#                 self.tf_buffer.lookup_transform(
#                     'world',
#                     self.drone_name,
#                     rclpy.time.Time()
#                 )
#                 break
#             except Exception:
#                 rclpy.spin_once(self, timeout_sec=0.1)

#         self.get_logger().info("TF detected.")

#         # ==============================
#         # TAKEOFF
#         # ==============================
#         self.get_logger().info("Sending takeoff command...")

#         req = Takeoff.Request()
#         req.height = self.hover_height
#         req.duration.sec = 3

#         future = self.takeoff_client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)

#         self.get_logger().info("Takeoff command sent. Waiting...")
#         time.sleep(3.5)


#         trans = self.tf_buffer.lookup_transform(
#                 'world',
#                 self.drone_name,
#                 rclpy.time.Time()
#             )

#         self.hover_x = trans.transform.translation.x
#         self.hover_y = trans.transform.translation.y

#         # ==============================
#         # Start control loop AFTER takeoff
#         # ==============================
#         self.timer = self.create_timer(self.dt, self.control_loop)

#         self.get_logger().info("Hover control started.")

#     # ==============================
#     # Read pose from TF
#     # ==============================
#     def get_current_z(self):
#         try:
#             trans = self.tf_buffer.lookup_transform(
#                 'world',
#                 self.drone_name,
#                 rclpy.time.Time()
#             )
#             return trans.transform.translation.z
#         except Exception:
#             return 0.0

#     # ==============================
#     # Control Loop
#     # ==============================
#     def control_loop(self):

#         msg = Position()
#         msg.x = self.hover_x
#         msg.y = self.hover_y
#         msg.z = self.hover_height
#         msg.yaw = 0.0
#         self.cmd_pub.publish(msg)

#         current_z = self.get_current_z()
#         self.get_logger().info(f"Current Z: {current_z:.3f}")


# def main(args=None):
#     rclpy.init(args=args)
#     node = SimpleHover()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()