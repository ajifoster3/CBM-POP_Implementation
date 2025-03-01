import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from rosgraph_msgs.msg import Clock

class KillRobotAtTime(Node):
    def __init__(self, wait_time, revive_time, agent_id):
        super().__init__('kill_robot_node')
        self.wait_time = wait_time  # Time to wait after first clock message
        self.revive_time = revive_time  # Time to wait after kill before reviving
        self.agent_id = agent_id  # ID of the robot to kill
        self.start_time = None  # Initialize start time
        self.kill_time = None  # Time when the robot was killed
        self.is_robot_killed = False
        self.is_robot_revived = False

        # Subscriber to simulation time
        self.clock_subscriber = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )

        # Publisher to send kill signal
        self.kill_publisher = self.create_publisher(
            Bool,
            f'/central_control/uas_{self.agent_id}/kill_robot',
            10
        )

        # Publisher to send revive signal
        self.revive_publisher = self.create_publisher(
            Bool,
            f'/central_control/uas_{self.agent_id}/revive_robot',
            10
        )

    def clock_callback(self, msg):
        current_time = msg.clock.sec + msg.clock.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = current_time  # Capture the initial time
            self.get_logger().info(f'Start time recorded: {self.start_time}')

        # if self.start_time is not None and (current_time - self.start_time) >= self.wait_time:
        #     if not self.is_robot_killed:
        #         self.publish_kill_signal()
        #         self.is_robot_killed = True
        #         self.kill_time = current_time  # Store the time when the robot was killed
        #
        # if self.is_robot_killed and not self.is_robot_revived and self.kill_time is not None:
        #     if (current_time - self.kill_time) >= self.revive_time:
        #         self.publish_revive_signal()
        #         self.is_robot_revived = True

    def publish_kill_signal(self):
        kill_msg = Bool()
        kill_msg.data = True
        self.get_logger().info(f'Publishing kill signal to robot {self.agent_id} after {self.wait_time} seconds')
        self.kill_publisher.publish(kill_msg)

    def publish_revive_signal(self):
        revive_msg = Bool()
        revive_msg.data = True
        self.get_logger().info(f'Publishing revive signal to robot {self.agent_id} after {self.revive_time} seconds')
        self.revive_publisher.publish(revive_msg)


def main(args=None):
    rclpy.init(args=args)

    wait_time = 100.0  # Time to wait after receiving first clock message
    revive_time = 100.0  # Time to wait after killing before reviving
    agent_id = random.randint(1, 5)  # The ID of the robot to kill

    kill_node = KillRobotAtTime(wait_time, revive_time, agent_id)
    rclpy.spin(kill_node)

    kill_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
