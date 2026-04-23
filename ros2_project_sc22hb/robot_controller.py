import math
import signal
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.exceptions import ROSInterruptException
from rclpy.node import Node
from sensor_msgs.msg import Image


class RobotController(Node):

    TRAVEL = 'travel'
    SCAN = 'scan'
    APPROACH_BLUE = 'approach_blue'
    DONE = 'done'

    WAYPOINT = (1.1209923028945923, -7.2677459716796875, 0.0)

    ROT_SPEED = 1.0
    FULL_ROTATION = 2.0 * math.pi + 0.2
    APPROACH_SPEED = 0.22
    TARGET_AREA = 520000
    DETECTION_THRESHOLD = 500
    CENTER_TOL = 0.1
    DETECTION_PAUSE = 1.0

    def __init__(self):
        super().__init__('robot_controller')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image_raw',
                                 self.camera_callback, 10)

        self.colors_detected = set()
        self.blue_found = False
        self.blue_area = 0
        self.blue_center_offset = 0.0
        self.pause_until = 0.0

        self.nav_sent = False
        self.scan_start_time = None
        self.state = self.TRAVEL

        cv2.namedWindow('RGB Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB Detection', 640, 480)

    def stop(self):
        self.cmd_vel_pub.publish(Twist())

    def camera_callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        display = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.blue_found = False

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255])),
        )
        self.detect_colour(red_mask, display, 'Red', (0, 0, 255))
        self.detect_colour(
            cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255])),
            display, 'Green', (0, 255, 0)
        )
        self.detect_colour(
            cv2.inRange(hsv, np.array([100, 100, 100]), np.array([140, 255, 255])),
            display, 'Blue', (255, 0, 0)
        )

        cv2.putText(display, f"Colors: {', '.join(sorted(self.colors_detected))}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.imshow('RGB Detection', display)
        cv2.waitKey(3)

    def detect_colour(self, mask, display, name, colour):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.DETECTION_THRESHOLD:
            return

        if self.state != self.TRAVEL and name not in self.colors_detected:
            self.colors_detected.add(name)
            self.pause_until = time.time() + self.DETECTION_PAUSE
            self.stop()
            self.get_logger().info(f'{name.upper()} DETECTED')
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(display, name.upper(), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        if name != 'Blue':
            return

        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        _, width = mask.shape[:2]
        self.blue_found = True
        self.blue_area = area
        self.blue_center_offset = (cx - width / 2.0) / (width / 2.0)
        cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)

    def send_nav_goal(self):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 server not available, scanning here')
            self.state = self.SCAN
            return

        x, y, yaw = self.WAYPOINT
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)

        self.get_logger().info(f'Navigating to ({x:.2f}, {y:.2f})')
        self.nav_client.send_goal_async(goal).add_done_callback(self.nav_response_cb)

    def nav_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav goal rejected, scanning here')
            self.state = self.SCAN
            return

        goal_handle.get_result_async().add_done_callback(self.nav_result_cb)

    def nav_result_cb(self, _future):
        self.get_logger().info('Starting scan')
        self.scan_start_time = None
        self.state = self.SCAN

    def scan_step(self):
        if time.time() < self.pause_until:
            self.stop()
            return

        if self.scan_start_time is None:
            self.scan_start_time = time.time()

        if self.blue_found and len(self.colors_detected) == 3:
            self.stop()
            self.state = self.APPROACH_BLUE
            return

        elapsed = time.time() - self.scan_start_time
        if elapsed * self.ROT_SPEED < self.FULL_ROTATION:
            twist = Twist()
            twist.angular.z = self.ROT_SPEED
            self.cmd_vel_pub.publish(twist)
            return

        self.stop()
        if 'Blue' in self.colors_detected:
            self.get_logger().info('Scan complete; approaching blue')
            self.state = self.APPROACH_BLUE
        else:
            self.get_logger().warn('Blue was not found')
            self.state = self.DONE

    def approach_blue_step(self):
        if time.time() < self.pause_until:
            self.stop()
            return

        if not self.blue_found:
            twist = Twist()
            twist.angular.z = self.ROT_SPEED * 0.4
            self.cmd_vel_pub.publish(twist)
            return

        if self.blue_area >= self.TARGET_AREA:
            self.get_logger().info(f'Stopped near blue box, area={int(self.blue_area)}')
            self.stop()
            self.state = self.DONE
            return

        twist = Twist()
        twist.angular.z = float(np.clip(-1.2 * self.blue_center_offset, -1.0, 1.0))
        twist.linear.x = self.APPROACH_SPEED
        if abs(self.blue_center_offset) > self.CENTER_TOL:
            twist.linear.x *= 0.5
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    robot = RobotController()

    def signal_handler(_sig, _frame):
        robot.stop()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    spin_thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok() and robot.state != RobotController.DONE:
            if robot.state == RobotController.TRAVEL and not robot.nav_sent:
                robot.send_nav_goal()
                robot.nav_sent = True
            elif robot.state == RobotController.SCAN:
                robot.scan_step()
            elif robot.state == RobotController.APPROACH_BLUE:
                robot.approach_blue_step()

            time.sleep(0.1)
    except ROSInterruptException:
        pass

    robot.stop()
    cv2.destroyAllWindows()
    robot.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
