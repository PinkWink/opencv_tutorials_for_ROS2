import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange
from rcl_interfaces.msg import SetParametersResult


class CannyEdgeDetector(Node):
    def __init__(self):
        super().__init__('canny_edge_detector')

        self.get_logger().info("Start Canny Edge Detector.")

        # Create subscriber
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        # Create publisher
        self.publisher = self.create_publisher(
            Image,
            '/image_edge',
            10
        )
        
        # Declare parameters (threshold1, threshold2)
        param_desc_threshold1 = ParameterDescriptor(
            description='Canny threshold1',
            integer_range=[IntegerRange(from_value=0, to_value=255, step=1)],
        )
        param_desc_threshold2 = ParameterDescriptor(
            description='Canny threshold2',
            integer_range=[IntegerRange(from_value=0, to_value=255, step=1)],
        )

        self.declare_parameter('threshold1', 100, param_desc_threshold1)
        self.declare_parameter('threshold2', 200, param_desc_threshold2)

        threshold1 = self.get_parameter('threshold1').value
        threshold2 = self.get_parameter('threshold2').value

        self.get_logger().info(f"Initial threshold1 : {threshold1}")
        self.get_logger().info(f"Initial threshold2 : {threshold2}")

        # Add on_set_parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

    def parameter_callback(self, params):
        for param in params:
            msg = f"{param.name} is changed to {param.value}"
            self.get_logger().info(msg)

        return SetParametersResult(successful=True)

    def image_callback(self, msg):
        # Get parameters
        threshold1 = self.get_parameter('threshold1').value
        threshold2 = self.get_parameter('threshold2').value

        # Convert ROS Image message to OpenCV image (BGR)
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny Edge Detection
        edges = cv2.Canny(gray_image, threshold1, threshold2)

        # Convert the result to a ROS Image message (mono8)
        edge_msg = self.cv_bridge.cv2_to_imgmsg(edges, encoding='mono8')

        # Publish the edge-detected image
        self.publisher.publish(edge_msg)


def main():
    rclpy.init()
    node = CannyEdgeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
