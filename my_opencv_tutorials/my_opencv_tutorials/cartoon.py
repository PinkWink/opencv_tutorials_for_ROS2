import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange
from rcl_interfaces.msg import SetParametersResult


class CartoonStyler(Node):
    def __init__(self):
        super().__init__('cartoon_styler')

        self.get_logger().info("Start Cartoon Styler.")

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
            '/image_cartoon',
            10
        )
        
        # Declare parameters (sigma_color, sigma_space)
        #   - sigma_color, sigma_space: bilateralFilter 관련 파라미터
        # sigmaColor : 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다
        # sigmaSpace : 공간 시그마 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다. 
        #               d>0 이면 영향을 받지 않고, 그 외에는 d 값에 비례한다. 
        #               d는 이미지 픽셀 간의 거리를 의미한다.   
        #               d가 0이면 이미지 픽셀 간의 거리를 계산하지 않는다.

        param_desc_sigma_color = ParameterDescriptor(
            description='Bilateral filter sigma_color',
            integer_range=[IntegerRange(from_value=0, to_value=100, step=1)],
        )
        param_desc_sigma_space = ParameterDescriptor(
            description='Bilateral filter sigma_space',
            integer_range=[IntegerRange(from_value=0, to_value=25, step=1)],
        )

        self.declare_parameter('sigma_color', 20, param_desc_sigma_color)
        self.declare_parameter('sigma_space', 5, param_desc_sigma_space)

        sigma_color = self.get_parameter('sigma_color').value
        sigma_space = self.get_parameter('sigma_space').value

        self.get_logger().info(f"Initial sigma_color : {sigma_color}")
        self.get_logger().info(f"Initial sigma_space : {sigma_space}")

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
        sigma_color = self.get_parameter('sigma_color').value
        sigma_space = self.get_parameter('sigma_space').value

        # Convert ROS Image message to OpenCV image (BGR)
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        blr = cv2.bilateralFilter(cv_image, -1, sigma_color, sigma_space)
        
        edge = 255 - cv2.Canny(cv_image, 80, 120)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        cartoon_image = cv2.bitwise_and(blr, edge)

        # Convert result to ROS Image message
        cartoon_msg = self.cv_bridge.cv2_to_imgmsg(cartoon_image, encoding='bgr8')

        # Publish the cartoon-style image
        self.publisher.publish(cartoon_msg)


def main():
    rclpy.init()
    node = CartoonStyler()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
