import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class OpticalFlowPublisher(Node):
    def __init__(self):
        super().__init__('optical_flow_publisher')
        self.get_logger().info("Starting OpticalFlowPublisher...")

        # Subscriber
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10
        )

        # Publishers
        self.image_optical_vector_publisher = self.create_publisher(
            Image, '/image_optical_vector', 10
        )

        self.bridge = CvBridge()
        self.prev_gray = None

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image (BGR)
        current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # If this is the first frame, just store and return
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return

        # 1. Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,      # previous frame (gray)
            current_gray,        # current frame  (gray)
            None,                # flow (output)
            0.5,                 # pyr_scale
            3,                   # levels
            15,                  # winsize
            3,                   # iterations
            5,                   # poly_n
            1.2,                 # poly_sigma
            0                    # flags
        )

        # Publish original image with optical flow vectors (arrows)
        flow_vec_img = current_frame.copy()
        self.draw_flow_vectors(flow_vec_img, flow, step=16)
        flow_vec_msg = self.bridge.cv2_to_imgmsg(flow_vec_img, encoding='bgr8')
        flow_vec_msg.header = msg.header
        self.image_optical_vector_publisher.publish(flow_vec_msg)

        # Update prev_gray for next call
        self.prev_gray = current_gray

    def draw_flow_vectors(self, frame, flow, step=16):
        h, w = flow.shape[:2]

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Optical Flow (Δx, Δy)
                fx, fy = flow[y, x]
                end_x = int(x + fx)
                end_y = int(y + fy)

                # 화살표 그리기
                cv2.arrowedLine(
                    frame, (x, y), (end_x, end_y), color=(0, 255, 0), thickness=1, tipLength=0.4
                )

def main():
    rclpy.init()
    node = OpticalFlowPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

