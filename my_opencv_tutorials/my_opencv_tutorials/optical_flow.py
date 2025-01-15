import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
from builtin_interfaces.msg import Time


class OpticalFlowPublisher(Node):
    def __init__(self):
        super().__init__('optical_flow_publisher')
        self.get_logger().info("Starting OpticalFlowPublisher...")

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        # 1) Optical flow in PointCloud2 format
        self.flow_publisher = self.create_publisher(
            PointCloud2,
            '/optical_flow',
            10
        )

        # 2) Optical flow vectors over the original image
        self.image_optical_vector_publisher = self.create_publisher(
            Image,
            '/image_optical_vector',
            10
        )

        # CV Bridge
        self.bridge = CvBridge()

        # To hold the previous frame (grayscale)
        self.prev_gray = None

    def image_callback(self, msg):
        """Receives image frames from /image_raw, computes optical flow, and publishes."""
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

        # 2. Publish PointCloud2 (optical flow)
        cloud_msg = self.flow_to_pointcloud2(flow, msg.header.stamp)
        self.flow_publisher.publish(cloud_msg)

        # 3. Publish original image with optical flow vectors (arrows)
        flow_vec_img = current_frame.copy()
        self.draw_flow_vectors(flow_vec_img, flow, step=16)
        flow_vec_msg = self.bridge.cv2_to_imgmsg(flow_vec_img, encoding='bgr8')
        flow_vec_msg.header = msg.header
        self.image_optical_vector_publisher.publish(flow_vec_msg)

        # Update prev_gray for next call
        self.prev_gray = current_gray

    def flow_to_pointcloud2(self, flow, stamp):
        """
        flow.shape = (height, width, 2)
          flow[..., 0] = x방향 흐름(Δx)
          flow[..., 1] = y방향 흐름(Δy)

        PointCloud2로 변환 시,
          각 픽셀 (u, v)에 대해
            x = flow[v, u, 0]
            y = flow[v, u, 1]
            z = 0
        로 저장해봅니다.
        """

        height, width, _ = flow.shape

        # PointCloud2 메시지 생성
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = stamp
        cloud_msg.header.frame_id = "camera_optical_frame"  # 상황에 맞게 frame_id 지정
        cloud_msg.height = height
        cloud_msg.width = width

        # Fields 정의 (x, y, z => float32)
        cloud_msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12  # float32 x 3 = 12 bytes
        cloud_msg.row_step = cloud_msg.point_step * width
        cloud_msg.is_dense = True  # 유효하지 않은 데이터가 없다고 가정

        # data 채우기
        data = []
        for v in range(height):
            for u in range(width):
                fx = flow[v, u, 0]
                fy = flow[v, u, 1]
                # z는 0으로
                point_bytes = struct.pack('fff', fx, fy, 0.0)
                data.append(point_bytes)

        cloud_msg.data = b''.join(data)
        return cloud_msg

    def flow_to_color_image(self, flow):
        """
        Optical Flow를 시각적으로 표현하기 위해:
          - 흐름의 각 화소에 대해 magnitude(세기), angle(방향)을 구함
          - angle을 색상(Hue)으로, magnitude를 명도(Value)로 매핑
          - 결과를 HSV->BGR로 변환
        """
        h, w, _ = flow.shape

        # x, y 흐름 -> magnitude, angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # HSV 컬러맵 생성
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        # Hue: 0~180 범위이므로 angle/2
        hsv[..., 0] = (angle / 2).astype(np.uint8)
        # Saturation = 최대
        hsv[..., 1] = 255
        # Value = 흐름 세기를 0~255 범위로 정규화
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # HSV -> BGR
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_bgr

    def draw_flow_vectors(self, frame, flow, step=16):
        """
        주어진 프레임 위에 Optical Flow 벡터(화살표)를 일정 간격(step)으로 그립니다.
        flow.shape = (height, width, 2)
        """
        h, w = flow.shape[:2]

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Optical Flow (Δx, Δy)
                fx, fy = flow[y, x]
                end_x = int(x + fx)
                end_y = int(y + fy)

                # 화살표 그리기
                cv2.arrowedLine(
                    frame,
                    (x, y),
                    (end_x, end_y),
                    color=(0, 255, 0),
                    thickness=1,
                    tipLength=0.4
                )


def main():
    rclpy.init()
    node = OpticalFlowPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

