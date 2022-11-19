import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import time
from .submodules.yolov5_img_blur import  yolov5onnx
from rclpy.serialization import serialize_message
import rosbag2_py


from sensor_msgs.msg import Image


class Main(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', str()),
                ('write_bag', bool()),
                ('debug', bool()),
                ('bag_name', str()),
                ('save_directory', str()),
                ('camera_topic', str())
            ])

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.write_bag = self.get_parameter('write_bag').get_parameter_value().bool_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.bag_name = self.get_parameter('bag_name').get_parameter_value().string_value
        self.save_directory = self.get_parameter('save_directory').get_parameter_value().string_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        # Print parameters
        self.get_logger().info('****************************************************************')
        self.get_logger().info('model_path: %s' % self.model_path)
        self.get_logger().info('write_bag: %s' % self.write_bag)
        self.get_logger().info('debug: %s' % self.debug)
        self.get_logger().info('bag_name: %s' % self.bag_name)
        self.get_logger().info('save_directory: %s' % self.save_directory)
        self.get_logger().info('camera_topic: %s' % self.camera_topic)
        self.get_logger().info('****************************************************************')
        
        # Initialize yolov5
        self.counter = 0
        self.yolo = yolov5onnx(self.model_path, nc=1, model_hw=[640,640], mem_limit=800)

        # Initialize cv_bridge
        self.bridge = CvBridge()

        # Create subscriber
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.callback,
            10)
        self.subscription  # prevent unused variable warning

        # Initialize ROS2 bag recorder
        if self.write_bag:
            self.writer = rosbag2_py.SequentialWriter()

            storage_options = rosbag2_py._storage.StorageOptions(
                uri=self.bag_name,
                storage_id='sqlite3')
            converter_options = rosbag2_py._storage.ConverterOptions('', '')
            self.writer.open(storage_options, converter_options)

            topic_info = rosbag2_py._storage.TopicMetadata(
                name=self.camera_topic + '/blurred',
                type='sensor_msgs/msg/Image',
                serialization_format='cdr')
            self.writer.create_topic(topic_info)
        

    def callback(self, input_image):
        tic = time.perf_counter()
        img = self.bridge.imgmsg_to_cv2(input_image, desired_encoding="")
        toc = time.perf_counter()
        print(f"Recieved in {toc - tic:0.4f} seconds")
        self.get_logger().info('Recieved in %s seconds' % (toc - tic))

        tic = time.perf_counter()
        self.yolo.pre_processing(img)
        toc = time.perf_counter()
        print(f"Pre-processing in {toc - tic:0.4f} seconds")
        self.get_logger().info('Pre-processing in %s seconds' % (toc - tic))

        tic = time.perf_counter()
        self.yolo.inference()
        toc = time.perf_counter()
        print(f"Inference in {toc - tic:0.4f} seconds")
        self.get_logger().info('Inference in %s seconds' % (toc - tic))

        tic = time.perf_counter()
        self.yolo.post_processing(img)
        self.counter += 1
        toc = time.perf_counter()
        print(f"Post processing in {toc - tic:0.4f} seconds")
        self.get_logger().info('Post processing in %s seconds' % (toc - tic))

        tic = time.perf_counter()
        # Save to save_directory with counter
        cv2.imwrite(self.save_directory + "blur/" + str(self.counter) + ".jpg", img)
        toc = time.perf_counter()
        print(f"Saved in {toc - tic:0.4f} seconds")
        self.get_logger().info('Saved in %s seconds' % (toc - tic))

        # Convert to ROS2 message
        tic = time.perf_counter()
        ros2_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        toc = time.perf_counter()
        print(f"Converted to ROS2 message in {toc - tic:0.4f} seconds")
        self.get_logger().info('Converted to ROS2 message in %s seconds' % (toc - tic))

        if self.write_bag:
            self.writer.write(
                self.camera_topic + '/blurred',
                serialize_message(ros2_msg),
                self.get_clock().now().nanoseconds)



def main(args=None):
    print('Hi from getjpg.')

    rclpy.init(args=args)
    image_subscriber = Main()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

