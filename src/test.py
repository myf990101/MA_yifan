import rclpy
from rclpy.node import Node
import tf2_ros

def main():
    rclpy.init()
    node = Node("tf_test")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            frames = tf_buffer.all_frames_as_string()
            print("Available TF frames:\n", frames)
            trans = tf_buffer.lookup_transform('world', 'shoulder_link', rclpy.time.Time())
            print(trans)
            break
        except Exception as e:
            print("Waiting for TF...", e)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
