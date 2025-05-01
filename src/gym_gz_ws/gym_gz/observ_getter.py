#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from control_msgs.msg import DynamicJointState
import numpy as np
import tf2_ros
import rclpy.task
import rclpy.time
from gym_gz_ws.gym_gz.utils.utils import (
    array_to_point,
    array_to_quaternion,
    point_to_array,
    quaternion_to_array,
    vector3_to_array,
    quaternion_to_euler_np,
    quaternion_to_rotation_matrix,
)
from numpy.linalg import norm
import time
import threading
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from numpy.typing import NDArray
import uuid
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import json



from example_interfaces.srv import Trigger
class ObservationGetter(Node):
    def __init__(self):
        unique_node_name = f'Observation_getter_{uuid.uuid4().hex[:8]}'
        super().__init__(unique_node_name)
        # 创建订阅者，队列大小设为 10（根据需要调整）
        self.ros_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self.ros_thread.start()
        self.subscription = self.create_subscription(
            DynamicJointState,
            '/dynamic_joint_states',
            self.get_joint_posotions,
            1
        )
        self.srv = self.create_service(Trigger, 'get_joint_positions', self.get_joint_service)

        self._joint_position_pub = self.create_publisher(
            Float64MultiArray, "/position_controller/commands", 1)
        self.transforms_pub = self.create_publisher(
            String, "/transforms", 1)
        self.tcp_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._TF_LOOKUP_PERIOD: float = 0.01
        # self.buffer=[]
        self.joint_names= ["shoulder_link","upper_arm_link","forearm_link","wrist_1_link","wrist_2_link","wrist_3_link"]
        # self.wait_until_tf_data_is_available()
    def get_joint_posotions(self, msg: DynamicJointState):
        positions = []
        velocities = []
        efforts = []
        for index, joint_name in enumerate(msg.joint_names):
            # 提取每个关节的状态信息
            interface_names = msg.interface_values[index].interface_names
            values = msg.interface_values[index].values
            # 查找 'position', 'velocity', 'effort' 并提取值
            if 'position' in interface_names:
                positions.append(values[interface_names.index('position')])
            # if 'velocity' in interface_names:
            #     velocities.append(values[interface_names.index('velocity')])
            # if 'effort' in interface_names:
            #     efforts.append(values[interface_names.index('effort')])
        self.joint_position = np.array(positions)
        # self.velocities = np.array(velocities)
        # self.efforts = np.array(efforts)
        # print(f"Position: {self.joint_position}")
        stamp = msg.header.stamp
        timestamp = stamp.sec + stamp.nanosec * 1e-9

        # 将带有时间戳的数据存入缓冲区
        # self.buffer.append((timestamp, self.joint_position, self.tcp_position))

        try:
            self.tcp_position = self.get_position_and_rotation()
            # print(f"tcp_position: {self.tcp_position}")
        except Exception as e:
            return  0 
        
        return self.joint_position,self.tcp_position
    
    def pass_joint_positions(self):
        return self.joint_position,self.tcp_position
    

    # def get_all_transforms(self):
    #     for joint in self.joint_names:
    #         try:
    #             trans = self._tf_buffer.lookup_transform('world', joint, rclpy.time.Time())
    #             t = trans.transform.translation
    #             q = trans.transform.rotation

    #             # 转换四元数到旋转矩阵
    #             R_matrix = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    #             # 组成 4x4 变换矩阵
    #             T = np.eye(4)
    #             T[:3, :3] = R_matrix
    #             T[:3, 3] = [t.x, t.y, t.z]
                
    #             print(f"Transformation Matrix for {joint} (Gazebo):")
    #             print(T)
    #             print("-" * 50)

    #         except Exception as e:
    #             self.get_logger().info(f"Error for {joint}: {str(e)}")
    def get_all_transforms(self):
        transforms = {}

        for joint in self.joint_names:
            try:
                trans = self._tf_buffer.lookup_transform('world', joint, rclpy.time.Time())
                t = trans.transform.translation
                q = trans.transform.rotation

                R_matrix = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

                T = np.eye(4)
                T[:3, :3] = R_matrix
                T[:3, 3] = [t.x, t.y, t.z]

                transforms[joint] = T.tolist()
                # print(f"Transformation Matrix for {joint}:")
                # print(T)
                # print("-" * 50)

            except Exception as e:
                print(f"[WARN] Error for {joint}: {str(e)}")
        msg = String()
        msg.data = json.dumps(transforms)
        self.transforms_pub.publish(msg)

        return transforms

    def plot_robot(T_list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        points = []
        for name, T in T_list.items():
            pos = T[:3, 3]
            points.append(pos)
            ax.text(*pos, name, fontsize=8)

            # 绘制坐标轴
            origin = pos
            x_axis = origin + T[:3, 0] * 0.05
            y_axis = origin + T[:3, 1] * 0.05
            z_axis = origin + T[:3, 2] * 0.05
            ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')  # X轴
            ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')  # Y轴
            ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')  # Z轴

        # 绘制连接线
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color='k', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Robot Arm Pose from Transformation Matrices")
        ax.set_box_aspect([1, 1, 1])
        plt.show()


    def init_plot(self):
        """初始化图像窗口"""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Live Robot Pose")

    def get_position_and_rotation(self) :
        result = self._tf_buffer.lookup_transform(
            target_frame="base_link",
            source_frame="tool0",
            time=rclpy.time.Time(),
        ).transform
        self.tcp_position =np.concatenate((vector3_to_array(result.translation),
                                          quaternion_to_rotation_matrix(result.rotation)) )
        return self.tcp_position
    def get_obstacle_info(self):
        """
        Return obstacle information:
          - Obstacle position (randomly generated each call; adjust the coordinate range as needed)
          - Obstacle velocity (randomly generated each call)
          - Distance to the obstacle (for reward calculation)
          - Distance to the target (for reward calculation)
        """
        # Randomly generate obstacle position (example range: x in [0.5, 1.0], y in [0.0, 1.0], z in [0.0, 1.5])
        obstacle_position = np.random.uniform(low=[0.5, 0.0, 0.0], high=[1.0, 1.0, 1.5])
        # Randomly generate obstacle velocity (3-dimensional, range from -0.5 to 0.5)
        obstacle_velocity = np.random.uniform(low=-0.5, high=0.5, size=3)
        # Calculate distances using the current TCP position (assumes tcp_position is updated)
        distance_to_obstacle = norm(obstacle_position - self.tcp_position[:3])
        distance_to_target = norm(self.target - self.tcp_position[:3])
        return obstacle_position, obstacle_velocity, distance_to_obstacle, distance_to_target
    def wait_until_tf_data_is_available(self, timeout_sec: float = 10):
        # setup
        _robot_tf_available_future = rclpy.task.Future()

        def _check() -> None:
            try:
                self._tf_buffer.lookup_transform(
                    target_frame="base_link",
                    source_frame="tool0",
                    time=rclpy.time.Time(),
                )
                # if the above call does not throw an exception, it means the robot is reachable,
                # i.e. complete the future
                _robot_tf_available_future.set_result(True)
            except Exception:
                # Using a generic exception is not great, but can't seem to find out right now
                # how to import tf2.LookupException
                pass

        self._robot_tf_available_timer = self.create_timer(0.001, _check)

        # wait until tf data available or timeout
        rclpy.spin_until_future_complete(
            self, _robot_tf_available_future, timeout_sec=timeout_sec
        )

        if not _robot_tf_available_future.done():
            # this means the robot could not be set up / setup timed out
            print("Robot TF data not available. Exiting.")
        else:
            print("Robot TF data available. Continuing.")

        self._robot_tf_available_timer.destroy()
        delattr(self, "_robot_tf_available_timer")
    def reset_robot(self, joint_positions: NDArray[np.float64]):
        msg = Float64MultiArray(data=joint_positions)
        self._joint_position_pub.publish(msg)
          # 等待 2 秒
        return 0
        
    def move_to_joint_positions(self, actiongoal):
        
        msg = Float64MultiArray(data=actiongoal)
        self._joint_position_pub.publish(msg)
        # self.angle_diff(actiongoal)
        #     else:
        #         print("Still moving...")
        # # time.sleep(2)
        # return 0
    def get_joint_service(self,request, response):
        """
        Service 回调函数：返回当前存储的 joint position 数据（不包含时间戳）
        """
        if self.joint_position is None:
            response.success = False
            response.message = "当前无可用的关节位置信息"
        else:
            response.success = True
            response.message = np.array(self.joint_position)
        return response
    def angle_diff(self, b):
        
        timeout = 2  # 超时时间，可根据需要调整
        start_time = time.monotonic()
        movement_finished = False
        tolerance = 0.01  # 定义允许的角度差误差（弧度）
        while time.monotonic() - start_time < timeout:
            a = self.joint_position
            diff = (a - b + np.pi) % (2 * np.pi) - np.pi
            print(f"Angle difference: {diff}")
            if np.all(diff[:5]< tolerance):
                movement_finished = True
                print("Movement finished.")
                break  # 条件满足后退出循环
            time.sleep(0.01) 
        return np.abs(diff) 

def main(args=None):
    rclpy.init(args=args)
    node = ObservationGetter() 
    try:
        rclpy.spin(node)\
        
    except KeyboardInterrupt:
        node.get_logger().info("节点中断，准备关闭...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()