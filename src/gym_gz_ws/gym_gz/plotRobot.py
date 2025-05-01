import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import matplotlib.pyplot as plt
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import tf2_ros
import rclpy.time

class RobotPloter(Node):
    def __init__(self):
        super().__init__('robot_ploter')
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.transforms = {}
        self.subscription = self.create_subscription(
            String,
            '/transforms',
            self.plot_robot,
            10)
        self.joint_names= ["shoulder_link","upper_arm_link","forearm_link","wrist_1_link","wrist_2_link","wrist_3_link"]
        # 初始化图像窗口
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.get_logger().info("RobotPloter Node started and subscribed to /transforms")




    def plot_robot(self, msg,radius=0.1):
        
        try:
            T_list_raw = json.loads(msg.data)
            T_list = {
                name: np.array(T_raw).reshape(4, 4)
                for name, T_raw in T_list_raw.items()
            }
        except Exception as e:
            self.get_logger().error(f"Invalid T_list data: {e}")
            return

        self.ax.clear()
        points = []

        for name, T in T_list.items():
            pos = T[:3, 3]
            points.append(pos)

        points = np.array(points)
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], color='k', linewidth=2)

        T_list_keys = list(T_list.keys())

        # ✅ 添加地面到第一个关节的连杆（基座柱）
        world_origin = np.array([0.0, 0.0, 0.0])
        first_joint_pos = T_list[T_list_keys[0]][:3, 3]
        self.plot_cylinder(self.ax, world_origin, first_joint_pos, radius=radius)
        for i in range(len(T_list_keys) - 1):
            T_i = T_list[T_list_keys[i]]
            T_j = T_list[T_list_keys[i + 1]]

            p1 = T_i[:3, 3]  # 当前关节坐标系原点 = 圆柱起点
            p2 = T_j[:3, 3]  # 下一个关节的位置 = 圆柱终点
            self.plot_cylinder(self.ax, p1, p2, radius=radius)
            
        # # 自适应坐标轴范围
        # max_range = np.ptp(points, axis=0).max() / 2
        # mid = points.mean(axis=0)
        # self.ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        # self.ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        # self.ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        # 固定坐标轴范围（不变）
        self.ax.set_xlim(-0.7, 0.75)
        self.ax.set_ylim(-0.75, 0.75)
        self.ax.set_zlim(0, 1)  # 如果你的机械臂竖直工作，也可以设为 [0, 1] 或 [0, 2]



        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Live Robot Pose")
        self.ax.set_box_aspect([1, 1, 1])
        plt.draw()
        plt.pause(0.01)


    def plot_cylinder(self,ax, p1, p2, radius=0.01, resolution=20):
        """
        从点 p1 到 p2 画一个圆柱体，底部精确对齐 p1
        """
        v = np.array(p2) - np.array(p1)
        height = np.linalg.norm(v)
        if height < 1e-6:
            return

        # 单位圆柱（沿z轴）
        z = np.linspace(0, height, 2)
        theta = np.linspace(0, 2 * np.pi, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        # 点云 shape: (3, N)
        xyz = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=0)

        # z轴对齐目标向量
        direction = v / height
        rotation, _ = R.align_vectors([direction], [[0, 0, 1]])
        rotated = rotation.apply(xyz.T).T

        # ✅ 关键修正：将旋转后的圆柱平移到底部 p1（不是中点！）
        translated = rotated + p1.reshape(3, 1)

        # 重塑为表面
        x_final = translated[0].reshape(z_grid.shape)
        y_final = translated[1].reshape(z_grid.shape)
        z_final = translated[2].reshape(z_grid.shape)

        ax.plot_surface(x_final, y_final, z_final, color='gray', alpha=0.6, linewidth=0)


    def check_quit_key(self):
        """按下 q 键关闭窗口"""
        backend = matplotlib.get_backend()
        if 'TkAgg' in backend or 'QtAgg' in backend:
            self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == 'q':
            print("[INFO] Quit key pressed (q), closing window...")
            plt.close(self.fig)  # 关闭窗口
            rclpy.shutdown()     

def main(args=None):
    rclpy.init(args=args)
    node = RobotPloter()

   
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()