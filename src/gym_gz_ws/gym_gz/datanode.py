from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from rclpy.node import Node
import rclpy
from sensor_msgs_py import point_cloud2
import numpy as np
from control_msgs.msg import DynamicJointState
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from Pointnet_Pointnet2_pytorch.models.pointnet_cls import get_model
class DataSubnode(Node):

    def __init__(self):
        super().__init__('DataSubnode')

        # 订阅深度摄像头的点云数据
        self.sub_depth_camera = self.create_subscription(
            PointCloud2,
            '/depth_camera/points',
            self.depth_camera_pointcloud2_callback,
            10
        )
        self.sub_depth_camera  # 防止未使用变量警告

        # 订阅动态关节状态数据
        self._joint_positions_sub = self.create_subscription(
            DynamicJointState,
            '/dynamic_joint_states',
            self.get_joint_positions,
            1
        )

        # 设置保存路径
        self.save_path = os.path.expanduser('~/rosdata')  # 自适应路径
        self.get_logger().info('DataSubnode has been started')

    def depth_camera_pointcloud2_callback(self, msg):
        
        points = np.array(list(point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))
        points = points.view(np.float32).reshape(-1, 3) # transfer to numpy array
        # 1. filter out invalid points
        #self.visualize_pointcloud(points, title="Original Point Cloud")
        filtered_points = self.remove_invalid_points(points)
        # 2. voxel grid downsampling
        downsampled_points = self.voxel_grid_downsampling(filtered_points)
        self.get_logger().info(f'points:{downsampled_points.shape}')  
        self.visualize_pointcloud(downsampled_points, title="Downsampled Point Cloud")
        # 3. segmentation
        # obstacles = self.plane_segmentation(downsampled_points)

        #  4. save to PCD file
        # timestamp = self.get_clock().now().to_msg().sec
        # filename = os.path.join(self.save_path, f"obstacles_{timestamp}.pcd")
        # self.save_to_pcd(obstacles, filename)
    def get_joint_positions(self, msg):
        positions = []
        velocities = []
        efforts = []
        # extract position, velocity, effort
        for index, joint_name in enumerate(msg.joint_names):
            interface_names = msg.interface_values[index].interface_names
            values = msg.interface_values[index].values

            if 'position' in interface_names:
                positions.append(values[interface_names.index('position')])
            if 'velocity' in interface_names:
                velocities.append(values[interface_names.index('velocity')])
            if 'effort' in interface_names:
                efforts.append(values[interface_names.index('effort')])
        # transfer to numpy array
        positions = np.array(positions)
        velocities = np.array(velocities)
        efforts = np.array(efforts)
        # print to console
        self.get_logger().info(f"Position: {positions.tolist()}")

    def save_to_txt(self, data, filename, fmt='%.6f'):
        try:
            full_path = os.path.join(self.save_path, filename)
            os.makedirs(self.save_path, exist_ok=True)
            np.savetxt(full_path, data, fmt=fmt)
            self.get_logger().info(f'Successfully saved data to: {full_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save data: {str(e)}')

    def remove_invalid_points(self, points):
        """
        take out （NaN and Inf）。
        """
        valid_mask = np.isfinite(points).all(axis=1)
        filtered_points = points[valid_mask]
        
        return filtered_points

    def voxel_grid_downsampling(self, points, voxel_size=0.01):
        #downsampling
        voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        downsampled_points = points[unique_indices]

        # fix to 1024 points for PointNet++
        num_points = downsampled_points.shape[0]
        if num_points > 1024:
            indices = np.random.choice(num_points, 1024, replace=False)
            downsampled_points = downsampled_points[indices]
        elif num_points < 1024:
            indices = np.random.choice(num_points, 1024, replace=True)
            downsampled_points = downsampled_points[indices]
        return downsampled_points
    def visualize_pointcloud(self,points, title="Point Cloud"):
        """
        Visualize the point cloud in 3D space
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        plt.show()



    def plane_segmentation(self, points, distance_threshold=0.01):
     
        pass

    def save_to_pcd(self, points, filename):
        pass

    # def publish_obstacles(self, header, points):
    #     """
    #     发布处理后的障碍物点云到 ROS 话题
    #     """
    #     obstacle_msg = point_cloud2.create_cloud_xyz32(header, points)
    #     self.publisher.publish(obstacle_msg)
    #     self.get_logger().info("Published processed obstacles")

    # def visualize_pointcloud(self, points):
    #     """
    #     使用 PCL 可视化处理后的障碍物点云
    #     """
    #     cloud = pcl.PointCloud.PointXYZ()
    #     cloud.from_array(points.astype(np.float32))

    #     viewer = pcl.visualization.PCLVisualizer("Obstacle Viewer")
    #     viewer.setBackgroundColor(0, 0, 0)
    #     viewer.addPointCloud(cloud, "obstacles")
    #     viewer.setPointCloudRenderingProperties(
    #         pcl.visualization.PCLVisualizer.PCL_VISUALIZER_POINT_SIZE, 2, "obstacles"
    #     )
    #     viewer.addCoordinateSystem(1.0)
    #     viewer.initCameraParameters()

    #     while not viewer.wasStopped():
    #         viewer.spinOnce(10)


def main(args=None):
    rclpy.init(args=args)

    node = DataSubnode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 显式销毁节点
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
