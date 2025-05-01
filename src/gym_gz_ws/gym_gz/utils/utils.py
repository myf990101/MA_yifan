"""Module containing utility functions."""

import numpy as np
from numpy.typing import NDArray

from geometry_msgs.msg import Vector3, Point, Quaternion


def vector3_to_array(vec: Vector3) -> NDArray[np.float64]:
    """Convert a geometry_msgs.msg.Vector3 to a numpy array."""
    return np.array([vec.x, vec.y, vec.z], dtype=np.float64)


def point_to_array(p: Point) -> NDArray[np.float64]:
    """Convert a geometry_msgs.msg.Vector3 to a numpy array."""
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def quaternion_to_array(q: Quaternion) -> NDArray[np.float64]:
    """Convert a geometry_msgs.msg.Vector3 to a numpy array."""
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)


def array_to_point(position: list[float] | tuple[float] | NDArray[np.float64]) -> Point:
    """Convert a list or array of size 3 to a geometry_msgs.msg.Point"""
    if len(position) != 3:  # could be better, but good enough
        raise ValueError("'position' must contain exactly three float values.")

    return Point(
        x=position[0],
        y=position[1],
        z=position[2],
    )


def array_to_quaternion(
    rotation: list[float] | tuple[float] | NDArray[np.float64],
) -> Quaternion:
    """Convert a list or array of size 4 to a geometry_msgs.msg.Quaternion"""
    if len(rotation) != 4:  # could be better, but good enough
        raise ValueError("'position' must contain exactly three float values.")

    return Quaternion(
        x=rotation[0],
        y=rotation[1],
        z=rotation[2],
        w=rotation[3],
    )
def quaternion_to_euler_np(quaternion: Quaternion) -> np.ndarray:
    [x,y,z,w] = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=np.float64)
    # 计算 Roll (X 轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # 计算 Pitch (Y 轴旋转)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # 避免 gimbal lock
    else:
        pitch = np.arcsin(sinp)
    
    # 计算 Yaw (Z 轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])
def quaternion_to_rotation_matrix(quaternion: Quaternion) -> np.ndarray:
    [x,y,z,w] = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=np.float64)
    r00 = 1 - 2*(y**2 + z**2)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x**2 + z**2)
    r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x**2 + y**2)
    R = np.array([[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]])
    v = np.array([0, 0, 1])
    v_rotated = np.dot(R, v)
    return v_rotated

