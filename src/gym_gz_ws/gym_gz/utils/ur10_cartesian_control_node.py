# checked
from math import ceil

import numpy as np
import rclpy
import rclpy.task
import rclpy.time
import tf2_ros
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from numpy.typing import NDArray
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.node import Node
from std_msgs.msg import Header, Float64MultiArray
from control_msgs.msg import DynamicJointState
from controller_manager_msgs.srv import (
    ListControllers,
    SwitchController,
)
from .utils import (
    array_to_point,
    array_to_quaternion,
    point_to_array,
    quaternion_to_array,
    vector3_to_array,
    quaternion_to_euler_np
)
import logging
import time

logger = logging.getLogger(__name__)


class UR10CartesianControl(Node):
    def __init__(
        self,
        tolerance: float = 5e-5,
        no_movement_threshold: float = 5e-5,
        no_movement_max_sec: float = 1.0,
        log_level: LoggingSeverity = LoggingSeverity.INFO,
    ):
        """
        Cartesian Control of UR10 robot
        """
        super().__init__("ur10_cartesian_control")
        self._logger: RcutilsLogger = self.get_logger()
        self._logger.set_level(level=log_level)
        self._logger.debug("Node startup...")

        self._logger.debug("User facing parameters:")

        # node parameters
        self.tolerance: float = tolerance
        self.no_movement_threshold: float = no_movement_threshold
        self.no_movement_max_sec: float = no_movement_max_sec

        self._logger.debug(f"\t{self.tolerance = }")
        self._logger.debug(f"\t{self.no_movement_threshold = }")
        self._logger.debug(f"\t{self.no_movement_max_sec = }")

        # command publishing
        self._pub = self.create_publisher(
            PoseStamped, "/full_controller/target_frame", 1
        )
        self._joint_position_pub = self.create_publisher(
            Float64MultiArray, "/position_controller/commands", 1
        )
        # reset robot
        self._reset_node = rclpy.create_node("reset_joints_node")
        self._switch_controllers_client = self._reset_node.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
        )
        self._list_controllers_client = self._reset_node.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )
        self._reset_robot_pub = self.create_publisher(
            Float64MultiArray, "/position_controller/commands", 1
        )
       

        # state listening
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._TF_LOOKUP_PERIOD: float = 0.01

        self._logger.debug("Internal parameters:")
        self._logger.debug(f"\t{self._TF_LOOKUP_PERIOD = }")

        # node management
        self._movement_finished_future: rclpy.task.Future = None  # type: ignore

        # => for tracking target of the movement
        self._target_position: Point = None  # type: ignore
        self._target_rotation: Quaternion = None  # type: ignore

        # => for tracking whether robot still moves
        self._TF_HISTORY_SAMPLE_RATE: int = (
            5  # data of every nth lookup is stored in history
        )
        self._TF_HISTORY_SIZE: int = max(
            5,
            ceil(
                self.no_movement_max_sec
                / self._TF_LOOKUP_PERIOD
                / self._TF_HISTORY_SAMPLE_RATE
            ),
        )

        self._tf_lookups_since_sample: int = 0  # number of lookups since last sample, used for tracking when to sample position

        # NOTE: history arrays work as ring buffer
        self._tf_history_is_full: bool = False  # this will be set to True once the ...next_idx reaches ...HISTORY_SIZE, after that it will always be true
        self._tf_history_next_idx: int = 0
        self._tf_history_position = np.empty(
            (self._TF_HISTORY_SIZE, 3), dtype=np.float64
        )
        self._tf_history_rotation = np.empty(
            (self._TF_HISTORY_SIZE, 4), dtype=np.float64
        )

        self._logger.debug(f"\t{self._TF_HISTORY_SAMPLE_RATE = }")
        self._logger.debug(f"\t{self._TF_HISTORY_SIZE = }")

        # wait until robot transform data is available, then create timer for listening to transforms
        try:
            self._wait_until_tf_data_is_available()
        except NodeInitializationError as e:
            pass
        self._logger.debug("Transform data now available")
        self._robot_tf_listener = self.create_timer(
            self._TF_LOOKUP_PERIOD, self._check_movement_callback
        )
        self._logger.info("Node startup finished.")
        


    def _movement_setup(self, target_position: Point, target_rotation: Quaternion):
        self._movement_finished_future = rclpy.task.Future()

        # target tracking
        self._target_position = target_position
        self._target_rotation = target_rotation

        # robot-still-moving tracking
        self._tf_lookups_since_sample = 0
        self._tf_history_is_full = False
        self._tf_history_next_idx = 0

    def get_position_and_rotation(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        result = self._tf_buffer.lookup_transform(
            target_frame="base_link",
            source_frame="tool0",
            time=rclpy.time.Time(),
        ).transform

        return vector3_to_array(result.translation), quaternion_to_euler_np(
            result.rotation
        )

    def reset_joints(
        self,
        joint_positions: NDArray[np.float64],
        target_position: NDArray[np.float64],
        target_rotation: NDArray[np.float64],
        timeout_per_operation: float = 15.0,
    ):
        """Reset robot by switching to position controller, setting joints, and then switching back to cartesian controller"""
        # switch to position_controller
        start_ts = time.monotonic()
        op_success = False
        while not op_success and time.monotonic() - start_ts < timeout_per_operation:
            msg = SwitchController.Request(
                deactivate_controllers=["full_controller"],
                activate_controllers=["position_controller"],
                strictness=SwitchController.Request.STRICT,
            )

            future = self._switch_controllers_client.call_async(msg)
            rclpy.spin_until_future_complete(self._reset_node, future)

            if future.result().ok:  # type: ignore
                op_success = True
        if not op_success:
            raise Exception("Switching to 'position_controller' timed out, critical.")

        # reset joints
        # (setup)
        self._movement_setup(
            target_position=array_to_point(target_position),
            target_rotation=array_to_quaternion(target_rotation),
        )

        # publish command
        msg = Float64MultiArray(data=joint_positions)
        self._reset_robot_pub.publish(msg)
        # turn turn off cartesian controller

        # wait until arm has reached target position or does not move beyond tolerance
        # NOTE: Spinning this node activates the _robot_tf_listener timer!
        rclpy.spin_until_future_complete(
            self, self._movement_finished_future, timeout_sec=timeout_per_operation
        )
        if not self._movement_finished_future.done():
            raise Exception("Robot reset timed out, critical.")

        # switch back to cartesian controller
        start_ts = time.monotonic()
        op_success = False
        while not op_success and time.monotonic() - start_ts < timeout_per_operation:
            msg = SwitchController.Request(
                deactivate_controllers=["position_controller"],
                activate_controllers=["full_controller"],
                strictness=SwitchController.Request.STRICT,
            )

            future = self._switch_controllers_client.call_async(msg)
            rclpy.spin_until_future_complete(self._reset_node, future)

            if future.result().ok:  # type: ignore
                op_success = True

        if not op_success:
            raise Exception("Switching back to 'full_controller' timed out, critical.")
    def reset_robot(self, joint_positions: NDArray[np.float64]):
        msg = Float64MultiArray(data=joint_positions)
        self._reset_robot_pub.publish(msg)
        time.sleep(5)
    def move_to_joint_positions(self, action):
        
        msg = Float64MultiArray(data=action)
        self._joint_position_pub.publish(msg)
        time.sleep(0.8)
        return 0

    def get_active_controller(self) -> str:
        """获取当前 active 控制器名称"""
        future = self._list_controllers_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self._reset_node, future)

        if future.done() and future.result():
            for controller in future.result().controller:
                if controller.state == "active":
                    return controller.name
        return None

    def switch_controller(self, deactivate_list, activate_list) -> bool:
        # 发送切换请求
        start_ts = time.monotonic()
        op_success = False
        while not op_success and time.monotonic() - start_ts < 5.0:
            req = SwitchController.Request(
                deactivate_controllers=deactivate_list,
                activate_controllers=activate_list,
                strictness=SwitchController.Request.STRICT,
            )

            future = self._switch_controllers_client.call_async(req)
            rclpy.spin_until_future_complete(self._reset_node, future)

            if future.result().ok:  # type: ignore
                op_success = True
        if not op_success:
            raise Exception("Switching to 'position_controller' timed out, critical.")


    def move_to_cartesian_position(
        self, position: NDArray[np.float64], rotation: NDArray[np.float64]
    ) -> bool:
        # setup
        self._movement_setup(
            target_position=array_to_point(position),
            target_rotation=array_to_quaternion(rotation),
        )

        # publish command
        msg = PoseStamped(
            header=Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link"),
            pose=Pose(
                position=self._target_position,
                orientation=self._target_rotation,
            ),
        )
        self._pub.publish(msg)

        # wait until arm has reached target position or does not move beyond tolerance
        # NOTE: Spinning this node activates the _robot_tf_listener timer!
        rclpy.spin_until_future_complete(
            self, self._movement_finished_future, timeout_sec=10
        )
        if not self._movement_finished_future.done():
            self._logger.info("Robot movement timed out.")

            return False

        return self._movement_finished_future.result()  # type: ignore

    def _check_movement_callback(self) -> None:
        current_tf = self._tf_buffer.lookup_transform(
            target_frame="base_link",
            source_frame="tool0",
            time=rclpy.time.Time(),
        ).transform

        self._tf_lookups_since_sample += 1

        current_pos = vector3_to_array(current_tf.translation)
        current_rot = quaternion_to_array(current_tf.rotation)
        try:
            if np.all(
                np.abs(current_pos - point_to_array(self._target_position)) < self.tolerance
            ) and np.all(
                np.abs(current_rot - quaternion_to_array(self._target_rotation))
                < self.tolerance
            ):  # robot has reached goal
                self._movement_finished_future.set_result(True)
                self._logger.info(
                    "Robot successfully reached target position and rotation."
                )
            elif (
                self._add_sample_to_tf_history(current_pos, current_rot)
                and self._tf_history_is_full
            ):  # check if robot might be stuck / not moving
                if np.all(
                    self._tf_history_position.ptp(axis=0) < self.no_movement_threshold
                ) and np.all(
                    self._tf_history_rotation.ptp(axis=0) < self.no_movement_threshold
                ):
                    self._movement_finished_future.set_result(False)
                    self._logger.info(
                        "Robot movement long enough below threshold, assume that robot can't reach target position or rotation."
                    )
        except Exception as e:
            return 0 

    def _add_sample_to_tf_history(
        self, pos_sample: NDArray[np.float64], rot_sample: NDArray[np.float64]
    ) -> bool:
        """Checks whether a new tf sample is needed, adds sample to history if so, and then returns if a sample was added."""

        if self._tf_lookups_since_sample < self._TF_HISTORY_SAMPLE_RATE:
            return False

        # => sample needs to be added
        self._tf_history_position[self._tf_history_next_idx] = pos_sample
        self._tf_history_rotation[self._tf_history_next_idx] = rot_sample

        self._tf_history_next_idx += 1
        self._tf_lookups_since_sample = 0

        if self._tf_history_next_idx == self._TF_HISTORY_SIZE:
            self._tf_history_next_idx = 0  # start at beginning again (ring buffer)

            if not self._tf_history_is_full:
                self._tf_history_is_full = True

        return True

    def _wait_until_tf_data_is_available(self, timeout_sec: float = 10):
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
            raise NodeInitializationError("Lookup of robot transform data timed out.")

        self._robot_tf_available_timer.destroy()
        delattr(self, "_robot_tf_available_timer")

    def close(self):
        self.destroy_node()


class NodeInitializationError(Exception):
    """Raise when node initialization fails."""
