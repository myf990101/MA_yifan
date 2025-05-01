import numpy as np

import rclpy
import gymnasium as gym
from gymnasium import spaces

from rclpy.impl.logging_severity import LoggingSeverity

from ..utils.ur10_cartesian_control_node import UR10CartesianControl
from ..utils.sim_env_control_node import SimEnvControl

from numpy.linalg import norm
from pathlib import Path
import time
import json
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Timestamp types
Timestamp = Enum(
    "Timestamp",
    [
        "GYM_TIME_START",
        "GYM_TIME_END",
        "MODEL_UPDATE",
        "SIM_MOVEMENT_START",
        "SIM_MOVEMENT_END",
        "RESET_START",
        "RESET_END",
    ],
)


class GymGzEnv(gym.Env):
    def __init__(
        self,
        max_steps_per_episode: int,
        result_dir: Path,
        perform_measurements: bool,
        seed: int,
        gui: bool = False,
        max_reset_tries: int = 30,
        render_mode: str | None = None,
    ):
        # this file assumes ros2, install/setup.bash, and the venv are sourced

        ### measurements ###
        # perf
        self.result_dir = result_dir
        self.timestamps = []
        self.perform_measurements = perform_measurements

        def _measure(ts: Timestamp):  # defined here so everything is close
            if self.perform_measurements:
                self.timestamps.append((ts, time.perf_counter_ns()))

        self.measure = _measure

        self.steps = None
        self.episode_data = []
        ###
        self.MAX_STEPS_PER_EPISODE = max_steps_per_episode

        self.SEED = seed
        self.MAX_RESET_TRIES = max_reset_tries

        self.MAX_MOVEMENT = 0.0015
        self.TARGET_COORDINATES = np.array([0.8, 0.5, 0.9])  # old
        self.TARGET_TOLERANCE = 0.1
        self._last_dist = 0

        ### control
        rclpy.init()

        self._sim_env_control = SimEnvControl(
            controller_node_name="/full_controller",
            seed=self.SEED,
            gui=gui,
        )
        self._sim_env_control.restart_sim()

        self._ur_controller = UR10CartesianControl(log_level=LoggingSeverity.WARN)

        self.INITIAL_POSITION, self.INITIAL_ROTATION = (
            self._ur_controller.get_position_and_rotation()
        )
        logger.debug(f"Initial position: {self.INITIAL_POSITION}")
        self.INITIAL_JOINT_POSITIONS = np.array([0.23, -0.37, -1.05, 1.42, 0.22, 0.0])
        self._reset_robot_to_initial_state()
        ###

        # V1: Only position
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([8.0]),
        )

    def get_sim_timestamp(self, timeout_sec: float = 5.0) -> int:
        """Get current timestamp of simulation

        :return: Nanoseconds since beginning of simulation
        """
        return self._sim_env_control.get_sim_timestamp(timeout_sec=timeout_sec)

    def _get_obs_and_info(self):
        pos_obs, _ = self._ur_controller.get_position_and_rotation()
        
        distance_to_target = norm(pos_obs - self.TARGET_COORDINATES)
        obs = np.array([distance_to_target])
        
        info = {"dist": distance_to_target}

        return obs, info 

    def _reset_robot_to_initial_state(self):
        self._ur_controller.reset_joints(
            self.INITIAL_JOINT_POSITIONS,
            self.INITIAL_POSITION,
            self.INITIAL_ROTATION,
        )

    def reset(self, seed=None, options=None):
        if self.perform_measurements and self.steps is not None:
            logger.info("Episode finished.")
            logger.info(
                f"Previous run terminated after {self.steps} steps and "
                + (
                    "reached the target."
                    if self.target_reached
                    else "did not reach the target."
                )
            )
            self.episode_data.append((self.steps, self.target_reached, self._last_dist))

        logger.debug("Resetting environment...")

        ### start: reset env
        self.measure(Timestamp.RESET_START)
        super().reset(seed=self.SEED)

        self.steps = 0
        self.target_reached = False

        # reset the robot arm to initial position
        self._reset_robot_to_initial_state()

        obs, info = self._get_obs_and_info()
        self._last_dist = info["dist"]

        self.measure(Timestamp.RESET_END)
        ### end: reset env

        logger.debug("Resetting finished.")
        return obs, info

    def step(self, action):
        self.measure(Timestamp.GYM_TIME_END)
        ### end: gym time

        # perform action
        cur_pos, _ = self._ur_controller.get_position_and_rotation()
        next_pos = cur_pos + action * self.MAX_MOVEMENT

        ### start: simulation movement
        self.measure(Timestamp.SIM_MOVEMENT_START)

        success = self._ur_controller.move_to_cartesian_position(
            next_pos, self.INITIAL_ROTATION
        )

        self.measure(Timestamp.SIM_MOVEMENT_END)
        ### end: simulation movement

        if not success:
            logger.warning("Robot movement command not successful.")
            logger.warning("Ending episode.")

        # observe new state
        observation, info = self._get_obs_and_info()
        terminated = info["dist"] <= self.TARGET_TOLERANCE

        if self.steps % 100 == 0:  # type: ignore
            logger.debug(f"Dist: {info['dist']}")

        self.steps += 1  # type: ignore
        self.target_reached = terminated

        reward = 1 if info["dist"] < self._last_dist else -1
        self._last_dist = info["dist"]

        if terminated:
            reward = self.MAX_STEPS_PER_EPISODE
        if not success:
            reward = -self.MAX_STEPS_PER_EPISODE

        # start: gym time
        self.measure(Timestamp.GYM_TIME_START)
        return observation, reward, terminated, not success, info

    def render(self): ...

    def close(self):
        logger.info("Environment is closing..")
        # save measurements if applicable
        # if self.perform_measurements:
        #     # save timestamps
        #     with open(self.result_dir.joinpath("sim-timestamps.json"), "w") as file:
        #         json.dump(
        #             {
        #                 str(id): {"type": ts_type.name, "ts": str(ts)}
        #                 for id, (ts_type, ts) in enumerate(self.timestamps)
        #             },
        #             file,
        #         )

        #     # save steps needed per episode
        #     with open(
        #         self.result_dir.joinpath("sim-steps-per-episode.json"), "w"
        #     ) as file:
        #         minimum_steps_needed = str(
        #             int(
        #                 np.ceil(
        #                     (
        #                         norm(self.INITIAL_POSITION - self.TARGET_COORDINATES)
        #                         - self.TARGET_TOLERANCE
        #                     )
        #                     / self.MAX_MOVEMENT
        #                 )
        #             )
        #         )

        #         json.dump(
        #             {
        #                 "minimum_steps_needed": minimum_steps_needed,
        #                 "measurements": {
        #                     str(episode): {
        #                         "steps": steps,
        #                         "reached": str(reached),
        #                         "last_dist": str(last_dist),
        #                     }
        #                     for episode, (steps, reached, last_dist) in enumerate(
        #                         self.episode_data
        #                     )
        #                 },
        #             },
        #             file,
        #         )

        # cleanly stop env
        self._sim_env_control.stop_sim()
        self._ur_controller.close()
        try:
            rclpy.shutdown()
        except Exception as e:
            logger.critical("rclpy.shutdown failed...")
            logger.critical(e)
            # raise e
