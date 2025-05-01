import numpy as np
import rclpy
import gymnasium as gym
from gymnasium import spaces

from rclpy.impl.logging_severity import LoggingSeverity

from ..utils.ur10_cartesian_control_node import UR10CartesianControl
from ..utils.sim_env_control_node import SimEnvControl
from gym_gz_ws.gym_gz.observ_getter import Observation_getter
from numpy.linalg import norm
from pathlib import Path
import time
import json
from rclpy.executors import MultiThreadedExecutor
import logging
import threading
from stable_baselines3.common.logger import Logger


logger = logging.getLogger(__name__)

class UR5eEnvParallel(gym.Env):
    def __init__(
        self,
        max_steps_per_episode: int,
        result_dir: Path,
        perform_measurements: bool,
        seed: int,
        gui: bool = True,
        max_reset_tries: int = 30,
        render_mode: str | None = None,
    ):
        """
        Initialize the UR5e environment. The observation dimension is 18:
          - 6 joint positions,
          - 6 TCP (end-effector) info (3 for position + 3 for rotation),
          - 3 obstacle positions,
          - 3 obstacle velocities.
        """
        # Set observation dimension
        self.observation_dim = 13  
        self.result_dir = result_dir
        self.timestamps = []
        self.seed = seed
        self.gui = gui
        self.max_reset_tries = max_reset_tries
        self.max_steps_per_episode = max_steps_per_episode
        # Status flags
        self.collision = False
        self.over_max_steps = False
        self.reached_goal = False
        self.reward = 0.0
        self.steps = 0  # Initialize step counter
        self.distance_comparsion = []
        self.distance_log = []
        self.model = None
        self.move_done_cunter = 0
        self.move_not_done_counter = 0
        self.distance_to_target = 0.0
        self.last_distance_to_target = None
        self.stuck_threshold = 5  # 连续多少步没有明显进展认为卡死
        self.stuck_counter = 0
        self.min_progress = 0.001  # 两次距离差小于此值认为没有进展
        self.already_closed = False 
        self.is_stuck = False
        # Initialize logger

        # Initialize rclpy
        rclpy.init()
        # self.executor = MultiThreadedExecutor()
        # Set target position (adjust as needed)
        self.target = self.random_target()
        print(f"🎯 New target position: {self.target} with seed :{str(self.seed)} ")
        # Initialize simulation environment control
        self._sim_env_control = SimEnvControl(
            controller_node_name="/position_controller",
            seed=self.seed,
            gui=gui,
        )
        self._sim_env_control.restart_sim()
        # Wait for the simulation to start
        # Initialize robot controller
        # self.controller = UR10CartesianControl(log_level=LoggingSeverity.WARN)
        self.observation_getter = Observation_getter()
        # self.executor.add_node(self.controller)
        # self.executor.add_node(self.observation_getter)
        # self._executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        # self._executor_thread.start()
        # print("✅ Executor 线程已启动！")
        
        self.init_joint_position = np.array([0,-1.5708, 0, 0, 0, 0])
        
        # self.current_controller = self.controller.get_active_controller()
        # if self.current_controller == "full_controller":
        #     self.controller.switch_controller(["full_controller"], ["position_controller"])
        #     self.current_controller = self.controller.get_active_controller()
        #     print(f'current controller: {self.current_controller}')
        # Define observation space (18-dimensional continuous space)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        # Define action space based on UR5e joint limits (6 joints)
        # joints_low_limits = np.array([
        #     -6.28318531, -6.28318531, -3.14159265,
        #     -6.28318531, -6.28318531, -6.28318531
        # ])
        # joints_high_limits = np.array([
        #     6.28318531, 6.28318531, 3.14159265,
        #     6.28318531, 6.28318531, 6.28318531
        # ])
        delta_q_limit = 1
        joints_low_limits = np.full(6, -delta_q_limit)
        joints_high_limits = np.full(6, delta_q_limit)
        self.action_space = spaces.Box(
            low=joints_low_limits,
            high=joints_high_limits,
            dtype=np.float32
        )
    # def start_ros_thread(self):
    #     """✅ 在主进程运行 rclpy 线程"""
    #     if self.ros_thread is None:
    #         self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
    #         self.ros_thread.start()
    #         print("✅ ROS 线程已启动")
    def get_obstacle_info(self):
        """
        Return obstacle information:
          - Obstacle position (randomly generated each call; adjust the coordinate range as needed)
          - Obstacle velocity (randomly generated each call)
          - Distance to the obstacle (for reward calculation)
          - Distance to the target (for reward calculation)
        """
        # Randomly generate obstacle position (example range: x in [0.5, 1.0], y in [0.0, 1.0], z in [0.0, 1.5])
        obstacle_position = np.zeros(3)
        # Randomly generate obstacle velocity (3-dimensional, range from -0.5 to 0.5)
        # obstacle_velocity = np.random.uniform(low=-0.5, high=0.5, size=3)
        obstacle_velocity = np.zeros(3)
        # Calculate distances using the current TCP position (assumes tcp_position is updated)
        distance_to_obstacle = norm(obstacle_position - self.tcp_position[:3])
        self.distance_to_target = norm(self.target - self.tcp_position[:3])
        
        return obstacle_position, obstacle_velocity, distance_to_obstacle, self.distance_to_target

    def get_observation(self):
        """
        Construct an 18-dimensional observation vector:
          [joint positions (6), TCP info (6), obstacle position (3), obstacle velocity (3)]
        """
        # rclpy.spin_once(self.observation_getter, timeout_sec=0.5)
        self.joint_positions,self.tcp_position = self.observation_getter.pass_joint_posotions()
        obstacle_position, obstacle_velocity, distance_to_obstacle, self.distance_to_target = self.get_obstacle_info()
        # Concatenate into an observation vector
        observation = np.concatenate((self.joint_positions, self.tcp_position, np.array([self.distance_to_target])))
        return observation.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.
        """

        print("Resetting environment.")
        
        self.observation_getter.move_to_joint_positions(self.init_joint_position)
        current_position, _ = self.observation_getter.pass_joint_posotions()
        print("🔍 Current joint position after reset:", current_position)
        start_time = time.time()
        while time.time() - start_time < 8:
            current_position, _ = self.observation_getter.pass_joint_posotions()
            if np.allclose(current_position, self.init_joint_position, atol=0.05):
                print("✅ Robot successfully reset!")
                break
            print("⏳ Waiting for reset...")
            time.sleep(0.5)  # 间隔 0.5 秒检查一次

        else:
            print(f"⏰ Reset timeout! Robot did not reach the target position within 5 seconds.")
        self.collision = False
        self.over_max_steps = False
        self.reached_goal = False
        self.steps = 0  # Reset step counter
        self.stuck_counter = 0
        self.is_stuck = False
        
        if seed is not None:
            np.random.seed(seed)

        # Additional logic to reset the robot state can be added here
        
        obs = self.get_observation()
        return obs, {}  # Return observation and an empty info dictionary

    def step(self, action):
        
        obs = self.get_observation()
        self.last_distance_to_target = self.distance_to_target
        # Execute the action
        action=action*0.1
        action_goal = self.joint_positions+action
        self.observation_getter.move_to_joint_positions(action_goal)
        # 获取新的观测
        obs = self.get_observation()
        # print(f"obs: {obs}")
        self.check_stuck(self.distance_to_target)
        
        if obs[1]> 0.1 or obs[2] > 2.8  or obs[2]<-2.8 : ##or self.is_stuck: 
            self.collision = True
        # self.distance_comparsion.append(norm(self.target - self.tcp_position[:3]))
        print(f"Distance to target: {self.distance_to_target}")
        # self.check_move_done(self.joint_positions+action)
        self.check_reach_goal(self.distance_to_target)
        self.reward = self.calculate_reward()
        # print(f"Reward: {self.reward}")
        self.steps += 1
      
        done = self.collision  or self.reached_goal
        truncated = self.steps >= self.max_steps_per_episode
        
        info = {
        "target": self.target,
        "distance_to_target": self.distance_to_target,
        "steps": self.steps,
        "collision": self.collision,
        "reached_goal": self.reached_goal
    }


        return obs, self.reward, done, truncated, info

    def calculate_reward(self):
        """
        Calculate reward based on distances to the obstacle and the target.
        Example calculation:
          - Negative quadratic penalty for the target distance.
          - Inverse proportional penalty for the obstacle distance.
        """
        _, _, distance_to_obstacle, distance_to_target = self.get_obstacle_info()
        # reward_target = - 250*(distance_to_target) ** 2
        reward_target = np.exp(-0.7 * distance_to_target)
        reward_obstacle = -0.5 * (1 / (1 + distance_to_obstacle)) ** 35
        # reward = -500 * reward_target - 15 * reward_obstacle\
        
        self.reward = reward_target 
        if self.reached_goal:
            self.reward += 300
        
        if self.collision:
            self.reward -= 500
        # if not self.move_done:
        #     self.reward -= 10
        # if self.get_closer_to_goal:
        #     self.reward += 3
        # else:
        #     self.reward -= 1
        
        return self.reward
    def check_move_done(self, action):
        start_time = time.time()
        print(f"tolerance: {self.joint_positions-action}")
        while time.time() - start_time < 5:
            # current_joint_positions,_= self.observation_getter.pass_joint_posotions()
            error_threshold = max(0.05, 0.1 - self.steps * 0.001)  # 随训练逐渐收紧误差
            if np.allclose(self.joint_positions, action, atol=error_threshold):
                
                print("✅ robot has finished the movement")
                # self.move_done_cunter += 1
                # self.move_not_done_counter = 0
                self.move_done = True
                return self.move_done 
        self.move_done = False
        self.collision = True
        
        # print(f"Current Joint Positions: {current_joint_positions}")    
        print("❌ robot has not finished the movement")
        # self.move_not_done_counter += 1
        return self.move_done 
    def check_reach_goal(self,distance_to_target):
        """
        Check if the TCP is close enough to the target position.
        """
        if distance_to_target< 0.05:
            self.reached_goal = True
        else:self.reached_goal = False
        # if disntance_comparsion[1] < disntance_comparsion[0]:
        #     self.get_closer_to_goal = True
        # else:
        #     self.get_closer_to_goal = False
        # self.distance_comparsion = []
        return self.reached_goal
    def random_target(self):
    
        x = np.random.uniform(0.1, 0.45)
        y = np.random.uniform(0.1, 0.45)
        z = np.random.uniform(0.1, 0.45)
        return np.array([x, y, z])
    def check_stuck(self, current_distance: float) -> bool:
        """
        判断是否卡死：如果连续 self.stuck_threshold 步内，没有显著减少距离，认为卡死。
        返回 True/False 表示当前这一步是否卡死。
        """
        # 如果是本回合第一次调用，没有上一时刻距离，重置计数并初始化
        if self.last_distance_to_target is None:
            self.last_distance_to_target = current_distance
            self.stuck_counter = 0
            return False

        # 如果当前距离与上一时刻距离差别不足 min_progress，说明进展不够
        if np.abs(current_distance-self.last_distance_to_target) <= self.min_progress:
            self.stuck_counter += 1
        else:
            # 一旦产生明显进展，就重置计数
            self.stuck_counter = 0

        # 更新 last_distance
        self.last_distance_to_target = current_distance
        if self.stuck_counter >= self.stuck_threshold :
            self.is_stuck = True
        # 若 stuck_counter 超过阈值，返回 True
        return self.is_stuck
    def render(self):
        pass
    def close(self):
        if self.already_closed:
            return 0
        self.already_closed = True

        logger.debug("Environment is closing...")
        self._sim_env_control.stop_sim()
        # self.executor.shutdown()
        # self._executor_thread.join(timeout=5)
        # self.controller.destroy_node()
        self.observation_getter.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
        print("✅ rclpy shutdown completed.")
