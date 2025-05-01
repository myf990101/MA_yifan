import numpy as np
import rclpy
import gymnasium as gym
from gymnasium import spaces

from rclpy.impl.logging_severity import LoggingSeverity

from ..utils.ur10_cartesian_control_node import UR10CartesianControl
from ..utils.sim_env_control_node import SimEnvControl
from gym_gz_ws.gym_gz.observ_getter import ObservationGetter
from gym_gz_ws.gym_gz.plotRobot import RobotPloter
from numpy.linalg import norm
from pathlib import Path
import time
import json
from rclpy.executors import MultiThreadedExecutor
import logging
import threading
from stable_baselines3.common.logger import Logger


logger = logging.getLogger(__name__)

class UR5eEnv(gym.Env):
    def __init__(
        self,
        max_steps_per_episode: int,
        result_dir: Path,
        perform_measurements: bool,
        seed: int,
        gui: bool = True,
        max_reset_tries: int = 30,
        
    ):
        """
        Initialize the UR5e environment. The observation dimension is 18:
          - 6 joint positions,
          - 6 TCP (end-effector) info (3 for position + 3 for rotation),
          - 3 obstacle positions,
          - 3 obstacle velocities.
        """
        # Set observation dimension
        self.observation_dim = 18
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
        self.distance_to_target = 0.0
        self.obs = np.zeros(self.observation_dim)
        self.TCP_position = np.zeros(6)
        self.distance_comparsion = []
        self.min_value = np.array([0,0,0])
        self.GoalCounter = 0
        self.terminated = False
        self.truncated = False
        self.old_distance_to_target = 0.0
        # Initialize rclpy
        rclpy.init()
        # self.target = self.random_target()
        self.target = np.array([0.55243283 ,0.49988214, 0.19894235]) 
        print(f"🎯 New target position: {self.target} with seed :{str(self.seed)} ")
        # Initialize simulation environment control
        self._sim_env_control = SimEnvControl(
            controller_node_name="/position_controller",
            seed=self.seed,
            gui=True,
        )
        self._sim_env_control.restart_sim()
        self.observation_getter = ObservationGetter()
        
        self.init_joint_position = np.array([0,-1.5708, 1.5708, -1.5708, -1.5708, 0])
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32), # 3 dimensional action space due to inverse kinematics
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32))
        

         # joints (5), TCP (3), Goal Position (3)
        self.observation_space = spaces.Box(
            low=np.array( [-1,-1,-1,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-1,-1,-1,-1,-1,-1], dtype=np.float32),
            high=np.array([ 1, 1, 1, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1, 1, 1, 1, 1, 1], dtype=np.float32))
        
    # def get_obstacle_info(self):
    #     """
    #     Return obstacle information:
    #       - Obstacle position (randomly generated each call; adjust the coordinate range as needed)
    #       - Obstacle velocity (randomly generated each call)
    #       - Distance to the obstacle (for reward calculation)
    #       - Distance to the target (for reward calculation)
    #     """
    #     # Randomly generate obstacle position (example range: x in [0.5, 1.0], y in [0.0, 1.0], z in [0.0, 1.5])
    #     obstacle_position = np.zeros(3)
    #     # Randomly generate obstacle velocity (3-dimensional, range from -0.5 to 0.5)
    #     # obstacle_velocity = np.random.uniform(low=-0.5, high=0.5, size=3)
    #     obstacle_velocity = np.zeros(3)
    #     # Calculate distances using the current TCP position (assumes TCP_position is updated)
    #     distance_to_obstacle = norm(obstacle_position - self.TCP_position[:3])
    #     self.distance_to_target = norm(self.target - self.TCP_position[:3])
        
        # return obstacle_position, obstacle_velocity, distance_to_obstacle, self.distance_to_target

    def get_observation(self):
        # self.joint_positions = self.observation_getter.get_joint_service()
        # self.TCP_position = self.observation_getter.get_position_and_rotation()
        self.joint_positions,self.TCP_position = self.observation_getter.pass_joint_positions()
        # obstacle_position, obstacle_velocity, distance_to_obstacle, self.distance_to_target = self.get_obstacle_info()
        # Concatenate into an observation vector
        joint_noise = np.random.normal(0, 0.1, size=self.joint_positions.shape) 
        self.joint_positions = self.joint_positions + joint_noise  
        aobservation = np.concatenate((self.TCP_position,self.joint_positions,self.target,self.min_value))
         
        observation = 0
        return  np.array(aobservation, dtype=np.float32), np.array(observation, dtype=np.float32)  
    def get_action(self):
        if not hasattr(self, "action_file"):
            self.action_file = open("actions.txt", "r")
        
        line = self.action_file.readline()
        # 如果已到文件末尾，重置指针到开头并重新读取一行
        if not line:
            self.action_file.seek(0)
            line = self.action_file.readline()
        
        # 将读取的 JSON 字符串转换为 Python 对象，再转换为 numpy 数组
        action = json.loads(line.strip())
        return np.array(action, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.
        """

        # print("Resetting environment.")
        
        self.observation_getter.move_to_joint_positions(self.init_joint_position)
        
        self.angle_diff( self.init_joint_position)
        self.collision = False
        self.over_max_steps = False
        self.reached_goal = False
        self.steps = 0  # Reset step counter
        # self.target = self.random_target(0.4)
        # print(f"🎯 New target position: {self.target}")
        if seed is not None:
            np.random.seed(seed)
        self.obs,_ = self.get_observation()
        
        return self.obs ,{}  # Return observation and an empty info dictionary

    def step(self, action):
        self.get_observation()
        old_dist_to_goal =norm(self.TCP_position[:3] - self.target)
        self.distance_comparsion.append(old_dist_to_goal)
        coefficients = np.random.uniform(0.9, 1.1, size=action.shape)
        action=action*0.1
        self.joint_positions,_ = self.observation_getter.pass_joint_positions()
        actiongoal = self.joint_positions + action
        self.observation_getter.move_to_joint_positions(actiongoal)
        time.sleep(0.1)
        self.joint_positions,_ = self.observation_getter.pass_joint_positions()
        self.obs,_ = self.get_observation()
        self.observation_getter.get_all_transforms()  #self.TCP_position[2]< 0.1 or
        if  self.joint_positions[2] > 2.8  or self.joint_positions[2]<-2.8 or self.joint_positions[1]<-3.1415926 or self.joint_positions[1] > 0: ##or self.is_stuck: 
            print("self collision")
            self.collision = True
        new_dist_to_goal = norm(self.TCP_position[:3] - self.target)
        self.distance_comparsion.append(new_dist_to_goal)   
        self.reward = self.calculate_reward()
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

        self.old_distance_to_target = self.distance_to_target   
        return self.obs, self.reward, done, truncated, info

    def calculate_reward(self):
        # self.distance_to_target = norm(self.TCP_position[:3] - self.target)
        # # print(f"📏 Distance to target: {self.distance_to_target}")
        # self.angle=np.arccos(np.dot(self.TCP_position[3:6],([0,0,-1])))
        #     # reward = np.exp(-0.7*self.distance_to_target)
        # reward_position = - 0.001*(self.distance_to_target) ** 2 -np.log((self.distance_to_target)**2 + 0.0001) 
        # reward_orientation = 2*np.exp(-0.7*self.angle**2) -1
        # self.reward = reward_position + 0.5*reward_orientation
        # if self.distance_to_target < 0.2:
        #         if self.distance_comparsion[1] < self.distance_comparsion[0]:
        #             # print("getting closer to the goal")
        #             self.reward += 5
        #             if self.distance_to_target < 0.05:
        #                 self.GoalCounter += 1
        #                 self.reward += 200
        #                 if self.GoalCounter > 5:
        #                     print("Asymptotic stability")
        #                     self.reward += 300
        #                     self.reached_goal = True    
        #                     self.terminated = True
        #             else:
        #                     self.GoalCounter = 0
        # self.distance_comparsion = []
        # if self.collision :# or self.collision_ground or self.collision_obstacle:
        #     self.reward -= 100
        #     self.terminated = True

        # if self.steps > self.max_steps_per_episode:
        #     self.truncated = True
        
        # return self.reward
            self.distance_to_target = norm(self.TCP_position[:3] - self.target)
            self.angle=np.arccos(np.dot(self.TCP_position[3:6],([0,0,-1])))
            reward_position = - 0.001*(self.distance_to_target) ** 2 -np.log((self.distance_to_target)**2 + 0.0001) 
            reward_orientation = 2*np.exp(-0.7*self.angle**2) -1
            reward_obstacle = 0
            # if self.ObstacleEnv == True:
            #     for i in range(0,len(self.obstacles)):
            #         phi =max(0, 1 - (self.min_values[i])/0.05)
            #         reward_obstacle -= 0.1*phi
            # else: reward_obstacle = 0
            self.reward = reward_position  + reward_orientation + reward_obstacle
            # print(reward_position,reward_orientation,reward_obstacle)
            
            if self.distance_to_target < 0.2:
                if self.old_distance_to_target > self.distance_to_target:
                    # print(self.old_distance_to_target, self.distance_to_target)
                    # print("getting closer to the goal")
                    self.reward += 5
                    if self.distance_to_target < 0.05:
                        self.GoalCounter += 1
                        self.reward += 200
                        if self.GoalCounter > 5:
                            print("Asymptotic stability")
                            self.reward += 500
                            self.reached_goal = True    
                            self.terminated = True
                    else:
                            self.GoalCounter = 0
            self.distance_comparsion = []
            if self.collision :# or self.collision_ground or self.collision_obstacle:
                self.reward -= 50
                self.terminated = True

            if self.steps > self.max_steps_per_episode:
                self.truncated = True
            
            return self.reward
    def check_move_done(self, action):
        start_time = time.time()
        print(f"tolerance: {self.joint_positions-action}")
        while time.time() - start_time < 5:
            # current_joint_positions,_= self.observation_getter.pass_joint_posotions()
            error_threshold = max(0.05, 0.1 - self.steps * 0.001)  # 随训练逐渐收紧误差
            if np.allclose(self.joint_positions, action, atol=error_threshold):
                
                print("✅ robot has finished the movement")
                self.move_done = True
                return self.move_done 
        self.move_done = False
        self.collision = True
        
        # print(f"Current Joint Positions: {current_joint_positions}")    
        print("❌ robot has not finished the movement")
        # self.move_not_done_counter += 1
        return self.move_done 
    def random_target(self,interval):
        UpperBound = interval
        LowerBound = -interval
        x_base, y_base, z_base = 0.5, 0.3, 0.5

        # 生成新目标点
        x = np.random.uniform(x_base + LowerBound, x_base + UpperBound)
        y = np.random.uniform(y_base + LowerBound, y_base + UpperBound)
        z = np.random.uniform(z_base + LowerBound, z_base + UpperBound)

        # 若目标点不满足条件，则重新生成
        while np.sqrt(x**2 + y**2 + z**2) > 0.85 or z < 0.1:
            x = np.random.uniform(x_base + LowerBound, x_base + UpperBound)
            y = np.random.uniform(y_base + LowerBound, y_base + UpperBound)
            z = np.random.uniform(z_base + LowerBound, z_base + UpperBound)

        # 更新目标点
        self.new_goal = np.array([x, y, z])
        self.reached_goal_counter = 0 


        # print("Goal Position:", self.new_goal)
        return self.new_goal
    def render(self):
        pass

    def log_data(self, observation, action):
   
        import json
        # 如果为 numpy 数组，则转换为 list
        if isinstance(observation, np.ndarray):
            observation = observation.tolist()
        if isinstance(action, np.ndarray):
            action = action.tolist()
            
        # 将 observation 以 JSON 格式追加写入 observations.txt
        with open("joint_p.txt", "a") as obs_file:
            obs_file.write(json.dumps(observation) + "\n")
        
        # 将 action 以 JSON 格式追加写入 actions.txt
        with open("act.txt", "a") as act_file:
            act_file.write(json.dumps(action) + "\n")
    def log_joint(self,gazebo ,pybullet ):
   
        import json
        # 如果为 numpy 数组，则转换为 list
        if isinstance(pybullet, np.ndarray):
            pybullet = pybullet.tolist()
        if isinstance(gazebo, np.ndarray):
            gazebo = gazebo.tolist()
            
        # 将 observation 以 JSON 格式追加写入 observations.txt
        with open(f"p.txt", "a") as obs_file:
            obs_file.write(json.dumps(pybullet) + "\n")
        
        # 将 action 以 JSON 格式追加写入 actions.txt
        with open(f"g.txt", "a") as act_file:
            act_file.write(json.dumps(gazebo) + "\n")
    def angle_diff(self, b):
        
        timeout = 2  # 超时时间，可根据需要调整
        start_time = time.monotonic()
        movement_finished = False
        tolerance = 0.01  # 定义允许的角度差误差（弧度）
        while time.monotonic() - start_time < timeout:
            a,_=self.observation_getter.pass_joint_positions()
            diff = ( a- b + np.pi) % (2 * np.pi) - np.pi
            # print(f"diff: {diff}")
            if np.all(np.abs(diff[:5]) < tolerance):
                print("reset finished.")
                break  # 条件满足后退出循环
            time.sleep(0.1) 
        return np.abs(diff) 
    def close(self):

        logger.debug("Environment is closing...")
        self._sim_env_control.stop_sim()
        # self.executor.shutdown()
        # self._executor_thread.join(timeout=5)
        # self.controller.destroy_node()
        self.observation_getter.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
        print("✅ rclpy shutdown completed.")
