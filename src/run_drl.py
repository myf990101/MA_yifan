import gymnasium as gym
import torch 
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
from gym_gz_ws.gym_gz.envs.ur5e_env import UR5eEnv
# import coloredlogs
import logging
from datetime import datetime
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing
from stable_baselines3.common.vec_env import VecNormalize
# 日志
# coloredlogs.install(fmt="%(asctime)s %(levelname)-6s %(name)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
import shutil
from pathlib import Path

# 设定超参数
SEED = 2
EPISODES = 100000 # 提前终止
MAX_STEPS_PER_EPISODE = 150
NUM_ENVS = 1

RESULT_DIR = Path("ur5e_results")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成唯一时间戳
PPO_DIR = RESULT_DIR / f"PPO_{timestamp}"  # 创建独立的 PPO 文件夹
PPO_DIR.mkdir(parents=True, exist_ok=True)

# 设定所有存储路径（全部存 PPO_DIR 里）
MODEL_SAVE_PATH = PPO_DIR / "ppo_ur5e.zip"  # 存放训练好的模型
TENSORBOARD_LOG_DIR = PPO_DIR / "tensorboard"  # 存放 TensorBoard 日志
CHECKPOINT_DIR = PPO_DIR / "checkpoints"  # 存放训练过程中的模型检查点
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 设定 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 设定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

env = UR5eEnv(max_steps_per_episode=MAX_STEPS_PER_EPISODE,
            result_dir=RESULT_DIR,
            perform_measurements=True, 
            seed=SEED,
            gui=True 
            )
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE)
env = Monitor(env, str(PPO_DIR),info_keywords=('distance_to_target','collision','reached_goal',"target"),)  # 监控训练

# 模型回调
checkpoint_callback = CheckpointCallback(save_freq=5e5, save_path=str(CHECKPOINT_DIR), name_prefix="ppo_ur5e_checkpoint")
early_stop_callback = StopTrainingOnMaxEpisodes(max_episodes=EPISODES, verbose=1)
#hyperparameters
batch_size = 1024
gamma = 0.99
n_steps = 4096
learning_rate = 1e-4
ent_coef = 0.01
clip_range = 0.2
policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                       net_arch=dict(pi=[256,256, 256], vf=[256, 256,256]))  
custom_objects = {
    "clip_range": 0.2,
    "lr_schedule": lambda _: 1e-4  # 或者直接指定固定值
} 
old_model = PPO.load(
    str('/workspaces/gazebo/ur5e_results/good_result/Reach_random_goal.zip'),
    env=env,custom_objects=custom_objects)

# 设定 PPO 模型

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=SEED,
    device='cpu',
    batch_size=batch_size,  # 批量大小
    n_steps=n_steps,     # 每次更新步数
    gamma=gamma,      # 折扣因子
    learning_rate=learning_rate,  # 学习率
    ent_coef=ent_coef,   # 熵系数
    tensorboard_log=str(TENSORBOARD_LOG_DIR) ,
    clip_range=clip_range,  # 限制比例
    policy_kwargs= policy_kwargs  # 神经网络结构
)
model.policy.load_state_dict(old_model.policy.state_dict())

new_logger = configure(str(PPO_DIR), ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)  # 添加 Tensorboard 记录

model.logger.record("hyperparams/gamma", gamma)
model.logger.record("hyperparams/batch_size", batch_size)
model.logger.record("hyperparams/seed", SEED)
model.logger.record("hyperparams/n_steps", n_steps)
model.logger.record("hyperparams/learning_rate", learning_rate)
model.logger.record("hyperparams/ent_coef", ent_coef)
model.logger.record("hyperparams/clip_range", clip_range)
model.logger.dump(step=0)  


# 训练
try:
     model.learn(total_timesteps=int(1e6),
                tb_log_name="PPO_UR5e",
                callback=[checkpoint_callback, early_stop_callback],
                log_interval=1,
                progress_bar=True)
    
except KeyboardInterrupt:
    print("🛑 训练被手动中断，安全退出...")
    model.save(str(MODEL_SAVE_PATH))
finally:
    env.close()
    
# 保存模型



print(f"🎉 训练完成，模型已保存至: {MODEL_SAVE_PATH}")

def cleanup_empty_timestamp_folders(result_dir: Path):
    """
    Traverse through all folders in result_dir that start with "PPO_".
    If the 'checkpoints' folder does not exist or is empty, delete the entire timestamp folder.
    """
    # Traverse all subfolders starting with "PPO_"
    for folder in result_dir.glob("PPO_*"):
        checkpoint_dir = folder / "checkpoints"
        # Check if the 'checkpoints' folder exists and is a directory
        if checkpoint_dir.exists() and checkpoint_dir.is_dir():
            # Check if the 'checkpoints' folder contains any files or subdirectories
            if not any(checkpoint_dir.iterdir()):
                print(f"Deleting folder: {folder}, because its checkpoints folder is empty")
                shutil.rmtree(folder)
            else:
                print(f"Keeping folder: {folder}, because its checkpoints folder contains files")
        else:
            # If there is no 'checkpoints' folder, delete the entire folder
            print(f"Deleting folder: {folder}, because the checkpoints folder does not exist")
            shutil.rmtree(folder)
if __name__ == "__main__":
    result_directory = Path("ur5e_results")
    cleanup_empty_timestamp_folders(result_directory)