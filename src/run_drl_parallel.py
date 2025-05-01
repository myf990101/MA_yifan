import gymnasium as gym
import torch 
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
from gym_gz_ws.gym_gz.envs.ur5e_env_parallel import UR5eEnvParallel
# import coloredlogs
import logging
from datetime import datetime
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import multiprocessing
import shutil

# ---------------------------
# 配置日志系统
# ---------------------------
# coloredlogs.install(
#     fmt="%(asctime)s %(levelname)-6s %(name)s: %(message)s",
#     datefmt="%H:%M:%S",
#     level=logging.INFO
# )

# ---------------------------
# 全局配置参数
# ---------------------------
SEED = 42
MAX_STEPS_PER_EPISODE = 100
NUM_ENVS = 2  # ✅ 并行环境数，建议等于CPU物理核心数
TOTAL_TIMESTEPS = int(1e6)  # 总训练步数
SAVE_FREQ = 3000  # 每多少步保存一次检查点

# ---------------------------
# 路径配置
# ---------------------------
RESULT_DIR = Path("ur5e_results")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
PPO_DIR = RESULT_DIR / f"PPO_{timestamp}"
PPO_DIR.mkdir(parents=True, exist_ok=True)

# 各组件存储路径
MODEL_SAVE_PATH = PPO_DIR / "ppo_ur5e_final.zip"
TENSORBOARD_LOG_DIR = PPO_DIR / "tensorboard"
CHECKPOINT_DIR = PPO_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 环境创建函数（关键修改点）
# ---------------------------
def make_env(seed: int, rank: int) -> callable:
    """
    创建单个环境的工厂函数
    :param seed: 基础随机种子
    :param rank: 环境编号 (0 <= rank < NUM_ENVS)
    :return: 环境初始化函数
    """
    def _init() -> gym.Env:
        # 关键设置：只有第一个环境显示GUI
        current_seed = seed + rank
        print(f"Initializing environment {rank} with seed {current_seed}")
        env = UR5eEnvParallel(
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
            result_dir=RESULT_DIR,
            perform_measurements=True,
            seed=seed + rank,  # 确保不同环境不同随机性    # 仅第一个环境显示GUI
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE)
        env = Monitor(env, str(PPO_DIR), info_keywords=('distance_to_target', 'collision', 'reached_goal'))
        return env
    return _init

# ---------------------------
# 主训练流程
# ---------------------------
if __name__ == "__main__":  # ✅ Windows多进程必须的保护
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn')
    
    # 固定随机种子
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ---------------------------
    # 创建并行环境（核心修改）
    # ---------------------------
    env = SubprocVecEnv([make_env(SEED, i)for i in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)  # 自动归一化
    print(f"✅ 已创建并行环境数量: {env.num_envs}")
    # ---------------------------
    # 模型超参数（根据并行调整）
    # ---------------------------
    hyperparams = {
        "policy_type": "MlpPolicy",
        "batch_size": 128,      # 需能被 (n_steps * NUM_ENVS) 整除
        "n_steps": 256,         # 每个环境每轮步数 → 总步数=256*4=1024
        "gamma": 0.99,
        "learning_rate": 3e-4,  # 并行训练建议稍低的学习率
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "device": "auto",       # 自动选择GPU/CPU
        "verbose": 1
    }

    # ---------------------------
    # 初始化PPO模型
    # ---------------------------
    model = PPO(
        hyperparams["policy_type"],
        env,
        **{k: v for k, v in hyperparams.items() if k != "policy_type"},
        tensorboard_log=str(TENSORBOARD_LOG_DIR)
    )

    # ---------------------------
    # 回调函数配置
    # ---------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # 调整实际保存频率
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_ur5e_checkpoint",
        save_vecnormalize=True  # ✅ 关键：保存归一化参数
    )
    # early_stop_callback = StopTrainingOnMaxEpisodes(max_episodes=EPISODES, verbose=1)

    # ---------------------------
    # 配置日志记录
    # ---------------------------
    logger = configure(str(PPO_DIR), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # 记录超参数
    for key, value in hyperparams.items():
        model.logger.record(f"hyperparams/{key}", value)
    model.logger.record("hyperparams/num_envs", NUM_ENVS)
    model.logger.dump(step=0)

    # ---------------------------
    # 训练循环（带安全保存）
    # ---------------------------
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback],
            tb_log_name="PPO_UR5e",
            progress_bar=True  # 显示进度条
        )
    except KeyboardInterrupt:
        logging.warning("训练被中断，正在保存最新模型...")
    finally:
        # 确保最终保存
        model.save(str(MODEL_SAVE_PATH))
        env.save(str(PPO_DIR / "vec_normalize.pkl"))  # 保存最终归一化参数
        env.close()
        logging.info(f"模型已保存至: {MODEL_SAVE_PATH}")

# ---------------------------
# 清理空文件夹（可选）
# ---------------------------
def cleanup_empty_folders(result_dir: Path):
    """删除没有检查点的训练文件夹"""
    for folder in result_dir.glob("PPO_*"):
        checkpoint_dir = folder / "checkpoints"
        if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
            shutil.rmtree(folder, ignore_errors=True)

if __name__ == "__main__":
    cleanup_empty_folders(RESULT_DIR)