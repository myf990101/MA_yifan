import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(file_path):
    """
    读取数据文件并绘制 reward 随 episode 变化的折线图。
    :param file_path: 数据文件路径，每行格式应为 'reward,length,time'
    """
    rewards = []
    
    # 读取数据
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("r,l,t"):  # 跳过非数据行
                continue
            values = line.split(',')
            try:
                rewards.append(float(values[0]))  # 取第一列作为 reward
            except ValueError:
                print(f"跳过无效行: {line}")  # 调试时可查看哪些行出错
    
    if not rewards:
        print("错误：未能解析有效的 reward 数据，请检查文件格式！")
        return

    episodes = np.arange(1, len(rewards) + 1)

    # 绘制图像
    plt.figure(figsize=(20, 10))
    plt.plot(episodes, rewards, marker='', linestyle='-', label="Reward per Episode", color='blue')
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.title(f"Reward per Episode (Total Episodes: {len(rewards)})")
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例用法，替换为你的实际文件路径
plot_rewards('/workspaces/gazebo/ur5e_results/PPO_20250416_150347/monitor.csv')
