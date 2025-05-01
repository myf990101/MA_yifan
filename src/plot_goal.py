import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图

# 替换为你实际的 CSV 文件路径
csv_file_path = '/workspaces/gazebo/ur5e_results/PPO_20250416_150347/monitor.csv'

# 初始化三个列表用于保存不同分类的目标点（3D 坐标）以及对应的 distance_to_target 数值
collision_pts = []      # 碰撞（collision 为 "True"）的目标点
reached_pts   = []      # 完成任务（collision 为 "False" 且 reached_goal 为 "True"）的目标点
other_pts     = []      # 其它情况（既未碰撞也未完成任务）的目标点

collision_distances = []  # 对应的 distance_to_target 数值
reached_distances   = []
other_distances     = []

# 读取 CSV 文件，按行处理（跳过第一行表头）
with open(csv_file_path, 'r') as f:
    lines = f.readlines()

# 遍历数据（跳过表头）
for line in lines[10000:12000]:
    line = line.strip()
    if not line:
        continue
    parts = line.split(',')
    # 检查数据是否足够（至少有 7 列）
    if len(parts) < 7:
        continue

    # 第四列为 distance_to_target（索引3）
    try:
        distance_val = float(parts[3].strip())
    except ValueError:
        continue

    collision_val = parts[4].strip()
    reached_val   = parts[5].strip()

    # 解析 target 字段（例如 "[ 0.44879592 -0.07925901  0.53972998]"）
    target_field = parts[6].strip().strip('[]')
    try:
        coords = [float(x) for x in target_field.split() if x]
    except ValueError:
        continue
    if len(coords) < 3:
        continue

    # 取前三个数作为 (x, y, z)
    point = (coords[0], coords[1], coords[2])

    if collision_val == "True":
        collision_pts.append(point)
        collision_distances.append(distance_val)
    elif reached_val == "True":
        reached_pts.append(point)
        reached_distances.append(distance_val)
    else:
        other_pts.append(point)
        other_distances.append(distance_val)

# 转换为 numpy 数组（若列表为空则创建空数组）
collision_pts = np.array(collision_pts) if collision_pts else np.empty((0, 3))
reached_pts   = np.array(reached_pts)   if reached_pts   else np.empty((0, 3))
other_pts     = np.array(other_pts)     if other_pts     else np.empty((0, 3))

# 总数据点数量
total_count = collision_pts.shape[0] + reached_pts.shape[0] + other_pts.shape[0]

# 定义函数计算均值
def compute_avg(lst):
    if len(lst) == 0:
        return None
    return np.mean(lst)

collision_avg = compute_avg(collision_distances)
reached_avg   = compute_avg(reached_distances)
other_avg     = compute_avg(other_distances)

print("Average distance_to_target (Collision):", collision_avg)
print("Average distance_to_target (Reached Goal):", reached_avg)
print("Average distance_to_target (Other):", other_avg)

# 计算各组数量及占总体百分比
collision_count = collision_pts.shape[0]
reached_count   = reached_pts.shape[0]
other_count     = other_pts.shape[0]

collision_pct = collision_count / total_count * 100 if total_count > 0 else 0
reached_pct   = reached_count   / total_count * 100 if total_count > 0 else 0
other_pct     = other_count     / total_count * 100 if total_count > 0 else 0

# 使用 subplot 同时显示三种分类的 3D 散点图
fig = plt.figure(figsize=(18, 6))

# 绘制 Collision 的子图
ax1 = fig.add_subplot(131, projection='3d')
if collision_pts.shape[0] > 0:
    ax1.scatter(collision_pts[:, 0], collision_pts[:, 1], collision_pts[:, 2],
                c='red', s=50, label='Collision')
else:
    ax1.text2D(0.5, 0.5, "No Data", transform=ax1.transAxes,
               fontsize=14, ha='center')
ax1.scatter(0, 0, 0, c='black', marker='*', s=200, label='Origin')
title1 = "Collision"
if collision_avg is not None:
    title1 += f"\nAvg distance: {collision_avg:.3f}"
title1 += f"\nCount: {collision_count} ({collision_pct:.1f}%)"
ax1.set_title(title1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# 绘制 Reached Goal 的子图
ax2 = fig.add_subplot(132, projection='3d')
if reached_pts.shape[0] > 0:
    ax2.scatter(reached_pts[:, 0], reached_pts[:, 1], reached_pts[:, 2],
                c='green', s=50, label='Reached Goal')
else:
    ax2.text2D(0.5, 0.5, "No Data", transform=ax2.transAxes,
               fontsize=14, ha='center')
ax2.scatter(0, 0, 0, c='black', marker='*', s=200, label='Origin')
title2 = "Reached Goal"
if reached_avg is not None:
    title2 += f"\nAvg distance: {reached_avg:.3f}"
title2 += f"\nCount: {reached_count} ({reached_pct:.1f}%)"
ax2.set_title(title2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

# 绘制 Other 的子图
ax3 = fig.add_subplot(133, projection='3d')
if other_pts.shape[0] > 0:
    ax3.scatter(other_pts[:, 0], other_pts[:, 1], other_pts[:, 2],
                c='blue', s=50, label='No Collision & Not Reached')
else:
    ax3.text2D(0.5, 0.5, "No Data", transform=ax3.transAxes,
               fontsize=14, ha='center')
ax3.scatter(0, 0, 0, c='black', marker='*', s=200, label='Origin')
title3 = "Other"
if other_avg is not None:
    title3 += f"\nAvg distance: {other_avg:.3f}"
title3 += f"\nCount: {other_count} ({other_pct:.1f}%)"
ax3.set_title(title3)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.legend()

plt.tight_layout()
plt.show()
