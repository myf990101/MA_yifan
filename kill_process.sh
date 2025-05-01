#!/bin/bash

echo "Killing all ROS 2 and Gazebo processes..."

# 1. 终止 ROS 2 和 Gazebo 相关进程
pkill -9 -f "gz sim"
pkill -9 -f "gzserver"
pkill -9 -f "gzclient"
pkill -9 -f "ros2"
pkill -9 -f "roslaunch"
pkill -9 -f "rosmaster"
pkill -9 -f "roscore"
pkill -9 -f "gazebo"
pkill -9 -f "robot_state_publisher"
pkill -9 -f "ros_gz_bridge"

# # 2. 终止 VS Code 远程进程（如果有）
# pkill -9 -f "code-server"
# pkill -9 -f "node"

# # 3. 查找并终止 ROS 2 和 Gazebo 相关的进程（备用方案）
# for process in "gz sim" "gzserver" "gzclient" "ros2" "roslaunch" "rosmaster" "roscore" "gazebo" "robot_state_publisher" "ros_gz_bridge" "node"; do
#     pids=$(ps aux | grep "$process" | grep -v grep | awk '{print $2}')
#     if [ ! -z "$pids" ]; then
#         echo "Killing process: $process ($pids)"
#         kill -9 $pids
#     fi
# done

# # 4. 检查并终止 Docker 容器（如果有 ROS 2 容器在运行）
# running_containers=$(docker ps -q)
# if [ ! -z "$running_containers" ]; then
#     echo "Stopping running Docker containers..."
#     docker stop $running_containers
#     docker rm $running_containers
# fi

# # 5. 关闭 `tmux` 和 `screen` 会话（如果有）
# tmux list-sessions &>/dev/null && tmux kill-server
# screen -ls | grep "Detached" | cut -d. -f1 | awk '{print $1}' | xargs kill -9 &>/dev/null

# # 6. 重新启动 ROS 2 Daemon（清理缓存）
# ros2 daemon stop
# ros2 daemon start

echo "All ROS 2 and Gazebo processes cleaned up!"
