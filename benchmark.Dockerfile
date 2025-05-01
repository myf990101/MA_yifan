FROM ba-gazebo-image

SHELL [ "/bin/bash", "-c" ]

ENV PUID=1000
ENV PGID=1000

# SETUP WORKSPACE
WORKDIR /workspace
COPY . .

# INSTALL SYSTEM DEPENDENCIES (合并 apt 命令以减少层数)
RUN apt-get update && apt-get install -y \
    libpcl-dev libeigen3-dev cmake ninja-build python3-dev python3-pip git \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

# SETUP VIRTUAL ENVIRONMENT
ENV VIRTUAL_ENV=/workspace/gym-env
RUN python3 -m venv ${VIRTUAL_ENV} --system-site-packages
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# UPGRADE PIP AND INSTALL BASIC DEPENDENCIES
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy coloredlogs

# INSTALL PYTHON DEPENDENCIES (优先安装项目依赖)
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

# BUILD ROS2 CODE (确保 ROS 环境激活)
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd src/ros_gz_ws/ && \
    make build

# 克隆并安装 PointNet++ 仓库 (关键修正)

RUN cd Pointnet_Pointnet2_pytorch && \
    pip install -e .

# 安装项目自身模块 (确保在 PointNet++ 之后)
RUN pip install -e src/gym_gz_ws/

# 设置 Python 路径 (确保能找到 PointNet++)
ENV PYTHONPATH="/workspace/Pointnet_Pointnet2_pytorch/models:${PYTHONPATH}"
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# SETUP BENCHMARK
ENV BENCHMARK_RESULTS_DIR /benchmark-results
RUN mkdir -p ${BENCHMARK_RESULTS_DIR}

# 重置工作目录到 /workspace (避免后续路径问题)
WORKDIR /workspace

# RUN BENCHMARK (确保激活 ROS 环境)
CMD source /opt/ros/$ROS_DISTRO/setup.bash && \
    source src/ros_gz_ws/install/setup.bash && \
    python3 src/run_benchmark.py \
        --result-dir ${BENCHMARK_RESULTS_DIR} \
        --repetitions 5 \
        --seed 4848 \
        --verbose && \
    chown ${PUID}:${PGID} -R ${BENCHMARK_RESULTS_DIR}