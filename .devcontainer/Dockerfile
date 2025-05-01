FROM yifanma/gazebo-image:latest

# --- set up user
ARG user=leont
RUN usermod -l ${user} ubuntu 
RUN groupmod -n ${user} ubuntu 
ENV HOME=/home/${user}
ENV USER=${user}
RUN mv /home/ubuntu /home/${user}

# configure sudo
RUN usermod -aG sudo ${user}
RUN echo "$user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN touch /home/${user}/.sudo_as_admin_successful  
	# so that the "To run a command as administrator ..." message does not appear

# configure prompt (user and root)
# escape=`
RUN echo "\
PS1='\${debian_chroot:+(\$debian_chroot)}\[\033[01;36m\]\u \[\033[01;32m\]→ \[\033[01;34m\]\w\[\033[00m\]\$ '\
" >> /home/$user/.bashrc
RUN echo "\
source /opt/ros/jazzy/setup.bash \
" >> /home/$user/.bashrc

RUN echo "\
PS1='\${debian_chroot:+(\$debian_chroot)}\[\033[01;31m\]\u \[\033[01;32m\]→ \[\033[01;34m\]\w\[\033[00m\]\$ '\
" >> /root/.bashrc
RUN echo "\
source /opt/ros/jazzy/setup.bash \
" >> /root/.bashrc
# escape=\

# RUN yes | unminimize
# 先更新包列表，安装需要的软件属性管理工具
RUN apt-get update && \
    apt-get install -y software-properties-common

# 添加 PPA，然后再次更新 apt 包列表
RUN add-apt-repository ppa:neovim-ppa/unstable -y && \
    apt-get update

# tools
RUN apt-get install -y software-properties-common && \
	add-apt-repository ppa:neovim-ppa/unstable -y && \
	apt-get update

RUN apt-get install -y \
		bash-completion \
		man-db \
		neovim \
		tmux \
		kitty-terminfo \
		ripgrep \
		wget \
		fzf \
		npm \
		htop \
		ros-jazzy-ros2controlcli\
 && rm -rf /var/lib/apt/lists/*
		# nodejs \
		# python3-venv

	
ENV TERM=xterm-kitty
USER root

SHELL ["/bin/bash"]
