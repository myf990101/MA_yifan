# LOGGING
ORANGE=\033[38;5;214m
RESET=\033[0m

log = @echo -e "$(ORANGE)[STATUS] $(1)$(RESET)"

# BUILDING
SHELL := /bin/bash
.PHONY: benchmark-image clean-benchmark-image run-benchmark clean-benchmark-results

# This file is generally meant for use inside the devcontainer,
# except for the **benchmark-image** target, which needs to be run from outside,
# as the devcontainer does not support docker in docker

# build venv
gym-env: requirements.txt
	$(call log, Building gym-env...)
	python3 -m venv gym-env --system-site-packages  # flag gives access to ros packages
	source gym-env/bin/activate && \
		python3 -m pip install -r requirements.txt && \
		python3 -m pip install -e src/gym_gz_ws/
	touch -c gym-env # update time of folder, so that make does not try to rebuild
	$(call log, Finished building gym-env.)

# run benchmark
run-benchmark: gym-env
	cd src/ros_gz_ws/ && $(MAKE) build
	source gym-env/bin/activate && \
		source src/ros_gz_ws/install/setup.bash && \
		python3 src/run_benchmark.py \
			--result-dir benchmark-results \
			--repetitions 5 \
			--seed 4848 \
			--verbose

run-benchmark-gui: gym-env
	cd src/ros_gz_ws/ && $(MAKE) build
	source gym-env/bin/activate && \
		source src/ros_gz_ws/install/setup.bash && \
		python3 src/run_benchmark.py \
			--result-dir benchmark-results \
			--repetitions 2 \
			--seed 4848 \
			--verbose \
			--gui

run-ur5e-gui: #gym-env
	./kill_process.sh
	cd src/ros_gz_ws/ && $(MAKE) build 
	source gym-env/bin/activate && \
		source src/ros_gz_ws/install/setup.bash && \
		python3 src/run_drl.py \
			--result-dir benchmark-results \
			--repetitions 2 \
			--seed 4848 \
			--verbose \
			--gui
run-ur5e-parallel: gym-env
	./kill_process.sh
	cd src/ros_gz_ws/ && $(MAKE) build 
	source gym-env/bin/activate && \
		source src/ros_gz_ws/install/setup.bash && \
		python3 src/run_drl_parallel.py \
		
# build benchmark-image
# THIS SHOULD ONLY BE USED OUTSIDE A DOCKER CONTAINER
# (it's possible to use it from within, but not necessary here)
GAZEBO_BASE_IMAGE_PATH = ../../base-images/build/ba-gazebo-image.tar
benchmark-image: ${GAZEBO_BASE_IMAGE_PATH}
	$(call log, Building ba-gazebo-benchmark...)
	mkdir -p build
	docker load -i ${GAZEBO_BASE_IMAGE_PATH}
	docker build -t ba-gazebo-benchmark -f benchmark.Dockerfile .
	docker save ba-gazebo-benchmark -o build/ba-gazebo-benchmark.tar
	$(call log, Finished ba-gazebo-benchmark.)

clean-benchmark-image:
	rm -rf build
