SHELL := /bin/bash
.PHONY: clean run-setup build

# build: $(wildcard src/**/*)
build: 
	colcon build --packages-skip cartesian_controller_simulation cartesian_controller_tests --cmake-args -DCMAKE_BUILD_TYPE=Release
	@touch -c build # update time of build folder, so that make does not rebuild unnecessarily

run-setup:
	source install/setup.bash && \
		ros2 launch ur_sim setup_sim.launch.py

run-start:
	source install/setup.bash && \
		ros2 launch ur_sim start_sim.launch.py

clean:
	rm build/ install/ log/ -rf

clean-partial:
	rm build/ur_sim install/ur_sim log/ -rf
