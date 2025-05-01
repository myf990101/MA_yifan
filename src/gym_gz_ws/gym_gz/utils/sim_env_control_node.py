import shlex
import subprocess
import logging
from time import sleep

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.task import Future
from rosgraph_msgs.msg import Clock

logger = logging.getLogger(__name__)


class SimEnvControl:
    def __init__(self, controller_node_name: str, seed: int, gui: bool = True):
        self.GUI = gui
        self.SEED = seed

        self.SECS_BETWEEN_NODE_CHECKS: float = 1.0

        # simulation time
        self.clock_node: Node = rclpy.create_node(
            "clock_listener",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.clock_future: Future = None  # type: ignore
        self.clock_node_sub = self.clock_node.create_subscription(
            Clock, "/clock", self._set_clock_future, 1
        )

        # gui and bridge
        self._setup_node_names = ["/bridge_node", "/robot_state_publisher"]
        self._setup_p: subprocess.Popen = None  # type: ignore

        # sim server and ros2 control
        self._sim_server_node_names = [
            controller_node_name,
            "/gz_ros_control",
            "/controller_manager",
            "/joint_state_broadcaster",
        ]
        self._start_p: subprocess.Popen = None  # type: ignore

        self._start_sim_setup(timeout=30.0)
        print("SimEnvControl init done.")

    # public API

    ## sim control
    # WARN: This needs to be run in a sourced env, i.e. ros2 needs to exist
    def restart_sim(
        self, shutdown_timeout_sec: float = 30.0, startup_timeout_sec: float = 60.0
    ):
        """Restart simulation server and ros2 control"""
        logger.debug("(Re)Starting simulation server.")

        # issue shutdown command if any of the relevant nodes is active
        self._stop_sim_server(shutdown_timeout_sec)

        # start sim server (again)
        self._start_sim_server(startup_timeout_sec)

        # unpause sim
        logger.debug("<= Finished restarting simulation server.")

    def pause_sim(self, pause: bool):
        pause_service = "/world/default/control"

        if not self._check_if_gz_service_available(pause_service):
            raise RuntimeError("Pause service is not available.")

        result = subprocess.run(
            shlex.split(
                f"gz service -s {pause_service} "
                "--reqtype gz.msgs.WorldControl "
                "--reptype gz.msgs.Boolean "
                f'--req "pause: {pause}"'
            ),
            capture_output=True,
            text=True,
        )

        if result.stderr.strip() or "data: true" not in result.stdout:
            raise RuntimeError(
                f"Error while trying to (un-)pause sim:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

    def stop_sim(self, shutdown_timeout_sec: float = 60.0):
        logger.debug("Stopping simulation.")
        self.clock_node.destroy_node()
        # precaution, so that get_sim_timestamp can't be called by accident
        self.clock_node = None  # type: ignore

        self._stop_sim_server(shutdown_timeout_sec)
        self._stop_sim_setup(shutdown_timeout_sec)
        logger.debug("Finished stopping simulation.")

    def get_sim_timestamp(self, timeout_sec: float = 5.0) -> int:
        self.clock_future = Future()

        rclpy.spin_until_future_complete(
            self.clock_node, self.clock_future, timeout_sec=timeout_sec
        )

        if self.clock_future.done():
            ns = self.clock_future.result()
            logger.debug(f"Current sim time: {ns}")
            return ns  # type: ignore
        raise Exception("Could not get current sim time.")

    # private API
    ## clock
    def _set_clock_future(self, msg: Clock):
        self.clock_future.set_result(int(msg.clock.sec * 1e9 + msg.clock.nanosec))

    ## setup
    def _start_sim_setup(self, timeout: float):
        # issue start command
        logger.debug("Starting sim setup.")

        cmd = f"ros2 launch ur_sim setup_sim.launch.py gazebo_gui:=True seed:={self.SEED}"
        self._setup_p = subprocess.Popen(
            shlex.split(cmd),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        sleep(5)
        # logger.debug("=> Waiting until all setup nodes are up.")
        # self._wait_until_nodes_available(self._setup_node_names, timeout)
        # print("<= All sim setup nodes are up.")
        # logger.debug("<= All sim setup nodes are up.")

    def _stop_sim_setup(self, timeout: float):
        logger.debug("Stopping sim setup.")

        cmd = f"pkill -P {self._setup_p.pid}"
        subprocess.run(shlex.split(cmd), capture_output=True, text=True)

        logger.debug("=> Waiting until all setup nodes have shut down.")
        self._wait_until_nodes_shutdown(self._setup_node_names, timeout)
        logger.debug("<= All sim setup nodes have shut down.")

    ## sim server
    def _start_sim_server(self, timeout: float):
        print("Starting sim server.")

        cmd = f"ros2 launch ur_sim start_sim.launch.py seed:={self.SEED}"
        self._start_p = subprocess.Popen(
            shlex.split(cmd),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        sleep(10)
        # wait until all nodes are available again
        # print("=> Waiting until all server nodes are up.")
        # if not self._wait_until_nodes_available(self._sim_server_node_names, timeout):
        #     raise TimeoutError("Waiting for simulation to start up has timed out.")
        print("<= All sim server nodes have are up.")

    def _stop_sim_server(self, timeout: float):
        logger.debug("Stopping sim server.")

        if not any(self._check_if_nodes_active(self._sim_server_node_names)):
            logger.debug("Did not find anything to shut down.")
            return

        logger.debug("=> Found active nodes, shutting them down.")

        # issue stop command to gz
        result = subprocess.run(
            shlex.split(
                "gz service -s /server_control "
                "--reqtype gz.msgs.ServerControl "
                "--reptype gz.msgs.Boolean "
                '--req "stop: True"'
            ),
            capture_output=True,
            text=True,
        )

        if "data: true" not in result.stdout or result.stderr.strip():
            raise RuntimeError(
                f"Stopping simulation server failed, reason: {result.stderr}"
            )

        logger.debug("=> Stop command finished, wait until all nodes are shut down.")

        # wait until all relevant nodes have shut down
        if not self._wait_until_nodes_shutdown(self._sim_server_node_names, timeout):
            raise TimeoutError("Waiting for simulation to shut down has timed out.")
        logger.debug("<= All sim server nodes have shut down.")

    ## helper
    def _wait_until_nodes_shutdown(
        self, node_name_list: list[str], timeout: float
    ) -> bool:
        wait_iterations = 0
        while (
            not_timed_out := (self.SECS_BETWEEN_NODE_CHECKS * wait_iterations < timeout)
        ) and any(self._check_if_nodes_active(node_name_list)):
            sleep(self.SECS_BETWEEN_NODE_CHECKS)
            wait_iterations += 1

        return not_timed_out  # i.e. success

    def _wait_until_nodes_available(
        self, node_name_list: list[str], timeout: float
    ) -> bool:
        wait_iterations = 0
        while (
            not_timed_out := (self.SECS_BETWEEN_NODE_CHECKS * wait_iterations < timeout)
        ) and not all(self._check_if_nodes_active(node_name_list)):
            sleep(self.SECS_BETWEEN_NODE_CHECKS)
            wait_iterations += 1

        return not_timed_out  # i.e. success

    def _check_if_nodes_active(self, node_name_list: list[str]) -> list[bool]:
        result = subprocess.run(
            shlex.split("ros2 node list"), capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Error while trying to get active nodes."
                f"\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )

        active_nodes = list(
            filter(
                lambda line: line.startswith("/"),
                map(lambda line: line.strip(), result.stdout.split("\n")),
            )
        )

        return [node_name in active_nodes for node_name in node_name_list]

    def _check_if_gz_service_available(self, service_name: str) -> bool:
        result = subprocess.run(
            shlex.split("gz service -l"), capture_output=True, text=True
        )

        if result.stderr.strip():
            raise RuntimeError(f"Error while checking gz services: {result.stdout}")

        services = result.stdout.split("\n")

        return service_name in services
