from launch import (
    LaunchDescription,
    LaunchDescriptionEntity,
)
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    FileContent,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace
from launch.actions import SetLaunchConfiguration
from launch.substitutions import TextSubstitution

def setup_sim(context, *args, **kwargs) -> list[LaunchDescriptionEntity]:
    ld_entities: list[LaunchDescriptionEntity] = []

    # -- parameters
    # general
    robot_description = LaunchConfiguration("robot_description")
    seed = LaunchConfiguration("seed").perform(context)
    # gazebo
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    bridge_config_file = LaunchConfiguration("bridge_config_file")
    # bridge_name = f"bridge_node_{seed}"
    # robot_state_publisher_name = f"robot_state_publisher_{seed}"
    bridge_name = f"bridge_node"
    robot_state_publisher_name = f"robot_state_publisher"
    # alias
    _pkg__ros_gz_sim = FindPackageShare("ros_gz_sim")
    # Uncomment next line once launch files for ros_gz_bridge are included in the deb
    # _pkg__ros_gz_bridge = FindPackageShare("ros_gz_bridge")
    _pkg__ros_gz_bridge = FindPackageShare("hotfix_ros_gz_bridge")  # hotfix

    ############### THESE NEED TO BE ADDED BEFORE GAZEBO IS STARTED
    # Fix wayland display issue by unsetting WAYLAND_DISPLAY (i.e. using XWayland)
    ld_entities.append(AppendEnvironmentVariable("WAYLAND_DISPLAY", ""))
    ############### END

    # gazebo launch description (with or without gui)
    ld_entities.append(
        ExecuteProcess(
            cmd=["gz", "sim", "-g"],
            output="screen",
            condition=IfCondition(gazebo_gui),
        )
    )

    # ros_gz_bridge launch description
    ld_entities.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [_pkg__ros_gz_bridge, "launch", "ros_gz_bridge.launch.py"]
                )
            ),
            launch_arguments={
                "name": bridge_name,
                "config_file": bridge_config_file,
                "use_composition": "False",  # important, otherwise node does not start up
            }.items(),
        )
    )

    # start robot state publisher (ros)
    ld_entities.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name=robot_state_publisher_name,
            output="both",
            parameters=[{"use_sim_time": True, "robot_description": robot_description}],
        )
    )

    return ld_entities


def generate_launch_description():
    _pkg_name = "ur_sim"
    _pkg__self = FindPackageShare(_pkg_name)

    ld = LaunchDescription()
    
    # Robot
    # urdf created during build
    ld.add_action(
        SetLaunchConfiguration(
            "robot_description",
            FileContent(PathJoinSubstitution([_pkg__self, "urdf", "robot.urdf"])),
        )
    )

    # Gazebo
    ld.add_action(
        DeclareLaunchArgument(
            "gazebo_gui", default_value="true", description="Start gazebo with GUI?"
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "seed", default_value="s", description="Seed"))

    ld.add_action(
        DeclareLaunchArgument(
            "bridge_config_file",
            default_value=PathJoinSubstitution(
                [_pkg__self, "config", "ros_gz_bridge.yaml"]
            ),
            description="Path to the ros_gz_bridge config file (YAML)",
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            "bridge_name",
            default_value="bridge_node",
            description="Name of the ros_gz_bridge node",
        )
    )
    # ld.add_action(
    #     GroupAction(
    #         actions=[
    #             PushRosNamespace(PathJoinSubstitution(["sim_", LaunchConfiguration("seed")])),
    #             OpaqueFunction(function=setup_sim)
    #         ]
    #     )
    # )
    # ###
    ld.add_action(OpaqueFunction(function=setup_sim))

    return ld
