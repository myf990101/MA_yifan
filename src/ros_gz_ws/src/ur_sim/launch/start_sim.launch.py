from launch import (
    LaunchDescription,
    LaunchDescriptionEntity,
    Substitution,
)
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    FileContent,
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace
from launch.actions import SetLaunchConfiguration


def start_sim(context, *args, **kwargs) -> list[LaunchDescriptionEntity]:
    ld_entities: list[LaunchDescriptionEntity] = []

    # -- parameters
    # general
    # robot_name = LaunchConfiguration("robot_name")
    robot_name = TextSubstitution(text="ur")
    robot_description_sdf = LaunchConfiguration(
        "robot_description_sdf"
    )  # used for spawning robot in gz

    # gazebo
    world_file = LaunchConfiguration("world_file")
    gazebo_models_paths = LaunchConfiguration("gazebo_models_paths")
    seed = LaunchConfiguration("seed")

    # ros2control
    cartesian_controller_name = LaunchConfiguration("cartesian_controller_name")
    position_controller_name = LaunchConfiguration("position_controller_name")

    # alias
    _pkg__ros_gz_sim = FindPackageShare("ros_gz_sim")
    # Uncomment next line once launch files for ros_gz_bridge are included in the deb
    # _pkg__ros_gz_bridge = FindPackageShare("ros_gz_bridge")
    _pkg__ros_gz_bridge = FindPackageShare("hotfix_ros_gz_bridge")  # hotfix

    ############### THIS NEEDS TO BE ADDED BEFORE GAZEBO IS STARTED
    # Set resource paths for Gazebo, so it can find the sdfs
    ld_entities.append(
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", gazebo_models_paths)
    )
    ############### END

    # launch gazebo server
    ld_entities.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([_pkg__ros_gz_sim, "launch", "gz_sim.launch.py"])
            ),
            launch_arguments={
                "gz_args": f" -s -r --headless-rendering --seed {seed.perform(context)} -v 4 {world_file.perform(context)}",
                "on_exit_shutdown": "blablabyes",  # BUG: on_exit_shutdown true if value is not empty
            }.items(),
        )
    )

    # spawn robot in gazebo via ros_gz_sim package
    _spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-string",
            robot_description_sdf,
            "-name",
            robot_name,
            "-allow_renaming",
            "true",
            # "-z","0.0"
         
        ],
        output="screen",
    )
    ld_entities.append(_spawn_robot)

    # start joint state broadcaster after that has finished
    _start_joint_state_broadcaster = ExecuteProcess(
        cmd=[
            "ros2",
            "control",
            "load_controller",
            "--set-state",
            "active",
            "joint_state_broadcaster",
        ],
        output="screen",
    )
    ld_entities.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=_spawn_robot,
                on_exit=[_start_joint_state_broadcaster],
            )
        )
    )

    # start robot controllers after that has finished
    _start_cartesian_controller = ExecuteProcess(
        cmd=[
            "ros2",
            "control",
            "load_controller",
            "--set-state",
            "inactive",
            cartesian_controller_name,
        ],
        output="screen",
    )

    ld_entities.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=_start_joint_state_broadcaster,
                on_exit=[_start_cartesian_controller],
            )
        )
    )

    _start_position_controller = ExecuteProcess(
        cmd=[
            "ros2",
            "control",
            "load_controller",
            "--set-state",
            "active",
            position_controller_name,
        ],
        output="screen",
    )

    ld_entities.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=_start_cartesian_controller,
                on_exit=[_start_position_controller],
            )
        )
    )

    return ld_entities


def generate_launch_description():
    _pkg_name = "ur_sim"
    _pkg__self = FindPackageShare(_pkg_name)

    ld = LaunchDescription()

    # Robot
    # robot_description_sdf created during build
    ld.add_action(
        SetLaunchConfiguration(
            "robot_description_sdf",
            FileContent(PathJoinSubstitution([_pkg__self, "models", "robot.sdf"])),
        )
    )

    # Gazebo
    ld.add_action(
        DeclareLaunchArgument(
            "world_file",
            default_value="empty.sdf",
            description="Gazebo world file (absolute path or filename from the gazebosim worlds collection) containing a custom world.",
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            "gazebo_models_paths",
            default_value=_colon_join(
                [
                    PathJoinSubstitution(
                        [FindPackageShare("ur_description"), ".."]
                    ),  # fix, because filepaths in urdf start at 'ur_description'
                    PathJoinSubstitution([_pkg__self, "worlds"]),
                    PathJoinSubstitution([_pkg__self, "models"]),
                ]
            ),
            description="All paths to gazebo model ressources, including for the robot, the world, and all other models.",
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            "cartesian_controller_name",
            default_value="full_controller",
            description="Name of the cartesian controller (as specified in ros2 control config file).",
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            "position_controller_name",
            default_value="position_controller",
            description="Name of the position controller (as specified in ros2 control config file).",
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            "seed",
            default_value="1234",
            description="Seed that is passed to gazebo on launch",
        )
    )
    # ld.add_action(
    #     GroupAction(
    #         actions=[
    #             PushRosNamespace(PathJoinSubstitution(["sim_", LaunchConfiguration("seed")])),
    #             OpaqueFunction(function=start_sim)
    #         ]
    #     )
    # )
    ###
    ld.add_action(OpaqueFunction(function=start_sim))

    return ld


def _colon_join(subs: list[Substitution]) -> list[Substitution]:
    r = []
    for sub in subs:
        r.append(sub)
        r.append(TextSubstitution(text=":"))
    return r