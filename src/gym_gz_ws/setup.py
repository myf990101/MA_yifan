from setuptools import setup

setup(
    name="gym_gz",
    version="0.0.1",
    install_requires=["gymnasium==0.29.1"], # also rclpy, but is available via ros2
)
