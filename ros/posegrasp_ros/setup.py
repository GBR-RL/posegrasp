from glob import glob

from setuptools import find_packages, setup

PACKAGE = "posegrasp_ros"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Rohiith Gettala Balachandar",
    maintainer_email="gbrohiith@gmail.com",
    description="ROS 2 nodes of posegrasp: scene player, pose estimation, grasps, MoveIt 2 pick",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"bop_player = {PACKAGE}.bop_player:main",
            f"pose_estimator = {PACKAGE}.pose_node:main",
            f"grasp_planner = {PACKAGE}.grasp_node:main",
            f"pick = {PACKAGE}.pick_node:main",
        ],
    },
)
