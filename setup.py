from setuptools import find_packages, setup

package_name = 'ros2_project_sc22hb'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sc22hb',
    maintainer_email='sc22hb@leeds.ac.uk',
    description='COMP3631 ROS2 project — autonomous exploration and RGB detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = ros2_project_sc22hb.robot_controller:main',
        ],
    },
)
