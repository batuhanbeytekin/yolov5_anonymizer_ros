from setuptools import setup
import os
from glob import glob

package_name = 'yolov5_anonymizer_ros'
submodules = "yolov5_anonymizer_ros/submodules"
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools','rosbag2_py'],
    zip_safe=True,
    maintainer='batuhanbeytekin',
    maintainer_email='batuhanbeytekin@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_blur = yolov5_anonymizer_ros.yolov5_blur:main'
        ],
    },
)
