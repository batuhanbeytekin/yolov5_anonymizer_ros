# yolov5_anonymizer_ros

This is a ROS 2 package for real-time detected object anonymization with YOLOv5 on a ROS 2 bag. 

## Build from source

Create a new workspace:
```bash
  mkdir ~/catkin_ws ~/catkin_ws/src
  cd ~/catkin_ws/src
```

Clone this repository into the source folder:
```bash
  git clone https://github.com/batuhanbeytekin/yolov5_anonymizer_ros.git
```
Go to the project directory:

```bash
  cd ~/catkin_ws
```

Build package with this command:

```bash
  colcon build
```

Launch the node:

```bash
  ros2 launch yolov5_anonymizer_ros yolov5_blur.launch.py 
```
Open new terminal and launch the ROS 2 bag:

```bash
  ros2 bag play your/bag/file/path
```

## Parameters

#### Parameters in params.yaml file


| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model_path` | `string` | Weight path converted to onnx format  |
| `write_bag` | `bool` | If True, save new bag file from blurred objects |
| `debug` | `bool` | If True, check the time between processes |
| `save_img` | `bool` | If True, save the extracted images from bag file |
| `bag_name` | `string` | New bag file name |
| `save_directory` | `string` | New bag file path |
| `camera_topic` | `string` | Subscribe camera topic |

