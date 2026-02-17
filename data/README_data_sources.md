# Plan for Gathering Real Telemetry Data

This document outlines sources and procedures for obtaining real robotics and autonomous vehicle telemetry to use with H.E.I.M.D.A.L.L. The pipeline supports Parquet and CSV formats matching the telemetry schema (IMU, LiDAR, CAN, GPS, camera).

## Open Datasets

### nuScenes
CAN bus, LiDAR, camera, IMU from real autonomous vehicles. Requires conversion to the H.E.I.M.D.A.L.L schema (timestamp_ns, vehicle_id, sensor_type, and sensor-specific columns). Download from https://www.nuscenes.org/

### KITTI
LiDAR, camera, GPS/IMU from real driving. Raw and derived formats. https://www.cvlibs.net/datasets/kitti/

### Waymo Open Dataset
LiDAR, camera, map data autonomous vehicles. Public subset available. https://waymo.com/open/

## Simulators

### CARLA
CARLA Simulator with ROS2 bridge generates synthetic but realistic sensor streams. Record to ROS2 bags and convert to Parquet. https://carla.org/

### NVIDIA Isaac Sim
Robot and vehicle simulation with ROS2 support. Export telemetry in formats compatible with the pipeline.

### Gazebo
ROS2-native simulation. Record with `ros2 bag record` and convert topics to the unified schema.

## ROS2 Bag Files

Record from real or simulated robots:
```
ros2 bag record -a
```
Convert bag files to Parquet using rosbag2_py or custom scripts that map sensor_msgs/Imu, sensor_msgs/PointCloud2, nav_msgs/Odometry to the schema.

## OBD-II CAN Data

Real vehicle data via ELM327 adapters and tools like python-OBD. Map OBD-II PIDs to vehicle_speed_kmh, brake-related signals, throttle_position_pct, etc.

## Conversion Guidelines

1. Map timestamps to nanoseconds (int64).
2. Use vehicle_id for multi-vehicle datasets.
3. Set sensor_type to one of: imu, lidar, can, gps, camera.
4. Fill sensor-specific columns; use NaN for columns not applicable to a given sensor type.
5. Save as Parquet for best performance with cuDF.
