
FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
    python3-pip python3-rosbag ros-noetic-cv-bridge \
    ros-noetic-vision-opencv ros-noetic-image-transport && \
    pip3 install opencv-python numpy pandas

WORKDIR /work
CMD ["/bin/bash"]
