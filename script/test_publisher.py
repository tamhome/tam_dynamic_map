#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import rospy
import roslib
import pprint

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion

from tamlib.node_template import Node


class MarkerPublisherCore(Node):
    def __init__(self) -> None:
        self.yaml_path = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/furniture_templates/table/model.yaml"
        self.marker_array_publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

    def load_marker_poses(self, yaml_path: str) -> dict:
        """モデルとなるyamlを読み込む関数
        Args:
            yaml_path(str): 対象とするyamlファイルのpath
        Return:
            group以下の内容
        """
        with open(yaml_path, "r") as file:
            marker_poses = yaml.safe_load(file)["shape"]
        return marker_poses

    def run(self) -> None:
        marker_poses = self.load_marker_poses(self.yaml_path)
        marker_array = MarkerArray()

        for i, marker_info in enumerate(marker_poses["group"]):
            # yamlから必要な情報を抽出
            marker_type = list(marker_info.keys())[0]
            content = marker_info[marker_type]
            pose = content["pose"]
            size = content["size"]

            cube_marker = Marker()
            cube_marker.header.frame_id = "map"  # フレームIDを適切な値に変更
            cube_marker.header.stamp = rospy.Time.now()
            cube_marker.ns = "cube_markers"
            cube_marker.id = i
            cube_marker.type = Marker.CUBE
            cube_marker.action = Marker.ADD
            cube_marker.pose = Pose(Point(pose['x'], pose['y'], pose['z']), Quaternion(0, 0, 0, 1))
            cube_marker.scale = Point(size["x"], size["y"], size["z"])  # Cubeのサイズを設定
            cube_marker.color.r = 1.0
            cube_marker.color.g = 0.0
            cube_marker.color.b = 0.0
            cube_marker.color.a = 1.0
            marker_array.markers.append(cube_marker)

        # MarkerArrayをパブリッシュ
        self.marker_array_publisher.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node('test_marker_publisher')
    cls = MarkerPublisherCore()
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        cls.run()
        rospy.sleep(1)
