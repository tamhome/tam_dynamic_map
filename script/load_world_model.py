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


class LoadWorldModel(Node):
    def __init__(self) -> None:
        self.furniture_dir = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/furniture_templates/"
        self.yaml_path = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/map/sample.yaml"

        self.marker_array_publisher = rospy.Publisher('dynamic_map', MarkerArray, queue_size=10)
        # self.marker_array = MarkerArray()

    def load_marker_poses(self, model_type: str, marker_ns: str, offset_pose={"x": 0, "y": 0, "z": 0}, scale={"x": 1, "y": 1, "z": 1}) -> dict:
        """家具のモデルとなるyamlを読み込む関数
        Args:
            model_type(str): どの家具を読み込むのか
            yaml_path(str): マーカのid
        Return:
            group以下の内容
        """

        yaml_path = f"{self.furniture_dir}{model_type}/model.yaml"
        with open(yaml_path, "r") as file:
            marker_poses = yaml.safe_load(file)["shape"]

        marker_array = MarkerArray()

        for i, marker_info in enumerate(marker_poses["group"]):
            # yamlから必要な情報を抽出
            marker_type = list(marker_info.keys())[0]
            content = marker_info[marker_type]
            pose = content["pose"]
            size = content["size"]

            marker = Marker()
            marker.header.frame_id = "map"  # フレームIDを適切な値に変更
            marker.header.stamp = rospy.Time.now()
            marker.ns = marker_ns
            marker.id = i
            if marker_type == "box":
                marker.type = Marker.CUBE
            else:
                marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = Pose(Point(pose['x'] + offset_pose["x"], pose['y'] + offset_pose["y"], pose['z'] + offset_pose["z"]), Quaternion(0, 0, 0, 1))
            marker.scale = Point(size["x"], size["y"], size["z"])
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        # namespaceを合わせて配信
        marker = Marker()
        marker.header.frame_id = "map"  # フレームIDを適切な値に変更
        marker.header.stamp = rospy.Time.now()
        marker.ns = marker_ns
        marker.text = marker_ns
        marker.id = 9999
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose = Pose(Point(offset_pose["x"], offset_pose["y"], offset_pose["z"] + scale["z"] + 0.2), Quaternion(0, 0, 0, 1))
        marker.scale = Point(0, 0, 0.3)  # 文字の大きさ指定
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker_array.markers.append(marker)

        return marker_array

        # return marker_poses

    def load_world(self, yaml_path: str) -> dict:
        """マップモデルのyamlを読み込む関数
        Args:
            yaml_path(str): 対象とするyamlファイルのpath
        Return:
            group以下の内容
        """
        with open(yaml_path, "r") as file:
            world_model = yaml.safe_load(file)["world"]
        return world_model

    def run(self) -> None:
        world_model = self.load_world(self.yaml_path)
        print(world_model)

        for target_model in world_model:
            print(target_model["id"])
            print(target_model["pose"])
            marker_array = self.load_marker_poses(target_model["type"], target_model["id"], target_model["pose"])

            # MarkerArrayをパブリッシュ
            self.marker_array_publisher.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node('test_marker_publisher')
    cls = LoadWorldModel()
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        cls.run()
        rospy.sleep(1)
