#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import rospy
import roslib
import pprint

from hsrlib.hsrif import collision_world

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion

from tamlib.node_template import Node


class LoadWorldModel(Node):
    def __init__(self) -> None:
        super().__init__()
        self.furniture_dir = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/furniture_templates/"
        self.yaml_path = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/map/sample.yaml"

        self.marker_array_publisher = rospy.Publisher('/dynamic_map/semi_dynamic_markers', MarkerArray, queue_size=10)
        self.collision_world = collision_world.CollisionWorld()

        # collision_worldの初期化フラグ
        self.initialize = True

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

            pub_pose_x = offset_pose["x"] + (pose["x"] * scale["x"])
            pub_pose_y = offset_pose["y"] + (pose["y"] * scale["y"])
            pub_pose_z = offset_pose["z"] + (pose["z"] * scale["z"])

            marker = Marker()
            marker.header.frame_id = "map"  # フレームIDを適切な値に変更
            marker.header.stamp = rospy.Time.now()
            marker.ns = marker_ns
            marker.id = i
            if marker_type == "box":
                marker.type = Marker.CUBE
                if self.initialize:
                    self.loginfo(f"pub collision world: {marker_ns}")
                    self.collision_world.add_box(
                        size["x"] * scale["x"],
                        size["y"] * scale["y"],
                        size["z"] * scale["z"],
                        pose=[(pub_pose_x, pub_pose_y, pub_pose_z), (0, 0, 0, 1)],
                        name=marker_ns
                    )
            else:
                marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = Pose(Point(pub_pose_x, pub_pose_y, pub_pose_z), Quaternion(0, 0, 0, 1))
            marker.scale.x = size["x"] * scale["x"]
            marker.scale.y = size["y"] * scale["y"]
            marker.scale.z = size["z"] * scale["z"]
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        # namespaceを可視化
        marker = Marker()
        marker.header.frame_id = "map"  # フレームIDを適切な値に変更
        marker.header.stamp = rospy.Time.now()
        marker.ns = marker_ns
        marker.text = marker_ns
        marker.id = 9999
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose = Pose(Point(offset_pose["x"], offset_pose["y"], offset_pose["z"] + scale["z"] + 0.1), Quaternion(0, 0, 0, 1))
        marker.scale = Point(0, 0, 0.1)  # 文字の大きさ指定
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker_array.markers.append(marker)

        return marker_array

        # return marker_poses

    def load_world_model(self, yaml_path: str) -> dict:
        """マップモデルのyamlを読み込む関数
        Args:
            yaml_path(str): 対象とするyamlファイルのpath
        Return:
            group以下の内容
        """
        with open(yaml_path, "r") as file:
            world_model = yaml.safe_load(file)["world"]
        return world_model

    def set_collision_world(self) -> bool:
        """world_modelに配信されている情報をcollsion_worldにも登録する
        Args:

        Return:
            bool 登録に成功したらTrue
        """
        pass

    def run(self) -> None:
        if self.initialize:
            self.loginfo("remove all collison world's objetcs")
            self.collision_world.remove_all()
            rospy.sleep(5)

        world_model = self.load_world_model(self.yaml_path)
        print(world_model)

        for target_model in world_model:
            print(target_model["id"])
            print(target_model["pose"])
            if target_model != "wall":
                marker_array = self.load_marker_poses(target_model["type"], target_model["id"], target_model["pose"], target_model["scale"])
            else:
                # 壁は画像から読み込みを行う
                pass

            # MarkerArrayをパブリッシュ
            self.marker_array_publisher.publish(marker_array)

        self.initialize = False


if __name__ == "__main__":
    rospy.init_node('test_marker_publisher')
    cls = LoadWorldModel()
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        cls.run()
        rospy.sleep(1)
