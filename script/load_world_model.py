#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import rospy
import roslib
import pprint
import numpy as np
from typing import Optional

from hsrlib.hsrif import collision_world
from hsrlib.utils import locations

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion, Pose2D
from tam_dynamic_map.srv import GetObjectPose, GetObjectPoseResponse
from tam_dynamic_map.srv import GetNavigationGoal, GetNavigationGoalResponse
from tamlib.node_template import Node


class LoadWorldModel(Node):
    def __init__(self) -> None:
        super().__init__()
        self.p_yaml_path = rospy.get_param("~world_model_path", "sample01.yaml")

        self.furniture_dir = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/furniture_templates/"
        self.yaml_path = roslib.packages.get_pkg_dir("tam_dynamic_map") + f"/io/map/{self.p_yaml_path}"

        self.marker_array_publisher = rospy.Publisher('/tam_dynamic_map/semi_dynamic_markers', MarkerArray, queue_size=10)
        self.marker_nav_goal_publisher = rospy.Publisher('/tam_dynamic_map/navigation_goal', Marker, queue_size=10)

        # service
        rospy.Service("/tam_dynamic_map/get_obj_pose/service", GetObjectPose, self.cb_get_object_pose)
        rospy.Service("/tam_dynamic_map/get_navigation_goal/service", GetNavigationGoal, self.cb_get_navigation_goal)

        self.collision_world = collision_world.CollisionWorld()

        # collision_worldの初期化フラグ
        self.initialize = True

    def load_marker_poses(self, model_type: str, marker_ns: str, offset_pose={"x": 0, "y": 0, "z": 0}, scale={"x": 1, "y": 1, "z": 1}, quaternion={"x": 0, "y": 0, "z": 0, "w": 1}) -> dict:
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

            # if model_type == "unknown":
            if True:
                pub_pose_x = offset_pose["x"]
                pub_pose_y = offset_pose["y"]
                pub_pose_z = offset_pose["z"]
            else:
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
                        pose=[(pub_pose_x, pub_pose_y, pub_pose_z), (quaternion["x"], quaternion["y"], quaternion["z"], quaternion["w"])],
                        name=marker_ns
                    )
            else:
                marker.type = Marker.CUBE

            marker.action = Marker.ADD
            marker.pose = Pose(Point(pub_pose_x, pub_pose_y, pub_pose_z), Quaternion(quaternion["x"], quaternion["y"], quaternion["z"], quaternion["w"]))
            marker.scale.x = size["x"] * scale["x"]
            marker.scale.y = size["y"] * scale["y"]
            marker.scale.z = size["z"] * scale["z"]
            if model_type == "table":
                marker.color.r = 1.0
                marker.color.g = 0.7
                marker.color.b = 0.05
                marker.color.a = 1.0
            elif model_type == "chair":
                marker.color.r = 0.05
                marker.color.g = 0.7
                marker.color.b = 1.0
                marker.color.a = 1.0
            else:
                marker.color.r = 1.00
                marker.color.g = 0.7
                marker.color.b = 1.0
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

    def _show_arrow_marker(self, pose: Pose2D) -> bool:
        """目的地にVisualization Markerを配信
        Args:
            pose(Pose2D): Navigationの目的地の座標とTheta
        Returns:
            bool
        """
        marker_arrow = Marker()
        marker_arrow.header.frame_id = "map"  # フレームIDを適切な値に変更
        marker_arrow.header.stamp = rospy.Time.now()
        marker_arrow.ns = "navigation_goal"
        marker_arrow.type = Marker.ARROW
        marker_arrow.action = Marker.ADD
        marker_arrow.pose.position.x = pose.x
        marker_arrow.pose.position.y = pose.y
        marker_arrow.pose.position.z = 0

        # オイラー角 (roll, pitch, yaw) をQuaternionsに変換
        quaternion = Quaternion()
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.z = pose.theta
        quaternion.w = 1.0

        marker_arrow.pose.orientation = quaternion

        marker_arrow.scale = Point(1.0, 0.1, 0.1)  # 文字の大きさ指定
        marker_arrow.color.r = 0.95
        marker_arrow.color.g = 0.2
        marker_arrow.color.b = 0.8
        marker_arrow.color.a = 0.8
        marker_arrow.lifetime = rospy.Duration(10)

        self.marker_nav_goal_publisher.publish(marker_arrow)

        return True

    def _calc_nav_goal(self, obj_pose: Pose, scale: Optional[float]) -> Pose2D:
        """Navigation Goalを算出する関数
        Args:
            obj_pose(Pose): 対象とするオブジェクトのPose
            scale(list[x, y, z]): x, y, z方向それぞれにおける，オブジェクトの大きさ 
        Returns:
            Pose2D ナビゲーションの目的地
        """
        current_x, current_y, _ = locations.get_robot_position()
        self.nav_distance = 0.1
        target_pose_list = [
            [obj_pose.position.x + scale[0] + self.nav_distance, obj_pose.position.y, 3.14],
            [obj_pose.position.x - scale[0] - self.nav_distance, obj_pose.position.y, 0],
            [obj_pose.position.x, obj_pose.position.y + scale[1] + self.nav_distance, -1.57],
            [obj_pose.position.x, obj_pose.position.y - scale[1] - self.nav_distance, 1.57],
        ]

        min_distance = np.inf

        for index, target_pose in enumerate(target_pose_list):
            distance = ((target_pose[0] - current_x)**2 + (target_pose[1] - current_y)**2)**0.5
            if min_distance > distance:
                min_distance = distance
                target_index = index

        target_pose = target_pose_list[target_index]
        target_pose2d = Pose2D(target_pose[0], target_pose[1], target_pose[2])

        return target_pose2d

    def cb_get_object_pose(self, req: GetObjectPose) -> GetObjectPoseResponse:
        """objectの中心位置と大きさを返す関数
        Args:
            req(GetObjectPose): service message
        Returns:
            GetObjectPoseResponse
        """
        target_obj_id = req.obj_id
        response = GetObjectPoseResponse()
        world_model = self.load_world_model(self.yaml_path)
        # 見つからなかった場合
        point = Point(x=np.inf, y=np.inf, z=np.inf)
        quaternion = Quaternion(np.inf, np.inf, np.inf, np.inf)
        scale = [np.inf, np.inf, np.inf]

        for obj in world_model:
            if obj["id"] != target_obj_id:
                continue
            self.loginfo("found target object")
            point = Point(x=obj["pose"]["x"], y=obj["pose"]["y"], z=obj["pose"]["z"])
            quaternion = Quaternion(obj["quaternion"]["x"], obj["quaternion"]["y"], obj["quaternion"]["z"], obj["quaternion"]["w"])
            scale = [obj["scale"]["x"], obj["scale"]["y"], obj["scale"]["z"]]
            break

        if point.x == np.inf:
            self.logwarn(f"指定のオブジェクトは見つかりませんでした．指定されたオブジェクト名: {target_obj_id}")

        response.obj_pose = Pose(point, quaternion)
        response.scale = scale
        return response

    def cb_get_navigation_goal(self, req: GetNavigationGoal) -> GetNavigationGoalResponse:
        """Navigationの目的地を返す関数
        Args:
            GetNavigationGoal
        Returns:
            GetNavigationGoalResponse
        """
        target_obj_id = req.obj_id
        response = GetNavigationGoalResponse()
        world_model = self.load_world_model(self.yaml_path)
        point = Point(x=np.inf, y=np.inf, z=np.inf)
        quaternion = Quaternion(np.inf, np.inf, np.inf, np.inf)
        scale = [np.inf, np.inf, np.inf]

        for obj in world_model:
            if obj["id"] != target_obj_id:
                continue
            self.loginfo("found target object")
            point = Point(x=obj["pose"]["x"], y=obj["pose"]["y"], z=obj["pose"]["z"])
            quaternion = Quaternion(obj["quaternion"]["x"], obj["quaternion"]["y"], obj["quaternion"]["z"], obj["quaternion"]["w"])
            scale = [obj["scale"]["x"], obj["scale"]["y"], obj["scale"]["z"]]
            obj_pose = Pose(point, quaternion)
            break

        if point.x == np.inf:
            response.nav_goal = Pose2D(np.inf, np.inf, np.inf)

        target_pose2d = self._calc_nav_goal(obj_pose, scale=scale)
        self._show_arrow_marker(target_pose2d)
        response.nav_goal = target_pose2d

        return response

    def run(self) -> None:
        if self.initialize:
            self.loginfo("remove all collison world's objetcs")
            self.collision_world.remove_all()
            rospy.sleep(5)

        try:
            world_model = self.load_world_model(self.yaml_path)
        except FileNotFoundError as e:
            self.logtrace(e)
            self.logwarn(f"指定されたyamlファイルが存在しません．{self.yaml_path}")
            return
        except Exception as e:
            self.logwarn(e)
            self.logwarn("yamlファイルが指定のフォーマットになっていません．")
            return

        for target_model in world_model:
            if target_model != "wall":
                marker_array = self.load_marker_poses(target_model["type"], target_model["id"], target_model["pose"], target_model["scale"], target_model["quaternion"])
                # marker_array = self.load_marker_poses(target_model["type"], target_model["id"], target_model["pose"], target_model["scale"])
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
