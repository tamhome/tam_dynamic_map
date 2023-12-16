#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import yaml
import rospy
import roslib
import pprint
import numpy as np
from typing import List

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion, Pose2D
from nav_msgs.msg import OccupancyGrid
from tam_dynamic_map.srv import GetNavigationGoal, GetNavigationGoalResponse

from tamlib.node_template import Node


class GetNavigationGoalService(Node):
    def __init__(self) -> None:
        super().__init__(loglevel="DEBUG")
        self.yaml_path = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/furniture_templates/table/model.yaml"
        self.marker_array_publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

        ###################################################
        # ROSPARAMの読み込み
        ###################################################
        self.p_omni3d_marker_array = rospy.get_param("~omni3d_marker_array", "/omni3d/objects")
        self.p_omni3d_result_image = rospy.get_param("~omni3d_image", "/omni3d/result_image/image_raw")
        self.p_omni3d_pose_array = rospy.get_param("~omni3d_pose_array", "/tam_dynamic_map/omni3d_array")
        self.p_rgb_topic = rospy.get_param("~rgb_topic", "/relay/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed")
        self.p_camera_info_topic = rospy.get_param("~camera_info_topic", "/hsrb/head_rgbd_sensor/rgb/camera_info")
        self.p_yaml_path = rospy.get_param("~world_model_path", "hma_room05.yaml")
        self.topic_grid_map = rospy.get_param("~grid_map_topic", "/rtabmap/grid_map")
        self.topic_cost_map = rospy.get_param("~cost_map_topic", "/move_base_flex/global_costmap/costmap")

        self.world_model_path = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/map/" + self.p_yaml_path

        ###################################################
        # ROS_INTERFACE
        ###################################################
        self.srv_calc_goal = rospy.Service("tam_dynamic_map/get_navigation_goal/service", GetNavigationGoal, self.cb_calc_goal)
        self.map_data_msg = OccupancyGrid()
        self.sub_register("map_data_msg", self.topic_grid_map, queue_size=1, callback_func=self.run)

        self.loginfo("start calc nav goal service")

    def mapdata_to_image(self, map_msg: OccupancyGrid) -> np.array:
        """
        ラスタスキャンされたデータを読み込み、2次元配列に変換する関数
        """
        width, height = map_msg.info.width, map_msg.info.height

        map_data = np.where(map_msg.data == -1, 255, map_msg.data)  # -1（未知）の領域を255に置換する
        map_img = np.array(map_data, dtype=np.uint8).reshape(height, width)

        return map_img

    def get_point_label(self, occupancy_grid_msg: OccupancyGrid, x: float, y: float, input_type="coordinate") -> int:
        """
        マップ座標系で指定されたグリッドのラベルデータを返す関数
        Args:
            occupancy_grid_msg(OccupancyGrid): rosmsgで渡されるマップ情報
            x(float): ラベルデータを取得したい点(x)
            y(float): ラベルデータを取得したい点(y)
            input_type(str): x, yで指定した値の意味
                coordinate: マップ座標系における値
                grid: どのグリッドかを直接指定
        Returns:
            int: 座標に対応するラベル
        """

        # 必要な情報を抽出
        map_data = occupancy_grid_msg.data
        resolution = occupancy_grid_msg.info.resolution
        resolution = 0.05
        width = occupancy_grid_msg.info.width
        map_pose_x = occupancy_grid_msg.info.origin.position.x
        map_pose_y = occupancy_grid_msg.info.origin.position.y

        if input_type == "coordinate":
            # 対象とされたポイントがどのグリッドに対応するのかを求める
            target_map_x = -map_pose_x + x
            target_map_y = -map_pose_y + y

            target_grid_x = int(target_map_x / resolution)
            target_grid_y = int(target_map_y / resolution)
            # ラスタスキャンされているデータに対応する点を求める
            target_pose = (target_grid_y * width) + target_grid_x
        elif input_type == "grid":
            target_pose = (y * width) + x
        else:
            self.logwarn("input_typeは[coordinate]か[grid]かで指定してください")

        # 対象グリッドのラベルデータを取得する
        label_data = map_data[int(target_pose)]
        self.loginfo(label_data)

        return label_data

    def get_area_label(self, occupancy_grid_msg: OccupancyGrid, x: float, y: float, input_type="coordinate", return_size=35, debug=False):
        """
        領域で指定されたエリアのすべてのラベルデータを同一の配列形式で返す関数
        Args:
            occupancy_grid_msg(OccupancyGrid): rosmsgで渡されるマップ情報
            x(float): ラベルデータを取得したい点(x)
            y(float): ラベルデータを取得したい点(y)
            input_type(str): x, yで指定した値の意味
                coordinate: マップ座標系における値
                grid: どのグリッドかを直接指定
            return_size(int):
                指定した点を中心として，どこまでの範囲を取得するか
                （偶数を指定した場合は+1された範囲のデータが返ります）
        Returns:
            np.ndarray: 指定範囲のラベルデータをまとめたもの
        """

        # 必要な情報を抽出
        map_data = occupancy_grid_msg.data
        resolution = occupancy_grid_msg.info.resolution
        width = occupancy_grid_msg.info.width
        height = occupancy_grid_msg.info.height
        map_pose_x = occupancy_grid_msg.info.origin.position.x
        map_pose_y = occupancy_grid_msg.info.origin.position.y

        map_img = np.array(map_data, dtype=np.uint8).reshape(height, width)

        if input_type == "coordinate":
            # 対象とされたポイントがどのグリッドに対応するのかを求める
            target_map_x = -map_pose_x + x
            target_map_y = -map_pose_y + y
            search_area = int(return_size / 2)
            target_grid_x = int(target_map_x / resolution)
            target_grid_y = int(target_map_y / resolution)
            # ラスタスキャンされているデータに対応する点を求める
            # target_pose = (target_grid_y * width) + target_grid_x
        elif input_type == "grid":
            target_grid_x = x
            target_grid_y = y
        else:
            self.logwarn("input_typeは[coordinate]か[grid]かで指定してください")
            return False

        # オーバーフローへの対策
        if target_grid_x - search_area < 0:
            target_grid_x = search_area
        if target_grid_x + search_area > width:
            target_grid_x = width - search_area

        if target_grid_y - search_area < 0:
            target_grid_y = search_area
        if target_grid_y + search_area > height:
            target_grid_y = height - search_area

        # ほしい領域のみを抽出
        trim_map_img = map_img[(target_grid_y - search_area):(target_grid_y + search_area), (target_grid_x - search_area):(target_grid_x + search_area)]
        start_grid = (target_grid_x - search_area, target_grid_y - search_area)

        if debug:
            cv2.imshow("trim_map_img", trim_map_img)
            cv2.waitKey(0)

        return trim_map_img, start_grid

    def get_most_safe_point(self, pose: Pose2D, goal_torelance=7, refer_size=7) -> Pose2D:
        """
        指定した地点から，近傍にある安全なナビゲーション先を算出する関数
        Args:
            pose: ナビゲーションの目的地
                geometory_msgs.msgのPose2Dで指定
            goal_torelance(int): ナビゲーションの目的地から何グリッド分のずれを許容するか
            refer_size(int): 特定グリッドの危険度を算出する際に，どの範囲までのグリッドを参照するか
        Returns:
            Pose2D
        """
        # グローバルコストマップのトピックをサブスクライブ
        try:
            cost_map_data_msg: OccupancyGrid = rospy.wait_for_message(self.topic_cost_map, OccupancyGrid, timeout=5)
            self.loginfo("receive cost_map data")
        except rospy.exceptions.ROSException as e:
            # サブスクライブ出来なかった場合は入力された値をそのまま返す
            self.logdebug(e)
            self.logwarn("Could not receive cost map.")
            self.logwarn("Return original pose.")
            return pose

        # mapのlinkがはられている座標を取得
        map_pose_x = cost_map_data_msg.info.origin.position.x
        map_pose_y = cost_map_data_msg.info.origin.position.y

        # どの範囲のコストマップ情報を取得するのかを決定する
        return_size = 2 * (goal_torelance + refer_size + 1)
        trim_cost_map, start_grid = self.get_area_label(cost_map_data_msg, pose.x, pose.y, return_size=return_size, debug=self.debug_mode)
        self.logdebug("get selected area's cost map.")

        # 計算範囲の設定と計算用の変数初期化
        range_max = 2 * (goal_torelance + refer_size + 1) - refer_size
        calc_area = int(refer_size / 2)
        safe_grid = (0, 0)
        safe_area_cost = float("inf")
        origin_grid = int(return_size / 2)

        # 目的地のコストが一定以下だった場合，計算を行わない
        refer_cost_map = trim_cost_map[origin_grid:(origin_grid + calc_area), (origin_grid - calc_area):(origin_grid + calc_area)]
        if np.sum(refer_cost_map) < self.safe_th:
            self.loginfo("origin target place is safety")
            return pose

        # TODO: 本来のナビゲーションゴールからなるべく近い点を選ぶように重み付けを行う
        # 計算範囲内のコストの総和を算出し，最も低いものを採用する
        for y in range(int(refer_size / 2), range_max):
            for x in range(int(refer_size / 2), range_max):
                # 計算範囲のコストマップを
                refer_cost_map = trim_cost_map[(y - calc_area):(y + calc_area), (x - calc_area):(x + calc_area)]
                sum_cost_map = np.sum(refer_cost_map)

                # グリッド単位で見たときの，本来のナビゲーション目的地との距離
                orig_goal_distance = abs(x - goal_torelance) + abs(y - goal_torelance)

                # 距離によって，コスト値に重みをつける
                if orig_goal_distance < 4:
                    weight = 1.0
                elif orig_goal_distance < 8:
                    weight = 1.5
                elif orig_goal_distance < 16:
                    weight = 2.0
                else:
                    weight = 3.0

                sum_cost_map = weight * sum_cost_map

                # コストが最も低いデータを保存
                if sum_cost_map < safe_area_cost:
                    self.logdebug(sum_cost_map)
                    safe_area_cost = sum_cost_map
                    safe_grid = (x, y)

        # 得られたグリッド情報をマップ座標系に直す
        local_x = (start_grid[0] + safe_grid[0]) * cost_map_data_msg.info.resolution  # マップの原点 + 抽出した領域
        local_y = (start_grid[1] + safe_grid[1]) * cost_map_data_msg.info.resolution

        map_x = local_x + map_pose_x
        map_y = local_y + map_pose_y

        return Pose2D(map_x, map_y, pose.theta)

    def show_cost_map(self):
        map_data_msg = rospy.wait_for_message(self.topic_cost_map, OccupancyGrid)
        self.loginfo("receive map data")
        trim_map_img = self.get_area_label(map_data_msg, 3.7, -0.09, return_size=50, debug=self.debug_mode)
        return trim_map_img

    def run(self, msg):
        self.map_data_msg = msg
        map_img = self.mapdata_to_image(msg)
        return map_img

    def load_object_info(self, obj_id: str) -> dict:
        """Load the object information of the specified object.

        Parameters:
        - obj_id (str): The ID of the object to retrieve the pose for.

        Returns:
        - dict: The 2D pose of the specified object, or None if the object is not found.

        Raises:
        - Any Exception: Any exception that occurs during file reading or data processing.

        Note:
        - This function assumes that the world model data is stored in YAML format.
        """
        try:
            with open(self.world_model_path, 'r') as existing_file:
                world_model_data = yaml.safe_load(existing_file)
                print(world_model_data["world"])
        except Exception as e:
            self.logwarn(e)
            return None

        for data in world_model_data["world"]:
            if data["id"] != obj_id:
                continue
            return data

    def cb_calc_goal(self, req: GetNavigationGoal) -> GetNavigationGoalResponse:
        """ナビゲーションの目的地を算出する関数
        Args:
            req: (GetNavigationGoal)
        Returns:
            GetNavigationGoalResponse
        """
        obj_id = req.obj_id
        obj_ifno = self.load_obj_pose2d(obj_id=obj_id)
        if obj_ifno is None:
            return Pose2D(0, 0, 0)

        obj_pose2d = Pose2D()
        obj_pose2d.x = obj_ifno["pose"]["x"]
        obj_pose2d.y = obj_ifno["pose"]["y"]
        obj_pose2d.theta = obj_ifno["pose"]["z"]
        obj_scale = obj_ifno["scale"]


if __name__ == "__main__":
    rospy.init_node(os.path.basename(__file__).split(".")[0])
    p_loop_rate = rospy.get_param("~loop_rate", 30)
    loop_wait = rospy.Rate(p_loop_rate)

    cls = GetNavigationGoalService()
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        try:
            # rospy.sleep(0.1)
            pass
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"[{rospy.get_name()}]: FAILURE")
            rospy.logerr(e)
        loop_wait.sleep()
