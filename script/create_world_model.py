#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tf
import sys
import cv2
import math
import time
import yaml
import copy
import json
import rospy
import roslib
import logging
import numpy as np
# import open3d as o3d
from matplotlib import cm
import tf.transformations as tft
from typing import Optional, Any, List
from collections import OrderedDict
from image_geometry import PinholeCameraModel
from lib.MapObjectTracker import MapObjectTracker
from lib.DatasetUtils import get_cuboid_verts_faces, convert_3d_box_to_2d, getTrunc2Dbbox
from scipy.spatial.transform import Rotation as R

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

# sys.dont_write_bytecode = True
# sys.path.append(os.getcwd())
# np.set_printoptions(suppress=True)

# sys.path.append(roslib.packages.get_pkg_dir("tam_dynamic_map") + "/include/omni3d/")

# from cubercnn.config import get_cfg_defaults
# from cubercnn.modeling.proposal_generator import RPNWithIgnore
# from cubercnn.modeling.roi_heads import ROIHeads3D
# from cubercnn.modeling.meta_arch import RCNN3D, build_model
# from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
# from cubercnn import util, vis

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import UInt16, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tam_dynamic_map.msg import Omni3D, Omni3DArray

# from cv_bridge import CvBridge
from tamlib.node_template import Node
from tamlib.utils import Logger
from tamlib.tf import Transform, euler2quaternion
from hsrlib.utils import utils, description, joints, locations
from tamlib.cv_bridge import CvBridge


class CreateWorldModel(Node):

    def __init__(self):
        super().__init__(loglevel="DEBUG")

        ###################################################
        # ROSPARAMの読み込み
        ###################################################
        self.p_omni3d_marker_array = rospy.get_param("~omni3d_marker_array", "/omni3d/objects")
        self.p_omni3d_result_image = rospy.get_param("~omni3d_image", "/omni3d/result_image")
        self.p_omni3d_pose_array = rospy.get_param("~omni3d_pose_array", "/tam_dynamic_map/omni3d_array")
        self.p_rgb_topic = rospy.get_param("~rgb", "/hsrb/head_rgbd_sensor/rgb/image_raw")
        self.p_camera_info_topic = rospy.get_param("~camera_info_topic", "/hsrb/head_rgbd_sensor/rgb/camera_info")

        self.p_yaml_path = rospy.get_param("~world_model_path", "hma_room05.yaml")
        self.p_distance_th = rospy.get_param("~distance_th", 0.08)
        self.p_iou_th = rospy.get_param("~iou_th", 0.1)
        self.p_register_th = rospy.get_param("~distance_th", 100)

        # self.category_dict = {34: "table", 15: "chair"}
        self.category_dict = {34: "table"}
        self.category_dict = {34: "table", 46: "bin"}
        self.tracking_objects = {}  # tracking用に用いる制御変数
        self.registered_objects = {}  # trackingの結果実際にmapに登録されるobjをまとめる変数
        self.index = 0
        self.index_dict = {"table": 0, "chair": 0, "bin": 0, "unknown": 0}
        self.world_model_path = roslib.packages.get_pkg_dir("tam_dynamic_map") + "/io/map/" + self.p_yaml_path

        ###################################################
        # 制御用変数の初期化
        ###################################################
        self.tamtf = Transform()
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.description = description.load_robot_description()

        ###################################################
        # ROS interface
        ###################################################
        self.omni3d_poses_msg = Omni3DArray()
        self.sub_register("omni3d_poses_msg", self.p_omni3d_pose_array, callback_func=self.cb_omni3d_array)

    @staticmethod
    def calc_pose_distance(pose1: Pose, pose2: Pose) -> float:
        """
        Calculate the Euclidean distance between two Pose messages.

        Parameters:
        - pose1 (Pose): The first Pose message.
        - pose2 (Pose): The second Pose message.

        Returns:
        - float: The Euclidean distance between pose1 and pose2.
        """
        # Extracting position components from Pose messages
        x1, y1, z1 = pose1.position.x, pose1.position.y, pose1.position.z
        x2, y2, z2 = pose2.position.x, pose2.position.y, pose2.position.z

        # Calculate Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        return abs(distance)

    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) between two 3D bounding boxes.

        Parameters:
        - box1: Tuple representing the first bounding box (center_x, center_y, center_z, size_x, size_y, size_z).
        - box2: Tuple representing the second bounding box (center_x, center_y, center_z, size_x, size_y, size_z).

        Returns:
        - iou: Intersection over Union value.
        """

        # Extracting box information
        center_x1, center_y1, center_z1, size_x1, size_y1, size_z1 = box1
        center_x2, center_y2, center_z2, size_x2, size_y2, size_z2 = box2

        # Calculating box coordinates
        min_x1, max_x1 = center_x1 - size_x1 / 2, center_x1 + size_x1 / 2
        min_y1, max_y1 = center_y1 - size_y1 / 2, center_y1 + size_y1 / 2
        min_z1, max_z1 = center_z1 - size_z1 / 2, center_z1 + size_z1 / 2

        min_x2, max_x2 = center_x2 - size_x2 / 2, center_x2 + size_x2 / 2
        min_y2, max_y2 = center_y2 - size_y2 / 2, center_y2 + size_y2 / 2
        min_z2, max_z2 = center_z2 - size_z2 / 2, center_z2 + size_z2 / 2

        # Calculating intersection coordinates
        intersection_x = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
        intersection_y = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
        intersection_z = max(0, min(max_z1, max_z2) - max(min_z1, min_z2))

        # Calculating intersection volume
        intersection_volume = intersection_x * intersection_y * intersection_z

        # Calculating union volume
        box1_volume = size_x1 * size_y1 * size_z1
        box2_volume = size_x2 * size_y2 * size_z2
        union_volume = box1_volume + box2_volume - intersection_volume

        # Calculating IoU
        iou = intersection_volume / union_volume if union_volume > 0 else 0.0

        return iou

    def delete_entry_by_id(self, yaml_file_path, target_key) -> Any:
        """
        特定のkeyを持つエントリを削除
        Args:
            data(Any): yamlファイルで読み込んだデータ
            target_key(str): 削除対象のkey
        Returns:
            Any: 特定のキー以下の情報を削除したデータ
        """
        yaml_file_path = self.world_model_path
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        data['world'] = [entry for entry in data['world'] if entry.get('id') != target_key]

        # YAMLファイルに書き出す
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)
        return data

    def update(self, registered_key: str, obj: dict) -> str:
        """yamlにすでに登録されているobjの内容を修正する関数
        Args:
            registered_key(str) Optional: 登録名
            obj(dict): yamlファイルに書き込むためのobjectの座標状など
        Return:
            str(registered_key): 登録したkeyの名前
        """
        # # ファイル読み込み
        # yaml_file_path = self.world_model_path
        # with open(yaml_file_path, 'r') as file:
        #     data = yaml.safe_load(file)

        # data = self.delete_entry_by_id(data, target_key=registered_key)
        self.delete_entry_by_id(self.world_model_path, target_key=registered_key)
        self.register(obj=obj, registered_key=registered_key)

        return registered_key

    def register(self, obj: dict, registered_key=None) -> str:
        """objの内容をyamlファイルに登録する関数
        Args:
            obj(dict): yamlファイルに書き込むためのobjectの座標状など
            registered_key(str) Optional: 登録名
        Return:
            str(registered_key): 登録したkeyの名前
        """

        category_name = obj["category"]
        if registered_key is None:
            # 登録名の決定
            registered_key = str(f"{category_name}_{self.index_dict[category_name]}")
            # registered_key = str(f"table_{self.index_dict[category_name]}")
            self.index_dict[category_name] += 1
            self.loginfo(f"register new object {registered_key}")

        # 登録
        self.registered_objects[registered_key] = obj
        pose_xyz = {"x": obj["center_pose"].position.x, "y": obj["center_pose"].position.y, "z": obj["center_pose"].position.z}
        quaternion = {"x": obj["center_pose"].orientation.x, "y": obj["center_pose"].orientation.y, "z": obj["center_pose"].orientation.z, "w": obj["center_pose"].orientation.w}
        scale = {"x": obj["scale"][2], "y": obj["scale"][1], "z": obj["scale"][0]}
        registerd_info = {"id": registered_key, "type": category_name, "pose": pose_xyz, "scale": scale, "quaternion": quaternion}

        # 既存のYAMLファイルからデータを読み込む
        yaml_file_path = self.world_model_path
        try:
            with open(yaml_file_path, 'r') as existing_file:
                world_model_data = yaml.safe_load(existing_file)
                print(world_model_data["world"])
        except Exception as e:
            self.logwarn(e)
            self.logwarn(f"we create new yaml file: {yaml_file_path}")
            world_model_data = {"world": []}

        world_model_data["world"].append(registerd_info)

        # YAMLファイルに書き出す
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(world_model_data, yaml_file, default_flow_style=False)

        return registered_key

    # def tracker(self, center_pose: Pose, scale: Optional[float], category_id: int) -> bool:
    def tracker(self, detections: Optional[Omni3D]) -> None:
        for detect_obj in detections:
            center_pose: Pose = detect_obj.center_pose
            scale: Optional[np.float32] = detect_obj.scale
            category_id: np.int64 = detect_obj.category
            confidence: np.float32 = detect_obj.confidence

            try:
                category_name = self.category_dict[category_id]
            except KeyError as e:
                self.logwarn(f"{e}, ignore object")
                continue

            # 3次元領域のIoUに基づくトラッキング
            max_iou = 0
            target_key = -1
            for key, tracking_obj in self.tracking_objects.items():
                iou = self.calculate_iou([center_pose.position.x, center_pose.position.y, center_pose.position.z, scale[2], scale[1], scale[0]], [tracking_obj["center_pose"].position.x, tracking_obj["center_pose"].position.y, tracking_obj["center_pose"].position.z, tracking_obj["scale"][2], tracking_obj["scale"][1], tracking_obj["scale"][0]])
                if max_iou < iou:
                    max_iou = iou
                    target_key = key

            # IoUの値が一定以下の場合は，新規オブジェクトとしてトラッキングの対象とする
            if max_iou < self.p_iou_th:
                self.logdebug("register new objects for tracking")
                tracking_obj = {"center_pose": center_pose, "scale": scale, "category": category_name, "tracking_count": 0, "confidence": confidence, "registered_key": None}
                self.tracking_objects[self.index] = tracking_obj
                self.index += 1

            # 同一のオブジェクトと判断したとき
            else:
                update_flag = False
                target_obj = self.tracking_objects[target_key]
                target_obj["tracking_count"] += 1

                if target_obj["tracking_count"] == 100:
                    registered_key = self.register(target_obj)
                    target_obj["registered_key"] = registered_key

                # 情報のアップデート
                if target_obj["confidence"] <= confidence and iou > 0.3:
                    target_obj["center_pose"] = center_pose
                    target_obj["scale"] = scale
                    target_obj["confidence"] = confidence
                    update_flag = True

                if target_obj["tracking_count"] > 100 and update_flag:
                    self.update(target_obj["registered_key"], target_obj)

    def cb_omni3d_array(self, msg: Omni3DArray) -> None:
        detections: Optional[Omni3D] = msg.detections
        self.tracker(detections)
        # for detect_obj in detections:
        #     center_pose: Pose = detect_obj.center_pose
        #     scale: Optional[np.float32] = detect_obj.scale
        #     category: np.int64 = detect_obj.category
        #     confidence: np.float32 = detect_obj.confidence
        #     self.tracker(center_pose, scale, category)

    def run(self):
        pass


if __name__ == "__main__":
    rospy.init_node("create_world_model_node")
    cls = CreateWorldModel()
    rospy.on_shutdown(cls.__del__)

    while not rospy.is_shutdown():
        rospy.sleep(0.1)
