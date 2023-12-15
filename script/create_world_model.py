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

        # self.category_dict = {34: "table", 15: "chair"}
        self.category_dict = {34: "table"}
        self.tracking_objects = {}  # tracking用に用いる制御変数
        self.registered_objects = {}  # trackingの結果実際にmapに登録されるobjをまとめる変数
        self.index = 0
        self.index_dict = {"table": 0, "chair": 0}
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

    def register(self, obj: dict) -> str:

        # 登録名の決定
        category_name = obj["category"]
        registerd_key = str(f"{category_name}_{self.index_dict[category_name]}")
        self.index_dict[category_name] += 1
        self.loginfo(f"register new object {registerd_key}")

        # 登録
        self.registered_objects[registerd_key] = obj
        pose_xyz = {"x": obj["center_pose"].position.x, "y": obj["center_pose"].position.y, "z": obj["center_pose"].position.z}
        quaternion = {"x": obj["center_pose"].orientation.x, "y": obj["center_pose"].orientation.y, "z": obj["center_pose"].orientation.z, "w": obj["center_pose"].orientation.w}
        scale = {"x": obj["scale"][0], "y": obj["scale"][2], "z": obj["scale"][1]}
        registerd_info = {"id": registerd_key, "type": category_name, "pose": pose_xyz, "scale": scale, "quaternion": quaternion}

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


        return registerd_key

    # def tracker(self, center_pose: Pose, scale: Optional[float], category_id: int) -> bool:
    def tracker(self, detections: Optional[Omni3D]):
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

            # 距離によるトラッキング
            min_distance = np.inf
            target_key = -1
            for key, tracking_obj in self.tracking_objects.items():
                distance = self.calc_pose_distance(center_pose, tracking_obj["center_pose"])
                if distance < min_distance:
                    min_distance = distance
                    target_key = key

            # 距離がしきい値以上　新規オブジェクトとして登録
            if min_distance > self.p_distance_th:
                self.logdebug("register new objects for tracking")
                tracking_obj = {"center_pose": center_pose, "scale": scale, "category": category_name, "tracking_count": 0}
                self.tracking_objects[self.index] = tracking_obj
                self.index += 1
            else:
                target_obj = self.tracking_objects[target_key]
                target_obj["tracking_count"] += 1

                if target_obj["tracking_count"] == 100:
                    self.register(target_obj)

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
