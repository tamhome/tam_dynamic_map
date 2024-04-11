#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tf
import sys
import cv2
import time
import yaml
import copy
import json
import torch
import rospy
import roslib
import logging
import numpy as np
import open3d as o3d
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

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

sys.path.append(roslib.packages.get_pkg_dir("tam_dynamic_map") + "/include/omni3d/")

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
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


class RecogFurnitureNode(Logger):

    def __init__(self, args):
        super().__init__(loglevel="DEBUG")

        ###################################################
        # ROSPARAMの読み込み
        ###################################################
        self.p_omni3d_marker_array = rospy.get_param("~omni3d_marker_array", "/omni3d/objects")
        self.p_omni3d_result_image = rospy.get_param("~omni3d_image", "/omni3d/result_image/image_raw")
        self.p_omni3d_pose_array = rospy.get_param("~omni3d_pose_array", "/tam_dynamic_map/omni3d_array")
        self.p_rgb_topic = rospy.get_param("~rgb_topic", "/relay/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed")
        self.p_camera_info_topic = rospy.get_param("~camera_info_topic", "/hsrb/head_rgbd_sensor/rgb/camera_info")
        self.p_omni3d_threshold = rospy.get_param("~omni3d_th", 0.70)
        self.p_time_ignore_threshold = rospy.get_param("~omni3d_time_ignore_th", 0.15)
        self.ignore_threshold = rospy.Duration(self.p_time_ignore_threshold)

        ###################################################
        # Omni3dモデルの初期化
        ###################################################
        self.args = args
        self.cfg = self.setup(self.args)
        self.model = build_model(self.cfg)

        # logger.info("Model:\n{}".format(self.model))
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(self.cfg.MODEL.WEIGHTS, resume=True)

        self.model.eval()

        min_size = self.cfg.INPUT.MIN_SIZE_TEST
        max_size = self.cfg.INPUT.MAX_SIZE_TEST
        self.augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

        category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        self.metadata = util.load_json(category_path)
        self.cats = self.metadata['thing_classes']

        ###################################################
        # camera parameterの設定
        ###################################################
        self.camera_info = rospy.wait_for_message(self.p_camera_info_topic, CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)
        self.camera_frame_id = self.camera_info.header.frame_id
        self.camera_param_k = self.camera_info.K
        self.K = np.reshape(np.array([self.camera_param_k]), (3, 3))
        self.h = self.camera_info.height
        self.w = self.camera_info.width

        ###################################################
        # ROS interface
        ###################################################
        self.rgb_sub = rospy.Subscriber(self.p_rgb_topic, CompressedImage, self.callback)
        self.pred_pub_debug = rospy.Publisher(self.p_omni3d_result_image, Image, queue_size=10)
        self.marker_pub = rospy.Publisher(self.p_omni3d_marker_array, MarkerArray,  queue_size=10)
        self.full_pred_pub = rospy.Publisher(self.p_omni3d_pose_array, Omni3DArray, queue_size=1)

        ###################################################
        # 制御用変数の初期化
        ###################################################
        self.detections = []
        self.tamtf = Transform()
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.description = description.load_robot_description()
        self.clr = cm.rainbow(np.linspace(0, 1, len(self.cats)))
        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

        self.logsuccess("omni3d node is ready!")

    @staticmethod
    def setup(args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        get_cfg_defaults(cfg)

        config_file = args.config_file

        # store locally if needed
        if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
            config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

        cfg.merge_from_file(config_file)
        print(args.opts)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        return cfg

    def delete(self):
        pass

    # def get_camera_pose(self, target_frame: str = "/map", camera_frame: str = "/head_rgbd_sensor_rgb_frame") -> Optional[PoseStamped]:
    #     """
    #     カメラの姿勢を指定されたターゲットフレーム内で取得する関数

    #     Args:
    #         target_frame (str): 変換の対象となるフレーム。デフォルトは "/map" です。
    #         camera_frame (str): 変換のためのカメラフレーム。デフォルトは "/head_rgbd_sensor_rgb_frame" です。

    #     Return:
    #         Optional[PoseStamped]: ターゲットフレーム内でのカメラの変換された姿勢を含む PoseStamped メッセージ。
    #         TF変換例外（LookupException、ConnectivityException、ExtrapolationException）の場合は None を返します。
    #     """
    #     try:
    #         # waitForTransformメソッドで座標変換の準備ができるまで待つ
    #         self.listener.waitForTransform(target_frame, camera_frame, rospy.Time(), rospy.Duration(4.0))

    #         # lookupTransformメソッドで座標変換を行う
    #         (trans, rot) = self.listener.lookupTransform(target_frame, camera_frame, rospy.Time(0))

    #         # 変換された座標と姿勢をPoseStampedメッセージに格納
    #         camera_pose = PoseStamped()
    #         camera_pose.header.frame_id = target_frame
    #         camera_pose.header.stamp = rospy.Time(0)
    #         camera_pose.pose.position.x = trans[0]
    #         camera_pose.pose.position.y = trans[1]
    #         camera_pose.pose.position.z = trans[2]
    #         camera_pose.pose.orientation.x = rot[0]
    #         camera_pose.pose.orientation.y = rot[1]
    #         camera_pose.pose.orientation.z = rot[2]
    #         camera_pose.pose.orientation.w = rot[3]

    #         return camera_pose
    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         rospy.logwarn("TF transform exception")
    #         return None

    def pose_to_matrix(self, pose) -> np.ndarray:
        """
        PoseStampedメッセージを4x4変換行列に変換する関数

        Args:
            pose (PoseStamped): 変換する姿勢情報が格納されたPoseStampedメッセージ。

        Returns:
            np.ndarray: カメラの姿勢を表す4x4変換行列。

        注記:
        - 入力のPoseStampedメッセージは、ローカル座標系からグローバル座標系への変換に使用されます。
        """
        translation = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
        rotation = [pose.pose.orientation.x, pose.pose.orientation.y,
                    pose.pose.orientation.z, pose.pose.orientation.w]

        # ローカル座標系からグローバル座標系への変換行列
        translation_matrix = tft.translation_matrix(translation)
        rotation_matrix = tft.quaternion_matrix(rotation)

        # 4x4変換行列
        camera_matrix = np.dot(translation_matrix, rotation_matrix)

        return camera_matrix

    def create_cube_markers(self, objects: List[dict]) -> List[Marker]:
        """
        キューブのマーカー配列を作成する関数

        Args:
            objects (List[dict]): 各オブジェクトの情報が格納された辞書のリスト。
            各辞書は以下のキーを持つ:
            - "category" (str): オブジェクトのカテゴリ。
            - "pose" (Pose): オブジェクトの姿勢情報が格納されたPoseメッセージ。
            - "scale" (List[float]): オブジェクトのスケール。[長さ、幅、高さ]の順。

        Returns:
            List[Marker]: 作成されたキューブのマーカーのリスト。

        注記:
            - 各マーカーの色は、カテゴリに基づいて指定された色になります。
            - マーカーは "map" フレームに追加され、3秒間表示されます。
        """

        markers = []
        for obj in objects:
            color = self.clr[obj["category"]]
            marker = Marker()
            marker.header.frame_id = "map"
            marker.action = marker.ADD
            marker.type = marker.CUBE
            marker.pose = obj["pose"]

            marker.scale.x = obj["scale"][2]
            marker.scale.y = obj["scale"][1]
            marker.scale.z = obj["scale"][0]
            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.lifetime = rospy.Duration(3)

            markers.append(marker)

        return markers

    def callback(self, msg: CompressedImage) -> None:
        """画像が入力されたときのコールバック関数
        Args:
            msg (Image): Image_msg
        Return:
            None
        """
        debug_img = self.infer(msg)
        # image_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
        try:
            image_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="rgb8")
            image_msg.header.stamp = msg.header.stamp
            self.pred_pub_debug.publish(image_msg)
        except TypeError as e:
            self.logtrace(e)
            self.logdebug("cannot publish image")

    def infer(self, img_msg: CompressedImage) -> np.ndarray:
        """Omni3Dを利用した家具認識の実行
        Args:
            img_msg(Image): 対象とするimage_message
        Return:
            np.ndarray: 認識結果を描画した画像
        """
        img_timestamp = img_msg.header.stamp
        time_diff = rospy.Time.now() - img_timestamp
        if time_diff > self.ignore_threshold:
            self.logdebug("ignore image due to dime diff.")
            return

        batched = []
        self.detections.clear()
        debug_img = np.zeros((self.h, self.w, 3)).astype(np.uint8)
        imgs = []
        markerArray = MarkerArray()
        im = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        image_shape = im.shape[:2]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)
        aug_input = T.AugInput(im)
        im = aug_input.image
        batched.append(
            {
                'image': torch.as_tensor(np.ascontiguousarray(imgs[0].transpose(2, 0, 1))).cuda(),
                'height': image_shape[0],
                'width': image_shape[1],
                'K': self.K
            }
        )

        alldets = self.model(batched)

        omni_msg_array = []

        dets = alldets[0]['instances']
        n_det = len(dets)

        meshes = []
        meshes_text = []
        im = imgs[0]
        objects = []

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions,
                    dets.pred_pose, dets.scores, dets.pred_classes
            )):

                # skip
                if score < self.p_omni3d_threshold:
                    continue

                cat = self.cats[cat_idx]

                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)

                center = center_cam.tolist()
                scale = dimensions.tolist()
                rot_cam = pose.cpu().detach().numpy()
                rot = np.eye(4)
                rot[:3, :3] = rot_cam
                rot[3, 3] = 1
                q = tft.quaternion_from_matrix(rot)
                conf = score.cpu().detach().numpy()
                category = cat_idx.cpu().detach().numpy()

                # 座標変換(カメラ座標系→マップ座標系)
                map_pose: Pose = self.tamtf.get_pose_with_offset(
                    self.description.frame.map,
                    self.description.frame.rgbd,
                    offset=Pose(Point(x=center[0], y=center[1], z=center[2]), Quaternion(q[0], q[1], q[2], q[3])),
                )
                objects.append({"pose": map_pose, "category": category, "scale": scale})

                omni_msg = Omni3D()
                omni_msg.center_pose = map_pose
                omni_msg.scale = scale
                omni_msg.category = category
                omni_msg.confidence = conf
                omni_msg_array.append(omni_msg)

                xyxy, behind_camera, fully_behind = convert_3d_box_to_2d(self.K, bbox3D, rot_cam, clipw=self.w, cliph=self.h, XYWH=False, min_z=0.00)
                bbox2D_proj = xyxy.cpu().detach().numpy().astype(np.int32)

                if fully_behind:
                    continue

                bbox2D_trunc = getTrunc2Dbbox(bbox2D_proj, self.h, self.w)
                x1, y1, x2, y2 = bbox2D_trunc

                color = 255 * self.clr[category, :3]
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                text1 = "{}".format(self.cats[category])
                text2 = "{:.2f}".format(conf)
                cv2.putText(im, text1, (x1 + 2, y1 + 40), 0, 2.0, color, thickness=3)
                cv2.putText(im, text2, (x1 + 2, y1 + 100), 0, 2.0, color, thickness=3)

        debug_img = im

        markers = self.create_cube_markers(objects)
        markerArray.markers = markers
        for index, m in enumerate(markerArray.markers):
            m.id = index

        self.marker_pub.publish(markerArray)

        omni_array_msg = Omni3DArray()
        omni_array_msg.header.stamp = img_msg.header.stamp
        omni_array_msg.detections = omni_msg_array
        self.full_pred_pub.publish(omni_array_msg)

        return debug_img


class ArgsSetting():
    # omni3dの動作に必要な引数を設定するためのクラス
    def __init__(self) -> None:
        self.config_file = "cubercnn://omni3d/cubercnn_DLA34_FPN.yaml"
        self.input_folder = "../include/omni3d/datasets/hma"
        self.threshold = 0.70
        self.display = False
        self.eval_only = True
        self.num_gpus = 1
        self.num_machines = 1
        self.machine_rank = 0
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        self.dist_url = "tcp://127.0.0.1:{}".format(port),
        self.opts = ['MODEL.WEIGHTS', "cubercnn://omni3d/cubercnn_DLA34_FPN.pth", 'OUTPUT_DIR', 'output/demo']


if __name__ == "__main__":
    rospy.init_node("semantic_map")
    args = ArgsSetting()
    cls = RecogFurnitureNode(args)
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        rospy.sleep(0.1)
