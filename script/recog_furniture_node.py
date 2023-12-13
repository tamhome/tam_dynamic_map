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
from collections import OrderedDict
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

sys.path.append(roslib.packages.get_pkg_dir("semantic_map") + "/include/omni3d/")

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import UInt16, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tam_dynamic_map.msg import Omni3D, Omni3DArray

from cv_bridge import CvBridge
from tamlib.node_template import Node
# from tamlib.cv_bridge import CvBridge


class RecogFurnitureNode():

    def __init__(self, args):

        omniTopic = rospy.get_param('omniTopic', "/tam_dynamic_map/omni3d_array")
        # configPath = rospy.get_param('~configPath')
        # modelPath = rospy.get_param('~modelPath')
        # cameraTopic = rospy.get_param('cameraTopic')
        # jsonPath = rospy.get_param('jsonPath')
        # with open(jsonPath) as f:
        #     self.args = json.load(f)

        # args.config_file = configPath
        # args.opts = ['MODEL.WEIGHTS', modelPath]

        self.args = args
        self.cfg = self.setup(self.args)
        self.model = build_model(self.cfg)

        # logger.info("Model:\n{}".format(self.model))
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(self.cfg.MODEL.WEIGHTS, resume=True)

        self.model.eval()
        self.thres = args.threshold

        min_size = self.cfg.INPUT.MIN_SIZE_TEST
        max_size = self.cfg.INPUT.MAX_SIZE_TEST
        self.augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

        category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        self.metadata = util.load_json(category_path)
        self.cats = self.metadata['thing_classes']

        # self.min_z = self.args.min_z
        cam_h = self.args.camera_z + self.args.min_z
        self.K = np.reshape(np.array([self.args.K]), (3, 3))
        self.cam_poses = np.reshape(np.array([self.args.cam_poses]), (4, 4))
        self.cam_poses[:, 2] += cam_h

        self.camNum = 4
        self.bridge = CvBridge()
        self.detections = []
        self.cnt = 0

        self.rgb_topic = rospy.get_param("~rgb", "/hsrb/head_rgbd_sensor/rgb/image_raw")
        self.sub = rospy.Subscriber(self.rgb_topic, Image, self.callback)

        self.pred_pub = rospy.Publisher("omni3d", Float32MultiArray, queue_size=1)
        self.pred_pub_debug = rospy.Publisher('omni3d_debug', Image, queue_size=1)
        self.marker_pub = rospy.Publisher("omni3dMarkerTopic", MarkerArray,  queue_size=10)

        self.full_pred_pub = rospy.Publisher(omniTopic, Omni3DArray, queue_size=1)
        self.clr = cm.rainbow(np.linspace(0, 1, len(self.cats)))

        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

        self.listener = tf.TransformListener()

        rospy.loginfo("Omni3DNode::Ready!")

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

    def to_lab_coords(self, center, dim, rot):

        center = np.array([center]).T
        rot = np.array(rot)

        rot_R = self.origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
        center = (rot_R @ center).T.flatten()
        rot = rot_R @ rot

        return center, dim, rot

    def move_to_cam(self, gt, center, dim, rot):

        rot_T = np.eye(4)
        # self.R = origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
        rot_R = self.origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
        rot_T[:3, :3] = rot_R
        rot_T[0, 3] = gt[0]
        rot_T[1, 3] = gt[1]
        rot_T[2, 3] = gt[2]

        center = np.array([center[0], center[1], center[2], 1.0]).T
        rot = np.array(rot)

        center = (rot_T @ center).T.flatten()
        center = center[:3]
        rot = rot_R @ rot

        return center, dim, rot

    def get_camera_pose(self, target_frame="/map", camera_frame="/head_rgbd_sensor_link"):
        try:
            # waitForTransformメソッドで座標変換の準備ができるまで待つ
            self.listener.waitForTransform(target_frame, camera_frame, rospy.Time(), rospy.Duration(4.0))

            # lookupTransformメソッドで座標変換を行う
            (trans, rot) = self.listener.lookupTransform(target_frame, camera_frame, rospy.Time(0))

            # 変換された座標と姿勢を表示
            rospy.loginfo("Translation: %s, Rotation: %s", trans, rot)

            # 変換された座標と姿勢をPoseStampedメッセージに格納
            camera_pose = PoseStamped()
            camera_pose.header.frame_id = target_frame
            camera_pose.header.stamp = rospy.Time(0)
            camera_pose.pose.position.x = trans[0]
            camera_pose.pose.position.y = trans[1]
            camera_pose.pose.position.z = trans[2]
            camera_pose.pose.orientation.x = rot[0]
            camera_pose.pose.orientation.y = rot[1]
            camera_pose.pose.orientation.z = rot[2]
            camera_pose.pose.orientation.w = rot[3]

            return camera_pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF transform exception")
            return None

    def pose_to_matrix(self, pose):
        translation = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
        rotation = [pose.pose.orientation.x, pose.pose.orientation.y,
                    pose.pose.orientation.z, pose.pose.orientation.w]

        # ローカル座標系からグローバル座標系への変換行列
        translation_matrix = tft.translation_matrix(translation)
        rotation_matrix = tft.quaternion_matrix(rotation)

        # 4x4変換行列
        camera_matrix = np.dot(translation_matrix, rotation_matrix)

        return camera_matrix

    def createMarkers(self, mapObjs):

        markers = []

        for obj in mapObjs:

            marker = Marker()
            marker.header.frame_id = "map"
            marker.action = marker.ADD
            marker.type = marker.CUBE
            marker.pose.position.x = obj.center[0]
            marker.pose.position.y = obj.center[1]
            marker.pose.position.z = obj.center[2]

            r = R.from_matrix(obj.rot)
            q = r.as_quat()
            color = self.clr[obj.category]

            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = obj.dim[2]
            marker.scale.y = obj.dim[1]
            marker.scale.z = obj.dim[0]
            marker.color.a = 1.0
            marker.color.r = color[2]
            marker.color.g = color[1]
            marker.color.b = color[0]
            marker.lifetime = rospy.Duration(3)

            markers.append(marker)

        return markers

    def callback(self, msg: Image) -> None:
        """画像が入力されたときのコールバック関数
        Args:
            msg (Image): Image_msg
        Return:
            None
        """
        # camera座標を取得
        camera_pose = self.get_camera_pose()
        print(camera_pose)
        camera_matrix = self.pose_to_matrix(camera_pose)
        self.cam_poses = copy.deepcopy(camera_matrix)

        # self.min_z = self.args.min_z
        # cam_h = self.args.camera_z + self.args.min_z
        # self.K = np.reshape(np.array([self.args.K]), (3, 3))

        debug_img = self.infer(msg)
        image_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        image_msg.header.stamp = msg.header.stamp
        self.pred_pub_debug.publish(image_msg)

        msg = Float32MultiArray()
        msg.data = self.detections
        self.pred_pub.publish(msg)

    def infer(self, img_msg: Image):

        batched = []
        self.detections.clear()
        h = img_msg.height
        w = img_msg.width
        debug_img = np.zeros((h, w, 3)).astype(np.uint8)
        imgs = []
        markerArray = MarkerArray()

        im = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
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

        clr = cm.rainbow(np.linspace(0, 1, len(self.cats)))
        omniArray = []
        objs = []

        dets = alldets[0]['instances']
        n_det = len(dets)

        meshes = []
        meshes_text = []
        gt_ = self.cam_poses[0]
        gt = copy.deepcopy(gt_)
        print(gt)
        im = imgs[0]

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                    dets.pred_pose, dets.scores, dets.pred_classes
            )):

                # skip
                if score < self.thres:
                    continue

                cat = self.cats[cat_idx]

                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)

                center = center_cam.tolist()
                dim = dimensions.tolist()
                R_cam = pose.cpu().detach().numpy()
                conf = score.cpu().detach().numpy()
                category = cat_idx.cpu().detach().numpy()

                center, dim, rot = self.to_lab_coords(center, dim, R_cam)
                center, dim, rot = self.move_to_cam(gt, center, dim, rot)
                obj = MapObjectTracker(category, center, dim, rot, conf)
                objs.append(obj)

                rot = np.reshape(rot, (3, 3))
                box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
                verts, faces = get_cuboid_verts_faces(box3d, rot)
                verts = torch.unsqueeze(verts, dim=0)
                xyz = np.asarray(verts).reshape((8, 3))
                xyz = xyz[np.argsort(xyz[:, 2], axis=0)]
                m_xy = xyz[:4]

                omni_msg = Omni3D()
                omni_msg.center = center.flatten()
                omni_msg.dim = dim
                omni_msg.rot = rot.flatten()
                omni_msg.category = category
                omni_msg.confidence = conf
                omniArray.append(omni_msg)

                detc = xyz.flatten()
                detc = np.append(detc, conf)
                detc = np.append(detc, category)
                self.detections.extend(detc)

                xyxy, behind_camera, fully_behind = convert_3d_box_to_2d(self.K, bbox3D, R_cam, clipw=w, cliph=h, XYWH=False, min_z=0.00)
                bbox2D_proj = xyxy.cpu().detach().numpy().astype(np.int32)

                if fully_behind:
                    continue

                bbox2D_trunc = getTrunc2Dbbox(bbox2D_proj, h, w)
                x1, y1, x2, y2 = bbox2D_trunc

                color = 255 * clr[category, :3]
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                text1 = "{}".format(self.cats[category])
                text2 = "{:.2f}".format(conf)
                cv2.putText(im, text1, (x1 + 2, y1 + 40), 0, 2.0, color, thickness=3)
                cv2.putText(im, text2, (x1 + 2, y1 + 100), 0, 2.0, color, thickness=3)

        debug_img = im

        markers = self.createMarkers(objs)
        markerArray.markers = markers
        mid = 0
        for m in markerArray.markers:
            m.id = mid
            mid += 1
        self.marker_pub.publish(markerArray)

        omni_array_msg = Omni3DArray()
        omni_array_msg.header.stamp = img_msg.header.stamp
        omni_array_msg.detections = omniArray
        self.full_pred_pub.publish(omni_array_msg)

        return debug_img

    def run(self):
        pass

    def delete(self):
        pass


class ArgsSetting():
    # omni3dの動作に必要な引数を設定するためのクラス
    def __init__(self) -> None:
        self.config_file = "cubercnn://omni3d/cubercnn_DLA34_FPN.yaml"
        self.input_folder = "../include/omni3d/datasets/hma"
        self.threshold = 0.60
        self.display = False
        self.eval_only = True
        self.num_gpus = 1
        self.num_machines = 1
        self.machine_rank = 0
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        self.dist_url = "tcp://127.0.0.1:{}".format(port),
        self.opts = ['MODEL.WEIGHTS', "cubercnn://omni3d/cubercnn_DLA34_FPN.pth", 'OUTPUT_DIR', 'output/demo']
        self.cam_poses = [0.1, 0, 0, 0, 0, -0.1, 0, -1.5707963267948966, -0.1, 0, 0, 3.141592653589793, 0, 0.1, 0,  1.5707963267948966]
        self.min_z = -1.55708286
        self.camera_z = 0.93
        self.K = [538.2791736731888, 0.0, 318.6289689273318, 0.0, 539.083535340329, 235.7076528540864, 0.0, 0.0, 1.0]
        print(self.opts)


if __name__ == "__main__":
    rospy.init_node("semantic_map")
    args = ArgsSetting()
    cls = RecogFurnitureNode(args)
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        rospy.sleep(0.1)
