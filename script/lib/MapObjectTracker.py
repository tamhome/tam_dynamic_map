import numpy as np
import open3d as o3d
# import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d


class MapObjectTracker():

    uid = 0
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

    def __init__(self, category, center, dim, rot, conf, room=0, static=False, seen=1, skip=0, times_matched=1, times_skipped=0):
        self.category = category
        self.center = center
        self.dim = np.asarray(dim)
        self.rot = rot
        self.seen = seen
        self.times_matched = times_matched
        self.times_skipped = times_skipped
        self.skip = skip
        self.uid = MapObjectTracker.uid
        self.conf = conf
        self.static = static
        self.active = 1
        self.inactive = 0
        self.room = room
        self.volume = self.dim[0] * self.dim[1] * self.dim[2] 

        MapObjectTracker.uid += 1

    def predict(self, isActive):

        if isActive:
            self.active += 1
            self.skip += 1
            self.times_skipped += 1
            self.inactive = 0
        else:
            self.active = 0
            self.inactive += 1

    def mergeObject(self, mo):

        self.center = (self.center * self.times_matched + mo.center * mo.times_matched) / ( self.times_matched + mo.times_matched)

        test_rot = np.linalg.inv(self.rot) @ mo.rot
        r = R.from_matrix(test_rot)
        x, y, z = r.as_euler('xyz', degrees=False)
        nrm1 = np.linalg.norm(np.array([0, 0, z]))

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

        rot_180 = origin.get_rotation_matrix_from_xyz((0, 0,np.pi))
        test_rot = np.linalg.inv(self.rot) @ rot_180 @ mo.rot
        r = R.from_matrix(test_rot)
        x, y, z = r.as_euler('xyz', degrees=False)
        nrm2 = np.linalg.norm(np.array([0, 0, z]))

        best_rot = mo.rot
        if nrm2 < nrm1:
            best_rot = rot_180 @ mo.rot

        rot_avg = (self.rot * self.times_matched + best_rot * mo.times_matched) / ( self.times_matched + mo.times_matched)
        u, s, vh = np.linalg.svd(rot_avg, full_matrices=True)
        self.rot = u @ vh

        self.dim = (self.dim * self.times_matched + mo.dim * mo.times_matched) / ( self.times_matched + mo.times_matched)
        self.volume = self.dim[0] * self.dim[1] * self.dim[2]

        self.times_matched += mo.times_matched
        self.times_skipped = 0
        self.skip = 0
        self.inactive = 0

    def avgRotation(self, mo):

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

        test_rot = np.linalg.inv(self.rot) @ mo.rot
        r = R.from_matrix(test_rot)
        x, y, z = r.as_euler('xyz', degrees=False)
        nrmz1 = np.linalg.norm(np.array([0, 0, z]))

        rot_180 = origin.get_rotation_matrix_from_xyz((0, 0,np.pi))
        test_rot = np.linalg.inv(self.rot) @ rot_180 @ mo.rot
        r = R.from_matrix(test_rot)
        x, y, z = r.as_euler('xyz', degrees=False)
        nrmz2 = np.linalg.norm(np.array([0, 0, z]))

        best_rot = mo.rot
        if nrmz2 < nrmz1:
            best_rot = rot_180 @ mo.rot

        rot_avg = (self.rot * self.times_matched + best_rot * mo.times_matched) / ( self.times_matched + mo.times_matched)
        u, s, vh = np.linalg.svd(rot_avg, full_matrices=True)
        self.rot = u @ vh
