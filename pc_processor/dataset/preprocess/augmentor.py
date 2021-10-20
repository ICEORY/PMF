import numpy as np
from scipy.spatial.transform import Rotation as R
import random


class AugmentParams(object):
    def __init__(self, p_flipx=0., p_flipy=0.,
                 p_transx=0., trans_xmin=0., trans_xmax=0.,
                 p_transy=0., trans_ymin=0., trans_ymax=0.,
                 p_transz=0., trans_zmin=0., trans_zmax=0.,
                 p_rot_roll=0., rot_rollmin=0., rot_rollmax=0.,
                 p_rot_pitch=0., rot_pitchmin=0, rot_pitchmax=0.,
                 p_rot_yaw=0., rot_yawmin=0., rot_yawmax=0.):
        self.p_flipx = p_flipx
        self.p_flipy = p_flipy

        self.p_transx = p_transx
        self.trans_xmin = trans_xmin
        self.trans_xmax = trans_xmax

        self.p_transy = p_transy
        self.trans_ymin = trans_ymin
        self.trans_ymax = trans_ymax

        self.p_transz = p_transz
        self.trans_zmin = trans_zmin
        self.trans_zmax = trans_zmax

        self.p_rot_roll = p_rot_roll
        self.rot_rollmin = rot_rollmin
        self.rot_rollmax = rot_rollmax

        self.p_rot_pitch = p_rot_pitch
        self.rot_pitchmin = rot_pitchmin
        self.rot_pitchmax = rot_pitchmax

        self.p_rot_yaw = p_rot_yaw
        self.rot_yawmin = rot_yawmin
        self.rot_yawmax = rot_yawmax

    def setFlipProb(self, p_flipx, p_flipy):
        self.p_flipx = p_flipx
        self.p_flipy = p_flipy

    def setTranslationParams(self,
                           p_transx=0., trans_xmin=0., trans_xmax=0.,
                           p_transy=0., trans_ymin=0., trans_ymax=0.,
                           p_transz=0., trans_zmin=0., trans_zmax=0.):
        self.p_transx = p_transx
        self.trans_xmin = trans_xmin
        self.trans_xmax = trans_xmax

        self.p_transy = p_transy
        self.trans_ymin = trans_ymin
        self.trans_ymax = trans_ymax

        self.p_transz = p_transz
        self.trans_zmin = trans_zmin
        self.trans_zmax = trans_zmax

    def setRotationParams(self,
                        p_rot_roll=0., rot_rollmin=0., rot_rollmax=0.,
                        p_rot_pitch=0., rot_pitchmin=0, rot_pitchmax=0.,
                        p_rot_yaw=0., rot_yawmin=0., rot_yawmax=0.):

        self.p_rot_roll = p_rot_roll
        self.rot_rollmin = rot_rollmin
        self.rot_rollmax = rot_rollmax

        self.p_rot_pitch = p_rot_pitch
        self.rot_pitchmin = rot_pitchmin
        self.rot_pitchmax = rot_pitchmax

        self.p_rot_yaw = p_rot_yaw
        self.rot_yawmin = rot_yawmin
        self.rot_yawmax = rot_yawmax

    def __str__(self):
        print("=== Augmentor parameters ===")
        print("p_flipx: {}, p_flipy: {}".format(self.p_flipx, self.p_flipy))
        print("p_transx: {}, p_transxmin: {}, p_transxmax: {}".format(
            self.p_transx, self.trans_xmin, self.trans_xmax))
        print("p_transy: {}, p_transymin: {}, p_transymax: {}".format(
            self.p_transy, self.trans_ymin, self.trans_ymax))
        print("p_transz: {}, p_transzmin: {}, p_transzmax: {}".format(
            self.p_transz, self.trans_zmin, self.trans_zmax))
        print("p_rotroll: {}, rot_rollmin: {}, rot_rollmax: {}".format(
            self.p_rot_roll, self.rot_rollmin, self.rot_rollmax))
        print("p_rotpitch: {}, rot_pitchmin: {}, rot_pitchmax: {}".format(
            self.p_rot_pitch, self.rot_pitchmin, self.rot_pitchmax))
        print("p_rotyaw: {}, rot_yawmin: {}, rot_yawmax: {}".format(
            self.p_rot_yaw, self.rot_yawmin, self.rot_yawmax))


class Augmentor(object):
    def __init__(self, params: AugmentParams):
        self.parmas = params

    @staticmethod
    def flipX(pointcloud: np.ndarray):
        pointcloud[:, 0] = -pointcloud[:, 0]
        return pointcloud

    @staticmethod
    def flipY(pointcloud: np.ndarray):
        pointcloud[:, 1] = -pointcloud[:, 1]
        return pointcloud

    @staticmethod
    def translation(pointcloud: np.ndarray, x: float, y: float, z: float):
        pointcloud[:, 0] += x
        pointcloud[:, 1] += y
        pointcloud[:, 2] += z
        return pointcloud

    @staticmethod
    def rotation(pointcloud: np.ndarray, roll: float, pitch: float, yaw: float, degrees=True):
        rot_matrix = R.from_euler(
            "zyx", [yaw, pitch, roll], degrees=degrees).as_matrix()
        pointcloud[:, :3] = np.matmul(pointcloud[:, :3], rot_matrix.T)
        return pointcloud

    @staticmethod
    def randomRotation(pointcloud: np.ndarray):
        rot_matrix = R.random(random_state=1234).as_matrix()
        pointcloud[:, :3] = np.matmul(pointcloud[:, :3], rot_matrix.T)
        return pointcloud

    def doAugmentation(self, pointcloud):
        # flip augment
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_flipx:
            pointcloud = self.flipX(pointcloud)

        rand = random.uniform(0, 1)
        if rand < self.parmas.p_flipy:
            pointcloud = self.flipY(pointcloud)

        # translation
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_transx:
            trans_x = random.uniform(
                self.parmas.trans_xmin, self.parmas.trans_xmax)
        else:
            trans_x = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_transy:
            trans_y = random.uniform(
                self.parmas.trans_ymin, self.parmas.trans_ymax)
        else:
            trans_y = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_transz:
            trans_z = random.uniform(
                self.parmas.trans_zmin, self.parmas.trans_zmax)
        else:
            trans_z = 0
        pointcloud = self.translation(pointcloud, trans_x, trans_y, trans_z)

        # rotation
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_roll:
            rot_roll = random.uniform(
                self.parmas.rot_rollmin, self.parmas.rot_rollmax)
        else:
            rot_roll = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_pitch:
            rot_pitch = random.uniform(
                self.parmas.rot_pitchmin, self.parmas.rot_pitchmax)
        else:
            rot_pitch = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_yaw:
            rot_yaw = random.uniform(
                self.parmas.rot_yawmin, self.parmas.rot_yawmax)
        else:
            rot_yaw = 0
        pointcloud = self.rotation(pointcloud, rot_roll, rot_pitch, rot_yaw)

        return pointcloud