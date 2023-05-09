# -*- coding: utf-8 -*-
# @Organization  : changanqiche
# @Author        : An Chao
# @Time          : 2022/7/13 16:50
# @Function      : IE
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


class Camera_FC:#前视相机
    ImageW=1920
    ImageH=1080
    FocalLengthX=1030.09247393442
    FocalLengthY=1036.35696377094
    PrincipalPointX=1047.06480360665
    PrincipalPointY=566.132172532619
    CamX=2049.03262133755
    CamY=17.352042623378
    CamZ=1356.31612406103
    thetaX=-1.55183829005273
    thetaY=-0.0138965790066937
    thetaZ=-1.59854070996435

class Camera_FLC:#前视左相机
    ImageW=640
    ImageH=480
    FocalLengthX=400.00826492356
    FocalLengthY=399.915984756393
    PrincipalPointX=326.646809685459
    PrincipalPointY=221.08276032183
    CamX=2017.34432157132
    CamY=979.224293984271
    CamZ=936.277011786294
    thetaX=-1.4854977357196
    thetaY=0.0164573936279063
    thetaZ=-0.658202015034656

class Camera_FRC:#前视右相机
    ImageW=640
    ImageH=480
    FocalLengthX=400.211812785984
    FocalLengthY=400.12610236914
    PrincipalPointX=324.641495885601
    PrincipalPointY=237.086983491275
    CamX=2019.34648697419
    CamY=-933.846337499431
    CamZ=938.938632250135
    thetaX=-1.50100541419735
    thetaY=0.0140042956183853
    thetaZ=-2.62038537712517

class Camera_RLC:#尾视左相机
    ImageW=640
    ImageH=480
    FocalLengthX=400.355403116186
    FocalLengthY=400.273909037567
    PrincipalPointX=318.708717093506
    PrincipalPointY=241.323299814695
    CamX=2242.05700973482
    CamY=935.943405139994
    CamZ=875.810154515316
    thetaX=-1.620951835844
    thetaY=-0.00651279433932462
    thetaZ=0.714579763932213

class Camera_RRC:#尾视右相机
    ImageW=640
    ImageH=480
    FocalLengthX=401.313590276652
    FocalLengthY=401.257734752059
    PrincipalPointX=332.23617463629
    PrincipalPointY=232.398448623212
    CamX=2230.88961495329
    CamY=-1019.77001479576
    CamZ=873.600761059582
    thetaX=-1.5661584984683
    thetaY=-0.0167083582090326
    thetaZ=2.26528279331194

class Camera_RC:#尾视相机
    ImageW=640
    ImageH=480
    FocalLengthX=400.995135356981
    FocalLengthY=400.94658927364
    PrincipalPointX=329.038600523693
    PrincipalPointY=252.873620317622
    CamX=-841.504966427994
    CamY=8.605207316614
    CamZ=991.980298601423
    thetaX=-1.63253288243424
    thetaY=-0.000757329950091724
    thetaZ=1.56585301538608


def get_sigle_camIE(cam_info):
    fx = cam_info.FocalLengthX
    fy = cam_info.FocalLengthY
    u0 = cam_info.PrincipalPointX
    v0 = cam_info.PrincipalPointY

    thetaX = cam_info.thetaX
    thetaY = cam_info.thetaY
    thetaZ = cam_info.thetaZ
    tx = cam_info.CamX/1000
    ty = cam_info.CamY/1000
    tz = cam_info.CamZ/1000

    intrinsic = np.array([[fx, 0, u0], 
                [0, fy, v0], 
                [0, 0, 1]])

    R4 = R.from_euler('zyx', [thetaX, thetaY, thetaZ], degrees=False)
    Rm = R4.as_matrix()#旋转矩阵
    qual = R4.as_quat()#四元数
    trans = [tx, ty, tz]
    return intrinsic, R4, trans


def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose

def get_all_camIE(flat=False, inv=False):
    keys = ["Camera_FLC", "Camera_FC", "Camera_FRC", "Camera_RLC", "Camera_RC", "Camera_RRC"]
    w_scale = 1920.0 / 640.0
    h_scale = 1080. / 480.
    result = {}
    for idx, key in enumerate(keys):
        per_ie = {}
        ob = eval(key)
        I, quat, T = get_sigle_camIE(ob)
        Q = quat.as_quat()
        rm = quat.as_matrix()
        if key == 'Camera_FC':
            print(key)
            I[0, 0] /= w_scale
            I[0, 2] /= w_scale
            I[1, 1] /= h_scale
            I[1, 2] /= h_scale
        per_ie["camera_intrinsic"] = I

        # flat extrinsic
        t = np.array(T, dtype=np.float32) 
        if flat:
            yaw = Quaternion(Q).yaw_pitch_roll[0]
            R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
            rot_mat = get_transformation_matrix(R, t, inv=inv)
        else:
            rot = np.array(rm, dtype=np.float32)
            rot_mat = np.eye(4, dtype=np.float32)
            rot_mat[:3, :3] = rot
            rot_mat[:3, -1] = t

        per_ie["rotation"] = Q
        per_ie["translation"] = T
        per_ie["extrinsic"] = rot_mat
        result[key] = per_ie
        # print(key+"\n:", per_ie)
    return result


if __name__ == "__main__":
    result = get_all_camIE()
    
