import numpy as np 

from getActionID import getActionID
from table import *
from protocols import *


avg_skeletonal_legth = np.array([132.94859077417854, 132.94882565107528, 233.47061306301964,
 442.8946124486049, 454.2064471493663, 442.89441332923815, 454.20659046567124, 151.03143697339942, 151.03422571006854,
 121.13493767282833, 115.00222708021806, 278.89292359255, 251.72868026287756, 278.88277282953516, 251.73345095358297])

def MPJPE_P1_P2(pose_a,pose_b,debug = False,return_error_list = False):
    pose_a_j17 = pose_a[0:17]
    pose_b_j17 = pose_b[0:17]


    pose_a_j17 = pose_a_j17.reshape((-1,3))
    pose_b_j17 = pose_b_j17.reshape((-1,3))

    _, Z, T, b, c = compute_similarity_transform(pose_b_j17,pose_a_j17,compute_optimal_scale=True)
    out = (b*pose_a_j17.dot(T))+c
    protocol_error = np.average(np.sqrt(np.sum(np.square(out - pose_b_j17),axis=-1))) 

    pose_a_j17 = move_hip_to_origin(pose_a_j17)
    pose_b_j17 = move_hip_to_origin(pose_b_j17)

    error = np.average(np.sqrt(np.sum(np.square(pose_a_j17 - pose_b_j17),axis=-1))) 
    if debug:
        preJointError = np.sqrt( np.sum( np.square(pose_a_j17 - pose_b_j17) ,axis=-1)) 
        print ('preJointError',preJointError)
        print ('avg  JointError',error)
        print ('algin pose error',protocol_error)
    if return_error_list:
        return error,protocol_error,np.sqrt( np.sum( np.square(pose_a_j17 - pose_b_j17) ,axis=-1)) 
    return error,protocol_error

def move_hip_to_origin(pose):
    rootPos = pose[0]
    poseNew = np.zeros((pose.shape[0],3),dtype=float)
    for i in range(pose.shape[0]):
        poseNew[i,:] = pose[i,:] - rootPos
    return  poseNew

def from_normjoint_to_cropspace(joint3d):
    joint3d[:,:,:2] = (joint3d[:,:,:2] + 0.5 )*256.0
    return joint3d

def cropPoseToFullPose(cropPose,trans):
    fullPose = cropPose.copy()
    fullPose = fullPose / trans[2]
    fullPose[:,0] = fullPose[:,0] + trans[0]
    fullPose[:,1] = fullPose[:,1] + trans[1]

    return fullPose

def CamBackProj(cam_x, cam_y, depth, fx, fy, u, v,re =False):
    
    x = (cam_x - u) / fx * depth
    y = (cam_y - v) / fy * depth
    z = depth
    if re:
        return x, y, z,depth/fx,cam_x - u
    else:
        return x,y,z

def cal_skeleton_bone_length(joints):
    length = []
    for idx,(jt_a, jt_b) in enumerate(h36m_cons_n) :
        l = np.linalg.norm(joints[jt_a] - joints[jt_b])
        # print(bone_name[idx],l)
        length.append(l)
    return np.array(length)

def restore_cameraspace_3d_joints(joint2d,depth,fx = 1150,fy = 1150,u = 500,v = 500):

    # print('scale',scale)
    decenter_joint_x = joint2d[:,0] - u
    decenter_joint_y = joint2d[:,1] - v

    scale_init = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
    error_min = 9999
    error_min_idx = -1
    for idx,scale_rough in enumerate(scale_init):
        x = decenter_joint_x * (depth/fx + scale_rough)
        y = decenter_joint_y * (depth/fx + scale_rough)
        z = depth + scale_rough * 1150.0
        j3d = np.zeros((x.shape[0],3))
        for i in range(x.shape[0]):
            j3d[i,0] = x[i]
            j3d[i,1] = y[i]
            j3d[i,2] = z[i]
        bone_length_array = cal_skeleton_bone_length(j3d)
        error_ = np.abs(bone_length_array - avg_skeletonal_legth).mean()
        if error_min > error_:
            error_min = error_
            error_min_idx = idx 

    _scale_init = scale_init[error_min_idx]
    scale_delta_init = []
    for i in range(-5,5):
        scale_delta_init.append(0.1*i)

    error_min_idx = -1
    scale_opt = _scale_init

    for i in range(len(scale_delta_init)):
        scale_rough = _scale_init + scale_delta_init[i]

        x = decenter_joint_x * (depth/fx + scale_rough)
        y = decenter_joint_y * (depth/fy + scale_rough)
        z = depth + scale_rough * 1150.0
        j3d = np.zeros((x.shape[0],3))
        for i in range(x.shape[0]):
            j3d[i,0] = x[i]
            j3d[i,1] = y[i]
            j3d[i,2] = z[i]
        bone_length_array = cal_skeleton_bone_length(j3d)
        error_ = np.abs(bone_length_array - avg_skeletonal_legth).mean()
        if error_min > error_:
            error_min = error_
            error_min_idx = idx 
            scale_opt = scale_rough

    x = decenter_joint_x * (depth/fx + scale_opt) # gt_root / fx = scale_opt
    y = decenter_joint_y * (depth/fy + scale_opt)
    z = depth + scale_opt * 1150.0
    j3d = np.zeros((x.shape[0],3))
    for i in range(x.shape[0]):
        j3d[i,0] = x[i]
        j3d[i,1] = y[i]
        j3d[i,2] = z[i]

    return j3d

    

def invPoseToCamSpacePlus(cropPose3d,jointRoot,camID,trans):
    # convert normalized 3d joints to camera space 3d joints

    fullPose2d = cropPoseToFullPose(cropPose3d[:,0:2],trans)

    j3d = restore_cameraspace_3d_joints(fullPose2d,cropPose3d[:,2]  * 2000.0)

    return j3d,j3d,fullPose2d

def eval_metric(pred_joint3d_numpy,pred_joint3d_filp_numpy,gt_joint3d_numpy,camid_numpy,\
        trans_numpy,joint_root_numpy,gt_joint3d_j18_numpy,seqJsonDict=None,debug = False,return_viz_joints=False):
    
    pred_joint3d_numpy_crop = from_normjoint_to_cropspace(pred_joint3d_numpy)
    gt_joint3d_numpy_crop = from_normjoint_to_cropspace(gt_joint3d_numpy)
    pred_filp_joint3d_numpy_crop = from_normjoint_to_cropspace(pred_joint3d_filp_numpy)

    patch_width = 256.0

    for i in range(pred_joint3d_numpy.shape[0]):
        crop_pred_j3d = pred_joint3d_numpy_crop[i]
        crop_gt_j3d = gt_joint3d_numpy_crop[i]

        pipws_flip = pred_filp_joint3d_numpy_crop[i]
        pipws_flip[ :, 0] = patch_width - pipws_flip[ :, 0] - 1
        for pair in flip_pairs:
            tmp = pipws_flip[ pair[0], :].copy()
            pipws_flip[ pair[0], :] = pipws_flip[ pair[1], :].copy()
            pipws_flip[ pair[1], :] = tmp.copy()

        # blending flip 3D joints
        mixJoint = (pipws_flip + crop_pred_j3d) * 0.5

        # gt 3D joints
        gt_cam3d_j18 = gt_joint3d_j18_numpy[i]

        
        # restore the predicted camera spcae 3D joints
        pred_cam3d_unity,_,_ = invPoseToCamSpacePlus(mixJoint,joint_root_numpy[i],-1,trans_numpy[i])
        # protocol #1 & #2
        protocol_1m,protocol_2m = MPJPE_P1_P2(gt_cam3d_j18,pred_cam3d_unity)

        # get the mapping 
        camParaTmp = (camid_numpy[i]).astype(int)
        subject = int(camid_numpy[i,0])

        # get the action id (for evaluation) and video name
        actionID,videoName = getActionID(camParaTmp,seqJsonDict[subject],debug = debug)

        if return_viz_joints:
            crop_gt_j3d[:,2]*=128
            mixJoint[:,2]*=128
            return actionID,protocol_1m,protocol_2m,videoName,crop_gt_j3d,mixJoint

        return actionID,protocol_1m,protocol_2m,videoName

