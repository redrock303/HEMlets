import cv2
import os
import torch
import sys
import numpy as np
sys.path.append('./')
from table import *
from draw_figure import *
from getActionID import *
from metric_3d import * 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.gridspec as gridspec
import imageio_ffmpeg

import argparse

import time 
def normalized_to_original(image):
    image_numpy = image.cpu().numpy()
    image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))
    image_numpy = image_numpy * img_std + img_mean
    return image_numpy.astype(np.uint8)

def validate(model, val_loader, device,  subject=9,visualize = False):

    if visualize:
        fig = plt.figure( figsize=(19.2 / 2, 10.8 / 2) )
        gs1 = gridspec.GridSpec(2, 3) # 6 rows, 10 columns
        gs1.update(left=0.08, right=0.98,top=0.95,bottom=0.08,wspace=0.05, hspace=0.1)

        font = {'family' : 'serif',  
            'color'  : 'darkred',  
            'weight' : 'normal',  
            'size'   : 10,  
                }
        
        axImg=plt.subplot(gs1[0,0])
        axImg.axis('off')
        # axImg.set_title('Input Image' )#,fontdict=font)

        axPose3d_gt=plt.subplot(gs1[0,1],projection='3d')
        axPose3d_pred=plt.subplot(gs1[0,2],projection='3d')

        ax_mpjpe = plt.subplot(gs1[1,:])
    
        vw_pure = None
        vw_path = './hemlets.mp4'
    
        protocol_1_list = []
        protocol_2_list = []


    

    seqList = [11]
    seqJsonDict = {}
    for seq in seqList:
        seqJsonDict[seq] =  LoadSeqJsonDict(rootPath = './data/',subject=seq)

    # define action wise protocol dict
    action_wise_error_dict = {}
    for i in range(15):
        action_wise_error_dict[i] = [0.0, 0.0, 0.0]
    
    # result_pics_path = './pics/'

    # if os.path.exists(result_pics_path) is False:
    #     os.mkdir(result_pics_path)

    N_viz = val_loader.__len__() 
    idx_list = []
    for idx, data in enumerate(val_loader):
        if idx >= N_viz:
            break
        
        image,image_flip, trans, camid, joint3d, joint3d_camera,  root, name = data

        image = image.to(device)
        # for flip test to improve the accuracy of 3d pose prediction 
        image_flip = image_flip.to(device)

        joint3d = joint3d.to(device)

        # start_time = time.time()

        with torch.no_grad():
            pred_joint3d = model(image,val=True)
            pred_joint3d_flip = model(image_flip,val=True)

        
        pred_joint3d_flip_numpy = pred_joint3d_flip.cpu().numpy()
        pred_joint3d_numpy = pred_joint3d.cpu().numpy()

        # normalized (0-1) 3d joints
        gt_joint3d_numpy = joint3d.cpu().numpy()
        # camera space 3d joints
        joint3d_camera_numpy = joint3d_camera.cpu().numpy()

        # sequence encodered information [seqID,CamID,_,video_number]
        camid_numpy = camid.numpy()
        # transform matrix between the croppred (256*256) and org image (1000*1000)
        trans_numpy = trans.numpy()
        # abs joint root 
        joint_root_numpy = [None] 

        

        # calculate the MPJPE(Protocol #1) and MPJPE(Protocol #2)
        actionID, protocol_1,protocol_2,vedioName,gt_crop_3d_joint,pred_crop_3d_joint= \
                    eval_metric(pred_joint3d_numpy,pred_joint3d_flip_numpy,gt_joint3d_numpy,camid_numpy,\
                    trans_numpy,joint_root_numpy,joint3d_camera_numpy,seqJsonDict = seqJsonDict,debug = False,return_viz_joints=True)
        print('actionID, protocol_1,protocol_2,vedioName',actions[actionID], protocol_1,protocol_2,vedioName)
        # inference_time = time.time()-start_time


        if visualize:
            # convert image tensor to image numpy
            image_batch = normalized_to_original(image)

            # buffered the protocol #1 & #2 Error (in mm) for every frame.
            protocol_1_list.append(protocol_1)
            protocol_2_list.append(protocol_2)
            idx_list.append(idx)

            # start_img_time = time.time()

            img = image_batch[0]
            # draw gt 2d joint on the image  
            image_draw = drawSkeleton(img.copy(),gt_crop_3d_joint,JOINT_CONNECTIONS,JOINT_COLOR_INDEX)
            axImg.imshow(image_draw)
            axImg.axis('off')
            axImg.set_title('S_11: ({})'.format(vedioName),fontdict=font)
            # img_time = time.time() - start_img_time

            # start_skel_time = time.time()
            # draw 3d joints
            
            Draw3DSkeleton(gt_crop_3d_joint,axPose3d_gt,JOINT_CONNECTIONS,'GT_joint3d',fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
            Draw3DSkeleton(pred_crop_3d_joint,axPose3d_pred,JOINT_CONNECTIONS,'Pred_joint3d',fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
            # skel_time = time.time() - start_skel_time
            # plot the cruve of Protocol #1 and Protocol #2

            # start_line_time = time.time()

            ax_mpjpe.plot(idx_list[-3:],protocol_1_list[-3:],color='b',label='Protocol #1')
            ax_mpjpe.plot(idx_list[-3:],protocol_2_list[-3:],color='g',label='Protocol #2')

            # ax_mpjpe.plot([idx-0.5,idx+0.5],[protocol_1_list[-1],protocol_1_list[-1]],color='b',label='Protocol #1')
            # ax_mpjpe.plot([idx-0.5,idx+0.5],[protocol_2_list[-1],protocol_2_list[-1]],color='g',label='Protocol #2')

            # line_time = time.time() - start_line_time

            # start_save_time = time.time()

            # add legend annotations
            
            plt.legend(loc = 'upper right',edgecolor='blue')
            ax_mpjpe.set_ylabel('Pose Error (mm)',fontsize = 10)
            ax_mpjpe.set_xlabel('Frame',fontsize = 10)

            ax_mpjpe.set_xlim(0,N_viz)
            ax_mpjpe.set_ylim(7,65)

            if idx >= N_viz-1:
                ax_mpjpe.plot(idx_list,protocol_1_list,color='b',label='Protocol #1')
                ax_mpjpe.plot(idx_list,protocol_2_list,color='g',label='Protocol #2')

            plt.draw()             
            plt.pause(0.01)
            
            if idx == N_viz-1:
                input('check')
            # plt.savefig('{}/{}.png'.format(result_pics_path,str(idx).zfill(5)))
            plt.savefig('./temp.png')
            
            img_viz = cv2.imread('./temp.png')

            axImg.cla()
            axPose3d_gt.cla()
            axPose3d_pred.cla()
            ax_mpjpe.cla()
            # save_time = time.time() - start_save_time

            # print('time',inference_time,img_time,skel_time,line_time,save_time)
            #write the rendered frame into a video
            if vw_pure is None:# initialize the video writer 
                wSize = (img_viz.shape[1],img_viz.shape[0])
                fps = 25

                vw_pure = imageio_ffmpeg.write_frames(vw_path,
                                                        wSize,
                                                        fps=fps,
                                                        quality=8,
                                                        
                                                        ffmpeg_timeout=0,       # libx264 in default
                                                        macro_block_size=1) 
                vw_pure.send( None )

            # convert BGR to RGB 
            vw_pure.send(img_viz[:,:,::-1].copy())

    
        # cache the action-wise error 
        action_wise_error_dict[actionID][0] += protocol_1
        action_wise_error_dict[actionID][1] += protocol_2
        action_wise_error_dict[actionID][2] += 1

    if visualize:
        vw_pure.close()


    for k,_ in action_wise_error_dict.items():
        if action_wise_error_dict[k][2]>0:
            print(actions[k],action_wise_error_dict[k][0]/action_wise_error_dict[k][2],action_wise_error_dict[k][1]/action_wise_error_dict[k][2])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="HEMlets(pose) inference script")
    parser.add_argument("--ckpt_path", type=str,\
                        default="./ckpt/hemlets_h36m_lastest.pth", \
                        help='path to model of the pretrained model')
    parser.add_argument("--dataset_path", type=str,\
                        default="./data/S11/S_11_C_4_1_full.h5", \
                        help='path to a dataset')
    parser.add_argument("--video_id", type=int,\
                        default=1, \
                        help='video id (1-120) pre sequence of Human3.6M')
    parser.add_argument("--visualize", type=int,\
                        default=0, \
                        help='activate the function of visualize')
    parser.add_argument("--sequence_id", type=int,\
                        default=11, \
                        help='evaluation sequence id of the testing dataset')

    argspar = parser.parse_args()
    print('argspar',argspar)

    from config import config
    from network import Network
    import dataloader
    from model_opr import load_model
    from test_dataset import H36M

    # define network 
    model = Network(config)
    # device = torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)


    checkpoint_path = argspar.ckpt_path # './ckpt/hemlets_lastest.pth'
    tiny_dataset = argspar.dataset_path  #'./data/S11/S_11_C_4_1_full.h5'
    print('tiny_dataset',tiny_dataset)
    video_id = argspar.video_id # (1-120)

    visualize = True if argspar.visualize==1 else False
    print('argspar.visualize',argspar.visualize)


    # load model weights
    load_model(model, checkpoint_path, cpu=not torch.cuda.is_available())
    model.eval()


    subject = argspar.sequence_id #11

    # define dataset and dataloader
    val_dataset = H36M(h5_path = tiny_dataset,video_id=video_id,subject=subject, split='val')
    val_loader = dataloader.val_loader(val_dataset, config, 0, 1)

    # start evaluation...
    print(validate(model, val_loader, device, subject=subject,visualize=visualize))
