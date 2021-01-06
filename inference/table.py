import numpy as np 
import h5py 


H5DataSetType = h5py._hl.dataset.Dataset

H36M_TO_J18 = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27] # convert j32 of H36M to our j18 format

actions = ["Directions","Discussion","Eating","Greeting",
        "Phoning","Photo","Posing","Purchases",
        "Sitting","SittingDown","Smoking","Waiting",
        "WalkDog","Walking","WalkTogether"]

img_mean = np.array([123.675,116.280,103.530])
img_std = np.array([58.395,57.120,57.375])

flip_pairs = np.array([[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int)

JOINT_CONNECTIONS = [[1, 0], [4, 0], [7, 0],
                     [2,1], [3,2],
                     [5,4], [6,5],
                     [17,7], [8,17],
                     [14,8], [11,8], [9,8], [10,9],
                     [15,14], [16,15],
                     [12,11], [13,12]]

h36m_cons_n = [[1,0],[4,0],[7,0],
                [2,1],[3,2],
                [5,4],[6,5],
                [14,8],[11,8],[9,8],[10,9],
                [15,14],[16,15],
                [12,11],[13,12]] 
BONE_NAMES = ['r_hip', 'l_hip', 'low_tosor',
             'up_r_leg','low_r_leg',
             'up_l_leg','low_l_leg',
             'r_shoulder','l_shoulder','neck','head',
             'up_r_arm','low_r_arm',
             'up_l_arm','low_l_arm']
             
JOINT_COLOR_INDEX = [0, 2, 1,
                     0, 0,
                     2, 2,
                     1, 1,
                     0, 2, 1, 1,
                     0, 0,
                     2, 2]