from easydict import EasyDict as edict


class Config:
    # dataset
    DATASET = edict()

    
    # training image size
    DATASET.WIDTH = 256
    DATASET.HEIGHT = 256
    DATASET.SEED = 0

    

    # model
    MODEL = edict()
    
    MODEL.IN_CHANNEL = 3                       # rgb channel
    MODEL.num_layers = 50                      # resnet 50
    MODEL.res_nfeat = 2048
    MODEL.bra_nfeat = 256                      # the encodered fb feature channel
    MODEL.high_nfeat = 1024                    # high-level feautre channel
    
    MODEL.N_JOINT = 18                         # training joint count
    MODEL.fb_nfeat = 14 * 3 +  MODEL.N_JOINT   # fb feature channel
    MODEL.DIMS = 64                            # 3d volume dimension

    MODEL.DEVICE = 'cuda'

    VAL = edict()
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 0



config = Config()



