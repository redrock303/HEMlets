import torch 
import numpy as np
from resnet import * 
# from torchvision.models.resnet import model_zoo, model_urls
from torchvision.models.resnet import model_urls
from torch.utils import model_zoo


class TwoBranch(torch.nn.Module):
    def __init__(self,config,out_channel):
        super(TwoBranch,self).__init__()

        conv_body = []
        for i in range(3):
            if i == 0:
                conv_body.append(
                    torch.nn.ConvTranspose2d(config.MODEL.res_nfeat, config.MODEL.bra_nfeat,
                                             kernel_size=(4, 4), stride=2, padding=(1, 1), bias=False))
            else:
                conv_body.append(
                    torch.nn.ConvTranspose2d(config.MODEL.bra_nfeat, config.MODEL.bra_nfeat,
                                             kernel_size=(4, 4), stride=2, padding=(1, 1), bias=False))
            # conv_body.append(TwoBranchBlock(config.MODEL.bra_nfeat))
            conv_body.append(torch.nn.BatchNorm2d(config.MODEL.bra_nfeat))
            conv_body.append(torch.nn.ReLU(True))
        self.conv_body = torch.nn.Sequential(*conv_body)
        self.conv_tail = torch.nn.Conv2d(config.MODEL.bra_nfeat, out_channel, 3, 1,1)

    def forward(self,x):
        return self.conv_tail(self.conv_body(x) )


class CoordinateRegress(torch.nn.Module):
    def __init__(self, config):
        super(CoordinateRegress, self).__init__()
        '''
        Input: a 3d volume (4d tensor b,xyz_dim*n_joints,xyz_dim,xyz_dim)
        Dis_map: a 3d disparity map(1,n_joints,xyz_dim) tensor for soft-argmax operation to regress 3d joints
        '''
        self.config = config
        self.joint_num = config.MODEL.N_JOINT

        sum_step = config.MODEL.DIMS
        step = 1.0 / (sum_step-1)
        dis_map = np.array([i*step for i in range(sum_step)])

        dis_map = torch.from_numpy(dis_map).float()
        dis_map = dis_map.view(-1, 1, sum_step).repeat(1, self.joint_num, 1)
        self.register_buffer('dis_map', dis_map)
        

    def forward(self,volume):
        b, c, h, w = volume.size()
        volume = volume.view(b, self.joint_num, -1)
        volume = torch.nn.functional.softmax(volume, -1)
        volume = volume.view(b, self.joint_num, -1, h, w)

        dis_map = self.dis_map.repeat(b,1,1)
        acc_z = volume.sum(-1)
        acc_z = acc_z.sum(-1)
        # print('acc_z',acc_z.shape)
        j_z = (dis_map * acc_z).sum(-1)

        acc_y = volume.sum(-1)
        acc_y = acc_y.sum(2)
        # print('acc_y',acc_y.shape)
        j_y = (dis_map * acc_y).sum(-1)

        acc_x = volume.sum(2)
        acc_x = acc_x.sum(2)
        # print('acc_x',acc_x.shape)
        j_x = (dis_map * acc_x).sum(-1)

        joint_3d = torch.stack([j_x, j_y, j_z], -1)-0.5
        # print('joint_3d',joint_3d.shape)
        return joint_3d


class Network(torch.nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config

        self.backbone = get_resnet(config.MODEL.num_layers)

        # high-level feature encoder 
        self.conv_feature = TwoBranch(config, config.MODEL.high_nfeat)

        # HEMlets feature extraction 
        self.conv_FBI = TwoBranch(config, config.MODEL.fb_nfeat)

        # HEMlets feature encoder
        self.FBI_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(config.MODEL.fb_nfeat, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True)
        )

        # 3d volume regression  
        self.conv_tail = torch.nn.Sequential(
            torch.nn.Conv2d(256 + config.MODEL.high_nfeat, 1024, 1, 1, 0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(1024, config.MODEL.N_JOINT * config.MODEL.DIMS, 1, 1, 0)
        )
        # differential soft-argmax operation for 3d joint regression
        self.volume_reg = CoordinateRegress(config)

        for m in self.modules():   
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def init_backbone(self):
        print('init_backbone ...')
        _, _, _, name = resnet_spec[self.config.MODEL.num_layers]
        resnet_weights = model_zoo.load_url(model_urls[name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        resnet_weights.pop('fc.weight', None)
        resnet_weights.pop('fc.bias', None)
        self.backbone.load_state_dict(resnet_weights)

    def forward(self, x, gt_info=None, val=False):

        feature = self.backbone(x)
        high_feature = self.conv_feature(feature)
        fb_map = self.conv_FBI(feature)
        fb_map_feature = self.FBI_encoder(fb_map)

        feature_cat = torch.cat([high_feature,fb_map_feature],1)

        volume = self.conv_tail(feature_cat)
        # print('volume', volume.shape)
        joint3d = self.volume_reg(volume)
       
        return joint3d



if __name__ == '__main__':
    import torchvision 
    import numpy as np
    from config import config
    # net = Network(config ).cuda()
    ## to run in cpu ##
    net = Network(config)

    print("net have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters())/1000000.0))
    print("backbone have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.backbone.parameters())/1000000.0))
    print("conv_FBI have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.conv_FBI.parameters())/1000000.0))
    print("conv_feature have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.conv_feature.parameters())/1000000.0))
    # net.init_backbone()


    # init_model = torch.save(net.state_dict(), 'init.pth')
    ones = np.ones((2, 3, 256, 256))
    # input = torch.from_numpy(ones).float().cuda()
    ## to run in cpu ##
    input = torch.from_numpy(ones).float()

    out = net(input, val=True)
    print('out', out.shape, out.max(), out.min())
