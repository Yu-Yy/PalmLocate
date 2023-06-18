from .seg_hrnet import HighResolutionNet
import torch
import torch.nn as nn
from time import time




class LocateNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.hrnet_high = HighResolutionNet(cfg) 
        self.hrnet_low = HighResolutionNet(cfg)  # B, C, H, W
        # channel_wise_feature = cfg.DATASET.NUM_CLASSES # get the B, hxw, H, W
        self.final_feature_channel = int((cfg.TRAIN.PATCH_SIZE / 8) ** 2)
        self.channel_num = 16 # hyperparameter
        self.mnt_layer = nn.Sequential(nn.Conv2d(self.channel_num, 8, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
                                        nn.Sigmoid())


        self.hm_layer = nn.Sequential(nn.Conv2d(self.final_feature_channel, 8, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                        nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1))
        
        # self.angle_layer = nn.Sequential(nn.Conv2d(self.final_feature_channel, 8, kernel_size=3, stride=1, padding=1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, image, patch):
        large_scale_feature = self.hrnet_high(image)
        patch_feature = self.hrnet_low(patch)
        # get the attention map
        org_mnt = self.mnt_layer(large_scale_feature)
        patch_mnt = self.mnt_layer(patch_feature)

        # get the coherent feature, (B, C, H, W) dot (B, C, h ,w) --> (B, hxw, H, W)  
        _,_,H,W = large_scale_feature.shape
        _,_,h,w = patch_feature.shape  
        coh_feature = torch.einsum('bcde,bchw->bhwde', large_scale_feature, patch_feature) # d,e is H W
        coh_feature = coh_feature.view(-1, h*w, H, W)
        hm = self.hm_layer(coh_feature)
        # return hm, angle
        return hm, org_mnt, patch_mnt #, angle


