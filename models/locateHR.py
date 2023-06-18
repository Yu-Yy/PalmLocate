from .seg_hrnet import HighResolutionNet
import torch
import torch.nn as nn

class LocateHR(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.hrnet = HighResolutionNet(cfg) 
        # self.hrnet_low = HighResolutionNet(cfg)  # B, C, H, W
        # channel_wise_feature = cfg.DATASET.NUM_CLASSES # get the B, hxw, H, W
        self.final_feature_channel = int((cfg.TRAIN.PATCH_SIZE / 8) ** 2)
        self.hm_layer = nn.Sequential(nn.Conv2d(self.final_feature_channel, 8, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                        nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1))
        # self.angle_layer = nn.Sequential(nn.Conv2d(self.final_feature_channel, 8, kernel_size=3, stride=1, padding=1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, image, patch):
        large_scale_feature = self.hrnet(image)
        patch_feature = self.hrnet(patch)
        # get the coherent feature, (B, C, H, W) dot (B, C, h ,w) --> (B, hxw, H, W)  
        _,_,H,W = large_scale_feature.shape
        _,_,h,w = patch_feature.shape  
        coh_feature = torch.einsum('bcHW,bchw->bhwHW', large_scale_feature, patch_feature)
        coh_feature = coh_feature.view(-1, h*w, H, W)
        hm = self.hm_layer(coh_feature)
        # angle = self.angle_layer(coh_feature)
        return hm  #, angle


