import torch
import numpy as np
import torch.utils.data as data
from glob import glob
import os
import cv2
import copy
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn

class NormalizeModule(nn.Module):
    def __init__(self, m0, var0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        # x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        # x_var = x.var(dim=(1, 2, 3), keepdim=True)
        x_m = x.mean()
        x_var = x.var()
        y = (self.var0 * (x - x_m) ** 2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y) 
        return y

class THUPALM(data.Dataset):
    def __init__(self, cfg, root, is_train=True, transform=True):
        # irrelevant to the ppi's and lightness
        super(THUPALM, self).__init__()
        # self.cfg = cfg # light it after debug
        self.root = root
        self.is_train = is_train
        self.transform = transform
        # set the path resolution
        # self.image_folder = os.path.join(self.root, 'image_match') # unified or match
        # self.mask_folder = os.path.join(self.root, 'mask_match')
        self.image_folder = os.path.join(self.root, 'image_match') # unified or match
        self.mask_folder = os.path.join(self.root, 'mask_match')
        self.mnt_folder = os.path.join(self.root, 'mnt_match')
        self.image_list = glob(os.path.join(self.image_folder, '*.bmp'))
        self.image_list.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        # self.mask_list = glob(os.path.join(self.mask_folder, '*.png'))
        self.mask_list = glob(os.path.join(self.mask_folder, '*.bmp'))
        self.mask_list.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        samples_num = len(self.image_list) // 16
        if self.is_train:
            data_num = int(samples_num * 0.7)
            self.image_list = self.image_list[:data_num*16]
            self.mask_list = self.mask_list[:data_num*16]
        else:
            data_num = samples_num - int(samples_num * 0.7)
            self.image_list = self.image_list[-data_num*16:]
            self.mask_list = self.mask_list[-data_num*16:]
        
        self.invalid_sample_txt = os.path.join(self.root, 'invalid_list.txt')
        self.invalid_list = []
        self.image_input_size = np.array([1024,1024]) # 250
        # self.patch_input_size = np.array([cfg.TRAIN.PATCH_SIZE,cfg.TRAIN.PATCH_SIZE]) #np.array([128,128])
        self.patch_size = np.array([cfg.TRAIN.PATCH_SIZE,cfg.TRAIN.PATCH_SIZE])
        self.sigma_weight = 1
        with open(self.invalid_sample_txt, 'r') as f:
            for line in f.readlines():
                self.invalid_list.append(line.strip())
        # output the heatmap, original image and transformed patch image
        # self.image_mean = torch.tensor(0.48)
        # self.image_std = torch.tensor(0.22)
        self.normalize = NormalizeModule(0, 1)

        
    def __len__(self):
        return len(self.image_list)
    
    def get_patch_image(self, image, mnt_map, center, angle):
        # get the patch image from the original image with the center and angle
        # get the rotation matrix
        M = cv2.getRotationMatrix2D((center[1].astype(np.int16), center[0].astype(np.int16)), -angle, 1)
        # get the rotated image
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        rotated_mnt_map = cv2.warpAffine(mnt_map, M, (image.shape[1], image.shape[0]))
        # get the patch image
        patch_image = rotated_image[center[0]-self.patch_size[0]//2:center[0]+self.patch_size[0]//2,
                                    center[1]-self.patch_size[1]//2:center[1]+self.patch_size[1]//2]
        patch_mnt_map = rotated_mnt_map[center[0]-self.patch_size[0]//2:center[0]+self.patch_size[0]//2,
                                    center[1]-self.patch_size[1]//2:center[1]+self.patch_size[1]//2]
        return patch_image, patch_mnt_map
    
    def get_heatmap(self, center, heatmap_shape):
        # get the heatmap with gaussian blur acording to the center
        heatmap = np.zeros(heatmap_shape)
        sigma = self.patch_size[0] / 3 # path is square
        tmp_size = sigma * 3 * self.sigma_weight
        ul = [int(center[1]- tmp_size), int(center[0] - tmp_size)]
        br = [int(center[1] + tmp_size + 1), int(center[0] + tmp_size + 1)]
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(
                    -((x - x0)**2 + (y - y0)**2) / (2 * sigma ** 2))
        # Usable gaussian range
        g_x = max(0,
                    -ul[0]), min(br[0], heatmap_shape[0]) - ul[0]
        g_y = max(0,
                    -ul[1]), min(br[1], heatmap_shape[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_shape[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_shape[0])
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return heatmap

    def add_noise(self, image):
        # adaptive histogram equalization to gray image
        org_image = image.copy()
        patch_size = self.patch_size[0]
        # linspace
        grid_cols = np.linspace(0, patch_size, 25)
        grid_cols = grid_cols[1:-1].astype(np.int16)
        # random true or false        
        noise_col = np.random.choice([True, False], size=1, p=[0.5, 0.5])
        image[:, grid_cols] = org_image[:, grid_cols] ** 1.2
        if noise_col:
            image[:, grid_cols+1] = org_image[:, grid_cols+1] ** 1.2
        else:
            image[:, grid_cols-1] = org_image[:, grid_cols-1] ** 1.2
        
        # random int number from 1 to 4
        noise_row = np.random.randint(1, 5)
        # random value from half of the patch size to the patch size with the number defined by noise_row
        grid_row = np.random.choice(np.linspace(patch_size//2, patch_size-3, 5).astype(np.int16), size=noise_row, replace=False)
        image[grid_row, :] = org_image[grid_row, :] ** 1.5
        image[grid_row+2, :] = org_image[grid_row+2, :] ** 1.5
        image[grid_row-2, :] = org_image[grid_row-2, :] ** 1.5

        return image

    def create_mnt_map(self, shape, center):
        # center is N x 2
        center = center.astype(np.int16)
        mnt_map = np.zeros(shape)
        # delete the center out of the image
        center = center[np.where((center[:, 0] >= 0) & (center[:, 0] < shape[1]) & (center[:, 1] >= 0) & (center[:, 1] < shape[0]))]
        # center is x y coord, and make sure the center is in the image and value is 1
        mnt_map[center[:, 1],center[:, 0]] = 1
        return mnt_map


    def __getitem__(self, index):
        image_name = os.path.basename(self.image_list[index])
        image_tag = image_name.split('.')[0]
        group = image_tag.split('_')[0] + '_' + image_tag.split('_')[1]
        group_idx = int(image_tag.split('_')[2])
        if image_tag in self.invalid_list:
            return None, None, None, None
        patch_image_index = random.randint(1, 8)
        if patch_image_index != group_idx:
            self.sigma_weight = 1.5
        
        patch_image_source_file = os.path.join(self.image_folder, group + '_' + str(patch_image_index) + '.bmp') 
        patch_image_mask_file = os.path.join(self.mask_folder, group + '_' + str(patch_image_index) + '_mask' +'.bmp') # '.bmp'
        source_image_mask_file = os.path.join(self.mask_folder, group + '_' + str(group_idx) + '_mask' +'.bmp') # '.bmp'
        source_image_mnt_file = os.path.join(self.mnt_folder, group + '_' + str(group_idx) +'.txt') # '.txt'
        patch_image_mnt_file = os.path.join(self.mnt_folder, group + '_' + str(patch_image_index) +'.txt') # '.txt'
        # data gap by ','
        source_image_mnt = np.loadtxt(source_image_mnt_file, delimiter=',')
        patch_image_mnt = np.loadtxt(patch_image_mnt_file, delimiter=',')
        

        source_mask = cv2.imread(source_image_mask_file,flags=cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(self.image_list[index],flags=cv2.IMREAD_GRAYSCALE)
        # create the mnt_position_map
        source_mnt_map = self.create_mnt_map(image.shape, source_image_mnt)
        patch_mnt_map = self.create_mnt_map(image.shape, patch_image_mnt)
        image[source_mask==0] = 255

        patch_image_source = cv2.imread(patch_image_source_file,flags=cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # enhance the image
        patch_image_source = clahe.apply(patch_image_source) # adaptive histogram equalization
        image = clahe.apply(image)

        mask = cv2.imread(patch_image_mask_file,flags=cv2.IMREAD_GRAYSCALE)
        # non mask region is 255
        image = cv2.resize(image, (self.image_input_size[1], self.image_input_size[0])) # change into 250 ppi
        source_mnt_map = cv2.resize(source_mnt_map, (self.image_input_size[1], self.image_input_size[0]))
        
        # patch_image_source = cv2.resize(patch_image_source, (self.image_input_size[1], self.image_input_size[0])) # do not change
        # mask = cv2.resize(mask, (self.image_input_size[1], self.image_input_size[0]))

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.bool8) 
        patch_image_source = patch_image_source.astype(np.float32) / 255.0

        image_shape = patch_image_source.shape
        # uniform the image and patch_image_source using transform.Normalize
        
        # segment the patch from the valid_mask region
        index = np.argwhere(mask)
        # random select a index
        random_idx = random.randint(0, len(index)-1)
        # get the center of the patch
        center = index[random_idx]
        tmp_size = self.patch_size[0] // 2 #* self.sigma_weight
        ul = [int(center[1]- tmp_size), int(center[0] - tmp_size)]
        br = [int(center[1] + tmp_size + 1), int(center[0] + tmp_size + 1)]
        while True:
            if ul[0] >= image_shape[1] or ul[1] >= image_shape[0] or br[0] < 0 or br[1] < 0 or\
                br[0] >= image_shape[1] or br[1] >= image_shape[0] or ul[0] < 0 or ul[1] < 0:
                random_idx = random.randint(0, len(index)-1)
                center = index[random_idx]
                tmp_size = self.patch_size[0] * self.sigma_weight
                ul = [int(center[1]- tmp_size), int(center[0] - tmp_size)]
                br = [int(center[1] + tmp_size + 1), int(center[0] + tmp_size + 1)]
            else:
                break
        # set random angle
        if self.transform:
            angle = random.randint(-180, 180) # predict the angle_rotate?
        else:
            angle = 0
        # get the patch image from the original image with the center and angle
        patch_image, patch_mnt_map = self.get_patch_image(patch_image_source, patch_mnt_map, center, angle)
        # add the noise to the patch 
        patch_image = self.add_noise(patch_image)

        # save patch image as test
        # # changed back to 255
        # patch_image = patch_image * 255
        # patch_image = patch_image.astype(np.uint8)
        # cv2.imwrite('patch_image.png', patch_image)
        # import pdb;pdb.set_trace()
        
        # resize the patch image
        # patch_image = cv2.resize(patch_image, (self.patch_input_size[1], self.patch_input_size[0]))
        # get the heatmap with gaussian blur acording to the center
        heatmap = self.get_heatmap(center, image_shape)
        heatmap = cv2.resize(heatmap, (self.image_input_size[1], self.image_input_size[0]))
        # # # show the heatmap using plt
        # plt.imshow(heatmap, cmap='jet')
        # plt.savefig('heatmap2.png')
        # # draw the bbox with angle on orignal image according the center and patch_size
        # cv2.rectangle(image, (center[1]-self.patch_input_size[1]//2, center[0]-self.patch_input_size[0]//2), 
        #               (center[1]+self.patch_input_size[1]//2, center[0]+self.patch_input_size[0]//2), (0, 255, 0), 2)
        
        # # show the orignal image using plt
        # plt.imshow(image, cmap='gray')

        # plt.savefig('image2.png')

        # # show the patch image using plt
        # plt.imshow(patch_image, cmap='gray')
        # plt.savefig('patch_image2.png')

        # process into the torch
        image = torch.from_numpy(image).unsqueeze(0)
        patch_image = torch.from_numpy(patch_image).unsqueeze(0)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)
        angle = torch.from_numpy(np.array([angle])).float() / 180.0
        source_mnt_map = torch.from_numpy(source_mnt_map).unsqueeze(0)
        patch_mnt_map = torch.from_numpy(patch_mnt_map).unsqueeze(0)

        # Is it need to normalize? or standardize?
        # image = transforms.Normalize(self.image_mean, self.image_std)(image)
        # patch_image = transforms.Normalize(self.image_mean, self.image_std)(patch_image)
        image = self.normalize(image)
        patch_image = self.normalize(patch_image)

        return image, patch_image, heatmap, angle, source_mnt_map, patch_mnt_map

if __name__ == '__main__':
    Folder = '/disk1/panzhiyu/THUPALMLAB/'
    dataset = THUPALM(root=Folder, is_train=True, transform=True)
    image, patch_image, heatmap, angle = dataset[0]
    print(image.shape)
    print(patch_image.shape)
    print(heatmap.shape)
    