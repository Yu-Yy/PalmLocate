import torch
import numpy as np
import os 
from config import config
from config import update_config
import argparse
from models.seg_hrnet import get_hr_model
from models.LocateNet import LocateNet
import dataset.THUPALM as THUPALM
import random
from torch.utils.data.dataloader import default_collate
import torch.backends.cudnn as cudnn
from utils import utils
import shutil
import torch.nn.functional as F
import matplotlib.pyplot as plt

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument("--local_rank", type=int, default=-1)       
    # parser.add_argument('opts',
    #                     help="Modify config options using the command-line",
    #                     default=None,
    #                     nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def my_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)

def main():
    args = parse_args()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    create_exp_dir(config.OUTPUT_DIR)
    logging = utils.Logger(0, config.OUTPUT_DIR)
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    mkdir(os.path.join(config.OUTPUT_DIR, 'train_debug'))
    mkdir(os.path.join(config.OUTPUT_DIR, 'valid_debug'))
    
    gpus = [int(i) for i in config.GPUS.split(',')]
    # set the dataset
    logging.info('Loading the dataset')
    train_dataset = THUPALM.THUPALM(cfg = config, root=config.DATASET.ROOT, is_train=True, transform=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU* len(gpus),collate_fn=my_collate_fn,
                                              shuffle=True, pin_memory=True, num_workers = config.NUM_WORKERS, drop_last=False) #config.NUM_WORKERS
    
    valid_dataset = THUPALM.THUPALM(cfg = config, root=config.DATASET.ROOT, is_train=False, transform=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=2* len(gpus),collate_fn=my_collate_fn,
                                                shuffle=False, pin_memory=True, num_workers = config.NUM_WORKERS, drop_last=False) #
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info('Creating the model')
    # model = get_hr_model(config)
    model = LocateNet(config)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.95))

    checkpoint_file = os.path.join(config.OUTPUT_DIR, 'checkpoint.pt')
    best_file = os.path.join(config.OUTPUT_DIR, 'best_model.pt')
    min_loss = torch.tensor(np.inf).to(device)
    if config.TRAIN.RESUME:
        if os.path.isfile(checkpoint_file):
            logging.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            init_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            global_step = checkpoint['global_step']
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_file, checkpoint['epoch']))
            min_loss = valid(config, validloader, model, logging, -1)
        else:
            logging.info("=> no checkpoint found at '{}'".format(checkpoint_file))
            global_step = 0
            init_epoch = 0
    else:
        global_step = 0
        init_epoch = 0

    for ep in range(init_epoch,config.TRAIN.EPOCHS):
        logging.info(f'Epoch {ep}:')
        global_step = train(config, trainloader, model, optimizer, logging, global_step)
        torch.save({
            'epoch': ep+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step,
        }, checkpoint_file)

        if (ep+1) % 10 == 0:
            hm_loss = valid(config, validloader, model, logging, ep)
            if hm_loss < min_loss:
                min_loss = hm_loss
                torch.save({
                    'epoch': ep+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'global_step': global_step,
                    'min_loss': min_loss,
                }, best_file)
                logging.info(f'Best model saved at {best_file}, with loss {min_loss}')
        


def train(config, train_loader, model, optimizer, logging, global_step):
    loss_heatmap_collect = utils.AvgrageMeter()
    loss_mnt_source_collect = utils.AvgrageMeter()
    loss_mnt_patch_collect = utils.AvgrageMeter()
    loss_collect = utils.AvgrageMeter()
    model.train()
    for step, (img, patch_image, heatmap, angle, source_mnt, patch_mnt) in enumerate(train_loader):
        img = img.cuda()
        patch_image = patch_image.cuda()
        heatmap = heatmap.cuda()
        angle = angle.cuda()
        source_mnt = source_mnt.cuda()
        patch_mnt = patch_mnt.cuda()
        batch_size = img.size(0)
        heatmap_pred, mnt_source_pred, mnt_patch_pred = model(img, patch_image) #, anglemap_pred
        heatmap_downsample = F.interpolate(heatmap, size=heatmap_pred.size()[2:], mode='bilinear', align_corners=True)
        source_mnt_pred = F.interpolate(mnt_source_pred, size=source_mnt.size()[2:], mode='bilinear', align_corners=True)
        patch_mnt_pred = F.interpolate(mnt_patch_pred, size=patch_mnt.size()[2:], mode='bilinear', align_corners=True)

        loss_mnt_source = - torch.mean(torch.sum(100 * source_mnt * torch.log(source_mnt_pred + 1e-8) + 1 * (1-source_mnt) * torch.log(1-source_mnt_pred + 1e-8), dim=[-1,-2]) / (source_mnt.size()[-1] * source_mnt.size()[-2]))
        loss_mnt_patch = - torch.mean(torch.sum(100 * patch_mnt * torch.log(patch_mnt_pred+1e-8) + 1 * (1-patch_mnt) * torch.log(1-patch_mnt_pred + 1e-8), dim=[-1,-2]) / (patch_mnt.size()[-1] * patch_mnt.size()[-2]))
        loss_mnt_source_collect.update(loss_mnt_source.item())
        loss_mnt_patch_collect.update(loss_mnt_patch.item())

        # do the L2 loss between the heatmap and the pred one
        loss_heatmap = torch.mean(torch.pow(heatmap_pred - heatmap_downsample, 2))
        loss_heatmap_collect.update(loss_heatmap.item())
        # get the max value index of pred heatmap and do the L1 loss between the anglemap and the pred one
        # heatmap is B,1,H,W
        # heatmap_pred_flat = heatmap_pred.reshape(batch_size, -1)
        # heatmap_pred_index = heatmap_pred_flat.argmax(dim=1)
        # anglemap_pred = anglemap_pred.reshape(batch_size, -1)
        # anglemap_pred = anglemap_pred[torch.arange(batch_size), heatmap_pred_index]
        # loss_angle = torch.mean(torch.abs(anglemap_pred - angle))
        # loss_angle_collect.update(loss_angle.item())

        loss = 100 * loss_heatmap + loss_mnt_patch + loss_mnt_source #+ 0.1 * loss_angle
        loss_collect.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (global_step + 1) % config.PRINT_FREQ == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Loss_heatmap {loss_heatmap.val:.4f} ({loss_heatmap.avg:.4f})\t'
                         'Loss_mnt_org {loss_mnt_org.val:.4f} ({loss_mnt_org.avg:.4f})\t'
                        'Loss_mnt_patch {loss_mnt_patch.val:.4f} ({loss_mnt_patch.avg:.4f})\t'.format(
                          config.TRAIN.EPOCHS, step, len(train_loader), loss=loss_collect, loss_heatmap=loss_heatmap_collect, \
                            loss_mnt_org=loss_mnt_source_collect, loss_mnt_patch=loss_mnt_patch_collect))

        if (global_step + 1) % config.DEBUG_FREQ == 0:
            # save the heatmap
            heatmap_pred = heatmap_pred[0,0].detach().cpu().numpy()
            # plot the heatmap
            plt.imshow(heatmap_pred, cmap='jet')
            plt.savefig(os.path.join(config.OUTPUT_DIR, 'train_debug', 'heatmap_pred_{}.png'.format(global_step)))
            # save the gtheatmap
            heatmap = heatmap[0,0].detach().cpu().numpy()
            # plot the gtheatmap
            plt.imshow(heatmap, cmap='jet')
            plt.savefig(os.path.join(config.OUTPUT_DIR, 'train_debug', 'heatmap_{}_gt.png'.format(global_step)))

            
        global_step += 1
        del img, patch_image, heatmap, angle
    return global_step

def valid(config, valid_loader, model, logging, ep):
    loss_heatmap_collect = utils.AvgrageMeter()
    loss_angle_collect = utils.AvgrageMeter()
    loss_coord_collect = utils.AvgrageMeter()
    loss_collect = utils.AvgrageMeter()
    model.eval()
    for step, (img, patch_image, heatmap, angle, source_mnt, patch_mnt) in enumerate(valid_loader):
        img = img.cuda()
        patch_image = patch_image.cuda()
        heatmap = heatmap.cuda()
        angle = angle.cuda()
        batch_size = img.size(0)
        heatmap_pred,_,_ = model(img, patch_image) #, anglemap_pred
        # heatmap_pred_up = F.interpolate(heatmap_pred, size=heatmap.size()[2:], mode='bilinear', align_corners=True)
        heatmap_flat = heatmap.reshape(batch_size, -1)
        heatmap_flat_index = heatmap_flat.argmax(dim=1)
        indices_x = (heatmap_flat_index // heatmap.size(3)).reshape(batch_size, 1)
        indices_y = (heatmap_flat_index % heatmap.size(3)).reshape(batch_size, 1)
        indices_gt = torch.cat((indices_x, indices_y), dim=1)
        indices_gt_uni = indices_gt / heatmap.size(3) # square ok
        
        heatmap_downsample = F.interpolate(heatmap, size=heatmap_pred.size()[2:], mode='bilinear', align_corners=True)
        
        # do the L2 loss between the heatmap and the pred one
        loss_heatmap = torch.mean(torch.pow(heatmap_pred - heatmap_downsample, 2))
        loss_heatmap_collect.update(loss_heatmap.item())
        # get the max value index of pred heatmap and do the L1 loss between the anglemap and the pred one
        # heatmap is B,1,H,W
        heatmap_pred_flat = heatmap_pred.reshape(batch_size, -1)
        heatmap_pred_index = heatmap_pred_flat.argmax(dim=1)
        indices_x = (heatmap_pred_index // heatmap_pred.size(3)).reshape(batch_size, 1)
        indices_y = (heatmap_pred_index % heatmap_pred.size(3)).reshape(batch_size, 1)
        indices_pred = torch.cat((indices_x, indices_y), dim=1)
        indices_pred_uni = indices_pred / heatmap_pred.size(3)
        # get the unified coordinate loss
        loss_uni = torch.mean(torch.pow(indices_pred_uni - indices_gt_uni, 2))
        loss_coord_collect.update(loss_uni.item())

        # anglemap_pred = anglemap_pred.reshape(batch_size, -1)
        # anglemap_pred = anglemap_pred[torch.arange(batch_size), heatmap_pred_index]
        # loss_angle = torch.mean(torch.abs(anglemap_pred - angle))
        # loss_angle_collect.update(loss_angle.item())

        loss = 100 * loss_heatmap #+ 0.1 * loss_angle
        loss_collect.update(loss.item())

        if (step + 1) % 20 == 0:
            # logging show the coordinate loss
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Loss_heatmap {loss_heatmap.val:.4f} ({loss_heatmap.avg:.4f})\t'
                            'Loss_angle {loss_angle.val:.4f} ({loss_angle.avg:.4f})\t'
                            'Loss_coord {loss_coord.val:.4f} ({loss_coord.avg:.4f})\t'.format(
                            config.TRAIN.EPOCHS, step, len(valid_loader), loss=loss_collect, loss_heatmap=loss_heatmap_collect, loss_angle=loss_angle_collect, loss_coord=loss_coord_collect))
            # save the heatmap
            ind = random.randint(0,batch_size-1)
            heatmap_pred = heatmap_pred[ind,0].detach().cpu().numpy()
            # plot the heatmap
            plt.imshow(heatmap_pred, cmap='jet')
            plt.savefig(os.path.join(config.OUTPUT_DIR, 'valid_debug', f'heatmap_pred_{step}_{ep}.png'))
            heatmap = heatmap[ind,0].detach().cpu().numpy()
            plt.imshow(heatmap, cmap='jet')
            plt.savefig(os.path.join(config.OUTPUT_DIR, 'valid_debug', f'heatmap_{step}_{ep}_gt.png'))
        del img, patch_image, heatmap, angle
    
    return loss_coord_collect.avg


if __name__ == '__main__':
    main()