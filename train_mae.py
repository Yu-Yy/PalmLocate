import torch
import numpy as np
import torch.distributed as dist
import pickle
import os
import models.mae as mae
import random
import dataset.THUPALM as THUPALM
from torch.utils.data.dataloader import default_collate
import torch.backends.cudnn as cudnn
from utils import utils
import shutil
import cv2
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

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

def my_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def main():
    Folder = '/disk1/panzhiyu/THUPALMLAB/'
    train_resume = True
    learning_rate = 0.00001
    output_folder = 'output'
    create_exp_dir(output_folder)
    GPUS = '0'
    seed = 1
    batch_size = 8
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging = utils.Logger(0, output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    mkdir(os.path.join(output_folder, 'debug'))
    gpus = [int(i) for i in GPUS.split(',')]
    # set the dataset
    logging.info('Loading the dataset')
    train_dataset = THUPALM.THUPALM(root=Folder, is_train=True, transform=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size* len(gpus),collate_fn=my_collate_fn,
                                              shuffle=True, pin_memory=True, num_workers = 8, drop_last=False) #
    
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    logging.info('Creating the model')
    model = mae.__dict__['mae_vit_palm_model'](norm_pix_loss=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    # optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    checkpoint_file = os.path.join(output_folder, 'checkpoint.pt')
    if train_resume:
        if os.path.exists(checkpoint_file):
            logging.info('Loading checkpoint from: %s', checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            model.module.load_state_dict(checkpoint)
    global_step = 0
    for epoch in range(1000):
        logging.info('Epoch: %d', epoch)
        global_step = train(trainloader, model, optimizer, logging, global_step, output_folder)
        if epoch % 10 == 0:
            torch.save(model.module.state_dict(), checkpoint_file)
            logging.info('Saving checkpoint to: %s', checkpoint_file)

def train(train_loader, model, optimizer, logging, global_step, output_folder):
    model.train()
    for step, data in enumerate(train_loader):
        if len(data) == 0:
            continue
        images, _,_,_ = data
        images = images.cuda()
        batch_size = images.size(0)
        loss_rec, pred, mask = model(images, mask_ratio=0)
        loss_rec = loss_rec.mean()
        optimizer.zero_grad()
        loss_rec.backward()
        optimizer.step()
        if global_step % 100 == 0:
            logging.info('step: %d, loss_rec: %f', step, loss_rec.item())
        global_step += 1
        # do the debug
        if global_step % 100 == 0:
            # save the pred reconstuction image
            pred = pred.detach().cpu().numpy()
            # save the pred reconstuction image
            pred = (pred * 255).astype(np.uint8)
            pred = pred * mask.detach().cpu().numpy() + (1 - mask.detach().cpu().numpy()) * (images.detach().cpu().numpy() * 255).astype(np.uint8)
            batch_size_show = batch_size if batch_size < 4 else 4
            for b in range(batch_size_show):
                cv2.imwrite(os.path.join(output_folder,'debug',f'pred_{global_step}_{b}.png'), pred[b, 0, :, :])
            # save the input image   
        del images
        del pred
        del mask
    return global_step



if __name__ == '__main__':
    main()