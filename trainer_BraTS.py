import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic

from eval_BraTS import test_per_epoch, vis_per_epoch, calc_loss

def trainer_BraTS(args, model, snapshot_path, multimask_output, low_res):
    from dataset_BraTS import BraTS_dataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = BraTS_dataset(base_dir=args.root_path)
    db_test = BraTS_dataset(base_dir='/content/samed_codes/Slices/Test')
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=5, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=10, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    best_epoch, best_loss = 0.0, np.inf
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, (image_batch, label_batch) in enumerate(trainloader):
            # Original shape of image and label batch

            image_batch, label_batch = image_batch.unsqueeze(1).float().cuda(), label_batch.unsqueeze(1).cuda()
            image_batch = image_batch.repeat(1, 3, 1, 1)
            # Resize the target
            label_batch = F.interpolate(label_batch, size=(128, 128), mode='nearest') 
            label_batch = label_batch.squeeze(1)

            label_batch = torch.clamp(label_batch, 0, num_classes-1)

            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            # Check the shape and content of the model output

            loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, args.dice_param, num_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

        # Testing at the end of each epoch
        test_loss, test_loss_ce, test_loss_dice = test_per_epoch(model, testloader, multimask_output, args.img_size) 

        # Update best model if current epoch's loss is lower
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch_num
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'model_best_epoch_{:03d}.pth'.format(best_epoch)))
            logging.info("New best model saved with loss {:.4f}".format(best_loss))

        # Log at the end of each epoch
        logging.info(f'--- Epoch {epoch_num}/{args.max_epochs}: Training loss = {loss:.4f}, Testing loss = {test_loss:.4f}, Best loss = {best_loss:.4f}, Best epoch = {best_epoch}')

        if (epoch_num + 1) % 2 == 0 or epoch_num >= args.max_epochs - 1 or epoch_num >= args.stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_{:03d}.pth'.format(epoch_num))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Model saved to {}".format(save_mode_path))
            if epoch_num >= args.max_epochs - 1 or epoch_num >= args.stop_epoch - 1:
                iterator.close()
                break

    writer.close()
    return "Training Finished!"
