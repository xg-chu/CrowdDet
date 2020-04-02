import os
import argparse
from setproctitle import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import network
import misc_utils
from config import config
from dataset import train_dataset

def process(data_generater, num_gpus):
    images = []
    gt_boxes = []
    im_info = []
    for _ in range(num_gpus):
        data = data_generater.__next__()
        minibt_img, minibt_gt, minibt_info = \
                data['data'], data['boxes'], data['im_info']
        images.append(minibt_img)
        gt_boxes.append(minibt_gt)
        im_info.append(minibt_info)
    images = torch.Tensor(np.vstack(images)).cuda()
    gt_boxes = torch.Tensor(np.vstack(gt_boxes)).cuda()
    im_info = torch.Tensor(np.vstack(im_info)).cuda()
    return images, gt_boxes, im_info

def adjust_learning_rate(optimizer,epoch,lr):
    learning_rate = lr
    lr = learning_rate*(0.1**(epoch//10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(params):
    total_nr_iters = config.train_base_iters
    batch_per_gpu = config.train_batch_per_gpu
    base_lr = config.base_lr
    line = 'network.base_lr.{}.train_iter.{}'.format(base_lr, total_nr_iters)
    print(line)
    # set model save path and log save path
    saveDir = config.model_dir
    misc_utils.ensure_dir(saveDir)
    fpath = os.path.join(config.output_dir, line+'.log')
    fid_log = open(fpath,'a')
    # set data input pipe
    program_name = config.program_name
    # check gpus
    torch.set_default_tensor_type('torch.FloatTensor')
    if not torch.cuda.is_available():
        print('No GPU exists!')
        return
    else:
        num_gpus = torch.cuda.device_count()
        train_iter = total_nr_iters//(num_gpus*batch_per_gpu)
        train_lr_decay = np.array(config.lr_decay)//(num_gpus*batch_per_gpu)
        train_dump_interval = config.model_dump_interval//(num_gpus*batch_per_gpu)
    train_lr = base_lr * num_gpus
    bt_size = num_gpus * batch_per_gpu
    line = 'Num of GPUs:{}, learning rate:{:.5f}, batch size:{},\
            train_iter:{}, decay_iter:{}, dump_interval:{}'.format(
            num_gpus,train_lr,bt_size,train_iter,train_lr_decay, train_dump_interval)
    print(line)
    print("Building netowrk.")
    net = network.Network()
    net.cuda()
    if params.resume_weights:
        model_file = os.path.join(saveDir, 'dump-{}.pth'.format(params.resume_weights))
        check_point = torch.load(model_file)
        net.load_state_dict(check_point['state_dict'])
    net = nn.DataParallel(net)
    # set the optimizer, use momentum and weight_decay
    optimizer = optim.SGD(net.parameters(), lr=train_lr, momentum=config.momentum, \
        weight_decay=config.weight_decay)
    # check if resume training
    training_data = train_dataset()
    net.train()
    if(params.progressbar):
        tqdm.monitor_interval = 0
        pbar = tqdm(total=train_iter, leave=False, ascii=True)
    dump_num = 1
    start_iter = 0
    if params.resume_weights:
        start_iter = int(params.resume_weights) * train_dump_interval
        if(start_iter >= train_lr_decay[0]):
            optimizer.param_groups[0]['lr'] = train_lr / 10
        if(start_iter >= train_lr_decay[1]):
            optimizer.param_groups[0]['lr'] = train_lr / 100
        dump_num = int(params.resume_weights) + 1
    for step in range(start_iter, train_iter):
        # warm up
        if step < config.warm_iter:
            alpha = step / config.warm_iter
            lr_new = 0.1 * train_lr + 0.9 * alpha * train_lr
            optimizer.param_groups[0]['lr'] = lr_new
        elif step == config.warm_iter:
            optimizer.param_groups[0]['lr'] = train_lr
        if step == train_lr_decay[0]:
            optimizer.param_groups[0]['lr'] = train_lr / 10
        elif step == train_lr_decay[1]:
            optimizer.param_groups[0]['lr'] = train_lr / 100
        # get training data
        images, gt_boxes, img_info = process(training_data, num_gpus)
        optimizer.zero_grad()
        # forwad
        outputs = net(images, img_info, gt_boxes)
        # collect the loss
        total_loss = sum([outputs[key].mean() for key in outputs.keys()])
        total_loss.backward()
        optimizer.step()
        if(params.progressbar):
            pbar.update(1)
        # stastic
        if step % config.log_dump_interval == 0:
            stastic_total_loss = total_loss.cpu().data.numpy()
            line = 'Iter {}: lr:{:.5f}, loss is {:.4f}.'.format(
                step, optimizer.param_groups[0]['lr'], stastic_total_loss)
            print(outputs)
            print(line)
            fid_log.write(line+'\n')
            fid_log.flush()
        # save the model
        if (step + 1)%train_dump_interval==0:
            fpath = os.path.join(saveDir,'dump-{}.pth'.format(dump_num))
            dump_num += 1
            model = dict(epoch = step,
                state_dict = net.module.state_dict(),
                optimizer = optimizer.state_dict())
            torch.save(model,fpath)
    if(params.progressbar):
        pbar.close()
    fid_log.close()

def run_train():
    setproctitle('train ' + os.path.split(os.path.realpath(__file__))[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_weights', '-r', default=None,type=str)
    parser.add_argument('--progressbar', '-p', action='store_true', default=False)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    run_train()
