'''
Version description: 

# based on distill_v2s_lstm_t2s.py.

This is my final version (v0), may contain some problem, under development

With these features:

1. only one second-order temporal loss (fake_vd_feat and real vd_feat)
'''

#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, copy, pickle
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import datasets_multiple_wGT_full as datasets_multiple
import utils
from utils import *
import torch.nn.init as init
from torch.nn.init import *
import torchvision
from loss import *
from option import *
import torch.nn.functional as F
from IPython import embed
from cal_ssim import SSIM
import cv2

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def tensor2np(out):
    # out should be shape [bsz, channel, H, W], but only bsz==1 is supported
    out_np = (out[0].clamp(0.0,1.0).permute(1,2,0).cpu().numpy()*255.0).astype('uint8')[:,:,::-1]
    return out_np

def mask_generate(frame1, frame2, b, c, h, w, th):
    img_min = torch.min(frame1, dim=1)
    img_min = img_min[0]

    img_min2 = torch.min(frame2, dim=1)
    img_min2 = img_min2[0]

    frame_diff = img_min - img_min2
    frame_diff[frame_diff<0] = 0

    frame_mask = 1 - torch.exp(-frame_diff*frame_diff/th)
    frame_neg_mask = 1- frame_mask

    frame_neg_mask = frame_neg_mask.view(b, 1, h, w).detach()
    frame_mask = frame_mask.view(b, 1, h, w).detach()

    frame_neg_mask = torch.cat((frame_neg_mask, frame_neg_mask, frame_neg_mask), 1)
    frame_mask = torch.cat((frame_mask, frame_mask, frame_mask), 1)

    return frame_mask, frame_neg_mask

def prepare_frame(frame_i, flow_warping, FlowNet):
    frame_i0 = frame_i[0]
    frame_i1 = frame_i[1]
    frame_i2 = frame_i[2]
    frame_i3 = frame_i[3]

    frame_i3_target = gt_i[3]

    frame_i4 = frame_i[4]
    frame_i5 = frame_i[5]
    frame_i6 = frame_i[6]

    flow_i30 = FlowNet(frame_i3, frame_i0)
    flow_i31 = FlowNet(frame_i3, frame_i1)
    flow_i32 = FlowNet(frame_i3, frame_i2)


    flow_i34 = FlowNet(frame_i3, frame_i4)
    flow_i35 = FlowNet(frame_i3, frame_i5)
    flow_i36 = FlowNet(frame_i3, frame_i6)

    warp_i0 = flow_warping(frame_i0, flow_i30) 
    warp_i1 = flow_warping(frame_i1, flow_i31)
    warp_i2 = flow_warping(frame_i2, flow_i32)

    warp_i4 = flow_warping(frame_i4, flow_i34)
    warp_i5 = flow_warping(frame_i5, flow_i35)
    warp_i6 = flow_warping(frame_i6, flow_i36)

    warp_i0 = warp_i0.view(b, c, 1, h, w)
    warp_i1 = warp_i1.view(b, c, 1, h, w)
    warp_i2 = warp_i2.view(b, c, 1, h, w)
    warp_i3 = frame_i3.view(b, c, 1, h, w)
    warp_i4 = warp_i4.view(b, c, 1, h, w)
    warp_i5 = warp_i5.view(b, c, 1, h, w)
    warp_i6 = warp_i6.view(b, c, 1, h, w)

    return warp_i0, warp_i1, warp_i2, warp_i3, warp_i4, warp_i5, warp_i6

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fast Blind Video Temporal Consistency")
    parser.add_argument('-three_dim_model',   type=str,     default="TDModel",     help='Multi-frame models for hanlde videos')

    parser.add_argument('-nf',              type=int,     default=16,               help='#Channels in conv layer')
    parser.add_argument('-blocks',          type=int,     default=2,                help='#ResBlocks') 
    parser.add_argument('-norm',            type=str,     default='IN',             choices=["BN", "IN", "none"],   help='normalization layer')
    parser.add_argument('-model_name',      type=str,     default='none',           help='path to save model')

    parser.add_argument('-datasets_tasks',  type=str,     default='video_rain_removal',    help='dataset-task pairs list')
    parser.add_argument('-data_dir',        type=str,     default='data_heavy',     help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=32,               help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.4,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=7,                help='#frames for training')
        
    parser.add_argument('-alpha',           type=float,   default=50.0,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="L2",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-VGGLayers',       type=str,     default="1",              help="VGG layers for perceptual loss, combinations of 1, 2, 3, 4")

    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')

    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=20,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.01,             help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    
    parser.add_argument('-seed',            type=int,     default=9487,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=16,               help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')

    parser.add_argument('-list_filename',   type=str,      help='use cpu?')

    opts = parser.parse_args()

    # embed()
    # exit()

    S = SSIM().cuda()
    ssim_loss = lambda a, b: -S(a,b)

    

    # opts.checkpoint_dir = opt.checkpoint_dir
    opts.data_dir = opt.data_dir
    # opts.list_filename = opt.list_filename
    opts.model_name = opt.model_name
    opts.batch_size = 4#opt.batch_size
    #opts.crop_size = opt.crop_size
    opts.vgg_path = opt.vgg_path

    opts.lr_step = 15#opt.vgg_path#decay by hyz for faster speed


    opts.train_epoch_size = 200#opt.train_epoch_size # decrease by hyz to speed up the training time
    opts.valid_epoch_size = opt.valid_epoch_size
    opts.epoch_max = opt.epoch_max
    opts.threads = opt.threads
    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m   

    '''
    Modification for consequent prediction
    '''
    hidden_size = 16
    frame_per_video = 11
    opts.sample_frames = frame_per_video
    # embed()
    # exit()
    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix

    opts.size_multiplier = 2 ** 6
    print(opts)

    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)

    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    print('===> Initializing model from %s...' %opts.three_dim_model)
    three_dim_model = networks.__dict__["TDModel_3f"](opts, 3, 16)
    three_dim_model_vd = networks.__dict__[opts.three_dim_model](opts, 3, 64)
    lstm_net_t = networks.__dict__["ConvLSTM_sin"](2, hidden_size, 3) # temporal LSTM

    lstm_net_s_front = networks.__dict__["ConvLSTM"](1, hidden_size, 3)# scale LSTM forward
    lstm_net_s_back = networks.__dict__["ConvLSTM"](1, hidden_size, 3)# scale LSTM backward


    three_dim_model.apply(weight_init)

    ### Load pretrained FlowNet2
    opts.rgb_max = 1.0
    opts.fp16 = False

    FlowNet = networks.FlowNet2(opts, requires_grad=True)
    model_filename = os.path.join("./pretrained_models", "FlowNet2_checkpoint.pth.tar")
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    FlowNet.load_state_dict(checkpoint['state_dict'])

    if opts.solver == 'SGD':
        optimizer = optim.SGD(three_dim_model.parameters(), \
                              lr=opts.lr_init, momentum=opts.momentum, weight_decay= opts.weight_decay )
    elif opts.solver == 'ADAM':
        optimizer = optim.Adam([ \
                                {'params': three_dim_model.parameters(), 'lr': opts.lr_init }, \
                                {'params': lstm_net_t.parameters(), 'lr': opts.lr_init }, \
                                {'params': lstm_net_s_front.parameters(), 'lr': opts.lr_init }, \
                                {'params': lstm_net_s_back.parameters(), 'lr': opts.lr_init }, \
                               ], lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))


    else:
        raise Exception("Not supported solver (%s)" %opts.solver)

    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = 0

    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]

    print('=====================================================================')
    print('===> Resuming model' )
    print('=====================================================================')
    three_dim_model = utils.load_model(three_dim_model, "checkpoints/small2_full/derain_self/model_epoch_100.pth", opts, epoch_st)
    three_dim_model_vd = utils.load_model(three_dim_model_vd, "checkpoints/wGT_full_ft_x4/derain_self/model_epoch_100.pth", opts, epoch_st)

    

    num_params = utils.count_network_parameters(three_dim_model)

    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')

    loss_dir = os.path.join(opts.model_dir, 'loss')
    # loss_writer = SummaryWriter(loss_dir)

    device = torch.device("cuda" if opts.cuda else "cpu")

    print("Finish prepare vgg")

    three_dim_model_vd = three_dim_model_vd.cuda()
    three_dim_model = three_dim_model.cuda()

    FlowNet = FlowNet.cuda()
    print("Finish prepare flownet")


    # LSTM init
    lstm_net_t = lstm_net_t.cuda()
    lstm_net_s_front = lstm_net_s_front.cuda()
    lstm_net_s_back = lstm_net_s_back.cuda()


    lstm_net_t.train()
    lstm_net_s_front.train()
    lstm_net_s_back.train()
    

    three_dim_model_vd = three_dim_model_vd.eval()

    three_dim_model.train()
    FlowNet.train()

    train_dataset = datasets_multiple.MultiFramesDataset(opts, "train")

    print("Finish prepare dataset")

    loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

    loss_kd = nn.L1Loss()

    downler = nn.AvgPool2d(kernel_size=2, stride=2)
    upsampler = nn.Upsample(scale_factor=2)

    while three_dim_model.epoch < opts.epoch_max:
        three_dim_model.epoch += 1

        data_loader = utils.create_data_loader(train_dataset, opts, "train")
        current_lr = utils.learning_rate_decay(opts, three_dim_model.epoch)

        for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        
        # error_last = 1e8
        ts = datetime.now()

        for iteration, all_batch in enumerate(data_loader, 1):
            total_iter = (three_dim_model.epoch - 1) * opts.train_epoch_size + iteration

            batch = all_batch[0]
            gt_batch = all_batch[1]

            recon_loss_s = []
            temporal_distill_loss_s = []
            spatial_distill_loss_front_s = []
            spatial_distill_loss_back_s = []

            vd_lstm_stat_t = None
            id_lstm_stat_t = None

            [raw_b, raw_c, raw_h, raw_w] = batch[0].shape

            scale_lstm_stat_vd = (
                torch.zeros([raw_b, hidden_size, raw_h, raw_w]).cuda(),
                torch.zeros([raw_b, hidden_size, raw_h, raw_w]).cuda()
            )

            scale_lstm_stat_id = (
                torch.zeros([raw_b, hidden_size, raw_h, raw_w]).cuda(),
                torch.zeros([raw_b, hidden_size, raw_h, raw_w]).cuda()
            )
            
            frame_i = []
            gt_i = []
            for st in range(0, frame_per_video - 7):
                # update the input window
                if st == 0:
                    for t in range(7):
                        frame_i.append(batch[st+t].cuda())
                        gt_i.append(gt_batch[st+t].cuda())
                else:
                    frame_i = frame_i[1:]
                    gt_i = gt_i[1:]

                    frame_i.append(batch[st+7].cuda())
                    gt_i.append(gt_batch[st+7].cuda())

                data_time = datetime.now() - ts
                ts = datetime.now()

                [b, c, h, w] = frame_i[0].shape
                frame_i3_target = gt_i[3]
                frame_i3 = frame_i[3]
                flow_warping = Resample2d().cuda()
                warp_i0, warp_i1, warp_i2, warp_i3, warp_i4, warp_i5, warp_i6 = prepare_frame(frame_i, flow_warping, FlowNet)


                # -------------- video forward
                optimizer.zero_grad() # optimize-1: temporal distill
                with torch.no_grad():
                    frame_input = torch.cat((warp_i0.detach(), warp_i1.detach(), warp_i2.detach(), warp_i3.detach(), warp_i4.detach(), warp_i5.detach(), warp_i6.detach()), 2)
                    vd_pred, vd_feat = three_dim_model_vd(frame_input, collect_feat=True)

                    vd_pred = vd_pred.squeeze()
                    vd_feat = vd_feat.squeeze()
                    vd_feat = utils.get_Gmean(vd_feat)
                    vd_feat = vd_feat.unsqueeze(1)
                


                # -------------- student forward
                frame_input = torch.cat((warp_i2.detach(), warp_i3.detach(), warp_i4.detach()), 2)

                frame_pred, feat = three_dim_model(frame_input, collect_feat=True)

                frame_pred = frame_pred.squeeze()
                feat = feat.squeeze()
                id_feat = utils.get_Gmean(feat)
                id_feat = id_feat.unsqueeze(1)


                vd_lstm_input_t_input = vd_feat#torch.cat([vd_feat, scale_lstm_stat_vd[1]], 1)
                vd_lstm_input_t_input_fake = id_feat#torch.cat([id_feat, scale_lstm_stat_vd[1]], 1)

                vd_lstm_stat_t_fake = lstm_net_t(vd_lstm_input_t_input_fake, scale_lstm_stat_vd[1], vd_lstm_stat_t)
                vd_lstm_stat_t = lstm_net_t(vd_lstm_input_t_input, scale_lstm_stat_vd[1], vd_lstm_stat_t)


                id_lstm_input_t_input = id_feat#torch.cat([id_feat, scale_lstm_stat_id[1]], 1)
                # id_lstm_input_t_input_fake = vd_feat#torch.cat([vd_feat, scale_lstm_stat_id[1]], 1)

                # id_lstm_stat_t_fake = lstm_net_t(id_lstm_input_t_input_fake, scale_lstm_stat_id[1], id_lstm_stat_t)
                id_lstm_stat_t = lstm_net_t(id_lstm_input_t_input, scale_lstm_stat_id[1], id_lstm_stat_t)



                distill_loss_t = loss_kd(id_lstm_stat_t[1], vd_lstm_stat_t[1].detach()) + loss_kd(vd_lstm_stat_t_fake[1], vd_lstm_stat_t[1].detach()) # loss_kd(id_lstm_stat_t_fake[1], id_lstm_stat_t[1].detach()) +
                recon_loss = ssim_loss(frame_pred, frame_i3_target)

                # --------------- begin front scale-level distillation -------------------
                id_feat_list = [id_feat]
                vd_feat_list = [vd_feat]

                # the first scale distill
                vd_lstm_stat_s = vd_lstm_stat_t
                id_lstm_stat_s = id_lstm_stat_t

                vd_lstm_stat_s = lstm_net_s_front(vd_feat, vd_lstm_stat_s)
                id_lstm_stat_s = lstm_net_s_front(id_feat, id_lstm_stat_s)

                distill_loss_s = loss_kd(id_lstm_stat_s[1], vd_lstm_stat_s[1].detach())

                total_loss = recon_loss + distill_loss_t*0.1 + distill_loss_s*0.1
                total_loss.backward(retain_graph=True)
                optimizer.step() # finish optimized-1

                # write down data
                recon_loss_s.append(recon_loss.item())
                temporal_distill_loss_s.append(distill_loss_t.item())
                spatial_distill_loss_front_s.append(distill_loss_s.item())

                # begin downsampling
                vd_lstm_stat_s = [downler(vd_lstm_stat_s[0]), downler(vd_lstm_stat_s[1])]
                id_lstm_stat_s = [downler(id_lstm_stat_s[0]), downler(id_lstm_stat_s[1])]
                # finish the first scale distill

                # begin next two scales' distill
                warp_i0_ori = warp_i0 
                warp_i1_ori = warp_i1
                warp_i2_ori = warp_i2

                warp_i4_ori = warp_i4
                warp_i5_ori = warp_i5
                warp_i6_ori = warp_i6
                
                for scale_idx in range(1, 3):
                    optimizer.zero_grad() # optimize-2: forward scale distill

                    warp_i0 = downler(warp_i0_ori.squeeze())
                    warp_i1 = downler(warp_i1_ori.squeeze())
                    warp_i2 = downler(warp_i2_ori.squeeze())
                    
                    warp_i4 = downler(warp_i4_ori.squeeze())
                    warp_i5 = downler(warp_i5_ori.squeeze())
                    warp_i6 = downler(warp_i6_ori.squeeze())

                    frame_i3 = downler(frame_i3)
                    frame_i3_target = downler(frame_i3_target)

                    [b, c, h, w] = warp_i0.shape

                    # backup
                    warp_i0_ori = warp_i0 
                    warp_i1_ori = warp_i1
                    warp_i2_ori = warp_i2

                    warp_i4_ori = warp_i4
                    warp_i5_ori = warp_i5
                    warp_i6_ori = warp_i6

                    # reshape
                    warp_i0 = warp_i0.view(b, c, 1, h, w)
                    warp_i1 = warp_i1.view(b, c, 1, h, w)
                    warp_i2 = warp_i2.view(b, c, 1, h, w)
                    warp_i3 = frame_i3.view(b, c, 1, h, w)
                    warp_i4 = warp_i4.view(b, c, 1, h, w)
                    warp_i5 = warp_i5.view(b, c, 1, h, w)
                    warp_i6 = warp_i6.view(b, c, 1, h, w)


                    # -------------- video forward
                    with torch.no_grad():
                        frame_input = torch.cat((warp_i0.detach(), warp_i1.detach(), warp_i2.detach(), warp_i3.detach(), warp_i4.detach(), warp_i5.detach(), warp_i6.detach()), 2)
                        vd_pred, vd_feat = three_dim_model_vd(frame_input, collect_feat=True)

                        vd_pred = vd_pred.squeeze()
                        vd_feat = vd_feat.squeeze()
                        vd_feat = utils.get_Gmean(vd_feat)
                        vd_feat = vd_feat.unsqueeze(1)
                        vd_feat_list.append(vd_feat)

                    vd_lstm_stat_s = lstm_net_s_front(vd_feat, vd_lstm_stat_s)

                    # -------------- image forward
                    frame_input = torch.cat((warp_i2.detach(), warp_i3.detach(), warp_i4.detach()), 2)
                    frame_pred, feat = three_dim_model(frame_input, collect_feat=True)

                    frame_pred = frame_pred.squeeze()
                    feat = feat.squeeze()
                    id_feat = utils.get_Gmean(feat)
                    id_feat = id_feat.unsqueeze(1)
                    id_feat_list.append(id_feat)

                    id_lstm_stat_s = lstm_net_s_front(id_feat, id_lstm_stat_s)

                    distill_loss_s = loss_kd(id_lstm_stat_s[1], vd_lstm_stat_s[1].detach())
                    recon_loss = ssim_loss(frame_pred, frame_i3_target)

                    total_loss = recon_loss*0.5 + distill_loss_s*0.1
                    total_loss.backward(retain_graph=True)
                    optimizer.step() # finish optimized-2

                    # write down data
                    recon_loss_s.append(recon_loss.item())
                    spatial_distill_loss_front_s.append(distill_loss_s.item())
                    
                    # begin downsampling
                    if scale_idx < 2:
                        vd_lstm_stat_s = [downler(vd_lstm_stat_s[0]), downler(vd_lstm_stat_s[1])]
                        id_lstm_stat_s = [downler(id_lstm_stat_s[0]), downler(id_lstm_stat_s[1])]


                # begin backward scale-level distillation
                back_vd_lstm_stat_s = vd_lstm_stat_s
                back_id_lstm_stat_s = id_lstm_stat_s

                for back_scale_idx in range(2, -1, -1):
                    optimizer.zero_grad() # optimize-3: forward scale distill
                    vd_feat = vd_feat_list[back_scale_idx]
                    id_feat = id_feat_list[back_scale_idx]


                    back_vd_lstm_stat_s = lstm_net_s_back(vd_feat, back_vd_lstm_stat_s)
                    back_id_lstm_stat_s = lstm_net_s_back(id_feat, back_id_lstm_stat_s)

                    if back_scale_idx > 0:
                        back_vd_lstm_stat_s = [upsampler(back_vd_lstm_stat_s[0]), upsampler(back_vd_lstm_stat_s[1])]
                        back_id_lstm_stat_s = [upsampler(back_id_lstm_stat_s[0]), upsampler(back_id_lstm_stat_s[1])]

                    distill_loss_s_back = loss_kd(back_id_lstm_stat_s[1], back_vd_lstm_stat_s[1].detach())
                    total_loss = distill_loss_s_back*0.1

                    total_loss.backward(retain_graph=True)
                    optimizer.step()

                    spatial_distill_loss_back_s.append(distill_loss_s_back.item())

                # finish backward scale-level distillation

                # inherit scale state to the spatial-level
                scale_lstm_stat_vd = back_vd_lstm_stat_s
                scale_lstm_stat_id = back_id_lstm_stat_s



                # error_last_inner_epoch = overall_loss.item()
                network_time = datetime.now() - ts

            info = "[GPU %d]: " %(opts.gpu)
            info += "Epoch %d; Batch %d / %d; " %(three_dim_model.epoch, iteration, len(data_loader))

            batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
            info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
            info += "\tmodel = %s\n" %opts.model_name

            info += "\t\t%25s = %f\n" %("Recond Loss", np.mean(recon_loss_s))
            info += "\t\t%25s = %f\n" %("Temporal Distill Loss", np.mean(temporal_distill_loss_s))
            info += "\t\t%25s = %f\n" %("Scale Front Distill Loss", np.mean(spatial_distill_loss_front_s))
            info += "\t\t%25s = %f\n" %("Scale Back Distill Loss", np.mean(spatial_distill_loss_back_s))

            recon_loss_s = []
            temporal_distill_loss_s = []
            spatial_distill_loss_front_s = []
            spatial_distill_loss_back_s = []

            print(info)
            # error_last = error_last_inner_epoch

        utils.save_model6(three_dim_model, lstm_net_t, lstm_net_s_front, lstm_net_s_back, optimizer, opts)     

