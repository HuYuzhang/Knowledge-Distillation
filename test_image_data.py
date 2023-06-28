#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from skimage.measure import compare_psnr, compare_ssim
from tqdm import tqdm

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import utils
# import matplotlib.pyplot as plt
from IPython import embed

rain_root = "data_NTU/test/b1_Rain"
gt_root = "data_NTU/GT/b1_GT"

def align_to_64(frame_i0, divide):
    [b, c, h, w] = frame_i0.shape

    h_pad = int(np.floor(h/divide)+1)*divide
    w_pad = int(np.floor(w/divide)+1)*divide

    frame_i0_pad = F.pad(frame_i0, pad = [0, w_pad-w, 0, h_pad-h], mode='replicate')

    return frame_i0_pad, h_pad-h, w_pad-w

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    parser.add_argument('-method', type=str, required=True, help='test model name')
    parser.add_argument('-model_name', type=str, required=True, help='test model name')
    parser.add_argument('-epoch', type=int, required=True, help='epoch')
    parser.add_argument('-streak_tag',          type=str,     default=1,               help='Whether the model handle rain streak')
    parser.add_argument('-haze_tag',            type=str,     default=1,               help='Whether the model handle haze')
    parser.add_argument('-flow_tag',            type=str,     default=1,               help='Whether the model handle haze')

    parser.add_argument('-dataset', type=str, required=True, help='dataset to test')
    parser.add_argument('-phase', type=str, default="test", choices=["train", "test"])
    parser.add_argument('-data_dir', type=str, default='data', help='path to data folder')
    parser.add_argument('-list_dir', type=str, default='lists', help='path to list folder')

    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='path to checkpoint folder')
    parser.add_argument('-task', type=str, required=True, help='evaluated task')
    parser.add_argument('-list_filename', type=str, required=True, help='evaluated task')
    parser.add_argument('-redo', action="store_true", help='Re-generate results')
    parser.add_argument('-gpu', type=int, default=1, help='gpu device id')
    parser.add_argument('-vd', type=str, default="vis", help='path to store image')


    opts = parser.parse_args()
    opts.cuda = True

    opts.size_multiplier = 2 ** 3  ## Inputs to TransformNet need to be divided by 4

    print(opts)

    vis_dir = opts.vd

    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")

    opts_filename = os.path.join(opts.checkpoint_dir, opts.model_name, "opts.pth")
    print("Load %s" % opts_filename)
    with open(opts_filename, 'rb') as f:
        model_opts = pickle.load(f)

    model_filename = os.path.join(opts.checkpoint_dir, opts.model_name, "model_epoch_%d.pth" % opts.epoch)
    print("Load %s" % model_filename)
    state_dict = torch.load(model_filename)
    # embed()
    # exit()

    opts.rgb_max = 1.0
    opts.fp16 = False

    three_dim_model = networks.__dict__['TDModel'](opts, 3, 64)
    # fusion_model = networks.__dict__['TDModel'](opts, 3, 64)
    FlowNet = networks.FlowNet2(opts, requires_grad=False)

    three_dim_model.load_state_dict(state_dict['three_dim_model'])
    # fusion_model.load_state_dict(state_dict['fusion_model'])
    # FlowNet.load_state_dict(state_dict['flow_model'])

    device = torch.device("cuda" if opts.cuda else "cpu")
    three_dim_model = three_dim_model.cuda()
    # fusion_model = fusion_model.cuda()
    FlowNet = FlowNet.cuda()
    flow_warping = Resample2d().cuda()

    three_dim_model.eval()
    # fusion_model.eval()


    list_filename = opts.list_filename

    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    times = []

    for v in range(len(video_list)):
        video = video_list[v]

        print("Test %s on %s video %d/%d: %s" % (opts.dataset, opts.phase, v + 1, len(video_list), video))

        input_dir = rain_root #os.path.join(opts.data_dir,  opts.phase, video)
        output_dir = os.path.join(opts.data_dir, opts.phase, "output", opts.model_name, opts.method, "epoch_%d" % opts.epoch, opts.task,
                                  opts.dataset, video)

        # print(input_dir)
        # print(output_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        frame_list = glob.glob(os.path.join(input_dir, "*.jpg"))
        output_list = glob.glob(os.path.join(output_dir, "*.jpg"))
        # embed()
        # exit()
        if len(frame_list) == len(output_list) and not opts.redo:
            print("Output frames exist, skip...")
            continue

        psnr_s = []
        ssim_s = []

        for t in tqdm(range(1+3, len(frame_list)-3)):
            frame_i0 = utils.read_img(os.path.join(input_dir, "%05d.jpg" % (t-3)))#[:360,:640]
            frame_i1 = utils.read_img(os.path.join(input_dir, "%05d.jpg" % (t-2)))#[:360,:640]
            frame_i2 = utils.read_img(os.path.join(input_dir, "%05d.jpg" % (t-1)))#[:360,:640]
            frame_i3 = utils.read_img(os.path.join(input_dir, "%05d.jpg" % (t)))#[:360,:640]

            frame_i3_gt = utils.read_img(os.path.join(gt_root, "%05d.jpg" % (t)))

            frame_i4 = utils.read_img(os.path.join(input_dir, "%05d.jpg" % (t+1)))#[:360,:640]
            frame_i5 = utils.read_img(os.path.join(input_dir, "%05d.jpg" % (t+2)))#[:360,:640]
            frame_i6 = utils.read_img(os.path.join(input_dir, "%05d.jpg" % (t+3)))#[:360,:640]
            # embed()
            # exit()
            with torch.no_grad():
                frame_i0 = utils.img2tensor(frame_i0).cuda()
                frame_i1 = utils.img2tensor(frame_i1).cuda()
                frame_i2 = utils.img2tensor(frame_i2).cuda()
                frame_i3 = utils.img2tensor(frame_i3).cuda()
                frame_i4 = utils.img2tensor(frame_i4).cuda()
                frame_i5 = utils.img2tensor(frame_i5).cuda()
                frame_i6 = utils.img2tensor(frame_i6).cuda()

                frame_i0, f_h_pad, f_w_pad = align_to_64(frame_i0, 64)
                frame_i1, f_h_pad, f_w_pad = align_to_64(frame_i1, 64)
                frame_i2, f_h_pad, f_w_pad = align_to_64(frame_i2, 64)
                frame_i3, f_h_pad, f_w_pad = align_to_64(frame_i3, 64)
                frame_i4, f_h_pad, f_w_pad = align_to_64(frame_i4, 64)
                frame_i5, f_h_pad, f_w_pad = align_to_64(frame_i5, 64)
                frame_i6, f_h_pad, f_w_pad = align_to_64(frame_i6, 64)

                [b, c, h, w] = frame_i0.shape

                flow_warping = Resample2d().cuda()


                warp_i0 = frame_i3.view(b, c, 1, h, w)
                warp_i1 = frame_i3.view(b, c, 1, h, w)
                warp_i2 = frame_i3.view(b, c, 1, h, w)
                warp_i4 = frame_i3.view(b, c, 1, h, w)
                warp_i5 = frame_i3.view(b, c, 1, h, w)
                warp_i6 = frame_i3.view(b, c, 1, h, w)
                warp_i3 = frame_i3.view(b, c, 1, h, w)

                frame_input = torch.cat((warp_i0.detach(), warp_i1.detach(), warp_i2.detach(), warp_i3.detach(), warp_i4.detach(), warp_i5.detach(), warp_i6.detach()), 2)
                frame_pred = three_dim_model(frame_input)
               
                frame_pred_rf = frame_pred

                frame_pred_rf = frame_pred_rf.view(b, c, 1, h, w)
                frame_pred_rf = frame_pred_rf[:, :, :, 0:h-f_h_pad, 0:w-f_w_pad]

            fusion_frame_pred = utils.tensor2img(frame_pred_rf.view(1, c, h-f_h_pad, w-f_w_pad))

            frame_i3_gt = np.uint8(frame_i3_gt*255.0)
            fusion_frame_pred = np.uint8(fusion_frame_pred.clip(0,1)*255.0)
            
            psnr_s.append(compare_psnr(fusion_frame_pred, frame_i3_gt))
            ssim_s.append(compare_ssim(fusion_frame_pred, frame_i3_gt, multichannel=True))
            # embed()
            # exit()
            # output_filename = os.path.join(vis_dir, "%05d_res.png"%t)
            # utils.save_img(fusion_frame_pred, output_filename)

            # frame_input = utils.tensor2img(frame_i3.view(1, c, h, w))
            # output_filename = os.path.join(output_dir, "%05d_input.png"%t)
            # utils.save_img(frame_input, output_filename)
            
        print("PSNR: %.2f, SSIM: %.4f"%(np.mean(psnr_s), np.mean(ssim_s)))
    if len(times) > 0:
        time_avg = sum(times) / len(times)
        print("Average time = %f seconds (Total %d frames)" % (time_avg, len(times)))
