from selectors import EpollSelector
from tempfile import tempdir
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
from scipy import misc
from generate_PSF import PSF
from generate_trajectory import Trajectory
import csv

import argparse
# import config
from PIL import Image
import pdb
import random



def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./Toy_in")
    parser.add_argument("--out_root", type=str, default="./Toy_out")
    parser.add_argument("--inpint_root", type=str, default="./inp_out")
    parser.add_argument("--move_root", type=str, default="./move_out")
    parser.add_argument("--s", type=float, default=8.5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--tt", type=int, default=1)
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98
    return parser



def blurring(image,piece_, scale, canvas=64,max_len=60,params=[0.01, 0.009, 0.008, 0.007, 0.005, 0.003]):
    b, c ,h,w = image.shape
    trajectory = Trajectory(canvas=canvas,max_len=max_len,expl=np.random.choice(params)).fit()
    psf_ = PSF(canvas=canvas,trajectory=trajectory).fit()
    psf = psf_[0]
    k_y, k_x = psf.shape
    delta_y = h - k_y
    delta_x = h - k_x
    tmp = np.pad(psf, delta_y // 2, 'constant')
    cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    blured = np.array(signal.fftconvolve(image[0,0,:,:], tmp, 'same'))
    blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    blured = torch.from_numpy(np.abs(blured)).unsqueeze(0)
    mask_full = torch.zeros([c,h,w])
    mask_full[:,:,piece_[0]:piece_[0]+scale+1] = 1.0 
    mask_back = 1.0 - mask_full
    blured = (blured * mask_full).unsqueeze(0) + mask_back * image
    return blured


def inpainting_black(image,piece_,scale):
    b,c,h,w=image.shape
    mask_full = torch.ones([b,c,h,w])
    mask_full[:,:,:,piece_[0]:piece_[0]+scale] = 0. 
    image_db = image * mask_full
    return image_db

def inpainting(image,piece_,scale):
    b,c,h,w=image.shape
    mask_full = torch.zeros([b,c,h,w])
    mask_full[:,:,:,piece_[0]:piece_[0]+scale] = 1. 
    mask_back = 1.0 - mask_full
    image_db = (image *mask_back) + mask_full
    mask_inp = mask_full
    return image_db,mask_inp

def move_x(inputs_bd,shift_line_x,locs,dxs):
    
    inputs_bd = inputs_bd.unsqueeze(0)
    img_zeros = torch.zeros_like(inputs_bd)
    t = 0
    for i in range(0,256):
        if shift_line_x[i] == 0:
            img_zeros[:,:,:,i] = inputs_bd[:,:,:,i-t]
        else:
            t += 1
    for j in range(len(locs)):
        img_zeros[:,:,:,locs[j]:locs[j]+dxs[j]] = inputs_bd[:,:,:,locs[j]-dxs[j]:locs[j]]
    img_move_x = img_zeros
    return img_move_x


def check_back_x(inputs_bd,shift_line_x,locs,dxs):
    
    inputs_bd = inputs_bd.unsqueeze(0)

    img_zeros = torch.zeros_like(inputs_bd)
    t = 0
    for i in range(0,256):
        count = i+t
        if count < 256:
            if shift_line_x[i] == 0:
                img_zeros[:,:,:,i-t] = inputs_bd[:,:,:,i]
            else:
                t += 1
    img_move_x = img_zeros
    return img_move_x

    

def check_back_y(hh,ww,shift_line_x,locs,dxs,shift_line,inputs_bd):

    inputs_bd = inputs_bd.unsqueeze(0)
    
    shift_line = -1 * shift_line
    vectors = [torch.arange(0, s) for s in (hh,ww)]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)  # y, x
    grid = torch.unsqueeze(grid, 0)  # add batch
    grid = grid.type(torch.FloatTensor) 

    
    flow_expand = shift_line.unsqueeze(1).unsqueeze(1).expand(1,1,hh,ww)
    flow_x = torch.zeros_like(flow_expand)
    flow_ = torch.cat((flow_expand,flow_x),dim=1)
    new_locs = grid + flow_
    shape = flow_.shape[2:]
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
    new_locs = new_locs.permute(0, 2, 3, 1)
    new_locs = new_locs[..., [1, 0]]
    inputs_bd = F.grid_sample(inputs_bd, new_locs, mode='nearest')
    return inputs_bd

def deformable_tran(opt, image):
    """
    Apply warping to the input image based on a generated random noise grid.
    Args:
        opt: Parsed command line arguments
        image: Input image tensor of shape (C, H, W)
    Returns:
        Warped image tensor of shape (C, H, W)
        Warping grid used for visualization
    """
    c, h, w = image.shape

    kk = random.randint(4,20) 
    ss = random.randint(0.5,4.5) 
    ins = torch.rand(1, 2, kk, kk) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.interpolate(ins, size=(h, w), mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
    )

    array1d = torch.linspace(-1, 1, steps=h)
    array1d_w = torch.linspace(-1, 1, steps=w)
    x, y = torch.meshgrid(array1d, array1d_w, indexing='ij')
    identity_grid = torch.stack((y, x), 2)[None, ...]

    grid_temps = (identity_grid + ss * noise_grid / h) * opt.grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)


    inputs_bd = F.grid_sample(image.unsqueeze(0), grid_temps, align_corners=True)
    inputs_bd = inputs_bd.squeeze(0)
    return inputs_bd, grid_temps


def warping_img(opt,image,piece_,moving=False, painting=False,blur=False):
    c,h,w=image.shape
    inputs_bd = image
    H_center = 320
    W_center = 320
    mask_move = inputs_bd * 0.
    im_ori = image.unsqueeze(0)

    vectors = [torch.arange(0, s) for s in (h,w)]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids) 
    grid = torch.unsqueeze(grid, 0) 
    grid = grid.type(torch.FloatTensor) 

    shift_line_y = torch.zeros(1,w)
    shift_line_x = torch.zeros(1,w)
    
   
    if moving==True:
        
        inputs_bd = inputs_bd.unsqueeze(0)
        img_ones = torch.ones_like(inputs_bd)
        start_idx = piece_[0]
        end_idx = piece_[1]
        shift_y_int = random.randint(-15,15)

        shift_line_y[0][start_idx:end_idx] = shift_y_int
        flow_expand = shift_line_y.unsqueeze(1).unsqueeze(1).expand(1,1,w,w)
        flow_x = torch.zeros_like(flow_expand)
        flow_ = torch.cat((flow_expand,flow_x),dim=1)
        new_locs = grid + flow_
        shape = flow_.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        inputs_bd = F.grid_sample(inputs_bd, new_locs, mode='nearest')
        mask_move = F.grid_sample(img_ones, new_locs, mode='nearest')

        if painting == True:
            scale = random.randint(1,2)
            im_ori_paint,mask_inp = inpainting(im_ori,piece_,scale)
            inputs_bd,mask_inp = inpainting(inputs_bd,piece_,scale)
            shift_line_y[0][piece_[0]:piece_[0]+scale] = 0.

            inputs_bd = transforms.CenterCrop((H_center,W_center))(inputs_bd.squeeze(0))
            mask_move = transforms.CenterCrop((H_center,W_center))(mask_move.squeeze(0))
            im_ori_paint = transforms.CenterCrop((H_center,W_center))(im_ori_paint.squeeze(0))
            mask_inp  = transforms.CenterCrop((H_center,W_center))(mask_inp.squeeze(0))
        else: 
            inputs_bd = transforms.CenterCrop((H_center,W_center))(inputs_bd.squeeze(0))
            mask_move = transforms.CenterCrop((H_center,W_center))(mask_move.squeeze(0))
            im_ori_paint = transforms.CenterCrop((H_center,W_center))(im_ori.squeeze(0))
            mask_inp  = inputs_bd * 0
    
    else:
        inputs_bd = transforms.CenterCrop((H_center,W_center))(inputs_bd)
        mask_move = transforms.CenterCrop((H_center,W_center))(mask_move)
        im_ori_paint = transforms.CenterCrop((H_center,W_center))(image)
        mask_inp  = inputs_bd * 0.

    return inputs_bd,im_ori_paint,mask_inp,mask_move,shift_line_y




def main():

    opt = get_arguments().parse_args()
    ori_imgs=os.listdir(opt.data_root)

    tt = opt.tt
    count = 0

    for im in ori_imgs:
            count += 1
            print('tt:',tt, 'count:', count)
            im_local = Image.open(os.path.join(opt.data_root,im))
            transform = transforms.Compose([transforms.Resize(320),transforms.ToTensor() ]) 
            im_local=transform(im_local)

            warped_image, grid_temps = deformable_tran(opt, im_local)
            warped_image = torch.clamp(warped_image, 0, 1)

            im_local = warped_image
            c,h,w=im_local.shape
            tensor_to_image=transforms.ToPILImage()

            N = random.randint(4,8) 
            low  = 50       
            high = w-20    
   
            res = random.sample(range(low, high), N) 
            res.append(w)
            res.insert(0,0)
            res.sort()
            N0 = random.randint(4,N+2) # 
            res_ = random.sample(res, N0)  
            res_.sort()
            pieces = []
            warping_images = []
            im_ori_paints = []
            mask_inps  = []
            mask_moves = []
            shift_lines = []
            Dx = []
            dx = 0

            for i in range(N+1):
                
                moving = False
                moving_x = False
                painting = False
                blur = False
              
                if res[i] in res_:
                    if res[i] == 0:
                        moving = False
                        painting = False
                        blur = False
                        
                    else:

                        rand_blur = random.randint(-1,3) 
                        moving   = True
                        painting = True
                        if rand_blur > 0:
                            painting = True
                        else:
                            painting = False
                            blur = False

                piece_ = [res[i] ,res[i+1]]
                warping_im,im_ori_paint,mask_inp,mask_move,shift_line = warping_img(opt,im_local,piece_,moving=moving,painting=painting,blur=blur)

                shift_lines.append(shift_line)
                pieces.append(piece_)
                warping_images.append(warping_im)
                im_ori_paints.append(im_ori_paint)
                mask_inps.append(mask_inp)
                mask_moves.append(mask_move)
            H_center = 256
            W_center = 256
            new_im=torch.ones([c,h,w])
            new_mask_img=torch.ones([c,h,w])
            new_mask_inp = torch.zeros([c,h,w])
            new_mask_move = torch.zeros([c,h,w])
            new_shift = torch.zeros(1,w)

            for i in range(N+1):
                temp_line = shift_lines[i]
                temp_mask_inp  =  mask_inps[i]
                temp_mask_move =  1.0 - mask_moves[i]
                temp_img = warping_images[i]
                temp_img_paint = im_ori_paints[i]

                new_shift[0][res[i]:res[i+1]] = temp_line[0][res[i]:res[i+1]].detach()
                new_im[:,:,res[i]:res[i+1]] = temp_img[:,:,res[i]:res[i+1]].detach()
                new_mask_inp[:,:,res[i]:res[i+1]] = temp_mask_inp[:,:,res[i]:res[i+1]].detach()
                new_mask_img[:,:,res[i]:res[i+1]] = temp_img_paint[:,:,res[i]:res[i+1]].detach()
                new_mask_move[:,:,res[i]:res[i+1]] = temp_mask_move[:,:,res[i]:res[i+1]].detach()
            

            
            new_shift = transforms.CenterCrop((1,W_center))(new_shift) 

            new_im_0 = transforms.CenterCrop((H_center,W_center))(new_im)   
            new_mask_inp_0 = transforms.CenterCrop((H_center,W_center))(new_mask_inp)  
            new_im_paint_0 = transforms.CenterCrop((H_center,W_center))(new_mask_img) 
            im_local_crop_0 = transforms.CenterCrop((H_center,W_center))(im_local) 
            new_mask_move_0 = transforms.CenterCrop((H_center,W_center))(new_mask_move) 
            
            shift_line_x =  torch.zeros(1,w)
            N_x = random.randint(0,4)
            locs = []
            dxs = []
            move_points = random.sample(range(20, 246), N_x) 
            for mp in move_points:
                dx  = random.randint(1,10)
                shift_line_x[0][mp:mp+dx] = 1
                locs.append(mp)
                dxs.append(dx)

            new_im  = move_x(new_im_0,shift_line_x[0],locs,dxs)  
            new_mask_inp = move_x(new_mask_inp_0,shift_line_x[0],locs,dxs)   
            new_im_paint = move_x(new_im_paint_0,shift_line_x[0],locs,dxs) 

            im_local_crop =  im_local_crop_0
            new_mask_move = move_x(new_mask_move_0,shift_line_x[0],locs,dxs) 

            new_name = im[:-4] + '_train_' + str(tt) + im[-4:]
            old_name = im[:-4] + '_train_' + str(tt) + im[-4:]
            shift_name = im[:-4] + '_train_'+ str(tt) +'.pt'

            out_root_clear = opt.out_root + '/clean_img'
            out_root_blur = opt.out_root + '/artifact_img'
            out_root_paint = opt.out_root + '/clean_paint'
            out_root_mask =  opt.out_root + '/mask_paint'
            out_root_shift =  opt.out_root + '/shift'
            out_root_move =  opt.out_root + '/move'


            tensor_to_image(new_im[0]).save(os.path.join(out_root_blur,new_name))
            tensor_to_image(im_local_crop[0]).save(os.path.join(out_root_clear,old_name))
            tensor_to_image(new_im_paint[0]).save(os.path.join(out_root_paint,old_name))
            tensor_to_image(new_mask_inp[0]).save(os.path.join(out_root_mask,old_name))
            tensor_to_image(new_mask_move[0]).save(os.path.join(out_root_move,old_name))

            torch.save(new_shift,os.path.join(out_root_shift,shift_name))



if __name__ == "__main__":
    main()