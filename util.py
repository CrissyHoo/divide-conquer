import sys
import os
import time
import numpy as np
from os.path import join,exists
import glob
from tqdm import trange, tqdm
import cv2
import math
import scipy
import random
import torch
from torch.nn import functional as F
import json
import matplotlib.pyplot as plt
import warnings

def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)
    #print("xxxxxxxxxxx",x.size())
    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    filter_height, filter_width = 13, 13
    pad_w, pad_h = (filter_width-1)//2, (filter_height-1)//2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        if H % 3 != 0:
            r_h = 3 - (H % 3)
        if W % 3 != 0:
            r_w = 3 - (W % 3)
    x = F.pad(x, (pad_w, pad_w + r_w, pad_h, pad_h + r_h), 'reflect')

    gaussian_filter = torch.from_numpy(gkern(filter_height, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)#stride可以看成是缩小的倍数
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x

def automkdir(path):
    if not exists(path):
        os.makedirs(path)

def automkdirs(path):
    [automkdir(p) for p in path]

def compute_psnr_torch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1. / mse)
def compute_psnr_torch_list(img1,img2):
    ba=img1.shape[0]
    ans=[]
    for i in range(ba):
        bimg1=img1[i]
        bimg2=img2[i]
        #print(bimg1.shape,bimg2.shape)
        tmp=10*torch.log10(1./torch.mean((bimg1-bimg2)**2))
        #tmp=tmp/100#adjust to 0-1
        ans.append(tmp)
    return ans

def calcu_scope(psnrprogress,config,index,weight):
    warnings.simplefilter('ignore', np.RankWarning)
    le=len(psnrprogress[index])
    #print(psnrprogress[index])
    #print("le",le,"ylen",len(psnrprogress[index]))
    x=range(le)
    #print("psnrprogresslen",le)
    
    if le<10:
        return 1.,0.,0.,1.
    else:
        #print("inscppeaaaaaaaaa")
        #需要计算num内的scope
        minus=psnrprogress[index][-1]-psnrprogress[index][-5]
        win=(100-psnrprogress[index][-1])/100
        win=np.exp(win)
        #rate=getRate(weight)
        if minus>0:
            wde=1+5*np.log(1+np.abs(minus/10))
        else:
            wde=1+5*np.log(1+np.abs(minus/10))
        tmp=(50-psnrprogress[index][-1])/5
        alfa=1/(1+np.exp(-tmp+2))
        wcur=alfa*wde+(1-alfa)*win
        #print("alfa",alfa)
        itwei=0.7*wcur+0.3*weight[index][-1]
        return itwei,wde,win,alfa
def getRate(weight):
    maxweight=0
    minweight=10
    for itwei in weight:
        if itwei[-1]>maxweight:
            maxweight=itwei[-1]
        if itwei[-1]<minweight:
            minweight=itwei[-1]
    #借助上一波的weight的界限
    rate=maxweight-minweight
    if rate==0:
        rate=1
    return rate

        
        

        #另一种方式，poly函数拟合，效果不够好
        #fitting=np.polyfit(x,psnrprogress[index],config.train.polytimes)
        #derive=np.polyder(fitting)
        #deval=np.polyval(derive,le)
        #if deval>5000 or deval<-5000:
         #   deval=5000
        #return 1+np.log(abs(deval)+1)/6

def compute_psnr(img1, img2):
    mse=np.mean((img1 - img2) ** 2)
    return 10 * np.log(1. / mse) / np.log(10)

def makelr_fromhr_cuda(hr, scale=4, device=None, data_kind='single'):
    if data_kind == 'double' or isinstance(hr, (tuple, list)):
        return [i.to(device) for i in hr]
    else:
        hr = hr.to(device)
        #print("hrsizeeeeeeeee",hr.size())#bcthw
        lr = DUF_downsample(hr, scale)
        return lr, hr

def crop_cpu(img, crop_sz, step):
    # 64,56  按照我自己的逻辑crop，中间重叠8个pixel
    h, w, c = img.size()  # 960,1024,3

    left_H = h // step * step + crop_sz - h  # 960/56*56+64-960
    left_W = w // step * step + crop_sz - w

    img = torch.cat([img, torch.flip(img, [0])], 0)[:h + left_H, :]
    img = torch.cat([img, torch.flip(img, [1])], 1)[:, :w + left_W]

    # show_tensor_image(img.permute(2, 0, 1))

    # if left_H:
    #     left_H = torch.zeros(left_H, w, c).to(device)
    #     img = torch.cat((img, left_H), dim=0)
    #
    # if left_W:
    #     left_W = torch.zeros(h, left_W, c).to(device)
    #     img = torch.cat((img, left_W), dim=1)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            lr_list.append(crop_img)
    h = x + crop_sz
    w = y + crop_sz
    return lr_list, num_h, num_w, h, w


def combine(sr_list, num_h, num_w, h, w, patch_size, step, scale):
    index = 0
    sr_img = sr_list[index].new_zeros(h * scale, w * scale, 3)
    for i in range(num_h):
        for j in range(num_w):
            sr_img[i * step * scale:i * step * scale + patch_size * scale,
            j * step * scale:j * step * scale + patch_size * scale, :] += sr_list[index]
            index += 1
    # sr_img=sr_img.astype('float32')

    for j in range(1, num_w):
        sr_img[:, j * step * scale:j * step * scale + (patch_size - step) * scale, :] /= 2

    for i in range(1, num_h):
        sr_img[i * step * scale:i * step * scale + (patch_size - step) * scale, :, :] /= 2
    return sr_img


def evaluation(model, eval_data, config):
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    #psnr_all=[]
    scale = config.model.scale
    epoch = config.train.epoch
    device = config.device
    test_runtime=[]
    in_h=128
    in_w=240
    bd=2

    for iter_eval, (img_lq,img_hq) in enumerate(tqdm(eval_data)):
        #img_hq = img_hq[:, :, :, bd * scale: (bd + in_h) * scale, bd * scale: (bd + in_w) * scale]
        #img_lq, img_hq = makelr_fromhr_cuda(img_hq, scale, device, config.data_kind)
        # img_lq = img_lq[:, :, :, :in_h, :in_w]
        # img_hq = img_hq[:, :, :, :in_h*scale, :in_w*scale]

        B, C, T, H, W = img_hq.shape
        #print("hqshape",img_hq.shape)
        #T,C,H,W=img_lq.shape
        
        start.record()
        with torch.no_grad():
            print(img_hq.shape,img_lq.shape)
            img_clean = model(img_lq[0])
        end.record()
        torch.cuda.synchronize()
        test_runtime.append(start.elapsed_time(end) / T)

        #cleans = [_.permute(0,2,3,4,1) for _ in img_clean]
        #hr = img_hq.permute(0,2,3,4,1)

        #psnr_cleans, psnr_hr = cleans, hr
        #psnrs = [compute_psnr_torch(_, psnr_hr).cpu().numpy() for _ in psnr_cleans]
        psnr=compute_psnr_torch(img_clean,img_hq)
        clean = (np.round(np.clip(img_clean.cpu().numpy()* 255, 0, 255))).astype(np.uint8)
        cv2_imsave(join(config.path.eval_result,'{:0>4}.png'.format(iter_eval )), clean)
        psnr_all.append(psnr)

    psnrs = np.array(psnr_all)
    psnr_avg = np.mean(psnrs, 0, keepdims = False)

    with open(config.path.eval_file,'a+') as f:
        eval_dict = {'Epoch': epoch, 'PSNR': psnr_avg.tolist()}
        eval_json = json.dumps(eval_dict)
        f.write(eval_json)
        f.write('\n')
    print(eval_json)
    ave_runtime = sum(test_runtime) / len(test_runtime)
    print(f'average time cost {ave_runtime} ms')

    model.train()
    
    return psnr_avg

def test_video(model, path, savepath, config):
    model.eval()
    automkdir(savepath)
    scale = config.model.scale
    device = config.device
    # print(savepath)
    prefix = os.path.split(path)[-1]
    inp_type = 'original' if config.data_kind == 'single' else f'input{config.model.scale}'
    imgs=sorted(glob.glob(join(path, "truth", '*.png')))#udm10改成truth
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    test_runtime=[]
    
    if inp_type == 'original':
        img_hq = [cv2_imread(i) for i in imgs]#thwc
        #print("imagehqshape",torch.from_numpy(np.array(img_hq)/255.).float().shape)
        img_hq = torch.from_numpy(np.array(img_hq)/255.).float().permute(0,3,1,2).contiguous()
        img_hq = img_hq.to(device)
        img_lq = DUF_downsample(img_hq.unsqueeze(0), scale)
    else:
        img_lq = [cv2_imread(i) for i in imgs]
        img_lq = torch.from_numpy(np.array(img_lq)).float().permute(0,3,1,2).contiguous()/255.
        img_lq = img_lq.to(device).unsqueeze(0)#增加0维度
    b, t, c, h, w = img_lq.shape
    img_hq=img_hq.unsqueeze(0)


    #print("imglqshape",img_lq.shape)


    files_info = [os.path.split(_)[-1] for _ in imgs]

    #对数据划分
    lr = []
    hr=[]
    SR_list = []
    real_patch=[]
    for i in range(t):  # 默认只有一个video？
        LR = img_lq[:, i].squeeze().permute(1, 2, 0)
        HR=img_hq[:,i].squeeze().permute(1,2,0)

        hr_list,_,_,_,_=crop_cpu(HR,256,248)
        lr_list, num_h, num_w, H, W = crop_cpu(LR, 64, 62)
        #在lr list中随机选取patch来做补偿
        n=len(lr_list)
        patchindex=random.sample(range(n),4)
        framepatch=[]
        for j in patchindex:
            framepatch.append(lr_list[j])
        real_patch.append(framepatch)#t4hwc
        lr.append(lr_list)  # lr list是一个patch list
        hr.append(hr_list)
    for i in range(len(lr)):  # t
        lr[i] = torch.stack(lr[i]).permute(0, 3, 1, 2)  # phwc->pchw
        hr[i]=torch.stack(hr[i]).permute(0, 3, 1, 2)
        real_patch[i]=torch.stack(real_patch[i]).permute(0,3,1,2)
    lr = torch.stack(lr, 1)  # ptchw
    real_patch=torch.stack(real_patch).permute(1,0,2,3,4)#t4chw  需要->b4tchw
    # print("realpatchshape",real_patch.shape,lr.shape)
    hr=torch.stack(hr,1)
    #===按照flow的复杂度划分===
    needflowindex = []
    noflowindex = []
    for i in range(len(lr)):
        HQ = hr[i]
        # print("in calculating flow, lr size", LQ.shape)
        img = HQ.permute(0, 2, 3, 1).float()  # 转成thwc
        k=img.shape[0]//2
        img1 = img[k]
        img2 = img[k + 1]
        f1 = [img1, img2]#thwc
        f1 = torch.stack(f1).permute(3, 0, 1, 2)
        f1 = torch.unsqueeze(f1, 0)
        #print("f1shape",f1.shape)#1,3,2,32,32
        tmpflow = flowmodel(f1)
        #print("tmpflowsize",tmpflow.shape)
        tmpflow = torch.abs(torch.mean(tmpflow))
        print("the flow is",tmpflow)
        if (tmpflow < 0.02):
            noflowindex.append(i)
        else:
            needflowindex.append(i)

    #====ending================

    print("flow:",len(needflowindex),"noflow:",len(noflowindex))  # 15, 9，3，256，256
    start.record()
    with torch.no_grad():
        # img_clean,_ = model(img_lq)
        #=====分两组计算=====
        if needflowindex != []:
            needflowlr_ = torch.stack([lr[i] for i in needflowindex])  # 15，9，3，64，64

            output_needflow = model(needflowlr_)[0]  # ptchw

        if noflowindex != []:
            noflowlr_ = torch.stack([lr[i] for i in noflowindex])
            real_patch_=torch.stack([real_patch for i in noflowindex])
            output_noflow = singlemodel(noflowlr_,real_patch_)
            # output_noflow=[]
            # for k in noflowindex:
            #     frames=lr[k]#tchw
            #     videopatch=[]
            #     for frame in frames:
            #         videopatch.append(singlemodel(frame.unsqueeze(0)).data.squeeze().float())
            #     output_noflow.append(torch.stack(videopatch))
            # output_noflow=torch.stack(output_noflow)
            # # print(output_noflow.shape)


        img_clean = []
        for i in range(len(lr)):
            if i in needflowindex:
                img_clean.append(output_needflow[needflowindex.index(i)])
            elif i in noflowindex:
                img_clean.append(output_noflow[noflowindex.index(i)])

        p, t, _, _, _ = lr.shape
        for i in range(t):
            sr = []
            for j in range(p):
                sr.append(img_clean[j][i].permute(1, 2, 0))
            sr_img = combine(sr, num_h, num_w, H, W, 64, 62, 4)#这里combine是combine之前的size
            SR_ref = sr_img[0:h * 4, 0:w * 4].permute(2, 0, 1).unsqueeze(0)
            # print("SR", SR_ref.shape)
            SR_list.append(SR_ref)
        img_clean = torch.stack(SR_list).permute(1, 0, 2, 3, 4)
    end.record()
    torch.cuda.synchronize()
    test_runtime.append(start.elapsed_time(end))  # milliseconds


    if isinstance(img_clean, tuple):
        img_clean = img_clean[0]

    sr = img_clean[0].permute(0,2,3,1)#tchw
    sr = sr.cpu().numpy()
    sr = (np.round(np.clip(sr * 255, 0, 255))).astype(np.uint8)
    [cv2_imsave(join(savepath, files_info[i]), sr[i]) for i in range(t)]
    print('Cost {} ms in average.\n'.format(np.mean(test_runtime) / t))
    #print psnr

    return

def save_checkpoint(model, epoch, model_folder,psnrprogress,weight,wde,win,alphas):
    model_out_path = os.path.join(model_folder , '{:0>4}.pth'.format(epoch))
    state = {"epoch": epoch ,"model": model.state_dict(),"psnrprogress":psnrprogress,"weight":weight,"wde":wde,"win":win,"alfa":alphas}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    return

def load_checkpoint(network=None, resume='', path='', weights_init=None, rank=0):
    try:
        num_resume = int(resume[resume.rfind('/')+1:resume.rfind('.')])
    except Exception as e:
        num_resume = 0
    finally:
        if num_resume < 0:
            checkpointfile = sorted(glob.glob(join(path,'*')))
            if len(checkpointfile)==0:
                resume = 'nofile'
            else:
                resume = checkpointfile[-1]
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage.cuda(rank))
            start_epoch = checkpoint["epoch"]
            network.load_state_dict(checkpoint["model"])
            #psnrprogress=checkpoint["psnrprogress"]
            #weight=checkpoint["weight"]
            #wde=checkpoint["wde"]
            #win=checkpoint["win"]
            #alpharecord=checkpoint["alfa"]
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            if weights_init is not None:
                network.apply(weights_init)
            start_epoch = 0
            psnrprogress=[]
            weight=[]
            wde=[]
            win=[]
            alpharecord=[]
            for i in range(522):
                tmp=[]
                psnrprogress.append(tmp)
            for i in range(522):
                tmp=[]
                weight.append(tmp)
            for i in range(522):
                tmp=[]
                wde.append(tmp)
            for i in range(522):
                tmp=[]
                win.append(tmp)
            for i in range(522):
                tmp=[]
                alpharecord.append(tmp)

    return start_epoch#,psnrprogress,weight,wde,win,alpharecord

def adjust_learning_rate(init_lr, final_lr, epoch, epoch_decay, iteration, iter_per_epoch, optimizer, ifprint=False):
    """Sets the learning rate to the initial LR decayed by 10"""
    #print("lrepoch",epoch)
    if epoch>750:
        lr=0.6e-5 
    elif epoch>680:
        lr=1.5e-5
    elif epoch>600:
        lr=3e-5
    elif epoch>550:
        lr=5e-5
#550之后lr变为5e-5 641 2.5e-5 706 1.15e-5 720 1e-5
    #elif epoch>650:
     #   lr=3.e-5
    #elif epoch>550:
    #    lr=4.e-5
    #elif epoch>450:
     #   lr=5.e-5
    else:
        lr = (init_lr-final_lr) * max((1 - (epoch + iteration / iter_per_epoch) / epoch_decay), 0)+final_lr
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr
        
    if ifprint:
        print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    return lr

def cv2_imsave(img_path, img):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def cv2_imread(img_path):
    img=cv2.imread(img_path)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    return img

def save_distribution(epoch,psnrdis):
    # observe the distribution of the eval data while training
    figname = 'psnr_distribution_epoch{}'.format(epoch)
    pathfig = os.path.join("./out/psnrdistri")
    if not os.path.exists(pathfig):
        os.makedirs(pathfig)
    fig = plt.figure()
    plt.title(figname)
    plt.hist(psnrdis)
    plt.xlabel('PSNR')
    plt.ylabel("number")
    plt.savefig(os.path.join(pathfig, '{}.png'.format(epoch)))
    plt.close(fig)
    # ending observe

class DICT2OBJ(object):
    def __init__(self, obj, v=None):
        # if not isinstance(obj, dict):
        #     setattr(self, obj, v)
        #     return 
        for k, v in obj.items():
            if isinstance(v, dict):
                # print('dict', k, v)
                setattr(self, k, DICT2OBJ(v))
            else:
                # print('no dict', k, v)
                setattr(self, k, v)


if __name__=='__main__':
    pass

