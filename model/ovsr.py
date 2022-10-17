import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.common import generate_it, UPSCALE, PFRB


class UNIT(nn.Module):
    def __init__(self, kind='successor', basic_feature=64, num_frame=3, num_b=5, scale=4, act=nn.LeakyReLU(0.2, True)):
        super(UNIT, self).__init__()
        self.bf = basic_feature
        self.nf = num_frame
        self.num_b = num_b
        self.scale = scale
        self.act = act
        self.kind = kind
        if kind == 'precursor':
            self.conv_c = nn.Conv2d(3, self.bf, 3, 1, 3 // 2)
            self.conv_sup = nn.Conv2d(3 * (num_frame - 1), self.bf, 3, 1, 3 // 2)

        else:
            self.conv_c = nn.Sequential(*[nn.Conv2d((3 + self.bf), self.bf, 3, 1, 3 // 2) for i in range(num_frame)])
        self.blocks = nn.Sequential(*[PFRB(self.bf, 3, act) for i in range(num_b)])
        self.merge = nn.Conv2d(3 * self.bf, self.bf, 3, 1, 3 // 2)
        self.upscale = UPSCALE(self.bf, scale, act)
        print(kind, num_b)

    def forward(self, it, ht_past, ht_now=None, ht_future=None):

        if self.kind == 'precursor':
            B, C, T, H, W = it.shape  # 这里T应该是=3的

            it_c = it[:, :, T // 2]
            index_sup = list(range(T))
            index_sup.pop(T // 2)
            it_sup = it[:, :, index_sup]
            it_sup = it_sup.view(B, C * (T - 1), H, W)
            # print("!!!!!!!!",it_sup.shape)
            hsup = self.act(self.conv_sup(it_sup))
            hc = self.act(self.conv_c(it_c))
            inp = [hc, hsup, ht_past]
            # print(hc.shape,hsup.shape,ht_past.shape)这三者的size是一样大小的。tchw

        elif self.kind == 'successor':
            ht = [ht_past, ht_now, ht_future]
            it_c = [torch.cat([it[:, :, i, :, :], ht[i]], 1) for i in range(3)]
            inp = [self.act(self.conv_c[i](it_c[i])) for i in range(3)]

        inp = self.blocks(inp)

        ht = self.merge(torch.cat(inp, 1))
        it_sr = self.upscale(ht)
        # print(it_sr.shape)

        return it_sr, ht


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.bf = config.model.basic_filter
        self.num_pb = config.model.num_pb
        self.num_sb = config.model.num_sb

        self.scale = config.model.scale
        self.nf = config.model.num_frame
        self.kind = config.model.kind  # local or global
        self.act = nn.LeakyReLU(0.2, True)
        self.precursor = UNIT('precursor', self.bf, self.nf, self.num_pb, self.scale, self.act)
        self.successor = UNIT('successor', self.bf, self.nf, self.num_sb, self.scale, self.act)

        print(self.kind, '{}+{}'.format(self.num_pb, self.num_sb))

        params = list(self.parameters())
        pnum = 0
        for p in params:
            l = 1
            for j in p.shape:
                l *= j
            pnum += l
        print('Number of parameters {}'.format(pnum))

    def forward(self, x, start=0):
        if x == []:
            return [[]]
        x = x.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = x.shape
        start = max(0, start)
        end = T - start
        sr_all = []
        pre_sr_all = []
        pre_ht_all = []
        ht_past = torch.zeros((B, self.bf, H, W), dtype=torch.float, device=x.device)

        # precursor
        for idx in range(T):
            t = idx if self.kind == 'local' else T - idx - 1
            insert_idx = T + 1 if self.kind == 'local' else 0

            it = generate_it(x, t, self.nf, T)  # 这个就是产生连续的nf帧图像，返回一个组
            # print("itititittititititi",it.shape)

            it_sr_pre, ht_past = self.precursor(it, ht_past, None, None)  # 如果flow为true的话，只需要precursor的结果
            pre_ht_all.insert(insert_idx, ht_past)
            pre_sr_all.insert(insert_idx, it_sr_pre)

        # successor
        ht_past = torch.zeros((B, self.bf, H, W), dtype=torch.float, device=x.device)
        for t in range(end):
            it = generate_it(x, t, self.nf, T)
            ht_future = pre_ht_all[t] if t == T - 1 else pre_ht_all[t + 1]
            it_sr, ht_past = self.successor(it, ht_past, pre_ht_all[t], ht_future)
            sr_all.append(it_sr + pre_sr_all[t])

        sr_all = torch.stack(sr_all, 2)[:, :, start:]
        pre_sr_all = torch.stack(pre_sr_all, 2)[:, :, start:end]
        sr_all = sr_all.permute(0, 2, 1, 3, 4)
        pre_sr_all = pre_sr_all.permute(0, 2, 1, 3, 4)
        return sr_all, pre_sr_all


class UNIT_single(nn.Module):
    def __init__(self, kind='successor', basic_feature=64, num_frame=5, num_b=5, scale=4, act=nn.LeakyReLU(0.2, True)):
        super(UNIT_single, self).__init__()
        self.bf = basic_feature
        self.nf = num_frame
        self.num_b = num_b
        self.scale = scale
        self.act = act
        self.kind = kind
        if self.kind == "precursor":
            self.conv_first = nn.Conv2d(3, self.bf, 3, 1, 3 // 2)
            self.conv_sup = nn.Conv2d(3 * 4, self.bf * 4, 3, 1, 1)  # 这里的4指的是补偿的部分只有4个patch
            self.sattention = globalAttention(num_feat=self.bf)
            self.tattention = globalAttention(num_feat=self.bf)
            self.conv_last1 = nn.Conv2d(self.bf, 80, 3, 1, 1)
            self.conv_last2 = nn.Conv2d(self.bf, 80, 3, 1, 1)
        else:
            self.conv_c = nn.Sequential(*[nn.Conv2d(83, 80, 3, 1, 3 // 2) for i in range(3)])
        self.blocks = nn.Sequential(*[PFRB(80, 3, act) for i in range(num_b)])
        self.merge = nn.Conv2d(3 * 80, 80, 3, 1, 3 // 2)
        self.upscale = UPSCALE(80, scale, act)

    def forward(self, it, patchpool, hiddeninfo=None, ht_now=None, ht_future=None):
        # hidden指的是patch pool
        # input是相邻的三帧，划分成patch后可以使用nonlocal进行补偿，这个是补偿部分。另外fusion部分就不使用多帧
        B, C, T, H, W = it.shape  # 15,3,5,64,64

        # print("patchpoolshape",patchpool.shape)#15,4,3,64,64
        if self.kind == 'precursor':
            b, t, c, h, w = patchpool.shape
            it = it.permute(0, 2, 1, 3, 4)
            it = it.reshape(B * T, C, H, W)
            patchpool = patchpool.reshape(b, c * t, h, w)

            # print(it.shape)
            conv_fea = self.act(self.conv_first(it)).view(B, T, -1, H, W)
            ht = self.act(self.conv_sup(patchpool)).view(b, t, -1, h, w)
            ht1 = ht[:, 0:2]
            ht2 = ht[:, 2:]
            conv_feacenter = torch.unsqueeze(conv_fea[:, T // 2], 1)
            # print("in ovsr", it.shape, conv_feacenter.shape, ht1.shape,ht2.shape)
            sinp = torch.cat((ht1, conv_feacenter, ht2), 1)

            tatte = self.tattention(conv_fea)  # b,5,64,64,64
            tatte_c = tatte[:, T // 2]

            satte = self.sattention(sinp)
            satte_c = satte[:, T // 2]

            satte_c = self.conv_last1(satte_c)
            tatte_c = self.conv_last2(tatte_c)
            if hiddeninfo == None:
                hiddeninfo = tatte_c
            inp = [satte_c, tatte_c, hiddeninfo]  # bchw
        else:
            # successor
            if hiddeninfo == None:
                hiddeninfo = ht_now
            ht = [hiddeninfo, ht_now, ht_future]
            it_c = [torch.cat([it[:, :, i, :, :], ht[i]], 1) for i in range(3)]  # 三个通道
            inp = [self.act(self.conv_c[i](it_c[i])) for i in range(3)]

        inp = self.blocks(inp)
        hiddeninfo = self.merge(torch.cat(inp, 1))
        it_sr = self.upscale(hiddeninfo)
        # print("patchpool,sr shape",patchpool.shape,it_sr.shape)

        return it_sr, hiddeninfo


class singleNet(nn.Module):
    def __init__(self, config):
        super(singleNet, self).__init__()
        self.bf = config.model.single_bf
        self.nf = config.model.single_nf
        self.num_si = config.model.num_si
        self.scale = config.model.scale
        self.act = nn.LeakyReLU(0.2, True)
        self.num_sb = config.model.num_sb
        self.successor = UNIT_single('successor', self.bf, self.nf, self.num_sb, self.scale, self.act)
        self.precursor = UNIT_single('precursor', self.bf, self.nf, self.num_si, self.scale, self.act)
        params = list(self.parameters())
        pnum = 0
        for p in params:
            l = 1
            for j in p.shape:
                l *= j
            pnum += l
        print('singleNet, Number of parameters {}'.format(pnum))

    def forward(self, x, realpatch):
        if x == []:
            return [[]]
        x = x.permute(0, 2, 1, 3, 4)
        b, c, t, h, w = x.shape
        # print(x.shape,realpatch.shape)
        sr_all = []
        pre_sr_all = []
        pre_ht_all = []
        # print("xshape",x.shape)
        ht = None
        # precursor
        for i in range(t):  # 用连续的五帧输出一帧
            it = generate_it(x, i, self.nf, t)
            patchcompen = realpatch[:, :, i]  # b,4,3,64,64
            sr_it, ht = self.precursor(it, patchcompen, ht)
            pre_sr_all.append(sr_it)
            pre_ht_all.append(ht)

        # successor
        ht = None
        for i in range(t):
            it = generate_it(x, i, self.nf, t)
            ht_future = pre_ht_all[i] if i == t - 1 else pre_ht_all[i + 1]
            it_sr, ht = self.successor(it, None, ht, pre_ht_all[i], ht_future)
            sr_all.append(it_sr + pre_sr_all[i])

        sr_all = torch.stack(sr_all, 2);
        # print("srallshape",sr_all.shape)#b,c,t,h,w
        sr_all = sr_all.permute(0, 2, 1, 3, 4)
        return sr_all


class globalAttention(nn.Module):
    def __init__(self, num_feat=64, patch_size=8, heads=1):
        super(globalAttention, self).__init__()
        self.heads = heads
        self.dim = patch_size ** 2 * num_feat
        self.hidden_dim = self.dim // heads
        self.num_patch = (64 // patch_size) ** 2

        self.to_q = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_k = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_v = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)
        self.patch2feat = torch.nn.Fold(output_size=(64, 64), kernel_size=patch_size, padding=0, stride=patch_size)

    def forward(self, x):
        b, t, c, h, w = x.shape  # B, 5, 64, 64, 64
        # print(x.shape)
        H, D = self.heads, self.dim
        n, d = self.num_patch, self.hidden_dim

        q = self.to_q(x.view(-1, c, h, w))  # [B*5, 64, 64, 64]

        k = self.to_k(x.view(-1, c, h, w))  # [B*5, 64, 64, 64]
        v = self.to_v(x.view(-1, c, h, w))  # [B*5, 64, 64, 64]

        unfold_q = self.feat2patch(q)  # [B*5, 8*8*64, 8*8]

        unfold_k = self.feat2patch(k)  # [B*5, 8*8*64, 8*8]
        unfold_v = self.feat2patch(v)  # [B*5, 8*8*64, 8*8]

        unfold_q = unfold_q.view(b, t, H, d, n)  # [B, 5, H, 8*8*64/H, 8*8]
        unfold_k = unfold_k.view(b, t, H, d, n)  # [B, 5, H, 8*8*64/H, 8*8]
        unfold_v = unfold_v.view(b, t, H, d, n)  # [B, 5, H, 8*8*64/H, 8*8]

        unfold_q = unfold_q.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 8*8]
        unfold_k = unfold_k.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 8*8]
        unfold_v = unfold_v.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 8*8]

        unfold_q = unfold_q.view(b, H, d, t * n)  # [B, H, 8*8*64/H, 5*8*8]
        unfold_k = unfold_k.view(b, H, d, t * n)  # [B, H, 8*8*64/H, 5*8*8]
        unfold_v = unfold_v.view(b, H, d, t * n)  # [B, H, 8*8*64/H, 5*8*8]

        attn = torch.matmul(unfold_q.transpose(2, 3), unfold_k)  # [B, H, 5*8*8, 5*8*8]
        attn = attn * (d ** (-0.5))  # [B, H, 5*8*8, 5*8*8]
        attn = F.softmax(attn, dim=-1)  # [B, H, 5*8*8, 5*8*8]

        attn_x = torch.matmul(attn, unfold_v.transpose(2, 3))  # [B, H, 5*8*8, 8*8*64/H]
        attn_x = attn_x.view(b, H, t, n, d)  # [B, H, 5, 8*8, 8*8*64/H]
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()  # [B, 5, H, 8*8*64/H, 8*8]
        attn_x = attn_x.view(b * t, D, n)  # [B*5, 8*8*64, 8*8]
        feat = self.patch2feat(attn_x)  # [B*5, 64, 64, 64]

        out = self.conv(feat).view(x.shape)  # [B, 5, 64, 64, 64]
        out += x  # [B, 5, 64, 64, 64]

        return out

