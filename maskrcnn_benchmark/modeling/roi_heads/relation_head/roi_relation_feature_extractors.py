# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
import torch.fft as fft

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor

from torchvision.ops import deform_conv2d
from mmcv.ops import DeformConv2dPack as DCN

from maskrcnn_benchmark.modeling.roi_heads.relation_head.net_dwt import ResNet50

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
    

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.unsqueeze(dim=1).expand(-1, c, -1, -1, -1).view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).view(b, c, h, w, N)
        return x_offset


    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.view(b, c, h*ks, w*ks)
        return x_offset

        


    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                g_rb.unsqueeze(dim=1) * x_q_rb + \
                g_lb.unsqueeze(dim=1) * x_q_lb + \
                g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        *,
        offset_groups=1,
        with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class DSConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSConv, self).__init__()
        self.conv2d = depthwise_separable_conv(in_channel, out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)

        return x

class EnhanceMLP(nn.Module):
    def __init__(self, in_features=64*64, hidden_features=1024, act=nn.GELU, drop=0.1):
        super(EnhanceMLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act()
        self.fc2 = nn.Linear(hidden_features, hidden_features*4)
        self.drop = nn.Dropout(drop)
        self.layer = nn.LayerNorm(hidden_features*4)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.layer(x)

        return x

class FrequencyDomain(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self,ratio=0.5,rate=0.2,D=50,h=64,w=64,in_ch=1,out_ch=24,hid_dim=1024):
        super(FrequencyDomain, self).__init__()
        self.freq_transfer=EnhanceMLP()
        self.ratio=ratio
        self.rate=rate
        self.D=D
        self.h=h
        self.w=w
        self.conv0_1=nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv0_2=nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.conv1=nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.conv2=nn.Conv2d(out_ch, in_ch, 3, 1, 1)

        self.deconv1 = DCN(out_ch, out_ch, kernel_size=3, stride=1, padding=1, deform_groups=1)
        self.cnn_block=nn.Sequential(self.conv1,torch.nn.LeakyReLU(),self.conv2)        
        self.decnn_block1=nn.Sequential(self.conv0_1,torch.nn.LeakyReLU(),self.deconv1) 
        self.layer=nn.LayerNorm(self.h*self.w)


        self.dwtnet=ResNet50()
        self.expand=nn.Linear(128,4096)

        self.num=0

    
    def gaussian_filter_high_f(self, fshift, D):
        _, h, w = fshift.shape
        x, y = torch.meshgrid(torch.arange(h), torch.arange(w))
        center = (int((h - 1) / 2), int((w - 1) / 2))
        dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
        template = torch.exp(- dis_square / (2 * D ** 2)).to(fshift.device)

        return template * fshift
    
    def gaussian_filter_low_f(self, fshift, D):  #高通/低通高斯滤波
        _, h, w = fshift.shape
        x, y = torch.meshgrid(torch.arange(h), torch.arange(w))
        center = (int((h - 1) / 2), int((w - 1) / 2))
        dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
        template = 1 - torch.exp(- dis_square / (2 * D ** 2)).to(fshift.device)
    
        return template * fshift
    
    def circle_filter_high_f(self, fshift, radius_ratio):
        filter_img = np.zeros(fshift.shape, np.uint8)
        row, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  
        radius = int(radius_ratio * fshift.shape[0] / 2)
        if len(fshift.shape) == 3:
            cv2.circle(filter_img , (row, col), radius, (1, 1, 1), -1)
        else:
            cv2.circle(filter_img , (row, col), radius, 1, -1)
        return filter_img * fshift
    
    
    def circle_filter_low_f(self, fshift, radius_ratio):
        filter_img = np.ones(fshift.shape, np.uint8)
        row, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
        radius = int(radius_ratio * filter_img.shape[0] / 2)
        if len(filter_img.shape) == 3:
            cv2.circle(filter_img, (row, col), radius, (0, 0, 0), -1)
        else:
            cv2.circle(filter_img, (row, col), radius, 0, -1)
        return filter_img * fshift
    
    
    def ifft(self, fshift):
        """
        傅里叶逆变换
        """
        _, H,W=fshift.shape
        #ishift = fft.ifftshift(fshift)  # 把低频部分sift回左上角
        ishift=torch.roll(fshift,shifts=(H//2,W//2),dims=(1,2)) 
        iimg = fft.ifftn(ishift)  # 出来的是复数，无法显示
        iimg = torch.abs(iimg)    # 返回复数的模
        return iimg
    
    def get_low_high_f(self, img, radius_ratio, D):
        """
        获取低频和高频部分图像
        """
        # 傅里叶变换
        _, H,W=img.shape
        f = fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
        fshift = torch.roll(f,shifts=(-H//2,-W//2),dims=(1,2)) 
        #fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频
    
        # 获取低频和高频部分
        # hight_parts_fshift = circle_filter_low_f(fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
        # low_parts_fshift = circle_filter_high_f(fshift.copy(), radius_ratio=radius_ratio)
        hight_parts_fshift =  self.gaussian_filter_low_f(fshift, D=D)
        low_parts_fshift = self.gaussian_filter_high_f(fshift, D=D)
    
        low_parts_img = self.ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
        high_parts_img = self.ifft(hight_parts_fshift)
    
        # 显示原始图像和高通滤波处理图像
        img_new_low = (low_parts_img - torch.min(low_parts_img)) / (torch.max(low_parts_img) - torch.min(low_parts_img) + 0.00001)   #归一化 
        img_new_high = (high_parts_img - torch.min(high_parts_img) + 0.00001) / (
                    torch.max(high_parts_img) - torch.min(high_parts_img) + 0.00001)
    
        # uint8
        # img_new_low = np.array(img_new_low * 255, np.uint8)
        # img_new_high = np.array(img_new_high * 255, np.uint8)
        return img_new_low, img_new_high 



    def forward(self, ubox, images):
        '''
        images: 3 * h * w
        ''' 
        box_images=[]
        if ubox.size(0)==0:
            return torch.tensor([]).to(images.device)
        for i in range(ubox.size(0)):
            ubox[i]=torch.clamp(ubox[i],0,10000)
            box_image = images[:,torch.floor(ubox[i,1]).int():torch.floor(ubox[i,3]).int(),torch.floor(ubox[i,0]).int():torch.floor(ubox[i,2]).int()].permute(1,2,0)

            box_image_gray = (0.114*box_image[:,:,0]+0.587*box_image[:,:,1]+0.299*box_image[:,:,2])/255
            he,we=box_image_gray.shape   #灰度图
            box_image_gray=box_image_gray.unsqueeze(0).unsqueeze(1)
            box_image_gray=F.interpolate(box_image_gray, size=(self.h*(he//self.h+1),self.w*(we//self.w+1)),mode="bilinear").squeeze(0)
            pool2d = nn.AvgPool2d((he//self.h+1, we//self.w+1), padding=(0,0), stride=(he//self.h+1, we//self.w+1))
            box_image_grey =pool2d(box_image_gray.float()) 
            box_images.append(box_image_grey.squeeze(0))
        
        box_gray_image=torch.stack(box_images, dim=0)
        
        fmode="fft"
        if fmode=="dwt":  
            union_freq_feature=self.dwtnet(box_gray_image.unsqueeze(1))
            union_freq_feature=self.expand(union_freq_feature)
        if fmode=="fft":  
            low_freq_part_img, high_freq_part_img = self.get_low_high_f(box_gray_image, self.ratio, self.D)  

            freq_feature_rep=self.cnn_block(self.decnn_block1(high_freq_part_img.unsqueeze(1).to(ubox.device)))
            fres=(freq_feature_rep.squeeze(1)+high_freq_part_img).view(-1,self.h*self.w)
            union_freq_feature=self.freq_transfer(fres)

        return union_freq_feature




@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separete spatial     
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim//2), nn.ReLU(inplace=True),
                                              make_fc(out_dim//2, out_dim), nn.ReLU(inplace=True),
                                            ])

        # union rectangle size
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            ])
        self.freq_domain=FrequencyDomain()
        

    def forward(self, x, proposals, images, rel_pair_idxs=None):
        device = x[0].device
        union_proposals = []
        union_freq_features = []
        rect_inputs = []
        for i, (proposal, rel_pair_idx) in enumerate(zip(proposals, rel_pair_idxs)):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            #union_proposal = union_proposal.resize((self.rect_size, self.rect_size))
            ubox = union_proposal.bbox
            union_freq_feature=self.freq_domain(ubox, images[i])

            union_proposals.append(union_proposal)
            union_freq_features.append(union_freq_feature)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            # resize bbox to the scale rect_size
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
            head_rect = ((dummy_x_range >= head_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= head_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= head_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= head_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= tail_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= tail_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= tail_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()

            rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 4, rect_size, rect_size)
            rect_inputs.append(rect_input)

        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_freq_features=torch.cat(union_freq_features, dim=0)
        rect_inputs = torch.cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        # merge two parts
        if self.separate_spatial:
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + rect_features
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)
            
        return union_features, union_freq_features


def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
