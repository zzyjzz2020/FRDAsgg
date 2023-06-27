# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext

 
from .model_transformer import TransformerContext    
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .vae import VAE

from .utils_motifs import obj_edge_vectors
import json
import torch.distributions as tdist
import collections


num_relation = 51

# count_total = torch.zeros((1,num_relation), requires_grad = False)
# count_total2 = torch.zeros((1,num_relation), requires_grad = False)
transfer_dim=1024
class CovEstimator():
    def __init__(self, feature_dim, class_num):
        super(CovEstimator, self).__init__()
        self.class_num = class_num
        self.CoVariance = (torch.rand(class_num, feature_dim)*0.01).cuda()
        self.Ave = torch.zeros(class_num, feature_dim).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)
        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1
        ave_CxA = features_by_sort.sum(0) / Amount_CxA 
        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)
        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)
        var_temp=torch.where(var_temp == torch.tensor(0,dtype=torch.float32,device=var_temp.device),
        torch.tensor(0.0000000001,dtype=torch.float32,device=var_temp.device), var_temp)


        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1).expand(C, A)+0.0000000001)
        weight_CV[weight_CV != weight_CV] = 0
        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))
        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)).detach() + additional_CV.detach()
        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()
        self.Amount += onehot.sum(0)
        


class BalanceLabelAugmentation(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(BalanceLabelAugmentation, self).__init__()

        self.estimator = CovEstimator(feature_dim, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()
        self.augs=torch.tensor([0.8,1,2])

    def data_generation(self,k,x,labels,ave,cov,groups):
        if len(groups[k])==0:
            aug_feat=torch.zeros(1,transfer_dim).to(x.device)
            aug_label=torch.zeros(1,1).to(x.device)
            return aug_feat,aug_label
        switch=torch.tensor([i in groups[k].to(x.device) for i in labels[0]]).to(x.device)
        switch=(labels*switch).squeeze(0)
        switch_out=torch.tensor([i not in labels[0] for i in groups[k].to(x.device)]).to(x.device)
        switch_out=(groups[k].to(x.device)*switch_out).view(-1)


        feat=x[switch!=0]
        label=labels[0,switch!=0]
        label_out=groups[k][switch_out!=0]

        if self.augs[k]<1:
            flag=torch.rand(1)
            if flag>self.augs[k]:
                aug_feat=torch.zeros(1,transfer_dim).to(x.device)
                aug_label=torch.zeros(1,1).to(x.device)
            else:
                aug_feat=feat
                aug_label=label.view(1,-1)
        else:
            if len(label_out)==0:
                aug_feat=feat
                aug_label=label.view(1,-1)
            else:
                n = tdist.Normal(ave[label_out],cov[label_out])

                aug_feats=n.sample((self.augs[k].int(),)).to(x.device)
                aug_feats=aug_feats.view(-1,transfer_dim)
                aug_labels=label_out.unsqueeze(0).repeat(self.augs[k].int(),1).view(1,-1)
                aug_feat=torch.cat((feat,aug_feats),dim=0)
                aug_label=torch.cat((label,aug_labels[0].to(x.device)),dim=0).view(1,-1)


        return aug_feat,aug_label

    def data_aug(self, x, labels, ave, cov, groups, ratio):
        aug_feat=[]
        aug_label=[]
        for i in range(3):
            part_feat,part_label=self.data_generation(i,x,labels,ave,cov,groups)

            aug_feat.append(part_feat)
            aug_label.append(part_label)
        aug_feat=torch.cat(aug_feat,dim=0)
        aug_label=torch.cat(aug_label,dim=1)
        

        return aug_feat,aug_label

    def forward(self, x_t, x, target_x, groups, fc, ba_coeffi, ratio=0.5):

        with torch.no_grad():
            self.estimator.update_CV(x, target_x)

        aug_f,aug_label = self.data_aug(x, target_x, self.estimator.Ave, self.estimator.CoVariance, groups, ratio)
        aug_y=fc(aug_f)

        aug_y=aug_y-ba_coeffi.to(aug_y.device)
        loss = self.cross_entropy(aug_y,aug_label.view(-1).long())

        return loss


class BalanceLabelAugmentation2(nn.Module):
    def __init__(self):
        super(BalanceLabelAugmentation2, self).__init__()

    def forward(self, feat, label, fc_o, fc, groups):
        feat_unlabel=feat[label==0,:]
        
        feat_o=feat[label!=0,:]
        label_o=label[label!=0].view(-1,1)

        feat_cat,label_cat=self.CrossMixup(feat_unlabel,fc_o,groups,feat_o,label_o)
        logits=fc(feat_cat)
        logits=F.softmax(logits,dim=1)
        loss=(-torch.log(logits)*label_cat).sum(1).mean()

        return loss
    def CrossMixup(self,feat,fc_o,groups,feat_o,label_o):
        with torch.no_grad():
            logits = fc_o(feat)
        logits_test = F.softmax(logits, dim=1)
        ranks=logits_test.sort(1,True)[1]
        score=logits_test.sort(1,True)[0]
        heads=[]
        mids=[]
        tails=[]

        for i in range(logits.shape[0]):
            if (ranks[i][0] in groups[0].to(logits.device)) and (score[i][0]>0.7):
                heads.append(i)
            if (ranks[i][0] in groups[1].to(logits.device)) and (score[i][0]>0.5): 
                mids.append(i)  
            if (ranks[i][0] in groups[2].to(logits.device)) and (score[i][0]>0.3): 
                tails.append(i)  

        mid=torch.tensor(mids).view(-1)
        mid_feat=feat[mid.long(),:].repeat(2,1)
        mid_label=ranks[mid.long(),0].repeat(2,1).view(-1,1)
        soft_mid_label=torch.zeros((mid_label.shape[0],num_relation),requires_grad=False,device=logits.device).scatter_(1,mid_label,1)

        tail=torch.tensor(tails).view(-1)
        tail_feat=feat[tail.long(),:].repeat(3,1)
        tail_label=ranks[tail.long(),0].repeat(3,1).view(-1,1)
        soft_tail_label=torch.zeros((tail_label.shape[0],num_relation),requires_grad=False,device=logits.device).scatter_(1,tail_label,1)

        soft_o_label=torch.zeros((label_o.shape[0],num_relation),requires_grad=False,device=logits.device).scatter_(1,label_o,1)

        feat_cat=[]
        label_cat=[]
        idx_m=torch.randint((feat_o.shape[0]),(mid_feat.shape[0],1)).view(-1)
        feat_cat.append(0.7*feat_o[idx_m,:]+0.3*mid_feat)
        label_cat.append(0.7*soft_o_label[idx_m,:]+0.3*soft_mid_label)
        idx_t=torch.randint((feat_o.shape[0]),(tail_feat.shape[0],1)).view(-1)
        feat_cat.append(0.7*feat_o[idx_t,:]+0.3*tail_feat)
        label_cat.append(0.7*soft_o_label[idx_t,:]+0.3*soft_tail_label)

        feat_cat=torch.cat(feat_cat,dim=0)
        label_cat=torch.cat(label_cat,dim=0)
        if(feat_cat.shape[0]==0):
            feat_cat=torch.rand(1,1024).to(feat.device)
            label_cat=torch.tensor([[0]]).to(feat.device)
            
        return feat_cat,label_cat




def adjust_sample(feat,label,feat_o,label_o,groups=None):

    feat_l_p = feat[label != 0, :]
    feat_nl_p = feat[label == 0, :]
    label_l_p = label[label != 0].view(1,-1)
    label_nl_p = label[label == 0].view(1,-1)
    tails=[]

    for i in range(feat_l_p.shape[0]):
        if ((label_l_p[0][i] in groups[1].to(feat_o.device)) or (label_l_p[0][i] in groups[2].to(feat_o.device))):
            tails.append(i)
 

    tail=torch.tensor(tails).view(-1)
    tail_feat=feat_l_p[tail.long(),:].repeat(2,1)
    tail_label=label_l_p[0,tail.long()].repeat(2,1).view(1,-1)
    
    feat_1 = torch.cat((feat_o[label_o != 0, :],tail_feat),dim=0)
    label_1 = torch.cat((label_o[label_o != 0].view(1,-1),tail_label),dim=1)

    feat_2 = torch.cat((feat_o,feat_l_p),dim=0)
    label_2 = torch.cat((label_o,label_l_p.view(-1)),dim=0)

    return feat_1,label_1,feat_2,label_2


def divide_group(feat,label,exset,imset):
    im=[]
    im_idx=[]
    im_lab=[]

    for i in range(label.shape[0]):
        
        if label[i] in imset.to(label.device):
            im.append(feat[i])
            im_lab.append(torch.where(imset.to(label.device)==label[i]))
            im_idx.append(i)
        else:
            im_idx.append(-1)

    if im==[]:
        im.append(torch.zeros(1,2048))
        im_lab.append(0)
    return torch.stack(im,dim=0).view(-1,2048),torch.tensor(im_lab).view(-1),torch.tensor(im_idx)


feat_dim=36
mid_dim=128
ind_dim=8
reduce_mode="bdc"
def divide_eigroup(feat,label,exset,imset):
    ex=[]
    im=[]
    ex_idx=[]
    im_idx=[]
    for i in range(label.shape[1]):

        if label[0,i] in exset:
            ex.append(feat[i])
            ex_idx.append(label[0,i])
        else:
            im.append(feat[i])
            im_idx.append(label[0,i])
            pass
    if im==[]:
        im.append(torch.zeros(1,mid_dim))
        im_idx.append(0)
    return torch.stack(ex,dim=0),torch.stack(im,dim=0).view(-1,mid_dim),torch.tensor(ex_idx),torch.tensor(im_idx)

def divide_set(feat,label,div=0.7):
    num=torch.floor(torch.tensor(feat.shape[0]*0.7)).to(int)
    ex_train=feat[:num]
    ex_test=feat[num:]
    ex_train_lab=label[:num]
    ex_test_lab=label[num:]

    return ex_train,ex_train_lab,ex_test,ex_test_lab
def divide_shift(feat,label,div=0.7):
    num=torch.floor(torch.tensor(feat.shape[0]*0.7)).to(int)
    idx = torch.randperm(feat.shape[0])
    feat_shuffle=feat[idx]
    label_shuffle=label[idx]
    ex_train=feat_shuffle[:num]
    ex_test=feat_shuffle[num:]
    ex_train_lab=label_shuffle[:num]
    ex_test_lab=label_shuffle[num:]


    return ex_train,ex_train_lab,ex_test,ex_test_lab

def meta_train_support(feat,label,indicator,mode):
    center=indicator(feat,label,mode,"sup")
    return center
def meta_train_query(feat,label,center,indicator,mode):
    label=label.view(-1)
    num=feat.shape[0]
    meta_feat=indicator(feat,label,mode,"que")
    feat_1=meta_feat.unsqueeze(1)
    feat_2=meta_feat.unsqueeze(0)
    center=center.unsqueeze(0)
    sim_to_center=torch.abs(torch.cosine_similarity(center,feat_1,dim=2))

    
    sim_to_sample=torch.abs(torch.cosine_similarity(feat_2,feat_1,dim=2))
    loss=0
    for i in range(num):
        loss_inter=(1-sim_to_sample[i,label==label[i]]).sum() 
        margin=0.5
        zero = torch.zeros_like(sim_to_sample[i,label!=label[i]])
        loss_mid = torch.where(sim_to_sample[i,label!=label[i]] > margin, sim_to_sample[i,label!=label[i]]-margin, zero)
        loss_intra=loss_mid.sum() 

        sim_dist=F.softmax(sim_to_center[i])
        loss = loss+loss_inter+loss_intra-torch.log(sim_dist[label[i]])

    return loss/num
def meta_test_support(feat,label,indicator,mode):
    center=indicator(feat,label,mode,"sup")
    return center



class BDC(nn.Module):
    def __init__(self, is_vec=True, input_dim=(1,6,32), activate='relu'):
        super(BDC, self).__init__()
        self.is_vec = is_vec
        self.activate = activate
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = BDCovpool(x)
        if self.is_vec:
            x = Triuvec(x)
        else:
            x = x.reshape(x.shape[0], -1)
        return x

def BDCovpool(x):
    batchSize, dim, M = x.shape  
    x = x.reshape(batchSize, dim, M)

    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
    I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
    x_pow2 = x.bmm(x.transpose(1, 2))
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    
    dcov = torch.clamp(dcov, min=0.0)
    temperature = nn.Parameter(torch.log((1. / (2 * dim* M)) * torch.ones(1,1))).to(dcov.device)
    dcov = torch.exp(temperature)* dcov
    dcov = torch.sqrt(dcov + 1e-5)
    t = dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)

    return t


def Triuvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y

class MetricIndicator(nn.Module):
    def __init__(self, n_components=feat_dim):    
        super(MetricIndicator, self).__init__()
        self.n_components = n_components
        self.bdc=BDC(is_vec=True)

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix.float())   
        idx = torch.argsort(-eigenvalues.to(torch.float16))         
        eigenvectors = eigenvectors[:, idx].to(torch.float16)
        self.proj_mat = eigenvectors[:, 0:self.n_components]  

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)
    def forward(self,X,label,mode="pca",sets="sup"):
        center=torch.zeros(num_relation,feat_dim).to(X.device)
        if mode=="pca":
            self.fit(X)
            trans_X = self.transform(X)
            if sets=="que":
                return trans_X
            idx=torch.unique(label)
            for i in idx:
                rep=trans_X[label==i].mean(0)
                center[i]+=rep
            return center
        if mode=="bdc":
            if sets=="que":
                indivisual_feat=X.reshape(X.shape[0],ind_dim,-1)
                indivisual_rep=self.bdc(indivisual_feat)
                return indivisual_rep
            for i in torch.unique(label):
                class_feat=X[(label==i).squeeze(0)]
                class_feat=class_feat.reshape(1,ind_dim,-1)
                center_rep=self.bdc(class_feat)
                center[i]+=center_rep

            return center





@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.config = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        # obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']   #todo
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        # assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.exset=torch.tensor([8,20,21,22,29,30,31,38,48,1,43,50])
            self.imset=torch.tensor([ 2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35,
                        36, 37, 39, 40, 41, 42, 44, 45, 46, 47, 49])
            self.unseen=torch.tensor([ 3,15,17,18,26,27,32,34,36,37,39,45])
        else:
            self.exset=torch.tensor(torch.tensor(torch.arange(1,26)))
            self.imset=torch.tensor(torch.tensor(torch.arange(26,101)))
            self.unseen=torch.tensor([ 3,15,17,18,26,27,32,34,36,37,39,45])
        self.imnum=self.imset.shape[0]
        self.unnum=self.unseen.shape[0]
        self.metric_indicator=MetricIndicator()
        self.ct=nn.Embedding(num_relation,feat_dim)
        ct_ini=torch.zeros(num_relation,feat_dim)
        with torch.no_grad():
            self.ct.weight.copy_(ct_ini, non_blocking=True)
        self.ct.weight.requires_grad_(False)


        self.indicator1 = nn.Linear(1024,1024)
        self.indicator2 = nn.Linear(1024,1024)   
        self.indicator3 = nn.Linear(1024,1024)
        self.indicator4 = nn.Linear(1024,mid_dim)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.pooling_dim,self.hidden_dim * 2)
        self.post_cat1 = nn.Linear(self.pooling_dim,self.hidden_dim * 2)
        self.rel_compress = nn.Linear(self.hidden_dim * 4, self.imnum)
        self.rel_compress2 = nn.Linear(self.hidden_dim * 4, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed_ini = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed_ini.weight.copy_(embed_vecs, non_blocking=True)
        self.embed_transfer1 = nn.Linear(200 *2 ,512)
        self.embed_transfer2 = nn.Linear(1024 ,512)
        self.embed_transfer3 = nn.Linear(1024 ,1024)

        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.rel_compress2, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        # layer_init(self.frq_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.post_cat1, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)



        self.info_ini=torch.ones(1,num_relation, requires_grad=False)  
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            info = json.load(open("datasets/vg/VG-SGG-dicts-with-attri.json", 'r'))
            count = info['predicate_count']
        else:
            info = json.load(open("datasets/gqa/GQA_200_ID_Info.json", 'r'))
            count = info['rel_count']
            count.pop("__background__")
        sort_value=sorted(count.values())
        total=torch.tensor(sort_value).sum()
        sort_dict=dict(sorted(count.items()))
        with torch.no_grad():
            for id,i in enumerate(sort_dict.keys()):
                self.info_ini[0,id+1]=torch.sqrt(torch.tensor(sort_dict[i]/total))
            self.info_ini=-torch.log(self.info_ini)/2
        self.alpha=nn.Embedding(1,num_relation,_weight=self.info_ini)
        self.beta=nn.Embedding(1,num_relation)

        self.transfer_sub=nn.Linear(512,512,bias=True)
        self.transfer_obj=nn.Linear(512,512,bias=True)
        shared_weight=torch.normal(0, 0.1, size=(512,512), requires_grad=True)     
        self.transfer_sub.weight = nn.Parameter(shared_weight)
        self.transfer_obj.weight = nn.Parameter(shared_weight)
        self.transfer_rel=nn.Linear(1024,transfer_dim,bias=True)
        self.balance_compress=nn.Linear(transfer_dim,num_relation)
        layer_init(self.balance_compress, xavier=True)
        self.balance_compress_mirror=nn.Linear(transfer_dim,num_relation)
        layer_init(self.balance_compress_mirror, xavier=True)
        self.pesudo_aug=BalanceLabelAugmentation(transfer_dim,num_relation)
        self.balance_compress_2=nn.Linear(transfer_dim,num_relation)
        layer_init(self.balance_compress_2, xavier=True)
        self.pesudo_aug2=BalanceLabelAugmentation2()

        self.transfernet1 = nn.Linear(1024,4096,bias=True)
        self.transfernet2 = nn.Linear(4096,1024,bias=True)
        self.vote1 = nn.Linear(num_relation,num_relation,bias=True)
        self.vote2 = nn.Linear(num_relation,num_relation,bias=True)
        basel1 = torch.ones((num_relation,num_relation),requires_grad=True)
        basel2 = torch.ones((num_relation,num_relation),requires_grad=True)
        self.vote1.weight = nn.Parameter(0.7*basel1)
        self.vote2.weight = nn.Parameter(0.3*basel2)

        self.vae=VAE()




        if True:
            pred_adj_np = np.load('./conf_mat_freq_train.npy')
            # pred_adj_np = 1.0 - pred_adj_np
            pred_adj_np[0, :] = 0.0
            pred_adj_np[:, 0] = 0.0
            pred_adj_np[0, 0] = 1.0
            # adj_i_j means the baseline outputs category j, but the ground truth is i.
            pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
            pred_adj_np = self.adj_normalize(pred_adj_np)
            self.pred_adj = torch.from_numpy(pred_adj_np).float()

    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        mx_output = r_mat_inv.dot(mx)
        return mx_output

    def adj_normalize(self,adj):
        adj = self.normalize(adj + np.eye(adj.shape[0]))
        return adj        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, union_freq_features, enhance, groups, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)


        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        pair_semas = []
        loss_neigh1=0
        rwt=False   

        if self.config.MODEL.DEBIAS_MUTILLABEL:
            sub_reps=[]
            obj_reps=[]
            for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
                prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
                pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
                pair_semas.append(torch.cat((self.obj_embed_ini(pair_idx[:,0]), self.obj_embed_ini(pair_idx[:,1])), dim=-1))
                sub_reps.append(head_rep[pair_idx[:,0]])
                obj_reps.append(tail_rep[pair_idx[:,1]])
            prod_rep = cat(prod_reps, dim=0)
            pair_pred = cat(pair_preds, dim=0)
            pair_sema = cat(pair_semas, dim=0)

            pair_rep=prod_rep


            if self.training:
                rel_labels=torch.cat(rel_labels,dim=0)

                new_feats,loss_neigh1 = self.vae(pair_rep)  
                new_feats = new_feats*0.3 + pair_rep*0.7
                new_labels = rel_labels
                feat_1,label_1,feat_2,label_2 = adjust_sample(new_feats,new_labels,pair_rep,rel_labels,groups)


                # 伪标签分支1
                loss_neigh1+=self.pesudo_aug(pair_rep,feat_1,label_1,groups,self.balance_compress,self.info_ini)
                # 伪标签分支2
                loss_neigh1+=self.pesudo_aug2(feat_2,label_2,self.balance_compress,self.balance_compress_2,groups)
                
                
                #动态融合
                res = self.vote1(self.balance_compress(pair_rep).detach())+self.vote2(self.balance_compress_2(pair_rep).detach())
                res = res + self.freq_bias.index_with_rwt_labels(pair_pred,gt=rel_labels)
                cross_entropy = nn.CrossEntropyLoss()
                loss_neigh1+=5*cross_entropy(res,rel_labels.view(-1).long()) 
                

                rel_labels = rel_labels.split(num_rels, dim=0)
                pass
            else:
                pseudo_dist=self.vote1(self.balance_compress(pair_rep))+self.vote2(self.balance_compress_2(pair_rep)) + \
                self.freq_bias.index_with_labels(pair_pred) 
                pass

        else:
            for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
                prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
                pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
                pair_semas.append(torch.cat((self.obj_embed_ini(pair_idx[:,0]), self.obj_embed_ini(pair_idx[:,1])), dim=-1))
            prod_rep = cat(prod_reps, dim=0)
            pair_pred = cat(pair_preds, dim=0)
            pair_sema = cat(pair_semas, dim=0)


        if True:
            pair_sema=self.embed_transfer1(pair_sema)
            prod_reps=self.embed_transfer2(prod_rep)
            ub_prod_rep=self.context_layer.decoder_edge(prod_reps, num_rels, pair_sema)
            prod_reps=torch.cat((prod_reps,ub_prod_rep),dim=1)
            prod_rep=self.embed_transfer3(prod_reps)


        union_features = self.post_cat(union_features)
        union_freq_features = self.post_cat1(union_freq_features)

        
        spat_freq_rep = torch.cat((union_features,union_freq_features),dim=1)
        loss_freq=0
        loss_metric=0
        loss_neigh2=0
        if self.training:
            rel_dists2 = self.ctx_compress(prod_rep) + 0.1*self.freq_bias.index_with_labels(pair_pred)  
            if self.config.MODEL.DEBIAS_MUTILLABEL:
                rel_dists2 = self.vote1(self.balance_compress(pair_rep))+self.vote2(self.balance_compress_2(pair_rep)) \
                + self.freq_bias.index_with_labels(pair_pred)
            rel_dists = rel_dists2
            if self.config.MODEL.FREQUENCYBRANCH:
                if not self.config.MODEL.DEBIAS_MUTILLABEL:
                    rel_labels=torch.cat(rel_labels,dim=0)
                    spat_freq_rep,spat_freq_lab,spat_freq_idx = divide_group(spat_freq_rep,rel_labels,self.exset,self.imset)
                    rel_dists1 = self.rel_compress(spat_freq_rep.to(union_freq_features.device)) 
                    
                    criterion_loss=nn.CrossEntropyLoss()
                    loss_freq = criterion_loss(rel_dists1, spat_freq_lab.long().to(union_freq_features.device))
                    rel_labels = rel_labels.split(num_rels, dim=0)
                   
                else:
                    rel_labels=torch.cat(rel_labels,dim=0)
                    rel_dists = self.ctx_compress(prod_rep) + self.rel_compress2(spat_freq_rep.to(union_freq_features.device)) 
                    rel_dists += self.freq_bias.index_with_rwt_labels(pair_pred,gt=rel_labels)
                    criterion_loss=nn.CrossEntropyLoss()
                    loss_freq = criterion_loss(rel_dists, rel_labels.long().to(union_freq_features.device))
                    rel_labels = rel_labels.split(num_rels, dim=0)

                    # rel_labels=torch.cat(rel_labels,dim=0)
                    # spat_freq_rep,spat_freq_lab,spat_freq_idx = divide_group(spat_freq_rep,rel_labels,self.exset,self.imset)
                    # rel_dists1 = self.rel_compress(spat_freq_rep.to(union_freq_features.device)) 
                    # criterion_loss=nn.CrossEntropyLoss()
                    # loss_freq = criterion_loss(rel_dists1, spat_freq_lab.long().to(union_freq_features.device))
                    # rel_dists = self.ctx_compress(prod_rep) + self.freq_bias.index_with_rwt_labels(pair_pred,gt=rel_labels)
                    # loss_freq += criterion_loss(rel_dists, rel_labels.long().to(union_freq_features.device))
                    # rel_labels = rel_labels.split(num_rels, dim=0)


            if self.config.MODEL.MMLEARNING:
                edge_feat=prod_rep.clone().detach()   
                rel_labels=torch.cat(rel_labels,dim=0)
                Yt=rel_labels[rel_labels!=0].view(1,-1)
                edge_pos=edge_feat[rel_labels!=0,:]
                edge_pos=self.indicator4(self.indicator3(self.indicator2(self.indicator1(edge_pos))+edge_pos))

                ex,im,ex_label,im_label=divide_eigroup(edge_pos,Yt,self.exset.to(Yt.device),self.imset.to(Yt.device))
                ex_train,ex_train_label,ex_test,ex_test_label=divide_shift(ex,ex_label)   
                center_rep=meta_train_support(ex_train,ex_train_label,self.metric_indicator,reduce_mode)
                self.ct.weight=self.ct.weight.to(edge_pos.device) 
                with torch.no_grad():
                    exnum=torch.unique(ex_train_label)
                    for idx_num in exnum:
                        self.ct.weight[idx_num]=0.9*self.ct.weight[idx_num]+0.1*center_rep[idx_num]
                im_center_rep=meta_test_support(im.to(edge_pos.device),im_label,self.metric_indicator,reduce_mode)
                with torch.no_grad():  
                    imnum=torch.unique(im_label)
                    for idx_num in imnum:
                        self.ct.weight[idx_num]=0.7*self.ct.weight[idx_num]+0.3*im_center_rep[idx_num]
                loss_metric=meta_train_query(ex_test,ex_test_label,self.ct.weight,self.metric_indicator,reduce_mode)
                
                rel_labels = rel_labels.split(num_rels, dim=0)
                # torch.save(self.ct.weight,"center.pt")


            if rwt:
                finalcenter=torch.load("center.pt")
                c1=finalcenter.unsqueeze(1)
                c2=finalcenter.unsqueeze(0)
                sim=torch.abs(torch.cosine_similarity(c2,c1,dim=2))
                idx_c=sim.sort(1,True)[1]
                rel_dists_new=rel_dists.clone().detach()
                rel_dists_bas=rel_dists.clone().detach()
                rel_dists_bias=rel_dists.clone().detach()
                rel_dists_new=rel_dists_bas*self.alpha.weight+ self.beta.weight
            
                infoes=rel_dists_bas*self.info_ini.to(rel_dists.device)

                k=7
                idx_re=idx_c[:,1:k+1]
                seq_info=infoes.sort(1,True)[1]
                seq_prod=rel_dists_bas[:,1:].sort(1,True)[1]+1
                possible=self.freq_bias.index_with_labels(pair_pred)-self.freq_bias.index_with_labels(pair_pred).min(1)[0].view(-1,1)
                self.unseen=groups[2].to(rel_dists.device)
                self.info_ini=self.info_ini.to(rel_dists.device)
                rel_dists_bias[:,self.unseen]=rel_dists_bas[:,self.unseen]*(possible[:,self.unseen]>0)* self.info_ini[0,self.unseen]* self.info_ini[0,self.unseen]

            
                
                for i in range(rel_dists.shape[0]):
                    for j in range(k):
                        if (seq_prod[i,j+1] in idx_re[seq_prod[i,0]]) and (infoes[i,seq_prod[i,j+1]]>infoes[i,seq_prod[i,0]]):
                            rel_dists_bias[i,seq_prod[i,j+1]] = ((rel_dists_bas[i,seq_prod[i,j+1]]*self.info_ini[0,seq_prod[i,j+1]])
                            if possible[i,seq_prod[i,j+1]]>0 else 1) 

                
                loss_neigh2=F.kl_div(rel_dists_new.softmax(dim=-1).log(), rel_dists_bias.softmax(dim=-1), reduction='mean')
                +0.5*(1/self.alpha.weight.mean()-torch.log(self.beta.weight.mean()))
          



        if not self.training:
            rel_dists=0
            rel_dists2=0
            if self.config.MODEL.FREQUENCYBRANCH:
                if not self.config.MODEL.DEBIAS_MUTILLABEL:
                    rel_dists1 = self.rel_compress(spat_freq_rep) 
                    rel_dists2 = self.ctx_compress(prod_rep) + 0.1*self.freq_bias.index_with_labels(pair_pred) 
                    r1=F.softmax(rel_dists1)
                    r1_total=torch.zeros(rel_dists1.shape[0],num_relation).to(spat_freq_rep.device)
                    r1_total[:,self.imset]+=r1

                    r1_total[r1_total.max(1)[0]<0.5]=0   
                    rel_dists=2*r1_total + rel_dists2  
                    if self.config.MODEL.MMLEARNING and not self.config.MODEL.DEBIAS_MUTILLABEL:
                        rel_dists=3*r1_total + rel_dists2 
                    if self.config.MODEL.DEBIAS_MUTILLABEL:
                        rel_dists=r1_total + 0.1*rel_dists2
                else:
                    rel_dists1 = self.rel_compress2(spat_freq_rep) 
                    rel_dists2 = self.ctx_compress(prod_rep) + self.freq_bias.index_with_labels(pair_pred) 
                    rel_dists = rel_dists1 + 0.5*rel_dists2
                    if self.config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
                        rel_dists = rel_dists1 + 0.1*rel_dists2   



            
            # meta-learning part
            if self.config.MODEL.MMLEARNING:
                label=None
                feat_rep=self.indicator4(self.indicator3(self.indicator2(self.indicator1(prod_rep))+prod_rep))
                metric_vec=self.metric_indicator(feat_rep,label,reduce_mode,"que")
                metric_vec=metric_vec.unsqueeze(1)
                centers=self.ct.weight.unsqueeze(0).to(metric_vec.device)
                sim=torch.cosine_similarity(centers,metric_vec,dim=2)
                rel_dists += sim

            
            
            if rwt:     
                rel_dists=(rel_dists+10*rel_dists*self.alpha.weight + self.beta.weight)/11  
                pass
            if self.config.MODEL.DEBIAS_MUTILLABEL:
                if self.config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
                    rel_dists += 1*pseudo_dist
                if self.config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
                    rel_dists += 1.5*pseudo_dist    
                if rwt:  
                    rel_dists = (self.pred_adj.to(pseudo_dist.device) @ rel_dists.T).T
                

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        rel_dists2 = rel_dists2.split(num_rels, dim=0)
        

        add_losses = {}
        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists,loss_freq,rel_dists2, add_losses,loss_metric,loss_neigh1,loss_neigh2




@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        

        #add
        self.balance_compress=nn.Linear(transfer_dim,num_relation)
        layer_init(self.balance_compress, xavier=True)
        self.pesudo_aug=BalanceLabelAugmentation(transfer_dim,num_relation)
        self.balance_compress_2=nn.Linear(transfer_dim,num_relation)
        layer_init(self.balance_compress_2, xavier=True)
        self.pesudo_aug2=BalanceLabelAugmentation2()
        self.transfer_s=nn.Linear(self.pooling_dim,1024)

        self.vae=VAE()

        self.info_ini=torch.ones(1,num_relation, requires_grad=False)  
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            info = json.load(open("datasets/vg/VG-SGG-dicts-with-attri.json", 'r'))
            count = info['predicate_count']
        else:
            info = json.load(open("datasets/gqa/GQA_200_ID_Info.json", 'r'))
            count = info['rel_count']
            count.pop("__background__")
        sort_value=sorted(count.values())
        total=torch.tensor(sort_value).sum()
        sort_dict=dict(sorted(count.items()))
        with torch.no_grad():
            for id,i in enumerate(sort_dict.keys()):
                self.info_ini[0,id+1]=torch.sqrt(torch.tensor(sort_dict[i]/total))
            self.info_ini=-torch.log(self.info_ini)/2
        self.vote1 = nn.Linear(num_relation,num_relation,bias=True)
        self.vote2 = nn.Linear(num_relation,num_relation,bias=True)
        basel1 = torch.ones((num_relation,num_relation),requires_grad=True)
        basel2 = torch.ones((num_relation,num_relation),requires_grad=True)
        self.vote1.weight = nn.Parameter(0.7*basel1)
        self.vote2.weight = nn.Parameter(0.3*basel2)



    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, groups, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)



        
        prod_rep_hl = self.transfer_s(prod_rep)
        loss_neigh1 = 0
        #add
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0)
            new_feats,loss_neigh1 = self.vae(prod_rep_hl)  
            new_feats = new_feats*0.3 + prod_rep_hl*0.7
            new_labels = rel_labels
            feat_1,label_1,feat_2,label_2 = adjust_sample(new_feats,new_labels,prod_rep_hl,rel_labels,groups)
            # 伪标签分支1
            loss_neigh1+=self.pesudo_aug(prod_rep_hl,feat_1,label_1,groups,self.balance_compress,self.info_ini)
            # 伪标签分支2
            loss_neigh1+=self.pesudo_aug2(feat_2,label_2,self.balance_compress,self.balance_compress_2,groups)

            res = self.vote1(self.balance_compress(prod_rep_hl).detach())+self.vote2(self.balance_compress_2(prod_rep_hl).detach())
            res = res + self.freq_bias.index_with_rwt_labels(pair_pred,gt=rel_labels)
            cross_entropy = nn.CrossEntropyLoss()
            loss_neigh1+=5*cross_entropy(res,rel_labels.view(-1).long()) 

            rel_labels = rel_labels.split(num_rels, dim=0)

        frq_dists = self.freq_bias.index_with_rwt_labels(pair_pred,gt=rel_labels)

        if self.training:
            rel_dists = rel_dists + frq_dists
        else:       
            rel_dists = rel_dists + frq_dists \
                        + self.vote1(self.balance_compress(prod_rep_hl)) \
                        + self.vote2(self.balance_compress_2(prod_rep_hl))


        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses, loss_neigh1


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)


        #add
        self.balance_compress=nn.Linear(transfer_dim,num_relation)
        layer_init(self.balance_compress, xavier=True)
        self.pesudo_aug=BalanceLabelAugmentation(transfer_dim,num_relation)
        self.balance_compress_2=nn.Linear(transfer_dim,num_relation)
        layer_init(self.balance_compress_2, xavier=True)
        self.pesudo_aug2=BalanceLabelAugmentation2()
        self.transfer_s=nn.Linear(self.pooling_dim,1024)

        self.vae=VAE()

        self.info_ini=torch.ones(1,num_relation, requires_grad=False)  
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            info = json.load(open("datasets/vg/VG-SGG-dicts-with-attri.json", 'r'))
            count = info['predicate_count']
        else:
            info = json.load(open("datasets/gqa/GQA_200_ID_Info.json", 'r'))
            count = info['rel_count']
            count.pop("__background__")
        sort_value=sorted(count.values())
        total=torch.tensor(sort_value).sum()
        sort_dict=dict(sorted(count.items()))
        with torch.no_grad():
            for id,i in enumerate(sort_dict.keys()):
                self.info_ini[0,id+1]=torch.sqrt(torch.tensor(sort_dict[i]/total))
            self.info_ini=-torch.log(self.info_ini)/2
        self.vote1 = nn.Linear(num_relation,num_relation,bias=True)
        self.vote2 = nn.Linear(num_relation,num_relation,bias=True)
        basel1 = torch.ones((num_relation,num_relation),requires_grad=True)
        basel2 = torch.ones((num_relation,num_relation),requires_grad=True)
        self.vote1.weight = nn.Parameter(0.7*basel1)
        self.vote2.weight = nn.Parameter(0.3*basel2)


        if True:
            pred_adj_np = np.load('./conf_mat_freq_train.npy')
            # pred_adj_np = 1.0 - pred_adj_np
            pred_adj_np[0, :] = 0.0
            pred_adj_np[:, 0] = 0.0
            pred_adj_np[0, 0] = 1.0
            # adj_i_j means the baseline outputs category j, but the ground truth is i.
            pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
            pred_adj_np = self.adj_normalize(pred_adj_np)
            self.pred_adj = torch.from_numpy(pred_adj_np).float()
    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        mx_output = r_mat_inv.dot(mx)
        return mx_output

    def adj_normalize(self,adj):
        adj = self.normalize(adj + np.eye(adj.shape[0]))
        return adj  


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, groups, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        prod_rep_hl = self.transfer_s(prod_rep)
        loss_neigh1 = 0


        #add
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0)
            new_feats,loss_neigh1 = self.vae(prod_rep_hl)  
            new_feats = new_feats*0.3 + prod_rep_hl*0.7
            new_labels = rel_labels
            feat_1,label_1,feat_2,label_2 = adjust_sample(new_feats,new_labels,prod_rep_hl,rel_labels,groups)
            # 伪标签分支1
            loss_neigh1+=self.pesudo_aug(prod_rep_hl,feat_1,label_1,groups,self.balance_compress,self.info_ini)
            # 伪标签分支2
            loss_neigh1+=self.pesudo_aug2(feat_2,label_2,self.balance_compress,self.balance_compress_2,groups)
            res = self.vote1(self.balance_compress(prod_rep_hl).detach())+self.vote2(self.balance_compress_2(prod_rep_hl).detach())
            res = res + self.freq_bias.index_with_rwt_labels(pair_pred,gt=rel_labels)
            cross_entropy = nn.CrossEntropyLoss()
            loss_neigh1+=5*cross_entropy(res,rel_labels.view(-1).long())
            rel_labels = rel_labels.split(num_rels, dim=0)


        frq_dists_ori = self.freq_bias.index_with_labels(pair_pred.long())     
        frq_dists = self.freq_bias.index_with_rwt_labels(pair_pred,gt=rel_labels)
        if self.training:
            rel_dists = ctx_dists + frq_dists
            #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists
        else:   
            rel_dists = ctx_dists + frq_dists_ori \
                        + 0.5*self.vote1(self.balance_compress(prod_rep_hl)) \
                        + 0.5*self.vote2(self.balance_compress_2(prod_rep_hl))

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses, loss_neigh1


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
