# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import numpy as np

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor
from maskrcnn_benchmark.data import get_dataset_statistics
import json
from torch.nn import functional as F

#todo
mid_num = 35
num_relation = 51
# mid_num = 60
# num_relation = 101
class GCNlayers(torch.nn.Module):
    def __init__(self,in_channels, out_channels):
        super(GCNlayers, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self,x,adj):
        adj_mtx=adj+torch.eye(adj.shape[0],device=adj.device)
        degree=adj_mtx.sum(0).view(1,-1)
        d=torch.diag_embed(degree).view(adj.shape[0],adj.shape[0])

        x=self.lin(torch.mm(1/degree.sum().sqrt()* adj_mtx, x))
        return x

        
class RelationEnhance(torch.nn.Module):
    def __init__(self,dim):
        super(RelationEnhance, self).__init__()
        self.expand=nn.Linear(dim*2, dim*2)
        self.gcn=GCNlayers(dim*2,dim)

    def local_info(self,roi_feature, rel_pair_idxs):
        num_obj=roi_feature.shape[0]
        adjmtx=torch.zeros((num_obj,num_obj), dtype=torch.int64,device=roi_feature.device)  
        adjmtx[rel_pair_idxs[:,0],rel_pair_idxs[:,1]]=1  
        edge_local_rep=self.gcn(roi_feature,adjmtx)
        return edge_local_rep


    def global_info(self, edge_feature, mode="cnn"):  
        if mode=="cnn":
            edge_global_rep=self.expand(edge_feature)
            return edge_global_rep
            
        if mode=="dilated_cnn":
            pass


class Debias(torch.nn.Module):
    def __init__(self,config):
        super(Debias, self).__init__()
        self.config = config
        self.statistics = get_dataset_statistics(config)
        self.tranfer_matrix=nn.Linear(num_relation,num_relation, bias=False)    
        nn.init.kaiming_uniform_(self.tranfer_matrix.weight, a=1)
        self.tranfer_mid1=nn.Linear(num_relation,1024) 
        self.tranfer_mid2=nn.Linear(1024,num_relation) 

        self.ba_freq=torch.zeros((1,num_relation),requires_grad=False)
        self.ba = torch.nn.Linear(1,num_relation,bias=False)  
        w=torch.ones(1,num_relation, requires_grad=False)  
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            info = json.load(open("datasets/vg/VG-SGG-dicts-with-attri.json", 'r'))
            count = info['predicate_count']
        else:
            info = json.load(open("datasets/gqa/GQA_200_ID_Info.json", 'r'))
            count = info['rel_count']
            count.pop("__background__")
        sort_value=sorted(count.values())
        media=sort_value[mid_num]
        sort_dict=dict(sorted(count.items()))
        for id,i in enumerate(sort_dict.keys()):
            w[0,id+1]=torch.sqrt(torch.tensor(media/sort_dict[i]))
        self.ba.weight=nn.Parameter(w, requires_grad=False)


    def generation_head_tail(self,pred,thre=5000,thre2=20000,dict_file="datasets/vg/VG-SGG-dicts-with-attri.json"):
        if self.config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            info = json.load(open(dict_file, 'r'))
            count = info['predicate_count']
            idx = info['predicate_to_idx']
        else:
            info = json.load(open("datasets/gqa/GQA_200_ID_Info.json", 'r'))
            count = info['rel_count']
            count.pop("__background__")
            idx = info['rel_name_to_id']
            thre = 3000
        idxs_tail=[]
        idxs_head=[]
        idxs_media=[]
        active=[]
        deactive=[]
        active_flag=(pred[1:,1:].max()>0.3)
        max_idx=pred[1:,1:].max(1)[1]
        max_val=pred[1:,1:].max(1)[0]

        max_idx_3=pred[1:,1:].sort(1,True)[1][:,:3]
        is_match=(max_idx.cpu()==torch.arange(num_relation-1)).view(-1)
        is_match_3=((max_idx_3.cpu()==torch.arange(num_relation-1).unsqueeze(1).repeat(1,3)).sum(1)).view(-1)
        for k,v in count.items():
            if (v > thre2) & int(is_match[idx[k]-1]):
                idxs_head.append(idx[k])
                if active_flag & (max_val[idx[k]-1]<0.3): 
                    active.append(idx[k])
                else:
                    deactive.append(idx[k])
            elif (v > thre) & int(is_match_3[idx[k]-1]):
                idxs_media.append(idx[k])
                if active_flag & (max_val[idx[k]-1]<0.2): 
                    active.append(idx[k])
                else:
                    deactive.append(idx[k])
            else:
                idxs_tail.append(idx[k])
                if active_flag & (max_val[idx[k]-1]<0.1):
                    active.append(idx[k])
                else:
                    deactive.append(idx[k])

        return torch.tensor(idxs_head), torch.tensor(idxs_media), torch.tensor(idxs_tail), torch.tensor(active) if active!=[] else None, torch.tensor(deactive) if active!=[] else None
    

    def generate_exp_adj(self,pred,gt,rel_count,istrain):
        if not istrain:
            rel_prod=rel_count
            return rel_count,rel_prod.to(pred[0].device)

        gt=torch.cat(gt,dim=0)
        num=pred.shape[0]
        pred_idx=pred[:,1:].max(-1)[1]+1

        rel_count0=torch.zeros(num_relation,num_relation).to(rel_count.device)
        for i in range(num):
            rel_count0[pred_idx[i],gt[i]]+=1
        idx=rel_count0.sum(1)/num
        idx=torch.where(idx > 0.0, 0.9, 1.0).to(rel_count.device)
        rel_count0[:,0]=0
        if rel_count.sum()==0:
            rel_count=rel_count0/(rel_count0.sum(1)[:,None].repeat(1,num_relation)+0.0000000001)
        else:
            rel_count=idx.view(-1,1)*rel_count+0.1*rel_count0/(rel_count0.sum(1)[:,None].repeat(1,num_relation)+0.0000000001)    
        rel_prod=rel_count
        return rel_count,rel_prod.to(pred.device)


class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        self.enhance=RelationEnhance(512) 
        self.debias=Debias(cfg)


        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

    def forward(self, features, proposals, images, targets=None, logger=None, mode=None, rel_count=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals, mode)

        # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, proposals)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)
        if self.use_union_box:
            union_features, union_freq_features = self.union_feature_extractor(features, proposals, images, rel_pair_idxs)
        else:
            union_features = None
            union_freq_features = None
        
        groups=self.debias.generation_head_tail(rel_count)  
        refine_logits, relation_logits,l1,r2, add_losses,l2,l3,l4 = self.predictor(
            proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, union_freq_features, self.enhance, groups, logger)
        
        # vctree
        # refine_logits, relation_logits, add_losses, l3 = self.predictor(
        #     proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, groups, logger)
        # r2 = relation_logits
        # vctree

        num_rels = [r.shape[0] for r in relation_logits]
        r2=torch.cat(r2,dim=0)    
        relation_logits=torch.cat(relation_logits,dim=0)
        with torch.no_grad():
            rel_count, rel_prod=self.debias.generate_exp_adj(r2,rel_labels,rel_count,self.training)
        
        
        relation_logits = relation_logits.split(num_rels, dim=0)   
        r2 = r2.split(num_rels, dim=0)                             


        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}, rel_count

        loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, r2, refine_logits,self.debias.ba_freq)

        output_losses = dict(loss_rel= l1 + l2 + l3, loss_refine_obj=loss_refine)   

        output_losses.update(add_losses)

        return roi_features, proposals, output_losses, rel_count



def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
