import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from torchvision import models
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList, Instances, Boxes


class AlexNetExtractor(nn.Module):
    def __init__(self):
        super(AlexNetExtractor, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet = self.alexnet.features[:11]

        for idx in range(7):
            for param in self.alexnet[idx].parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.alexnet(x)
        return {"conv5": x}

    def output_shape(self):
        return {"conv5": ShapeSpec(channels=256, stride=21)}


class SingleObjectDetector(nn.Module):
    def __init__(self, cfg, writer=None):
        super(SingleObjectDetector, self).__init__()
        self.cfg = cfg
        self.writer = writer
        self.alexnet = AlexNetExtractor()
        self.rpn = build_proposal_generator(cfg, self.alexnet.output_shape())

    def forward(self, data, train=True, data_idx=0, tmode=True):

        if train:
            search_img, exemplar_img, label, bbox_old = data['search'].cuda(), data['exemplar'].cuda(),\
                                                        data['label'].cuda(), data['bbox']
            s_img_list = ImageList(
                search_img,
                image_sizes=[
                    (self.cfg.TRAIN_SEARCH_H, self.cfg.TRAIN_SEARCH_W)
                ] * self.cfg.SOLVER.IMS_PER_BATCH)  # List[Tuple[int, int]]
            t_img_list = ImageList(
                exemplar_img,
                image_sizes=[
                    (self.cfg.TRAIN_TEMPLATE_H, self.cfg.TRAIN_TEMPLATE_W)
                ] * self.cfg.SOLVER.IMS_PER_BATCH)

            s_gt_instances = []
            bbox = Boxes(
                list(zip(bbox_old.x1, bbox_old.y1, bbox_old.x2,
                         bbox_old.y2))).to(torch.device('cuda'))
            for idx3, i3 in enumerate(range(search_img.shape[0])):
                s_gt = Instances(
                    (self.cfg.TRAIN_SEARCH_H,
                     self.cfg.TRAIN_SEARCH_W)).to(torch.device('cuda'))
                s_gt.gt_boxes = bbox[idx3]
                s_gt.label = [1] if label[idx3].cpu().numpy() == 1 else [0]
                s_gt_instances.append(s_gt)
        else:
            search_img, exemplar_img = data['search'].cuda(
            ), data['exemplar'].cuda()
            s_img_list = ImageList(search_img,
                                   image_sizes=[(self.cfg.TRAIN_SEARCH_H,
                                                 self.cfg.TRAIN_SEARCH_W)])
            t_img_list = ImageList(exemplar_img,
                                   image_sizes=[(self.cfg.TRAIN_TEMPLATE_H,
                                                 self.cfg.TRAIN_TEMPLATE_W)])
            s_gt_instances = None
            self.rpn.training = False

        search_features = self.alexnet(search_img)
        template_features = self.alexnet(exemplar_img)

        if tmode and data_idx % 100 == 0 and self.writer is not None:
            self.writer.add_images('search_images', search_img * 255, data_idx)
            self.writer.add_images('template_images', exemplar_img * 255,
                                   data_idx)

        proposals, losses = self.rpn(s_img_list.to(torch.device('cuda')),
                                     search_features,
                                     template_features,
                                     s_gt_instances,
                                     writer=self.writer,
                                     data_idx=data_idx,
                                     tmode=tmode)
        return proposals, losses
