from detectron2.modeling import RPN_HEAD_REGISTRY
from PIL import Image
import torch
from torch import nn as nn
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs, find_top_rpn_proposals
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
import torch.nn.functional as F
import torchvision.transforms.functional as TF


@RPN_HEAD_REGISTRY.register()
class SotRPNHead(nn.Module):
    """
    RPN cls and loc convolution head
    """
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.cfg = cfg

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(
            set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        num_cell_anchors = [
            len(self.cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]) *
            len(self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0])
        ]
        box_dim = 4
        assert (len(set(num_cell_anchors)) == 1
                ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.search_conv_cls = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(in_channels), nn.ReLU())

        self.search_conv_loc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(in_channels), nn.ReLU())

        self.template_conv_cls = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels * num_cell_anchors,
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(in_channels * num_cell_anchors), nn.ReLU())

        self.template_conv_loc = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels * box_dim * num_cell_anchors,
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(in_channels * box_dim * num_cell_anchors),
            nn.ReLU())

        for layer in [
                l for seq in [
                    self.template_conv_cls, self.template_conv_loc,
                    self.search_conv_cls, self.search_conv_loc
                ] for l in seq[:2]
        ]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features, t_s):
        """

        Args:
            features: Embedding features
            t_s: ["template", "search"]

        Returns:
            cls and loc rpn head results
        """
        if t_s == "template":
            t_cls = self.template_conv_cls(features)
            t_loc = self.template_conv_loc(features)
            return t_cls, t_loc
        elif t_s == "search":
            s_cls = self.search_conv_cls(features)
            s_loc = self.search_conv_loc(features)
            return s_cls, s_loc
        else:
            raise ValueError("proper mode not provided...")


@PROPOSAL_GENERATOR_REGISTRY.register()
class CorrelationalRPN(RPN):
    """
    Co-rrelational RPN
    """
    def __init__(self, cfg, input_shape):
        super(CorrelationalRPN, self).__init__(cfg, input_shape)
        self.cfg = cfg
        self.loc_adjust = nn.Conv2d(
            4 * len(self.cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]) *
            len(self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]),
            4 * len(self.cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]) *
            len(self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]),
            kernel_size=1,
            stride=1)

        self.conv_adjust = nn.Conv2d(
            len(self.cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]) *
            len(self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]),
            len(self.cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]) *
            len(self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]),
            kernel_size=1,
            stride=1)

    def fcorr(self, x, kernel):
        """group conv2d to calculate cross correlation, fast version"""

        # print(f"x shape {x.shape} and kernel {kernel.shape}")
        batch = kernel.size()[0]
        pk = kernel.view(-1,
                         x.size()[1],
                         kernel.size()[2],
                         kernel.size()[3])  # NOTE: x.size!!!
        px = x.view(1, -1, x.size()[2], x.size()[3])

        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def xcorr_slow(self, x, kernel):
        """for loop to calculate cross correlation, slow version
        """
        batch = x.size()[0]
        out = []
        for i in range(batch):
            px = x[i]
            pk = kernel[i]
            px = px.view(1, -1, px.size()[1], px.size()[2])
            pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
            po = F.conv2d(px, pk)
            out.append(po)
        out = torch.cat(out, 0)
        return out

    def forward(self,
                s_images,
                s_features,
                t_features,
                gt_instances=None,
                writer=None,
                data_idx=None,
                tmode=True):

        gt_boxes = [x.gt_boxes for x in gt_instances
                    ] if gt_instances is not None else None
        gt_labels = [x.label for x in gt_instances
                     ] if gt_instances is not None else None

        # in Training mode
        if gt_boxes is not None:
            gt_obj_logits = torch.Tensor(gt_labels).reshape(
                (self.cfg.SOLVER.IMS_PER_BATCH, 1))
        else:
            gt_obj_logits = None
        del gt_instances
        s_features = [s_features[f] for f in self.in_features]
        t_features = [t_features[f] for f in self.in_features]

        s_cls, s_loc = self.rpn_head(s_features[0], "search")
        t_cls, t_loc = self.rpn_head(t_features[0], "template")

        pred_objectness_logits = self.fcorr(s_cls, t_cls)

        if tmode and data_idx % 100 == 0 and writer is not None:
            # Create heatmap image in red channel
            heatmap = torch.nn.functional.interpolate(
                torch.sum(pred_objectness_logits, dim=1, keepdim=True),
                (511, 511)).cpu()

            for idx2 in range(len(s_images)):
                heatmap_i = heatmap[idx2]
                o_image = s_images[idx2]

                heatmap_i = torch.cat((heatmap_i, heatmap_i, heatmap_i), dim=0)

                heatmap_i = TF.to_pil_image(heatmap_i *
                                            255)  # assuming your image in x
                o_image = TF.to_pil_image(o_image.cpu())

                res = Image.blend(o_image, heatmap_i, 0.5)
                res = TF.to_tensor(res)
                o_image = TF.to_tensor(o_image)
                heatmap_i = TF.to_tensor(heatmap_i)
                writer.add_image(
                    f'pred_objectness_logits_{str(data_idx)}_{str(idx2)}/original',
                    o_image, data_idx)
                writer.add_image(
                    f'pred_objectness_logits_{str(data_idx)}_{str(idx2)}/heatmap',
                    heatmap_i, data_idx)
                writer.add_image(
                    f'pred_objectness_logits_{str(data_idx)}_{str(idx2)}/blended',
                    res, data_idx)

        pred_anchor_deltas = self.fcorr(s_loc, t_loc)
        pred_objectness_logits = pred_objectness_logits
        pred_anchor_deltas = self.loc_adjust(pred_anchor_deltas)

        pred_objectness_logits = [pred_objectness_logits]
        pred_anchor_deltas = [pred_anchor_deltas]

        anchors = self.anchor_generator(pred_anchor_deltas)
        outputs = RPNOutputs(self.box2box_transform, self.anchor_matcher,
                             self.batch_size_per_image, self.positive_fraction,
                             s_images, pred_objectness_logits,
                             pred_anchor_deltas, anchors,
                             self.boundary_threshold, gt_boxes,
                             self.smooth_l1_beta, gt_obj_logits)

        if self.training:
            losses = {
                k: v * self.loss_weight
                for k, v in outputs.losses().items()
            }
        else:
            losses = {}

        with torch.no_grad():
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                s_images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )

        return proposals, losses
