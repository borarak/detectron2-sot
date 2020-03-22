import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from detectron2.utils.events import EventStorage
from dsot.dataset import COCODataset

from dsot.net import SingleObjectDetector
from detectron2.config import _CNode as CN

_C = CN()
cfg = _C.clone()

# @TODO: move config out to yaml
_C.MODEL = CN()
_C.MODEL.LOAD_PROPOSALS = False
_C.MODEL.MASK_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

_C.MODEL.RPN.IN_FEATURES = ["conv5"]
_C.MODEL.PROPOSAL_GENERATOR.NAME = "CorrelationalRPN"
_C.MODEL.RPN.HEAD_NAME = "SotRPNHead"
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 64

_C.SOLVER.IMS_PER_BATCH = 16  # Typical batch size
_C.IMAGE_FOLDER_PER_BATCH = 16  # Number of folders that will be contained in each batch

_C.MAGIC_NUMBER = _C.SOLVER.IMS_PER_BATCH // _C.IMAGE_FOLDER_PER_BATCH

_C.NUM_IMAGE_FOLDERS = 15000  # Number of image folders considered in this train run
_C.EPOCHS_PER_FOLDER = 1  # Each folder will get trained for 5 epochs
_C.LR_STEP_SIZE = 7500

_C.MAX_EPOCHS = 5
_C.LR = 1e-06
_C.IDX_NUM = 1
_C.SAVE_IDX = 250
_C.VAL_IDX = 250
_C.WARM_START = False

writer = SummaryWriter()

model = SingleObjectDetector(cfg, writer).cuda().train()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=cfg.LR_STEP_SIZE,
                                            gamma=2)

step = 0


#@TODO: Tidy up train loop
def main():
    print("Starting training....")
    for epoch in range(cfg.MAX_EPOCHS):
        ds = COCODataset(config=cfg)
        datal = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=False,
            num_workers=7)

        ds_val = COCODataset(
            config=cfg,
            anno_file=
            "/home/rex/datasets/coco2017/SiamFCCrop511_anno/val2017.json",
            train_dir="/home/rex/datasets/coco2017/SiamFCCrop511/val2017",
            eval_mode=True)
        datal_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            num_workers=7)

        with EventStorage() as storage:
            print(f"Epoch: {str(epoch)}")
            try:
                loss_loc = 0
                loss_cls = 0
                for idx, data in enumerate(datal):
                    step = step + 1
                    proposals, losses = model(data, data_idx=idx, tmode=True)
                    loss_cls += losses['loss_rpn_cls']
                    loss_loc += losses['loss_rpn_loc']

                    if idx % cfg.IDX_NUM == 0 and idx != 0:
                        mean_loss_cls = loss_cls / cfg.IDX_NUM
                        mean_loss_loc = loss_loc / cfg.IDX_NUM
                        total_loss = mean_loss_cls * 0.25 + mean_loss_loc * 0.75

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        scheduler.step()

                        # if idx % IDX_NUM == 0 and idx != 0:
                        print(f"batch: {idx}, loss_rpn_cls: {mean_loss_cls},\
               loss_rpn_loc: {mean_loss_loc}, total loss: {total_loss}")
                        writer.add_scalar('loss_rpn_cls', mean_loss_cls * 0.5,
                                          step)
                        writer.add_scalar('loss_rpn_loc', mean_loss_loc * 0.5,
                                          step)
                        writer.add_scalar('total_loss', total_loss, step)
                        # writer.add_scalar('lr', scheduler.get_lr()[0], step)

                        loss_loc = 0
                        loss_cls = 0

                    if idx % cfg.SAVE_IDX == 0 and idx != 0:
                        torch.save(
                            {
                                'epoch': epoch,
                                'idx': idx,
                                'cfg': yaml.load(cfg.dump()),
                                'cfg2': cfg.dump(),
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                            },
                            f"/home/rex/workspace/single-object-tracker/checkpoints/model_e{str(epoch)}_i{str(idx)}.pth"
                        )

                    if idx % cfg.VAL_IDX == 0 and idx != 0:
                        # Run validation
                        print("Running validation...")
                        loss_loc_val = []
                        loss_cls_val = []

                        try:
                            for idx, data_val in enumerate(datal_val):
                                proposals_val, losses_val = model(data_val,
                                                                  data_idx=idx,
                                                                  tmode=False)
                                loss_cls_val.append(losses_val['loss_rpn_cls'].
                                                    cpu().detach().numpy())
                                loss_loc_val.append(losses_val['loss_rpn_loc'].
                                                    cpu().detach().numpy())
                                if idx == 2:
                                    break

                            mean_loss_cls_val = np.sum(loss_cls_val) / len(
                                loss_cls_val)
                            mean_loss_loc_val = np.sum(loss_loc_val) / len(
                                loss_loc_val)
                            total_loss_val = mean_loss_cls_val * 0.5 + mean_loss_loc_val * 0.5

                            writer.add_scalar('val_loss/loss_rpn_cls',
                                              mean_loss_cls_val * 0.5, step)
                            writer.add_scalar('val_loss/loss_rpn_loc',
                                              mean_loss_loc * 0.5, step)
                            writer.add_scalar('val_loss/total_loss',
                                              total_loss_val, step)
                        except:
                            mean_loss_cls_val = np.sum(loss_cls_val) / len(
                                loss_cls_val)
                            mean_loss_loc_val = np.sum(loss_loc_val) / len(
                                loss_loc_val)
                            total_loss_val = mean_loss_cls_val * 0.5 + mean_loss_loc_val * 0.5

                            writer.add_scalar('val_loss/loss_rpn_cls',
                                              mean_loss_cls_val * 0.5, step)
                            writer.add_scalar('val_loss/loss_rpn_loc',
                                              mean_loss_loc * 0.5, step)
                            writer.add_scalar('val_loss/total_loss',
                                              total_loss_val, step)
            except Exception as e:
                print("Error:", e)
                continue


if __name__ == "__main__":
    main()
