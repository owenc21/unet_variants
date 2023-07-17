import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim
import os
import cv2
import numpy as np

from utils.transform_utils import MultiCompose, ToTensor, EnforceFloat
from utils.losses import DiceLoss
from utils.geometric_utils import iou
from unet.coco_dataset import COCODataset
# from unet.unet_model import UNet
from unet.unet import UNet
from utils.dice_score import dice_coeff

import ipdb


def train():
    print("beginning training")

    train_dataset_path = os.path.join("/data/iphone_rgb_screw_aug/train")
    validation_dataset_path = os.path.join("/data/iphone_rgb_screw_aug/validation")
    model_save_path = os.path.join("/home/owenc21/multimodal/unet_variants/checkpoints")
    batch_size = 2
    lr = 0.00001
    step_size = 5
    gamma = 0.99
    momentum = 0.999

    # Set transformations
    transforms = MultiCompose(
        transforms=[
            ToTensor(),
            EnforceFloat()
        ]
    )

    # Training dataloader
    training_data = DataLoader(
        COCODataset(
            root_dir=train_dataset_path,
            transforms=transforms,
            input_size=640
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # Validation dataloader
    validation_data = DataLoader(
        COCODataset(
            root_dir=validation_dataset_path,
            transforms=transforms,
            input_size=640
        ),
        batch_size=2,
        shuffle=True
    )

    # Set up device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    # Set up optimizer, loss, and scheduler
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=lr, momentum=momentum
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Training loop
    best_iou = 0.0
    for epoch in range(100):
        num_epoch = epoch+1

        print(f"Epoch: {num_epoch}")
        print(f"TRAINING")
        print('-'*20)

        model.train()

        for idx, (img, gt) in enumerate(training_data):
            """
            img is of shape (N, in_c, h, w)
            where N is the batch size
            in_c is the number of channels (3 for rgb imgs)
            h,w is height and width of image

            gt is of shape N, h, num_c, w
            where N is batch size
            h is height of image
            num_c is the number of classes
            w is width of image
            """
            img, gt = img.to(device, dtype=torch.float32), gt.to(device, dtype=torch.long)

            prediction = model(img)

            # # For visualizaiton purposes
            # for batch in range(img.shape[0]):
            #     img_t = img[batch,:,:,:].cpu().numpy()
            #     img_t = img_t.transpose([1,2,0])
            #     gt_t = gt[batch,:,:,:].cpu().float().numpy()
            #     gt_t = gt_t.transpose([1,2,0])
            #     pred_t = prediction[batch,:,:,:].detach().cpu().numpy()
            #     pred_t = pred_t.transpose([1,2,0])
            #     cv2.imshow('Image', img_t)
            #     cv2.imshow('GT', gt_t)
            #     cv2.imshow('Pred', pred_t)
            #     cv2.waitKey(0)

            loss = loss_fn(prediction.squeeze(1), gt.squeeze(1).float())
            print(f"Loss1: {loss.item()}")
            diceloss = dice_loss(F.sigmoid(prediction.squeeze(1)), gt.squeeze(1).float(), multiclass=False)
            print(f"Dice loss: {diceloss}")
            loss += diceloss

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            # loss.backward()
            # optimizer.step()

            print(f"Training loss: {loss.item()}")
            iou_score = iou(prediction, gt)
            print(f"IoU: {iou_score}")


        # At this point, training epoch is complete, perform validation

        model.eval()

        print(f"VALIDATION")
        print("-"*20)
        for idx, (img, gt) in enumerate(validation_data):
            """
            img is of shape (N, in_c, h, w)
            where N is the batch size
            in_c is the number of channels (3 for rgb imgs)
            h,w is height and width of image

            gt is of shape N, h, num_c, w
            where N is batch size
            h is height of image
            num_c is the number of classes
            w is width of image
            """

            with torch.no_grad():
                img, gt = img.to(device, dtype=torch.float32), gt.to(device, dtype=torch.long)

                prediction = model(img)

                # For visualizaiton purposes
                for batch in range(img.shape[0]):
                    img_t = img[batch,:,:,:].cpu().numpy()
                    img_t = img_t.transpose([1,2,0])
                    gt_t = gt[batch,:,:,:].cpu().float().numpy()
                    gt_t = gt_t.transpose([1,2,0])
                    pred_t = prediction[batch,:,:,:].detach().cpu().numpy()
                    pred_t = pred_t.transpose([1,2,0])
                    cv2.imshow('Image', img_t)
                    cv2.imshow('GT', gt_t)
                    cv2.imshow('Pred', pred_t)
                    cv2.waitKey(0)

                    dice_score = dice_coeff(prediction.squeeze(1), gt.squeeze(1).float())
                    lr_scheduler.step(dice_score)

                iou_score = iou(prediction, gt)
                print(f"IoU: {iou_score}")

                # Save model based on IoU
                if iou_score > best_iou:
                    torch.save(model.state_dict(), os.path.join(model_save_path, "best.pth"))


if __name__ == "__main__":
    train()