from unet import UNet
import torch
from torch.utils.data import DataLoader
import torch.optim
from unet_utils import MultiCompose, ToTensor, EnforceFloat
from coco_dataset import COCODataset
import os
import cv2
import numpy as np


def train():
    print("beginning training")

    train_dataset_path = os.path.join("/data/iphone_rgb_screw_aug/train")
    validation_dataset_path = os.path.join("/data/iphone_rgb_screw_aug/validation")

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
        batch_size=4,
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

    # Set up optimizer, loss, and scheduler
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)

    # Training loop
    best_iou = 0.0
    best_iou_recall = 0.0
    for epoch in range(10):
        num_epoch = epoch+1

        print(f"Epoch: {num_epoch}")
        print(f"TRAINING")
        print('-'*20)
        for idx, (img, gt) in enumerate(training_data):
            print(f"Batch {idx}")
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
            img, gt = img.to(device), gt.to(device)

            prediction = model(img)

            # For visualizaiton purposes
            for batch in range(img.shape[0]):
                img_t = img[batch,:,:,:].cpu().numpy()
                img_t = img_t.transpose([1,2,0])
                gt_t = gt[batch,:,:,:].cpu().numpy()
                gt_t = gt_t.transpose([1,2,0])
                pred_t = prediction[batch,:,:,:].detach().cpu().numpy()
                pred_t = pred_t.transpose([1,2,0])
                cv2.imshow('Image', img_t)
                cv2.imshow('GT', gt_t)
                cv2.imshow('Pred', pred_t)
                cv2.waitKey(0)

            loss = loss_fn(prediction, gt)
            print(f"Training loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

        # At this point, training epoch is complete, perform validation

        print(f"VALIDATION")
        print("-"*20)
        for idx, (img, gt) in enumerate(validation_data):
            print(f"Batch {idx}")
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
                img, gt = img.to(device), gt.to(device)

                prediction = model(img)

                # # For visualizaiton purposes
                # for batch in range(img.shape[0]):
                #     img_t = img[batch,:,:,:].cpu().numpy()
                #     img_t = img_t.transpose([1,2,0])
                #     gt_t = gt[batch,:,:,:].cpu().numpy()
                #     gt_t = gt_t.transpose([1,2,0])
                #     pred_t = prediction[batch,:,:,:].detach().cpu().numpy()
                #     pred_t = pred_t.transpose([1,2,0])
                #     cv2.imshow('Image', img_t)
                #     cv2.imshow('GT', gt_t)
                #     cv2.imshow('Pred', pred_t)
                #     cv2.waitKey(0)

                loss = loss_fn(prediction, gt)
                print(f"Validation loss: {loss}")

        


if __name__ == "__main__":
    train()