import os
import cv2
import json
import numpy as np
import pycocotools.mask
from torch.utils.data import Dataset

import ipdb


class COCODataset(Dataset):
    def __init__(self, root_dir, transforms, input_size, ext="JPG"):

        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "annotations")
        self.transforms = transforms
        self.ext = ext

        self.input_size = input_size

        self.mask_dict = json.load(open(os.path.join(self.mask_dir, "instances_default.json")))

        self.img_list = []
        images = self.mask_dict["images"]
        images.sort(key=lambda img: img["id"])

        for image in images:
            self.img_list.append((image['file_name'], image['id']))

        self.img_list.sort(key=lambda img: img[1])  # Sort the img_list based on image id

        self.classes = [category["name"] for category in self.mask_dict["categories"]]
        self.image_shape = self._get_image(0).shape

        print()
        print("COCO Dataset")
        print("="*10)
        print("Length: ", len(self))
        print("Classes:")
        for idx, class_name in enumerate(self.classes):
            print(idx, '::', class_name)

    def _get_image(self, idx):
        # print(self.img_list[idx][0])
        # cv2.imshow("image", cv2.imread(os.path.join(self.img_dir, self.img_list[idx][0]),0))
        # cv2.waitKey(0)
        return cv2.imread(os.path.join(self.img_dir, self.img_list[idx][0]),cv2.IMREAD_UNCHANGED)

    def get_image(self, idx):
        orig = self._get_image(idx)
        square = cv2.copyMakeBorder(
            orig,
            int((max(orig.shape) - orig.shape[0]) / 2),
            int((max(orig.shape) - orig.shape[0]) / 2), int((max(orig.shape) - orig.shape[1]) / 2),
            int((max(orig.shape) - orig.shape[1]) / 2), cv2.BORDER_CONSTANT, value=(255,255,255)
        )
        image = cv2.resize(square, (self.input_size, self.input_size))
        return image
    
    def get_mask(self, idx):
        component_masks = np.zeros((len(self.classes),) + self._get_image(idx).shape[:2])
        output_masks = np.zeros((len(self.classes), self.input_size, self.input_size))

        for i in range(len(self.classes)):
            for annotation in self.mask_dict["annotations"]:
                if int(annotation["image_id"]) == self.img_list[idx][1]:
                    if int(annotation["category_id"]) == i + 1:
                        if type(annotation["segmentation"]) == dict:
                            try:
                                temp = pycocotools.mask.decode(annotation["segmentation"])
                                temp = cv2.resize(temp, component_masks[i].shape[::-1])
                                component_masks[i] = np.add(component_masks[i], temp)
                            except TypeError:
                                foo = pycocotools.mask.frPyObjects(annotation["segmentation"], annotation["segmentation"]['size'][0], annotation["segmentation"]['size'][1])
                                temp = pycocotools.mask.decode(foo)
                                component_masks[i] = np.add(component_masks[i], temp)
                        else:
                            x_pts = np.array(annotation["segmentation"]).squeeze()[0::2]
                            y_pts = np.array(annotation["segmentation"]).squeeze()[1::2]
                            pts = [np.stack((x_pts, y_pts), 1).astype(int)]

                            component_masks[i] = cv2.drawContours(component_masks[i], pts, -1, color=(1), thickness=-1)

            temp = cv2.copyMakeBorder(
                component_masks[i], 
                int((max(component_masks[i].shape) - component_masks[i].shape[0]) / 2),
                int((max(component_masks[i].shape) - component_masks[i].shape[0]) / 2),
                int((max(component_masks[i].shape) - component_masks[i].shape[1]) / 2),
                int((max(component_masks[i].shape) - component_masks[i].shape[1]) / 2),
                cv2.BORDER_REPLICATE
            )
            output_masks[i] = cv2.resize(temp, (self.input_size, self.input_size))

        output_masks = np.asarray([output_masks for output_masks in output_masks])

        # To get the masks in the format (num_channel, H, W)
        output_masks = output_masks.transpose([1,2,0])

        return output_masks

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        return self.transforms(self.get_image(idx), self.get_mask(idx))

