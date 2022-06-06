import re, random, torch, cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np

from glob import glob
from albumentations.pytorch import ToTensorV2 as ToTensor
from cv2 import imread
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
Activations,
AddChanneld,
AsDiscrete,
Compose,
LoadImaged,
RandCropByPosNegLabeld,
RandRotate90d,
ScaleIntensityd,
EnsureTyped,
EnsureType,
) 

writer = SummaryWriter()

def custom_collate(batch):
    images = torch.cat([torch.as_tensor(np.transpose(item_["img"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    segs = torch.cat([torch.as_tensor(np.transpose(item_["seg"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    
    return [images, segs]

if __name__ == "__main__":
    images = sorted(glob("images/*_ct.nii.gz"))
    
    # HYPERPARAMETERS
    train_split = 0.7
    test_split = 0.1
    batch_size = 2

    random.seed(2020)
    train_images = random.sample(images, int(train_split * len(images)))
    val_images = list(set(images) - set(train_images))
    test_images = random.sample(val_images, int(test_split * len(images)))
    val_images = list(set(val_images) - set(test_images))
    
    train_segs = [img.replace("ct", "seg") for img in train_images]
    val_segs = [img.replace("ct", "seg") for img in val_images]
    test_segs = [img.replace("ct", "seg") for img in test_images]
    
    train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_segs)]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_segs)]
    test_files = [{"img": img, "seg": seg} for img, seg in zip(test_images, test_segs)]
    
    # define transforms for image and segmentation
    train_transforms = Compose(
    [
    LoadImaged(keys = ["img", "seg"]),
    AddChanneld(keys = ["img", "seg"]),
    ScaleIntensityd(keys = ["img", "seg"]),
    RandCropByPosNegLabeld(
    keys=["img", "seg"], label_key = "seg", spatial_size=[512, 512, 1], pos = 2, neg = 1, num_samples = 8
    ),
    #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
    #EnsureTyped(keys=["img", "seg"]),
    ]
    )
    
    val_transforms = Compose(
    [
    LoadImaged(keys = ["img", "seg"]),
    AddChanneld(keys = ["img", "seg"]),
    ScaleIntensityd(keys = ["img", "seg"]),
    RandCropByPosNegLabeld(
    keys=["img", "seg"], label_key = "seg", spatial_size = [512, 512, 1], pos = 2, neg = 1, num_samples = 8
    )])
    
    # create a training data loader
    train_ds = monai.data.Dataset(data = train_files, transform = train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
    train_ds,
    batch_size = 2,
    shuffle = True,
    num_workers = 4,
    collate_fn = custom_collate,
    pin_memory = torch.cuda.is_available(),
    )
    
    # create a validation data loader
    val_ds = monai.data.Dataset(data = val_files, transform = val_transforms)
    val_loader = DataLoader(val_ds, batch_size = 2, num_workers = 4, collate_fn = custom_collate) 
    
    model = smp.FPN(encoder_name = "resnet34",
                    classes = 1,
                    encoder_weights = "imagenet",
                    in_channels = 1,
                    activation = "sigmoid"
                    )

    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold = 0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr = 1e-3)])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss = loss,
        metrics = metrics,
        optimizer = optimizer,
        device = "cuda",
        verbose = True)

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss = loss,
        metrics = metrics,
        device = "cuda",
        verbose = True)

    # train model for 40 epochs
    max_score = 0

    for i in range(0, 20):

        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        writer.add_scalars("Loss/DiceLoss", {"train": train_logs["dice_loss"],
                                            "valid": valid_logs["dice_loss"]}, i)
        writer.add_scalars("Score/IoU", {"train": train_logs["iou_score"],
                                        "valid": valid_logs["iou_score"]}, i)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]

    writer.flush()