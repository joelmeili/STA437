from platform import architecture
import re, random, torch, cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np

from glob import glob
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
    EnsureType) 

def custom_collate(batch):
    images = torch.cat([torch.as_tensor(np.transpose(item_["img"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    segs = torch.cat([torch.as_tensor(np.transpose(item_["seg"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    
    return [images, segs]

def create_data_splits(train_split, tets_split):
    # get path to all volumes
    images = sorted(glob("images/*_ct.nii.gz"))

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

    return train_files, val_files, test_files

if __name__ == "__main__":    
    # HYPERPARAMETERS
    train_splits = [0.5, 0.7]
    test_split = 0.1
    batch_sizes = [2, 4]
    learning_rates = [1e-4, 1e-5]
    optimizers = ["adam", "SGD"]
    dropout = [None, 0.5]
    architecture = ["resnet34", "resnet50"]
    batch_norm = [True, False]

    train_splits = [0.7]
    batch_sizes = [2]
    learning_rates = [1e-4]
    optimizers = ["adam"]
    
    # define transforms for image and segmentation
    train_transforms = Compose([
        LoadImaged(keys = ["img", "seg"]),
        AddChanneld(keys = ["img", "seg"]),
        ScaleIntensityd(keys = ["img", "seg"]),
        RandCropByPosNegLabeld(
        keys=["img", "seg"], label_key = "seg", spatial_size=[512, 512, 1], pos = 2, neg = 1, num_samples = 8
        ),
        #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        #EnsureTyped(keys=["img", "seg"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys = ["img", "seg"]),
        AddChanneld(keys = ["img", "seg"]),
        ScaleIntensityd(keys = ["img", "seg"]),
        RandCropByPosNegLabeld(
        keys=["img", "seg"], label_key = "seg", spatial_size = [512, 512, 1], pos = 2, neg = 1, num_samples = 8
    )])

    test_transforms = Compose([
        LoadImaged(keys = ["img", "seg"]),
        AddChanneld(keys = ["img", "seg"]),
        ScaleIntensityd(keys = ["img", "seg"]),
        RandCropByPosNegLabeld(
        keys=["img", "seg"], label_key = "seg", spatial_size = [512, 512, 1], pos = 3, neg = 1, num_samples = 8
    )])

    # select best overall model
    ##### best score = 0.0

    for batch_size in batch_sizes:

        for train_split in train_splits:

            train_files, val_files, test_files = create_data_splits(train_split, test_split)
            
            # create a training data loader
            train_ds = monai.data.Dataset(data = train_files, transform = train_transforms)

            # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
            train_loader = DataLoader(
            train_ds,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 8,
            collate_fn = custom_collate,
            pin_memory = torch.cuda.is_available())
            
            # create a validation data loader
            val_ds = monai.data.Dataset(data = val_files, transform = val_transforms)
            val_loader = DataLoader(val_ds, batch_size = batch_size, num_workers = 8, shuffle = False, collate_fn = custom_collate)

            # create a test data loader
            test_ds = monai.data.Dataset(data = test_files, transform = test_transforms)
            test_loader = DataLoader(test_ds, batch_size = batch_size, num_workers = 8, shuffle = False, collate_fn = custom_collate)
            
            for opt in optimizers:

                for lr in learning_rates:

                    model = smp.FPN(encoder_name = "resnet34",
                                classes = 1,
                                encoder_weights = "imagenet",
                                in_channels = 1,
                                activation = "sigmoid",
                                aux_params = dict(
                                    pooling = "avg",
                                    dropout = 0.5,
                                    activation = "sigmoid",
                                    classes = 1))

                    writer = SummaryWriter(log_dir = "runs/" + "batch=" + str(batch_size) + "_train=" + str(train_split) + 
                    "_test=" + str(test_split) + "_opt=" + opt + "_lr=" + str(lr))

                    loss = smp.utils.losses.DiceLoss()

                    metrics = [
                    smp.utils.metrics.IoU(threshold = 0.5)
                    ]

                    if opt == "adam":
                        optimizer = torch.optim.Adam([
                            dict(params = model.parameters(), lr = lr)])

                    elif opt == "SGD":
                        optimizer = torch.optim.SGD([
                            dict(params = model.parameters(), lr = lr, momentum = 0.9)])
                    
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

                        writer.add_scalar("Loss/Train", train_logs["dice_loss"], i)
                        writer.add_scalar("Loss/Valid", valid_logs["dice_loss"], i)

                        writer.add_scalar("Score/Train", train_logs["iou_score"], i)
                        writer.add_scalar("Score/Valid", valid_logs["iou_score"], i)
                        
                        # do something (save model, change lr, etc.)
                        if max_score < valid_logs["iou_score"]:
                            max_score = valid_logs["iou_score"]
                            torch.save(model, "models/batch=" + str(batch_size) + "_train=" + str(train_split) + 
                            "_test=" + str(test_split) + "_opt=" + opt + "_lr=" + str(lr) + ".pth")

                    writer.flush()

                    # inference
                    model = torch.load("models/batch=" + str(batch_size) + "_train=" + str(train_split) + 
                            "_test=" + str(test_split) + "_opt=" + opt + "_lr=" + str(lr) + ".pth")
                    pred = model(test_loader)

                    print(pred, pred.shape)