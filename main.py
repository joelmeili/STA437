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


best_score = 0.0

def custom_collate(batch):
    images = torch.cat([torch.as_tensor(np.transpose(item_["img"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    segs = torch.cat([torch.as_tensor(np.transpose(item_["seg"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    
    return [images, segs]

def create_data_splits(train_split, test_split):
    # get path to all volumes
    images = sorted(glob("images/*_ct.nii.gz"))

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

    return train_files, val_files, test_files

def train_model(train_loader, valid_loader, test_loader, name = "base_line", opt = "adam", 
dropout = 0.2, architecture = "resnet34", lr = 1e-4):
    
    model = smp.FPN(encoder_name = architecture,
                classes = 1,
                encoder_weights = "imagenet",
                in_channels = 1,
                activation = "sigmoid",
                decoder_dropout = dropout)

    writer = SummaryWriter(log_dir = "runs/" + name + ".pth")

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

    test_epoch = smp.utils.train.ValidEpoch(
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
        test_logs = valid_epoch.run(test_loader)

        writer.add_scalar("Loss/Train", train_logs["dice_loss"], i)
        writer.add_scalar("Loss/Valid", valid_logs["dice_loss"], i)

        writer.add_scalar("Score/Train", train_logs["iou_score"], i)
        writer.add_scalar("Score/Valid", valid_logs["iou_score"], i)
        writer.add_scalar("Score/Test", test_logs["iou_score"], i)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, "models/" + name + ".pth")

        if max_score > best_score:
            torch.save(model, "models/best_model.pth")
            torch.save(test_loader, "test_loader_for_inference.pth")

    writer.flush()         


if __name__ == "__main__":       
    # define transforms for image and segmentation
    train_transforms = Compose([
        LoadImaged(keys = ["img", "seg"]),
        AddChanneld(keys = ["img", "seg"]),
        ScaleIntensityd(keys = ["img", "seg"]),
        RandCropByPosNegLabeld(
        keys=["img", "seg"], label_key = "seg", spatial_size=[512, 512, 1], pos = 2, neg = 1, num_samples = 8
        )
    ])

    train_transforms_augmented = Compose([
        LoadImaged(keys = ["img", "seg"]),
        AddChanneld(keys = ["img", "seg"]),
        ScaleIntensityd(keys = ["img", "seg"]),
        RandCropByPosNegLabeld(
        keys=["img", "seg"], label_key = "seg", spatial_size=[512, 512, 1], pos = 2, neg = 1, num_samples = 8
        ),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        EnsureTyped(keys=["img", "seg"]),
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

    # run baseline model
    train_split = 0.7
    test_split = 0.1
    batch_size = 2

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

    # train base model
    train_model(train_loader = train_loader, valid_loader = val_loader, test_loader = test_loader)

    # train different architecture model
    train_model(train_loader = train_loader, valid_loader = val_loader, 
    test_loader = test_loader, architecture = "resnet50", name = "architecture")
    
    # train different learning rate model
    train_model(train_loader = train_loader, valid_loader = val_loader, 
    test_loader = test_loader, lr = 1e-5, name = "learning_rate")

    # train different drop out rate
    train_model(train_loader = train_loader, valid_loader = val_loader, 
    test_loader = test_loader, dropout = 1e-4, name = "drop_out")
    
    # train different optimizer
    train_model(train_loader = train_loader, valid_loader = val_loader, 
    test_loader = test_loader, opt = "SGD", name = "optimizer")

    # train with augmentation model
    train_ds_aug = monai.data.Dataset(data = train_files, transform = train_transforms_augmented)

    train_loader_aug = DataLoader(
    train_ds_aug,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
    collate_fn = custom_collate,
    pin_memory = torch.cuda.is_available())

    train_model(train_loader = train_loader_aug, valid_loader = val_loader,
    test_loader = test_loader, name = "augmented")

    # train different train split
    train_files, val_files, test_files = create_data_splits(train_split = 0.5, test_split = test_split)
    
    train_ds = monai.data.Dataset(data = train_files, transform = train_transforms)

    train_loader = DataLoader(
    train_ds,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
    collate_fn = custom_collate,
    pin_memory = torch.cuda.is_available())
    
    val_ds = monai.data.Dataset(data = val_files, transform = val_transforms)
    val_loader = DataLoader(val_ds, batch_size = batch_size, num_workers = 8, shuffle = False, collate_fn = custom_collate)

    test_ds = monai.data.Dataset(data = test_files, transform = test_transforms)
    test_loader = DataLoader(test_ds, batch_size = batch_size, num_workers = 8, shuffle = False, collate_fn = custom_collate)   

    train_model(train_loader = train_loader, valid_loader = val_loader,
    test_loader = test_loader, name = "split")