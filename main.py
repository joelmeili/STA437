import re, random, torch, cv2
import albumentations as albu
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

# set transforms
IMAGE_SIZE = 256
writer = SummaryWriter()

train_transforms = albu.Compose(
    [
        albu.Resize(IMAGE_SIZE, IMAGE_SIZE),
        #albu.Rotate(limit=35, p=1.0),
        #albu.HorizontalFlip(p=0.5),
        #albu.VerticalFlip(p=0.1),
        albu.Normalize(),
        ToTensor(transpose_mask=True)
    ]
)

val_transforms = albu.Compose(
    [
        albu.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albu.Normalize(),
        ToTensor(transpose_mask=True)
    ]
)


# data set class
class SegmentationDataSet(Dataset):
    def __init__(self, images, masks, transforms):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = imread(self.images[idx])
        mask = imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        augmented = self.transforms(image=image, mask=mask)
        augmented["mask"] = torch.where(augmented["mask"] == 255, 1, 0)

        return augmented["image"], augmented["mask"].unsqueeze(0)
    """
def custom_collate(batch):
    images = torch.stack([torch.as_tensor(np.transpose(item_, (2, 0, 1))) for item in batch for item_ in item[0]["img"]], 0).contiguous()
    segs = torch.stack([torch.as_tensor(np.transpose(item_, (2, 0, 1))) for item in batch for item_ in item[0]["seg"]], 0).contiguous()
    
    return [images, segs]
    """

def custom_collate(batch):
    images = torch.cat([torch.as_tensor(np.transpose(item_["img"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    segs = torch.cat([torch.as_tensor(np.transpose(item_["seg"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    
    return [images, segs]

if __name__ == "__main__":
    images = sorted(glob("images/*_ct.nii.gz"))
    segs = sorted(glob("images/*_seg.nii.gz"))
    
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:140], segs[:140])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[140:160], segs[140:160])]
    
    # define transforms for image and segmentation
    train_transforms = Compose(
    [
    LoadImaged(keys=["img", "seg"]),
    AddChanneld(keys=["img", "seg"]),
    ScaleIntensityd(keys=["img", "seg"]),
    RandCropByPosNegLabeld(
    keys=["img", "seg"], label_key="seg", spatial_size=[512, 512, 1], pos=2, neg=1, num_samples=8
    ),
    #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
    #EnsureTyped(keys=["img", "seg"]),
    ]
    )
    
    val_transforms = Compose(
    [
    LoadImaged(keys=["img", "seg"]),
    AddChanneld(keys=["img", "seg"]),
    ScaleIntensityd(keys=["img", "seg"]),
    RandCropByPosNegLabeld(
    keys=["img", "seg"], label_key="seg", spatial_size=[512, 512, 1], pos=2, neg=1, num_samples=8
    )])
    
    # create a training data loader
    train_ds = monai.data.Dataset(data = train_files, transform = train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate,
    pin_memory=torch.cuda.is_available(),
    )
    
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=custom_collate) 
    
    """
    all_images = sorted(glob("cts_sliced/*.png"))
    all_masks = sorted(glob("masks_sliced/*.png"))

    # train, validation and test split - 70%, 20% and 10% respectively
    subjects = list(set([re.search("cts_sliced/(.+?)-slice", image).group(1) for image in all_images]))

    random.seed(2020)
    random.shuffle(subjects)

    train_subjects = random.sample(subjects, int(0.7 * len(subjects)))
    val_subjects = list(set(subjects) - set(train_subjects))

    random.shuffle(val_subjects)
    test_subjects = random.sample(val_subjects, int(0.1 * len(subjects)))
    val_subjects = list(set(val_subjects) - set(test_subjects))

    train_images = [[image for image in all_images if subject in image] for subject in train_subjects]
    train_images = [item for sublist in train_images for item in sublist]

    val_images = [[image for image in all_images if subject in image] for subject in val_subjects]
    val_images = [item for sublist in val_images for item in sublist]

    test_images = [[image for image in all_images if subject in image] for subject in test_subjects]
    test_images = [item for sublist in test_images for item in sublist]

    train_masks = [[mask for mask in all_masks if subject in mask] for subject in train_subjects]
    train_masks = [item for sublist in train_masks for item in sublist]

    val_masks = [[mask for mask in all_masks if subject in mask] for subject in val_subjects]
    val_masks = [item for sublist in val_masks for item in sublist]

    test_masks = [[mask for mask in all_masks if subject in mask] for subject in test_subjects]
    test_masks = [item for sublist in test_masks for item in sublist]

    train_set = SegmentationDataSet(
        images=train_images,
        masks=train_masks,
        transforms=train_transforms)

    val_set = SegmentationDataSet(
        images=val_images,
        masks=val_masks,
        transforms=val_transforms)

    train = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=8)
    val = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=8)

    img, mask = next(iter(train))

    img_grid = make_grid(img, nrow=4, ncol=4)
    mask_grid = make_grid(mask, nrow=4, ncol=4)

    plt.imshow(img_grid.permute(1, 2, 0))
    plt.imshow(mask_grid.permute(1, 2, 0), alpha=0.6)
    """
    
    model = smp.FPN(encoder_name="resnet34",
                    classes=1,
                    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    activation='sigmoid'
                    )

    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=1e-4)])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device="cuda",
        verbose=True)

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device="cuda",
        verbose=True)

    # train model for 40 epochs
    max_score = 0

    for i in range(0, 20):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
