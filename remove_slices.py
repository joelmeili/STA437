import cv2

from glob import glob
from cv2 import imread

all_masks = sorted(glob("masks_sliced/*.png"))
empty_masks = []

for mask in all_masks:
  img = imread(mask, cv2.IMREAD_GRAYSCALE)
  if img.max() == 0:
    empty_masks.append(mask)
    
empty_images = ["cts_sliced/" + mask.split("/")[1] for mask in empty_masks]

print(empty_images)
