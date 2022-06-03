import os, re
from glob import glob

if not os.path.isdir("cts_sliced/"):
    os.mkdir("cts_sliced/")
    
if not os.path.isdir("masks_sliced/"):
    os.mkdir("masks_sliced/")

all_cts = sorted(glob("images/*_ct.nii.gz"))
all_masks = sorted(glob("images/*_seg.nii.gz"))

all_images = all_cts + all_masks

for image in all_images:
    subject_id = re.search("volume-covid19-(.+?)_", image).group(1)
    
    if "ct" in image:
        os.system("med2image -i" + " " + image + " " + "-d cts_sliced/" + " " + "-o " + subject_id + ".png")
        
    else:
        os.system("med2image -i" + " " + image + " " + "-d masks_sliced/" + " " + "-o " + subject_id + ".png")