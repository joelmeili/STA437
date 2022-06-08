import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

def custom_collate(batch):
    images = torch.cat([torch.as_tensor(np.transpose(item_["img"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    segs = torch.cat([torch.as_tensor(np.transpose(item_["seg"], (3, 0, 1, 2))) for item in batch for item_ in item], 0).contiguous()
    
    return [images, segs]

# inference
model = torch.load("models/best_model.pth", map_location = torch.device("cpu"))
test_loader = torch.load("test_loader_for_inference.pth", map_location = torch.device("cpu"))

scores = []
img = []
ground_truth = []
pred_mask = []

for image, mask in test_loader:
    pred = model(image)

    for i in range(len(image)):
        target = mask[i, :, :, :]
        pr_mask = pred[i, :, :, :]
        
        if target.max() > 0:
        
            temp_score = smp.utils.metrics.IoU().forward(pr_mask, target)
            scores.append(temp_score.cpu().numpy())
            img.append(image[i, :, :, :])
            ground_truth.append(target)
            pred_mask.append(pr_mask)        

max_values = np.argsort(scores)[-5:]

img = [img[i] for i in range(len(img)) if i in max_values]
ground_truth = [ground_truth[i] for i in range(len(ground_truth)) if i in max_values]
pred_mask = [pred_mask[i] for i in range(len(pred_mask)) if i in max_values]

print([scores[i] for i in range(len(scores)) if i in max_values])

for i in range(len(max_values)):
    image = img[i]
    truth = ground_truth[i]
    mask = pred_mask[i]

    # image
    plt.imshow(np.transpose(image))
    plt.set_title("")
    plt.savefig("inference/image" + str(i) + ".png")

    # truth
    plt.imshow(np.transpose(truth))
    plt.set_title("Ground truth")
    plt.savefig("inference/truth" + str(i) + ".png")

    # pred
    plt.imshow(np.transpose(mask.detach()))
    plt.set_title("Predicted mask")
    plt.savefig("inference/pred" + str(i) + ".png")