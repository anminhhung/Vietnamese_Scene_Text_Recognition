import numpy as np 
import cv2
from PIL import Image
import torch 
import torchvision.transforms.functional as F

from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.resnet import resnext50_32x4d
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops

# our dataset has two classes only - background and text
# num_classes=2
# device = torch.device("cpu")

# backbone = resnext50_32x4d(pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
# backbone = BackboneWithFPN(backbone = backbone, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, 
#                              in_channels_list=[256, 512, 1024, 2048], out_channels=256, extra_blocks=LastLevelMaxPool())

# model = MaskRCNN(backbone, num_classes).to(device)

def detect_text_area(model, image_path, device):
    model.eval()
    model.half()

    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).half()

    # put the model in evaluation mode
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])

    idx_text = torch.logical_and(prediction[0]['labels'] == 1, prediction[0]["scores"] >= 0.5)
    scores = prediction[0]["scores"][idx_text].cpu().numpy()
    masks = torch.round(prediction[0]["masks"].squeeze(1))[idx_text].cpu().numpy()
    # masks = torch.round(prediction[0]["masks"].squeeze(1)).cpu().numpy()
    boxes = []
    for i in range(masks.shape[0]):
        mask = np.uint8(masks[i])
        contours, _ = cv2.findContours(mask.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        rotrect = cv2.minAreaRect(contour)
        box = np.int32(np.round(cv2.boxPoints(rotrect))) 
        boxes.append(box)

    boxes = np.array(boxes)
    # scores = prediction[0]["scores"].cpu().numpy()


    return boxes, scores 