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

from source.detect.mask_rcnn import detect_text_area
from utils.visualize import draw_text_bbox

# our dataset has two classes only - background and text
num_classes=2
device = torch.device("cuda")

backbone = resnext50_32x4d(pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
backbone = BackboneWithFPN(backbone = backbone, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, 
                             in_channels_list=[256, 512, 1024, 2048], out_channels=256, extra_blocks=LastLevelMaxPool())

detect_model = MaskRCNN(backbone, num_classes).to(device)
detect_model.load_state_dict(torch.load("models/mask_rcnn/resnext50.pth").to(device)) 

if __name__ == '__main__':
    image_path = "data/demo_images/im1518.jpg"
    list_boxes, list_scores = detect_text_area(detect_model, image_path, device)
    image = cv2.imread(image_path)
    image = draw_text_bbox(image, list_boxes)
    cv2.imwrite("demo.jpg", image)
    print(len(boxes))
