import numpy as np 
import cv2

from PIL import Image
from shapely.geometry import Polygon

import torch 
import torchvision.transforms.functional as F
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.resnet import resnext50_32x4d
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from source.detect.mask_rcnn import detect_text_area
from utils.visualize import draw_text_bbox
from utils.helper import crop_text_area, create_output_file

# Vietocr
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = "models/vietocr/transformerocr.pth"
config['cnn']['pretrained']=False
config['device'] = 'cuda:0' 
config['predictor']['beamsearch']=False
recognition_model = Predictor(config)

# our dataset has two classes only - background and text
num_classes=2
device = torch.device("cuda")

backbone = resnext50_32x4d(pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
backbone = BackboneWithFPN(backbone = backbone, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, 
                             in_channels_list=[256, 512, 1024, 2048], out_channels=256, extra_blocks=LastLevelMaxPool())

detect_model = MaskRCNN(backbone, num_classes).to(device)
detect_model.load_state_dict(torch.load("models/mask_rcnn/resnext50.pth")) 

def predict_image(image_path)
    image = cv2.imread(image_path)
    list_boxes, list_scores = detect_text_area(detect_model, image_path, device)
    for bbox in list_boxes:
        text_image = crop_text_area(image, bbox)
        text_image_pil = Image.fromarray(text_image)
        result_text = recognition_model.predict(text_image_pil)
        # write output file
        create_output_file("submit.txt", bbox, result_text)

    # image = draw_text_bbox(image, list_boxes)

if __name__ == "__main__":
    