import numpy as np 
import cv2
import os 

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
from configs.config import init_config

# Vietocr
CFG = init_config()
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = CFG['vietocr']['weights']
config['cnn']['pretrained'] = False
config['device'] = CFG['service']['device']
config['predictor']['beamsearch'] = False
recognition_model = Predictor(config)

# our dataset has two classes only - background and text
num_classes=2
device = torch.device(CFG['service']['device'])

backbone = resnext50_32x4d(pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
backbone = BackboneWithFPN(backbone = backbone, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, 
                             in_channels_list=[256, 512, 1024, 2048], out_channels=256, extra_blocks=LastLevelMaxPool())

detect_model = MaskRCNN(backbone, num_classes).to(device)
detect_model.load_state_dict(torch.load(CFG['mask_rcnn']['model_path'])) 

def predict_image(image_name, data_dir="data/TestA", result_dir="predicted", visual_dir="data/visual"):
    image_path = os.path.join(data_dir, image_name)
    image = cv2.imread(image_path)
    try:
        list_boxes, list_scores = detect_text_area(detect_model, image_path, device)
        list_result_text = []
        list_use_boxes = []
        for bbox in list_boxes:
            try:
                text_image = crop_text_area(image, bbox)
                text_image_pil = Image.fromarray(text_image)
                result_text = recognition_model.predict(text_image_pil)
                # write output file
                create_output_file(os.path.join(result_dir, "{}.txt".format(image_name)), bbox, result_text)
                list_use_boxes.append(bbox)
                list_result_text.append(result_text)
            except Exception as e:
                with open("error_maybe_in_bbox.txt", "a+") as f:
                    f.write("{}\t{}\n".format(e, image_name))
                continue
    except Exception as e:
        with open("error_in_detect_module.txt", "a+") as f:
            f.write("{}\t{}\n".format(e, image_name))
        pass

    image = draw_text_bbox(image, list_use_boxes, list_result_text)
    image_des_path = os.path.join(visual_dir, image_name)
    cv2.imwrite(image_des_path, image)

def create_submit(image_dir="data/TestA", result_dir="predicted"):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    if not os.path.exists("data/visual"):
        os.mkdir("data/visual")

    for image_name in os.listdir(image_dir):
        print("[INFO] Predicting {}........".format(image_name))
        predict_image(image_name)

    print("[INFO] Done")

if __name__ == "__main__":
    create_submit()