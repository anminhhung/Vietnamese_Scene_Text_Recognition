import numpy as np 
import cv2
import itertools 
import os 

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from tqdm import tqdm

import torch 
import torchvision.transforms.functional as F
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.resnet import resnext50_32x4d
from torchvision.models.vgg import vgg19_bn
from efficientnet_pytorch import EfficientNet
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
# from model_maskRCNN import *
# from model_maskRCNN import (b7_maskRCNN, b4_maskRCNN, resNext50_maskRCNN, resnet101_maskRCNN, vgg19_maskRCNN, mobile_maskRCNN)

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from sklearn.metrics import pairwise_distances

from source.detect.efficientNet import IntermediateLayerGetter, BackboneWithFPN_B7
from source.detect.mask_rcnn import detect_text_area
from utils.visualize import draw_text_bbox
from utils.helper import crop_text_area, create_output_file
from configs.config import init_config
from libs.box_ensemble import *
from libs.box_ensemble.custom_iou import *

# Vietocr
CFG = init_config()
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = CFG['vietocr']['weights']
config['cnn']['pretrained'] = False
config['device'] = CFG['service']['device']
config['predictor']['beamsearch'] = False
recognition_model = Predictor(config)

# our dataset has two classes only - background and text
num_classes = 2 + 80

# efficientNet b7
device = torch.device(CFG['service']['device'])

# backbone_b7 = EfficientNet.from_name('efficientnet-b7', include_top=False) 
# backbone__b7_fpn = BackboneWithFPN_B7(backbone = backbone_b7, return_layers = {"27":'0', "37":'1', "50":'2', "54":'3'}, 
#                              in_channels_list=[160, 224, 384, 640], out_channels=256, extra_blocks=LastLevelMaxPool())

# detect_model_b7 = MaskRCNN(backbone__b7_fpn, num_classes).to(device)
# detect_model_b7.load_state_dict(torch.load("models/efficientb7_fail.pth")) 

# resnext50 
backbone_resnext50 = resnext50_32x4d(pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
backbone_resnext50_fpn = BackboneWithFPN(backbone = backbone_resnext50, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, 
                             in_channels_list=[256, 512, 1024, 2048], out_channels=256, extra_blocks=LastLevelMaxPool())

detect_model = MaskRCNN(backbone_resnext50_fpn, num_classes).to(device)
detect_model.load_state_dict(torch.load("models/resnext50_crop.pth")) 

# resnet101
# detect_model = resnet101_maskRCNN(82).to(device)
# detect_model.load_state_dict(torch.load("models/resnet101_crop.pth")) 

# detect_model.load_state_dict(torch.load(CFG['mask_rcnn']['model_path'])) 

# iou_thr = 0.5
# skip_box_thr = 0.0001
# sigma = 0.1
# weights = [2, 1]

iou_thr = 0.1
conf_thr = 0.9

def postprocess(result):
    result = [obj for obj in result if len(obj[-1]) > 1 or obj[-1].upper()=='Ô' or obj[-1].isdigit()]
    # result = [obj for obj in result if len(obj[-1]) > 6 or not obj[-1].isdigit()]
    # for x in result:
    #     print(x)
    for i in range(len(result)):
        result[i] = [int(obj) for obj in result[i][:-1]] + [result[i][-1]]
        txt = result[i][-1].split(' ')
        if len(txt) > 1:
            if abs(result[i][2]-result[i][4]) < abs(result[i][2]-result[i][0]):
                x1,y1,x2,y2,x3,y3,x4,y4 = result[i][0:8]
            else:
                x1,y1,x2,y2,x3,y3 = result[i][2:8]
                x4,y4 = result[i][0:2]
            result = result[:i] + result[i+1:]
            xx1 = x1
            xx4 = x4
            deltaw1 = (x2-x1)/len(txt)
            deltaw2 = (x3-x4)/len(txt)
            for j in range(len(txt)):
                result.append([int(xx1), int(y1), int(xx1 + deltaw1), int(y2), int(xx4 + deltaw2), int(y3) ,int(xx4), int(y4), txt[j]])
                xx1 += deltaw1
                xx4 += deltaw2
    return result
        

def predict_image(detected_model, image_name, data_dir="data/TestA", result_dir="predicted", visual_dir="data/visual"):
    image_path = os.path.join(data_dir, image_name)
    image = cv2.imread(image_path)
    # print(image_path)
    # print(detect_text_area(detect_model, image_path, 'cuda'))

    try:
        result = []
        concatenated_detections = []
        list_boxes, _ = detect_text_area(detect_model, image_path, 'cuda')
        for bbox in list_boxes:
            try:
                bbox = bbox.reshape((4,2))
                text_image = crop_text_area(image, bbox)
                text_image_pil = Image.fromarray(text_image)
                result_text, prob = recognition_model.predict(text_image_pil, True)
                # write output file
                if prob > 0.9: 
                    concatenated_detections.append(list(itertools.chain(*bbox.tolist())) + [result_text])
            except Exception as e:
                continue

        concatenated_detections = postprocess(concatenated_detections)
        for res in concatenated_detections:
            result.append("{},{},{},{},{},{},{},{},{}".format(\
                        res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]))

        with open(os.path.join(result_dir, "{}.txt".format(image_name)),'w') as f:
            f.write('\n'.join(result))

    except Exception as e:
        pass

def create_submit(detected_model, image_dir="data/TestA", result_dir="predicted"):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    if not os.path.exists("data/visual"):
        os.mkdir("data/visual")

    for image_name in os.listdir(image_dir):
        # print("predict ", image_name)
        predict_image(detected_model, image_name, data_dir=image_dir, result_dir=result_dir)

    print("[INFO] Done")

if __name__ == "__main__":
    create_submit(detect_model)