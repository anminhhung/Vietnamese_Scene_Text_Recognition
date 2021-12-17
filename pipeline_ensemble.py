import numpy as np 
import cv2
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
from model_maskRCNN import *
from model_maskRCNN import (b7_maskRCNN, b4_maskRCNN, resNext50_maskRCNN, resnet101_maskRCNN, vgg19_maskRCNN, mobile_maskRCNN)

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
    list_result_text = []
    list_use_boxes = []
    final_list_boxes = np.array([])
    final_list_text = []

    try:
        # This snippet is used for wbf codebase      
        # list_boxes_resnet, list_scores_resnet = detect_text_area(detect_model_resnext50, image_path, 'cuda')
        # list_boxes_resnet = list(list_boxes_resnet.reshape((-1,8)))
        # list_boxes_b7, list_scores_b7 = detect_text_area(detect_model_b7, image_path, 'cuda')
        # list_boxes_b7 = list(list_boxes_b7.reshape((-1,8)))
        # list_scores_resnet = list(list_scores_resnet)
        # list_scores_b7 = list(list_scores_b7)
        # label_list = []
        # label_list.append([0 for j in range(len(list_boxes_resnet))])
        # label_list.append([0 for j in range(len(list_boxes_b7))])
        # list_boxes, list_scores, _ = weighted_boxes_fusion([list_boxes_resnet, list_boxes_b7], \
        #                                         [list_scores_resnet, list_scores_b7], \
        #                                         label_list, weights=weights, iou_thr=iou_thr)

        # This snippet is used for new code base 
        result = []
        bboxes = []
        texts = []
        final_list_boxes, _ = detect_text_area(detect_model, image_path, 'cuda')
        for bbox in final_list_boxes:
            try:
                bbox = bbox.reshape((4,2))
                text_image = crop_text_area(image, bbox)
                text_image_pil = Image.fromarray(text_image)
                result_text, prob = recognition_model.predict(text_image_pil, True)
                # write output file
                if prob > 0.9: # best 0.6
                    bboxes.append(bbox)
                    texts.append(result_text)
                    res = postprocess(bbox + [result_text])
                    result.append("{},{},{},{},{},{},{},{},{}".format(
                res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]))
            except Exception as e:
                with open("error_maybe_in_bbox.txt", "a+") as f:
                    f.write("{}\t{}\n".format(e, image_name))
                continue
        with open(os.path.join(result_dir, "{}.txt".format(image_name)),'w') as f:
            f.write('\n'.join(result))

        image = draw_text_bbox(image, bboxes, texts)
        image_des_path = os.path.join(visual_dir, image_name)
        cv2.imwrite(image_des_path, image)

    except Exception as e:
        pass

        # This snippet is used for ensemble b7 and resnet50
    #     list_boxes_resnet, list_scores_resnet = detect_text_area(detect_model_resnext50, image_path, 'cuda')
    #     list_boxes_b7, list_scores_b7 = detect_text_area(detect_model_b7, image_path, 'cuda')
    #     for list_boxes in [list_boxes_resnet, list_boxes_b7]:
    #         list_result_text_ = []
    #         list_use_boxes_ = []
            # for bbox in list_boxes:
            #     try:
            #         bbox = bbox.reshape((4,2))
            #         text_image = crop_text_area(image, bbox)
            #         text_image_pil = Image.fromarray(text_image)
            #         result_text, prob = recognition_model.predict(text_image_pil, True)
            #         # write output file
            #         if prob > 0.9: # best 0.6
            #             list_use_boxes_.append(bbox)
            #             list_result_text_.append(result_text)
            #             with open("prob_text.txt", "a+") as f:
            #                 f.write("{}\t{}\n".format(result_text, prob))

            #     except Exception as e:
            #         with open("error_maybe_in_bbox.txt", "a+") as f:
            #             f.write("{}\t{}\n".format(e, image_name))
            #         continue
            # list_result_text_ = np.array(list_result_text_)
            # list_use_boxes_ = np.array(list_use_boxes_)
            # list_use_boxes.append(list_use_boxes_)
            # list_result_text.append(list_result_text_)
    #     try:
    #         aff_mat = pairwise_distances(list_use_boxes[0].reshape(-1,8), list_use_boxes[1].reshape(-1,8), metric=quadrangle_intersection_over_union)
    #         unmatched_b7_index = np.where(aff_mat.max(axis=0) < iou_thr)
    #         final_list_boxes = np.concatenate((list_use_boxes[0], list_use_boxes[1][unmatched_b7_index]), axis = 0)
    #         final_list_text = np.concatenate((list_result_text[0], list_result_text[1][unmatched_b7_index]), axis = 0)
    #     except Exception as e:
    #         pass 
        
    # except Exception as e:
    #     with open("error_in_detect_module.txt", "a+") as f:
    #         f.write("{}\t{}\n".format(e, image_name))
    #     pass

    # image = draw_text_bbox(image, final_list_boxes, final_list_text)
    # image_des_path = os.path.join(visual_dir, image_name)
    # cv2.imwrite(image_des_path, image)

    # for bbox, result_text in zip(final_list_boxes ,final_list_text):
    #     create_output_file(os.path.join(result_dir, "{}.txt".format(image_name)), bbox, result_text)

def create_submit(detected_model, image_dir="data/TestA", result_dir="predicted"):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    if not os.path.exists("data/visual"):
        os.mkdir("data/visual")

    for image_name in os.listdir(image_dir):
        print("predict ", image_name)
        predict_image(detected_model, image_name)

    print("[INFO] Done")

if __name__ == "__main__":
    create_submit(detect_model)